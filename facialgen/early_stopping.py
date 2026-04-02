from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_tree
from scipy.stats import rankdata

from .evaluation import symmetrize_transition_scores, transition_count_matrix_from_walks


def _to_undirected_simple_csr(A: sp.spmatrix) -> sp.csr_matrix:
    A = sp.csr_matrix(A)
    A = A.maximum(A.T).astype(np.float64)
    A.setdiag(0.0)
    A.eliminate_zeros()
    if A.nnz > 0:
        A.data[:] = 1.0
    return A


def _upper_triangle_edges(A: sp.spmatrix) -> np.ndarray:
    upper = sp.triu(_to_undirected_simple_csr(A), k=1).tocoo()
    if upper.nnz == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack((upper.row, upper.col)).astype(np.int64)


def _sample_non_edges(
    A: sp.spmatrix,
    *,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    A = _to_undirected_simple_csr(A)
    n = A.shape[0]
    chosen: set[tuple[int, int]] = set()
    existing = {tuple(edge) for edge in _upper_triangle_edges(A).tolist()}

    while len(chosen) < num_samples:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u == v:
            continue
        edge = (u, v) if u < v else (v, u)
        if edge in existing or edge in chosen:
            continue
        chosen.add(edge)

    return np.asarray(sorted(chosen), dtype=np.int64)


def connected_link_prediction_split(
    A: sp.spmatrix,
    *,
    val_fraction: float = 0.10,
    test_fraction: float = 0.05,
    seed: int | None = None,
) -> dict[str, np.ndarray | sp.csr_matrix]:
    """
    Split edges into connected train graph, validation edges, and test edges.

    The train graph is kept connected by retaining a BFS spanning tree and
    sampling validation/test edges only from the remaining edges.
    """
    A = _to_undirected_simple_csr(A)
    rng = np.random.default_rng(seed)

    all_edges = _upper_triangle_edges(A)
    num_edges = int(all_edges.shape[0])
    if num_edges == 0:
        raise ValueError("Graph must contain at least one edge.")

    tree = breadth_first_tree(A, i_start=0, directed=False)
    tree = _to_undirected_simple_csr(tree)
    tree_edges = {tuple(edge) for edge in _upper_triangle_edges(tree).tolist()}
    removable_edges = [
        tuple(edge) for edge in all_edges.tolist()
        if tuple(edge) not in tree_edges
    ]

    num_val = int(round(val_fraction * num_edges))
    num_test = int(round(test_fraction * num_edges))
    num_holdout = num_val + num_test
    if num_holdout > len(removable_edges):
        raise ValueError(
            "Not enough removable edges to keep the train graph connected while "
            "holding out the requested validation/test fractions."
        )

    perm = rng.permutation(len(removable_edges))
    held_out = [removable_edges[i] for i in perm[:num_holdout]]
    val_edges = np.asarray(held_out[:num_val], dtype=np.int64)
    test_edges = np.asarray(held_out[num_val:], dtype=np.int64)

    holdout_set = {tuple(edge) for edge in held_out}
    train_edges = np.asarray(
        [tuple(edge) for edge in all_edges.tolist() if tuple(edge) not in holdout_set],
        dtype=np.int64,
    )

    rows = np.concatenate((train_edges[:, 0], train_edges[:, 1]))
    cols = np.concatenate((train_edges[:, 1], train_edges[:, 0]))
    data = np.ones(rows.shape[0], dtype=np.float64)
    train_adj = sp.coo_matrix((data, (rows, cols)), shape=A.shape).tocsr()

    val_non_edges = _sample_non_edges(A, num_samples=len(val_edges), rng=rng)
    test_non_edges = _sample_non_edges(A, num_samples=len(test_edges), rng=rng)

    return {
        "train_adj": train_adj,
        "val_edges": val_edges,
        "val_non_edges": val_non_edges,
        "test_edges": test_edges,
        "test_non_edges": test_non_edges,
    }


def edge_overlap_ratio(
    A_generated: sp.spmatrix,
    A_reference: sp.spmatrix,
) -> float:
    """
    Compute edge overlap as |E_gen ∩ E_ref| / |E_ref|.
    """
    A_generated = _to_undirected_simple_csr(A_generated)
    A_reference = _to_undirected_simple_csr(A_reference)

    ref_edges = _upper_triangle_edges(A_reference)
    gen_edges = {
        tuple(edge) for edge in _upper_triangle_edges(A_generated).tolist()
    }
    if ref_edges.shape[0] == 0:
        return 0.0

    overlap = sum(tuple(edge) in gen_edges for edge in ref_edges.tolist())
    return float(overlap / ref_edges.shape[0])


def _scores_for_edge_pairs(S: sp.spmatrix, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.asarray(S[pairs[:, 0], pairs[:, 1]]).ravel().astype(np.float64)


def roc_auc_score_from_edge_scores(
    pos_scores: Sequence[float],
    neg_scores: Sequence[float],
) -> float:
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    if pos_scores.size == 0 or neg_scores.size == 0:
        return float("nan")

    y = np.concatenate((np.ones_like(pos_scores), np.zeros_like(neg_scores)))
    scores = np.concatenate((pos_scores, neg_scores))
    ranks = rankdata(scores)
    n_pos = pos_scores.size
    n_neg = neg_scores.size
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_from_edge_scores(
    pos_scores: Sequence[float],
    neg_scores: Sequence[float],
) -> float:
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    if pos_scores.size == 0 or neg_scores.size == 0:
        return float("nan")

    y_true = np.concatenate((np.ones_like(pos_scores), np.zeros_like(neg_scores)))
    scores = np.concatenate((pos_scores, neg_scores))
    order = np.argsort(-scores, kind="stable")
    y_sorted = y_true[order]
    scores_sorted = scores[order]

    # Group tied scores so AP is not artificially inflated by the original
    # positive-first concatenation order.
    total_pos = float(pos_scores.size)
    tp = 0.0
    fp = 0.0
    prev_recall = 0.0
    ap = 0.0

    idx = 0
    while idx < y_sorted.size:
        score = scores_sorted[idx]
        j = idx
        block_tp = 0.0
        block_fp = 0.0
        while j < y_sorted.size and scores_sorted[j] == score:
            if y_sorted[j] == 1:
                block_tp += 1.0
            else:
                block_fp += 1.0
            j += 1

        tp += block_tp
        fp += block_fp
        recall = tp / total_pos
        precision = tp / max(tp + fp, 1.0)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        idx = j

    return float(ap)


def link_prediction_scores_from_transition_matrix(
    S: sp.spmatrix,
    *,
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
) -> dict[str, float]:
    S = symmetrize_transition_scores(S)
    pos_scores = _scores_for_edge_pairs(S, positive_edges)
    neg_scores = _scores_for_edge_pairs(S, negative_edges)
    return {
        "roc_auc": roc_auc_score_from_edge_scores(pos_scores, neg_scores),
        "average_precision": average_precision_from_edge_scores(
            pos_scores,
            neg_scores,
        ),
    }


def link_prediction_scores_from_walks(
    walks: Iterable[Sequence[int]],
    *,
    num_nodes: int,
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
) -> dict[str, float]:
    S = transition_count_matrix_from_walks(walks, num_nodes=num_nodes)
    return link_prediction_scores_from_transition_matrix(
        S,
        positive_edges=positive_edges,
        negative_edges=negative_edges,
    )


def _sample_constrained_facial_batch(
    model,
    *,
    batch_size: int,
    max_length: int,
    bos_token_id: int,
    pad_token_id: int,
    model_block_size: int,
    device,
    progress_bar=None,
) -> list[list[int]]:
    import torch

    def make_prompt_tokens(sequence: list[int]) -> list[int]:
        if len(sequence) <= model_block_size:
            return list(sequence)
        payload_capacity = max(model_block_size - 1, 0)
        suffix_len = payload_capacity - (payload_capacity % 2)
        if suffix_len == 0:
            return [int(bos_token_id)]
        return [int(bos_token_id), *sequence[-suffix_len:]]

    sequences: list[list[int]] = [[int(bos_token_id)] for _ in range(batch_size)]
    sampled_vertices: list[list[int]] = [[] for _ in range(batch_size)]
    pending_copy = np.zeros(batch_size, dtype=bool)
    finished = np.zeros(batch_size, dtype=bool)

    while not np.all(finished):
        progressed = False
        num_appended_tokens = 0

        for row_idx in range(batch_size):
            if finished[row_idx] or not pending_copy[row_idx]:
                continue
            verts = sampled_vertices[row_idx]
            copied = verts[0] if len(verts) == 3 else verts[-2]
            if len(sequences[row_idx]) >= max_length:
                finished[row_idx] = True
                continue
            sequences[row_idx].append(int(copied))
            num_appended_tokens += 1
            pending_copy[row_idx] = False
            progressed = True
            if len(sequences[row_idx]) >= max_length:
                finished[row_idx] = True

        need_model_rows = [
            row_idx for row_idx in range(batch_size)
            if not finished[row_idx] and not pending_copy[row_idx]
        ]
        if not need_model_rows:
            if not progressed:
                break
            continue

        prompt_lengths = [
            min(len(sequences[row_idx]), model_block_size)
            for row_idx in need_model_rows
        ]
        prompt_max_len = max(prompt_lengths)
        input_ids = torch.full(
            (len(need_model_rows), prompt_max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(need_model_rows), prompt_max_len),
            dtype=torch.bool,
            device=device,
        )
        for batch_row, row_idx in enumerate(need_model_rows):
            prompt = make_prompt_tokens(sequences[row_idx])
            prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device)
            input_ids[batch_row, : prompt_tensor.numel()] = prompt_tensor
            attention_mask[batch_row, : prompt_tensor.numel()] = True

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=None,
        )
        logits = outputs["logits"]
        next_token_logits = logits[
            torch.arange(len(need_model_rows), device=device),
            torch.tensor(prompt_lengths, device=device) - 1,
        ]
        masked_logits = next_token_logits.clone()

        # Only vertex ids are valid generation targets during evaluation.
        # We intentionally disallow EOS here so every sampled sequence grows
        # to the requested graph-scale budget.
        masked_logits[:, bos_token_id:] = -float("inf")
        for batch_row, row_idx in enumerate(need_model_rows):
            verts = sampled_vertices[row_idx]
            if not verts:
                continue
            if len(verts) <= 2:
                partner = int(verts[0])
            else:
                partner = int(verts[-1])
            if 0 <= partner < bos_token_id:
                masked_logits[batch_row, partner] = -float("inf")
        probs = torch.softmax(masked_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()

        for batch_row, row_idx in enumerate(need_model_rows):
            token = int(next_tokens[batch_row])
            sequences[row_idx].append(token)
            num_appended_tokens += 1
            progressed = True

            sampled_vertices[row_idx].append(token)
            if len(sampled_vertices[row_idx]) >= 3:
                pending_copy[row_idx] = True
            if len(sequences[row_idx]) >= max_length:
                finished[row_idx] = True

        if not progressed:
            break

        if progress_bar is not None and num_appended_tokens > 0:
            progress_bar.update(int(num_appended_tokens))

    return sequences


def sample_model_walks(
    model,
    *,
    num_samples: int,
    max_length: int,
    bos_token_id: int,
    eos_token_id: int | None,
    device,
    batch_size: int = 128,
    show_progress: bool = False,
    progress_desc: str = "sampling walks",
) -> list[list[int]]:
    """
    Sample token sequences with constrained facial-walk decoding.

    The faithful facial-walk encoding alternates model-chosen vertex tokens with
    deterministic copied vertices that preserve the dart structure. This decoder
    only samples at the model-chosen positions, inserts the copied vertices
    itself, and disallows `EOS` so generation runs to the requested cap.
    """
    import torch
    from tqdm.auto import tqdm

    model.eval()
    walks: list[list[int]] = []
    remaining = int(num_samples)
    total_token_budget = max(0, remaining * max(max_length - 1, 0))
    hf_config = getattr(model, "hf_config", None)
    model_block_size = int(
        getattr(getattr(model, "config", None), "block_size", 0)
        or getattr(hf_config, "max_position_embeddings", 0)
        or max_length
    )
    pad_token_id = int(
        getattr(getattr(model, "config", None), "pad_token_id", -1)
        if getattr(getattr(model, "config", None), "pad_token_id", None) is not None
        else getattr(hf_config, "pad_token_id", bos_token_id)
    )
    pbar = tqdm(
        total=total_token_budget,
        desc=progress_desc,
        disable=not show_progress,
        unit="tok",
    )

    with torch.inference_mode():
        while remaining > 0:
            cur_batch = min(batch_size, remaining)
            sequences = _sample_constrained_facial_batch(
                model,
                batch_size=cur_batch,
                max_length=max_length,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                model_block_size=model_block_size,
                device=device,
                progress_bar=pbar,
            )
            walks.extend(sequences)

            remaining -= cur_batch
            if show_progress:
                pbar.set_postfix_str(
                    f"walks={len(walks)}/{num_samples}",
                    refresh=False,
                )

    pbar.close()
    return walks


@dataclass
class EarlyStoppingState:
    mode: str
    patience: int
    min_delta: float = 0.0
    best_value: float = -float("inf")
    best_step: int = -1
    num_bad_evals: int = 0

    def update(self, value: float, *, step: int) -> bool:
        improved = value > self.best_value + self.min_delta
        if improved:
            self.best_value = value
            self.best_step = step
            self.num_bad_evals = 0
            return False

        self.num_bad_evals += 1
        return self.num_bad_evals >= self.patience
