from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_tree, connected_components, minimum_spanning_tree
from scipy.stats import rankdata
import warnings

from .evaluation import aggregate_transition_scores, transition_count_matrix_from_walks


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


def _edges_to_sparse(edges: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)
    rows = edges[:, 0]
    cols = edges[:, 1]
    data = np.ones(rows.shape[0], dtype=np.float64)
    return sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()


def connected_link_prediction_split(
    A: sp.spmatrix,
    *,
    val_fraction: float = 0.10,
    test_fraction: float = 0.05,
    seed: int | None = None,
) -> dict[str, np.ndarray | sp.csr_matrix]:
    """
    NetGAN-style train/validation/test split for an undirected connected graph.

    This mirrors the logic in `train_val_test_split_adjacency(..., every_node=True,
    connected=True, undirected=True, use_edge_cover=True, set_ops=True)`.
    """
    A = _to_undirected_simple_csr(A)
    if val_fraction + test_fraction <= 0:
        raise ValueError("val_fraction + test_fraction must be positive.")
    if A.nnz == 0:
        raise ValueError("Graph must contain at least one edge.")
    if A.diagonal().sum() != 0:
        raise ValueError("Graph must not contain self-loops.")
    deg = A.sum(0).A1 + A.sum(1).A1
    if np.any(deg == 0):
        raise ValueError("Graph must not contain dangling/isolated nodes.")
    if connected_components(A)[0] != 1:
        raise ValueError("Graph must be connected.")

    rng = np.random.RandomState(seed)
    # NetGAN utils: for undirected graphs only keep one triangular half so
    # each undirected edge is represented once during splitting.
    A_half = sp.tril(A).tocsr()
    A_half.eliminate_zeros()

    E = int(A_half.nnz)
    N = int(A_half.shape[0])
    s_train = int(E * (1.0 - float(val_fraction) - float(test_fraction)))

    # Hold a spanning tree first to keep the training graph connected.
    A_hold = minimum_spanning_tree(A_half)
    A_hold[A_hold > 1] = 1
    A_hold.eliminate_zeros()
    A_sample = A_half - A_hold
    s_train = s_train - int(A_hold.nnz)

    if s_train < 0:
        raise ValueError(
            "Training percentage too low to keep the graph connected after splitting."
        )

    idx_ones = rng.permutation(A_sample.nnz)
    ones = np.column_stack(A_sample.nonzero()).astype(np.int64)
    train_ones = ones[idx_ones[:s_train]]
    held_out_ones = ones[idx_ones[s_train:]]

    # Return back held spanning-tree edges.
    hold_edges = np.column_stack(A_hold.nonzero()).astype(np.int64)
    if hold_edges.size > 0:
        train_ones = np.row_stack((train_ones, hold_edges)).astype(np.int64)

    n_test = int(len(held_out_ones))
    random_sample = rng.randint(0, N, [int(2.3 * n_test), 2])
    random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
    test_zeros = random_sample[A_half[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
    test_zeros = np.row_stack(test_zeros)[:n_test].astype(np.int64)
    if test_zeros.shape[0] != n_test:
        raise RuntimeError("Failed to sample enough non-edges in NetGAN-style split.")

    s_val_ones = int(len(held_out_ones) * float(val_fraction) / (float(val_fraction) + float(test_fraction)))
    s_val_zeros = int(len(test_zeros) * float(val_fraction) / (float(val_fraction) + float(test_fraction)))

    val_edges_half = held_out_ones[:s_val_ones].astype(np.int64)
    test_edges_half = held_out_ones[s_val_ones:].astype(np.int64)
    val_non_edges_half = test_zeros[:s_val_zeros].astype(np.int64)
    test_non_edges_half = test_zeros[s_val_zeros:].astype(np.int64)

    # Symmetrize back to a full undirected training adjacency, but keep the
    # positive/negative edge-pair outputs as single undirected pairs (u < v).
    train_edges_half = train_ones.astype(np.int64)
    if train_edges_half.size == 0:
        train_adj = sp.csr_matrix(A.shape, dtype=np.float64)
    else:
        train_edges_full = np.row_stack(
            (train_edges_half, np.column_stack((train_edges_half[:, 1], train_edges_half[:, 0])))
        ).astype(np.int64)
        train_adj = _edges_to_sparse(train_edges_full, N)
        train_adj = _to_undirected_simple_csr(train_adj)

    # Ensure outputs are upper-triangle undirected pairs to match our scorer.
    def _canon(pairs: np.ndarray) -> np.ndarray:
        pairs = np.asarray(pairs, dtype=np.int64)
        if pairs.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        lo = np.minimum(pairs[:, 0], pairs[:, 1])
        hi = np.maximum(pairs[:, 0], pairs[:, 1])
        return np.column_stack((lo, hi)).astype(np.int64)

    return {
        "train_adj": train_adj,
        "val_edges": _canon(val_edges_half),
        "val_non_edges": _canon(val_non_edges_half),
        "test_edges": _canon(test_edges_half),
        "test_non_edges": _canon(test_non_edges_half),
    }


def connected_train_subsample(
    A_train: sp.spmatrix,
    *,
    train_fraction: float,
    seed: int | None = None,
) -> sp.csr_matrix:
    """
    Keep a connected subset of the train graph edges.

    A BFS spanning tree is always retained. Additional edges are sampled
    uniformly from the remaining removable train edges until the requested
    fraction of train edges is reached.
    """
    A_train = _to_undirected_simple_csr(A_train)
    if not (0.0 < float(train_fraction) <= 1.0):
        raise ValueError("train_fraction must lie in (0, 1].")
    if float(train_fraction) >= 1.0:
        return A_train

    rng = np.random.default_rng(seed)
    all_edges = _upper_triangle_edges(A_train)
    num_edges = int(all_edges.shape[0])
    if num_edges == 0:
        return A_train

    tree = breadth_first_tree(A_train, i_start=0, directed=False)
    tree = _to_undirected_simple_csr(tree)
    tree_edges = {tuple(edge) for edge in _upper_triangle_edges(tree).tolist()}
    removable_edges = [
        tuple(edge) for edge in all_edges.tolist()
        if tuple(edge) not in tree_edges
    ]

    min_keep = len(tree_edges)
    target_keep = max(int(round(float(train_fraction) * num_edges)), min_keep)
    target_keep = min(target_keep, num_edges)
    extra_needed = max(target_keep - min_keep, 0)

    if extra_needed > len(removable_edges):
        extra_needed = len(removable_edges)

    chosen_extra: list[tuple[int, int]] = []
    if extra_needed > 0:
        perm = rng.permutation(len(removable_edges))
        chosen_extra = [removable_edges[i] for i in perm[:extra_needed]]

    kept_edges = np.asarray(sorted([*tree_edges, *chosen_extra]), dtype=np.int64)
    rows = np.concatenate((kept_edges[:, 0], kept_edges[:, 1]))
    cols = np.concatenate((kept_edges[:, 1], kept_edges[:, 0]))
    data = np.ones(rows.shape[0], dtype=np.float64)
    return sp.coo_matrix((data, (rows, cols)), shape=A_train.shape).tocsr()


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
    walk_type: str = "facial",
    score_symmetrization: str | None = None,
) -> dict[str, float]:
    walk_type = str(walk_type)
    if walk_type not in {"random", "facial"}:
        raise ValueError(f"Unsupported walk_type={walk_type!r}")
    if score_symmetrization is None:
        score_symmetrization = "sum" if walk_type == "random" else "none"
    S = aggregate_transition_scores(S, mode=score_symmetrization)
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
    walk_type: str = "facial",
    score_symmetrization: str | None = None,
) -> dict[str, float]:
    S = transition_count_matrix_from_walks(
        walks,
        num_nodes=num_nodes,
        walk_type=walk_type,
    )
    return link_prediction_scores_from_transition_matrix(
        S,
        positive_edges=positive_edges,
        negative_edges=negative_edges,
        walk_type=walk_type,
        score_symmetrization=score_symmetrization,
    )


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
