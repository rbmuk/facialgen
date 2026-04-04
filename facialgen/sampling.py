from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _get_hf_causal_lm(model):
    return getattr(model, "model", None)


def _update_transition_counts(
    counts: dict[tuple[int, int], float],
    sequences: list[list[int]],
    *,
    num_nodes: int,
    walk_type: str,
) -> None:
    walk_type = str(walk_type)
    if not sequences:
        return

    try:
        arr = np.asarray(sequences, dtype=np.int64)
    except ValueError:
        arr = None

    if arr is not None and arr.ndim == 2:
        if walk_type == "facial":
            verts = arr[:, 1:]
            even_len = verts.shape[1] - (verts.shape[1] % 2)
            if even_len > 0:
                u = verts[:, :even_len:2].reshape(-1)
                v = verts[:, 1:even_len:2].reshape(-1)
                valid = (
                    (0 <= u) & (u < num_nodes)
                    & (0 <= v) & (v < num_nodes)
                    & (u != v)
                )
                if np.any(valid):
                    flat = u[valid] * int(num_nodes) + v[valid]
                    uniq, freq = np.unique(flat, return_counts=True)
                    for key, value in zip(uniq.tolist(), freq.tolist()):
                        edge = (int(key // num_nodes), int(key % num_nodes))
                        counts[edge] = counts.get(edge, 0.0) + float(value)
                return
        elif walk_type == "random":
            verts = arr[:, 1:]
            if verts.shape[1] >= 2:
                u = verts[:, :-1].reshape(-1)
                v = verts[:, 1:].reshape(-1)
                valid = (
                    (0 <= u) & (u < num_nodes)
                    & (0 <= v) & (v < num_nodes)
                    & (u != v)
                )
                if np.any(valid):
                    flat = u[valid] * int(num_nodes) + v[valid]
                    uniq, freq = np.unique(flat, return_counts=True)
                    for key, value in zip(uniq.tolist(), freq.tolist()):
                        edge = (int(key // num_nodes), int(key % num_nodes))
                        counts[edge] = counts.get(edge, 0.0) + float(value)
                return

    for sequence in sequences:
        vertices = [int(v) for v in sequence if 0 <= int(v) < num_nodes]
        if walk_type == "facial":
            even_len = len(vertices) - (len(vertices) % 2)
            for idx in range(0, even_len, 2):
                u = int(vertices[idx])
                v = int(vertices[idx + 1])
                if u == v:
                    continue
                counts[(u, v)] = counts.get((u, v), 0.0) + 1.0
        elif walk_type == "random":
            for idx in range(0, max(len(vertices) - 1, 0)):
                u = int(vertices[idx])
                v = int(vertices[idx + 1])
                if u == v:
                    continue
                counts[(u, v)] = counts.get((u, v), 0.0) + 1.0
        else:
            raise ValueError(f"Unsupported walk_type={walk_type!r}")


def _counts_dict_to_csr(
    counts: dict[tuple[int, int], float],
    *,
    num_nodes: int,
) -> sp.csr_matrix:
    if not counts:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    rows = np.fromiter((k[0] for k in counts.keys()), dtype=np.int64)
    cols = np.fromiter((k[1] for k in counts.keys()), dtype=np.int64)
    vals = np.fromiter(counts.values(), dtype=np.float64)
    return sp.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()


def _sample_constrained_facial_batch(
    model,
    *,
    batch_size: int,
    max_length: int,
    bos_token_id: int,
    pad_token_id: int,
    model_block_size: int,
    device,
) -> list[list[int]]:
    import torch

    hf_model = _get_hf_causal_lm(model)
    if hf_model is None or max_length > model_block_size:
        return _sample_constrained_facial_batch_legacy(
            model,
            batch_size=batch_size,
            max_length=max_length,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            model_block_size=model_block_size,
            device=device,
        )

    sequences: list[list[int]] = [[int(bos_token_id)] for _ in range(batch_size)]
    sampled_vertices: list[list[int]] = [[] for _ in range(batch_size)]
    pending_copy = np.zeros(batch_size, dtype=bool)
    finished = np.zeros(batch_size, dtype=bool)

    input_ids = torch.full(
        (batch_size, 1),
        fill_value=int(bos_token_id),
        dtype=torch.long,
        device=device,
    )
    outputs = hf_model(
        input_ids=input_ids,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

    while not np.all(finished):
        need_model_rows = np.flatnonzero(~finished & ~pending_copy)
        if need_model_rows.size > 0:
            masked_logits = next_token_logits[need_model_rows].clone()

            # Only vertex ids are valid generation targets during evaluation.
            # We intentionally disallow EOS here so every sampled sequence grows
            # to the requested cap.
            masked_logits[:, bos_token_id:] = -float("inf")
            for batch_row, row_idx in enumerate(need_model_rows.tolist()):
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
            sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            sampled_tokens = None

        step_tokens = torch.full(
            (batch_size,),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        active_any = False

        for batch_row, row_idx in enumerate(need_model_rows.tolist()):
            token = int(sampled_tokens[batch_row].item())
            sequences[row_idx].append(token)
            sampled_vertices[row_idx].append(token)
            step_tokens[row_idx] = token
            active_any = True

            if len(sampled_vertices[row_idx]) >= 3:
                pending_copy[row_idx] = True
            if len(sequences[row_idx]) >= max_length:
                finished[row_idx] = True
                pending_copy[row_idx] = False

        copy_rows = np.flatnonzero(~finished & pending_copy)
        for row_idx in copy_rows.tolist():
            verts = sampled_vertices[row_idx]
            copied = verts[0] if len(verts) == 3 else verts[-2]
            sequences[row_idx].append(int(copied))
            step_tokens[row_idx] = int(copied)
            pending_copy[row_idx] = False
            active_any = True

            if len(sequences[row_idx]) >= max_length:
                finished[row_idx] = True

        if not active_any:
            break

        outputs = hf_model(
            input_ids=step_tokens.unsqueeze(1),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

    return sequences


def _sample_constrained_facial_batch_legacy(
    model,
    *,
    batch_size: int,
    max_length: int,
    bos_token_id: int,
    pad_token_id: int,
    model_block_size: int,
    device,
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

        for row_idx in range(batch_size):
            if finished[row_idx] or not pending_copy[row_idx]:
                continue
            verts = sampled_vertices[row_idx]
            copied = verts[0] if len(verts) == 3 else verts[-2]
            if len(sequences[row_idx]) >= max_length:
                finished[row_idx] = True
                continue
            sequences[row_idx].append(int(copied))
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
        # to the requested cap.
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
            progressed = True

            sampled_vertices[row_idx].append(token)
            if len(sampled_vertices[row_idx]) >= 3:
                pending_copy[row_idx] = True
            if len(sequences[row_idx]) >= max_length:
                finished[row_idx] = True

        if not progressed:
            break

    return sequences


def _sample_random_walk_batch(
    model,
    *,
    batch_size: int,
    max_length: int,
    bos_token_id: int,
    pad_token_id: int,
    model_block_size: int,
    device,
) -> list[list[int]]:
    import torch

    hf_model = _get_hf_causal_lm(model)
    if hf_model is None or max_length > model_block_size:
        return _sample_random_walk_batch_legacy(
            model,
            batch_size=batch_size,
            max_length=max_length,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            model_block_size=model_block_size,
            device=device,
        )

    sequences: list[list[int]] = [[int(bos_token_id)] for _ in range(batch_size)]

    input_ids = torch.full(
        (batch_size, 1),
        fill_value=int(bos_token_id),
        dtype=torch.long,
        device=device,
    )
    outputs = hf_model(
        input_ids=input_ids,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

    for _ in range(max_length - 1):
        masked_logits = next_token_logits.clone()
        masked_logits[:, bos_token_id:] = -float("inf")

        for row_idx in range(batch_size):
            seq = sequences[row_idx]
            if len(seq) >= 2:
                prev_vertex = int(seq[-1])
                if 0 <= prev_vertex < bos_token_id:
                    masked_logits[row_idx, prev_vertex] = -float("inf")

        probs = torch.softmax(masked_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        for row_idx in range(batch_size):
            sequences[row_idx].append(int(next_tokens[row_idx].item()))

        outputs = hf_model(
            input_ids=next_tokens.unsqueeze(1),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

    return sequences


def _sample_random_walk_batch_legacy(
    model,
    *,
    batch_size: int,
    max_length: int,
    bos_token_id: int,
    pad_token_id: int,
    model_block_size: int,
    device,
) -> list[list[int]]:
    import torch

    sequences: list[list[int]] = [[int(bos_token_id)] for _ in range(batch_size)]

    while True:
        active_rows = [
            row_idx for row_idx in range(batch_size)
            if len(sequences[row_idx]) < max_length
        ]
        if not active_rows:
            break

        prompt_lengths = [
            min(len(sequences[row_idx]), model_block_size)
            for row_idx in active_rows
        ]
        prompt_max_len = max(prompt_lengths)
        input_ids = torch.full(
            (len(active_rows), prompt_max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(active_rows), prompt_max_len),
            dtype=torch.bool,
            device=device,
        )
        for batch_row, row_idx in enumerate(active_rows):
            prompt = sequences[row_idx][-model_block_size:]
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
            torch.arange(len(active_rows), device=device),
            torch.tensor(prompt_lengths, device=device) - 1,
        ]
        masked_logits = next_token_logits.clone()
        masked_logits[:, bos_token_id:] = -float("inf")

        for batch_row, row_idx in enumerate(active_rows):
            seq = sequences[row_idx]
            if len(seq) >= 2:
                prev_vertex = int(seq[-1])
                if 0 <= prev_vertex < bos_token_id:
                    masked_logits[batch_row, prev_vertex] = -float("inf")

        probs = torch.softmax(masked_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()

        for batch_row, row_idx in enumerate(active_rows):
            sequences[row_idx].append(int(next_tokens[batch_row]))

    return sequences


def sample_model_walks(
    model,
    *,
    num_samples: int,
    max_length: int,
    bos_token_id: int,
    device,
    walk_type: str = "facial",
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
    walk_type = str(walk_type)
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
        total=remaining,
        desc=progress_desc,
        disable=not show_progress,
        unit="walk",
    )

    with torch.inference_mode():
        while remaining > 0:
            cur_batch = min(batch_size, remaining)
            if walk_type == "facial":
                sequences = _sample_constrained_facial_batch(
                    model,
                    batch_size=cur_batch,
                    max_length=max_length,
                    bos_token_id=bos_token_id,
                    pad_token_id=pad_token_id,
                    model_block_size=model_block_size,
                    device=device,
                )
            elif walk_type == "random":
                sequences = _sample_random_walk_batch(
                    model,
                    batch_size=cur_batch,
                    max_length=max_length,
                    bos_token_id=bos_token_id,
                    pad_token_id=pad_token_id,
                    model_block_size=model_block_size,
                    device=device,
                )
            else:
                raise ValueError(f"Unsupported walk_type={walk_type!r}")
            walks.extend(sequences)

            remaining -= cur_batch
            pbar.update(cur_batch)

    pbar.close()
    return walks


def sample_model_transition_counts(
    model,
    *,
    num_samples: int,
    max_length: int,
    bos_token_id: int,
    num_nodes: int,
    device,
    walk_type: str = "facial",
    batch_size: int = 128,
    show_progress: bool = False,
    progress_desc: str = "sampling walks",
    log_every_samples: int | None = None,
) -> sp.csr_matrix:
    """
    Sample token sequences batch-by-batch and accumulate the transition count
    matrix directly, without retaining all sampled walks in memory.
    """
    import torch
    from tqdm.auto import tqdm

    model.eval()
    remaining = int(num_samples)
    walk_type = str(walk_type)
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
    counts: dict[tuple[int, int], float] = {}
    pbar = tqdm(
        total=remaining,
        desc=progress_desc,
        disable=not show_progress,
        unit="walk",
    )

    with torch.inference_mode():
        while remaining > 0:
            cur_batch = min(batch_size, remaining)
            if walk_type == "facial":
                sequences = _sample_constrained_facial_batch(
                    model,
                    batch_size=cur_batch,
                    max_length=max_length,
                    bos_token_id=bos_token_id,
                    pad_token_id=pad_token_id,
                    model_block_size=model_block_size,
                    device=device,
                )
            elif walk_type == "random":
                sequences = _sample_random_walk_batch(
                    model,
                    batch_size=cur_batch,
                    max_length=max_length,
                    bos_token_id=bos_token_id,
                    pad_token_id=pad_token_id,
                    model_block_size=model_block_size,
                    device=device,
                )
            else:
                raise ValueError(f"Unsupported walk_type={walk_type!r}")
            _update_transition_counts(
                counts,
                sequences,
                num_nodes=num_nodes,
                walk_type=walk_type,
            )
            remaining -= cur_batch
            pbar.update(cur_batch)
            if log_every_samples is not None and log_every_samples > 0:
                sampled = int(num_samples) - int(remaining)
                prev_sampled = sampled - int(cur_batch)
                if sampled == int(num_samples) or sampled // log_every_samples > prev_sampled // log_every_samples:
                    print(f"{progress_desc}: sampled {sampled}/{int(num_samples)}")

    pbar.close()
    return _counts_dict_to_csr(counts, num_nodes=num_nodes)
