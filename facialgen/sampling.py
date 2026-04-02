from __future__ import annotations

import numpy as np


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


def sample_model_walks(
    model,
    *,
    num_samples: int,
    max_length: int,
    bos_token_id: int,
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
            sequences = _sample_constrained_facial_batch(
                model,
                batch_size=cur_batch,
                max_length=max_length,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                model_block_size=model_block_size,
                device=device,
            )
            walks.extend(sequences)

            remaining -= cur_batch
            pbar.update(cur_batch)

    pbar.close()
    return walks
