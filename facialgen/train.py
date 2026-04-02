from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .curvature import largest_connected_component, resistance_curvature
from .data import (
    CyclicFaceChunkDataset,
    FacialWalkVertexDataset,
    load_graph_dataset_sparse,
    make_face_chunk_dataloader,
)
from .early_stopping import (
    EarlyStoppingState,
    connected_link_prediction_split,
    edge_overlap_ratio,
    link_prediction_scores_from_walks,
    sample_model_walks,
)
from .evaluation import compute_graph_statistics, reconstruct_graph_from_generated_walks
from .models import FacialGen, FacialGenConfig


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="coraml",
        choices=["coraml", "cora_ml", "citeseer", "polblogs"],
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-sign-configs", type=int, default=8)
    parser.add_argument("--sign-seed", type=int, default=2026)
    parser.add_argument("--epoch-seed", type=int, default=99)
    parser.add_argument("--vertex-context-size", type=int, default=32)
    parser.add_argument("--dart-stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--resume-from-latest", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--early-stop-mode",
        type=str,
        choices=["none", "val", "edge_overlap"],
        default="none",
    )
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--eval-generated-walks", type=int, default=4096)
    parser.add_argument("--eval-max-length", type=int, default=None)
    parser.add_argument("--target-edge-overlap", type=float, default=0.5)
    parser.add_argument("--use-link-prediction-split", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 style facial-walk language model on CoraML."
    )
    add_training_args(parser)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _vertex_context_size_from_args(args: argparse.Namespace) -> int:
    if hasattr(args, "vertex_context_size"):
        return int(args.vertex_context_size)
    return int(args.context_size)


def _dart_stride_from_args(args: argparse.Namespace) -> int | None:
    if hasattr(args, "dart_stride"):
        value = args.dart_stride
    else:
        value = getattr(args, "stride", None)
    return None if value is None else int(value)


def default_face_generation_max_length(
    vertex_context_size: int,
) -> int:
    """
    Default short-fragment generation cap.

    Evaluation generation should stay aligned with the training regime:
    `BOS` followed by as many complete darts as fit in the vertex context.
    Since one dart is represented by two vertex tokens and we reserve one token
    for `BOS`, the cap is:

        1 + 2 * floor((vertex_context_size - 1) / 2)

    So with `vertex_context_size = 17`, evaluation generation emits exactly
    `8` darts (`17` tokens total including `BOS`).
    """
    vertex_context_size = int(vertex_context_size)
    desired_darts = max(vertex_context_size - 1, 0) // 2
    return max(3, 1 + 2 * desired_darts)


def build_training_objects(args: argparse.Namespace) -> tuple[
    CyclicFaceChunkDataset,
    torch.utils.data.DataLoader,
    FacialGen,
    dict[str, object],
]:
    vertex_context_size = _vertex_context_size_from_args(args)
    dart_stride = _dart_stride_from_args(args)
    A_full, X, y = load_graph_dataset_sparse(
        args.dataset_name,
        data_dir=args.data_dir,
    )
    A_lcc, nodes_lcc = largest_connected_component(A_full)
    n_lcc = A_lcc.shape[0]
    ref_num_edges = int(sp.triu(A_lcc, k=1).nnz)

    lp_split = None
    train_adj = A_lcc
    if args.early_stop_mode == "val" or args.use_link_prediction_split:
        lp_split = connected_link_prediction_split(
            A_lcc,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.split_seed,
        )
        train_adj = lp_split["train_adj"]
        split_reason = (
            "VAL early stopping"
            if args.early_stop_mode == "val"
            else "link-prediction evaluation"
        )
        print(
            f"Using connected train split for {split_reason}: "
            f"train_edges={int(sp.triu(train_adj, k=1).nnz)}, "
            f"val_edges={len(lp_split['val_edges'])}, "
            f"test_edges={len(lp_split['test_edges'])}"
        )

    curvature = resistance_curvature(
        train_adj,
        use_lcc=False,
        solver="lstsq",
        rhs_scale=float(n_lcc),
    )

    face_ds = FacialWalkVertexDataset(
        train_adj,
        curvature,
        num_sign_configs=args.num_sign_configs,
        sign_seed=args.sign_seed,
    )
    chunk_ds = CyclicFaceChunkDataset(
        face_ds,
        vertex_context_size=vertex_context_size,
        dart_stride=dart_stride,
        epoch_seed=args.epoch_seed,
    )
    loader = make_face_chunk_dataloader(
        chunk_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = FacialGen(
        FacialGenConfig(
            vocab_size=chunk_ds.pad_token_id + 1,
            block_size=vertex_context_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            bos_token_id=face_ds.bos_token_id,
            eos_token_id=face_ds.eos_token_id,
            pad_token_id=chunk_ds.pad_token_id,
        )
    )

    print(f"Dataset: {args.dataset_name}")
    print(f"LCC nodes: {n_lcc}")
    print(f"Full face sequences: {len(face_ds)}")
    print(f"Chunk samples @ T={vertex_context_size}: {len(chunk_ds)}")
    print(f"Dart stride: {chunk_ds.dart_stride}")
    print(
        f"Vocab: {face_ds.vocab_size + 1} "
        f"(vertices + BOS + EOS + PAD)"
    )

    eval_info: dict[str, object] = {
        "reference_adj": A_lcc,
        "reference_labels": np.asarray(y)[nodes_lcc],
        "train_adj": train_adj,
        "num_nodes": n_lcc,
        "num_reference_edges": ref_num_edges,
        "bos_token_id": face_ds.bos_token_id,
        "eos_token_id": face_ds.eos_token_id,
        "link_prediction_split": lp_split,
    }

    return chunk_ds, loader, model, eval_info


def maybe_save_checkpoint(
    model: FacialGen,
    optimizer: AdamW,
    epoch: int,
    save_dir: str | None,
) -> None:
    if save_dir is None:
        return

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = out_dir / f"epoch_{epoch:03d}"
    model.save_pretrained(str(model_dir))
    torch.save(optimizer.state_dict(), out_dir / f"optimizer_epoch_{epoch:03d}.pt")


def maybe_resume_training(
    model: FacialGen,
    optimizer: AdamW,
    save_dir: str | None,
    resume_from_latest: bool,
    device: torch.device,
) -> int:
    if save_dir is None or not resume_from_latest:
        return 0

    out_dir = Path(save_dir)
    if not out_dir.exists():
        return 0

    epoch_dirs = sorted(out_dir.glob("epoch_*"))
    if not epoch_dirs:
        return 0

    latest_dir = epoch_dirs[-1]
    try:
        latest_epoch = int(latest_dir.name.split("_")[-1])
    except ValueError:
        return 0

    optimizer_path = out_dir / f"optimizer_epoch_{latest_epoch:03d}.pt"
    if not optimizer_path.exists():
        return 0

    resumed_model = FacialGen.from_pretrained(str(latest_dir))
    model.load_state_dict(resumed_model.state_dict())
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
    print(f"Resuming from checkpoint: {latest_dir}")
    return latest_epoch


def save_final_training_artifacts(
    model: FacialGen,
    history: list[dict[str, float]],
    args: argparse.Namespace,
    save_dir: str | None,
) -> None:
    if save_dir is None:
        return

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_dir = out_dir / "final"
    model.save_pretrained(str(final_dir))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "train_args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True)
    )


def save_history_snapshot(
    history: list[dict[str, float]],
    save_dir: str | None,
) -> None:
    if save_dir is None:
        return

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))


def add_generated_graph_stats_to_epoch_record(
    epoch_record: dict[str, float],
    A_hat: sp.csr_matrix,
    *,
    reference_labels: np.ndarray | None,
) -> None:
    graph_stats = compute_graph_statistics(A_hat, labels=reference_labels)
    for key, value in graph_stats.items():
        if value is None:
            continue
        epoch_record[f"generated_{key}"] = float(value)


def train_model(
    args: argparse.Namespace,
) -> tuple[FacialGen, dict[str, object], list[dict[str, float]]]:
    if hasattr(args, "seed") and args.seed is not None:
        seed_everything(args.seed)
    vertex_context_size = _vertex_context_size_from_args(args)
    device = resolve_device(args.device)
    chunk_ds, loader, model, eval_info = build_training_objects(args)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = maybe_resume_training(
        model,
        optimizer,
        args.save_dir,
        getattr(args, "resume_from_latest", False),
        device,
    )

    print(f"Training on device: {device}")
    print(
        f"Model config: layers={args.n_layer}, heads={args.n_head}, "
        f"embd={args.n_embd}, dropout={args.dropout}"
    )

    early_state = None
    if args.early_stop_mode == "val":
        early_state = EarlyStoppingState(
            mode="val",
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
        )

    eval_max_length = (
        int(args.eval_max_length)
        if args.eval_max_length is not None
        else default_face_generation_max_length(vertex_context_size)
    )
    print(f"Eval generation max_length: {eval_max_length}")
    history: list[dict[str, float]] = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        chunk_ds.set_epoch(epoch)

        running_loss = 0.0
        running_tokens = 0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]
            if loss is None:
                raise RuntimeError("Model returned no loss.")

            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                valid_targets = (labels[:, 1:] != -100).sum().item()
                running_loss += float(loss.detach()) * max(valid_targets, 1)
                running_tokens += max(valid_targets, 1)

            if step % args.log_every == 0 or step == len(loader):
                mean_nll = running_loss / max(running_tokens, 1)
                perplexity = math.exp(mean_nll) if mean_nll < 20 else float("inf")
                pbar.set_postfix(
                    loss=f"{mean_nll:.4f}",
                    ppl=f"{perplexity:.2f}",
                )

        epoch_nll = running_loss / max(running_tokens, 1)
        epoch_ppl = math.exp(epoch_nll) if epoch_nll < 20 else float("inf")
        print(
            f"epoch {epoch + 1}: "
            f"mean_nll={epoch_nll:.4f} "
            f"perplexity={epoch_ppl:.2f}"
        )
        epoch_record: dict[str, float] = {
            "epoch": float(epoch + 1),
            "mean_nll": float(epoch_nll),
            "perplexity": float(epoch_ppl),
        }

        if args.early_stop_mode != "none":
            walks = sample_model_walks(
                model,
                num_samples=args.eval_generated_walks,
                max_length=eval_max_length,
                bos_token_id=int(eval_info["bos_token_id"]),
                device=device,
                show_progress=True,
                progress_desc=f"eval sampling @ epoch {epoch + 1}",
            )

            if args.early_stop_mode == "val":
                lp_split = eval_info["link_prediction_split"]
                if lp_split is None:
                    raise RuntimeError("Missing link-prediction split for val criterion.")
                scores = link_prediction_scores_from_walks(
                    walks,
                    num_nodes=int(eval_info["num_nodes"]),
                    positive_edges=lp_split["val_edges"],
                    negative_edges=lp_split["val_non_edges"],
                )
                val_score = 0.5 * (
                    scores["roc_auc"] + scores["average_precision"]
                )
                print(
                    f"  val_roc_auc={scores['roc_auc']:.4f} "
                    f"val_ap={scores['average_precision']:.4f} "
                    f"val_score={val_score:.4f}"
                )
                epoch_record["val_roc_auc"] = float(scores["roc_auc"])
                epoch_record["val_ap"] = float(scores["average_precision"])
                epoch_record["val_score"] = float(val_score)
                A_hat, _ = reconstruct_graph_from_generated_walks(
                    walks,
                    num_nodes=int(eval_info["num_nodes"]),
                    target_num_edges=int(eval_info["num_reference_edges"]),
                    seed=args.split_seed + epoch,
                )
                add_generated_graph_stats_to_epoch_record(
                    epoch_record,
                    A_hat,
                    reference_labels=eval_info["reference_labels"],
                )
                should_stop = early_state.update(val_score, step=epoch + 1)
                history.append(epoch_record)
                save_history_snapshot(history, args.save_dir)
                if should_stop:
                    print(
                        "Early stopping triggered by VAL criterion at "
                        f"epoch {epoch + 1}. Best epoch was {early_state.best_step} "
                        f"with score {early_state.best_value:.4f}."
                    )
                    maybe_save_checkpoint(model, optimizer, epoch + 1, args.save_dir)
                    break

            elif args.early_stop_mode == "edge_overlap":
                A_hat, _ = reconstruct_graph_from_generated_walks(
                    walks,
                    num_nodes=int(eval_info["num_nodes"]),
                    target_num_edges=int(eval_info["num_reference_edges"]),
                    seed=args.split_seed + epoch,
                )
                add_generated_graph_stats_to_epoch_record(
                    epoch_record,
                    A_hat,
                    reference_labels=eval_info["reference_labels"],
                )
                overlap = edge_overlap_ratio(A_hat, eval_info["reference_adj"])
                print(
                    f"  edge_overlap={overlap:.4f} "
                    f"(target={args.target_edge_overlap:.4f})"
                )
                epoch_record["edge_overlap"] = float(overlap)
                history.append(epoch_record)
                save_history_snapshot(history, args.save_dir)
                if overlap >= args.target_edge_overlap:
                    print(
                        "Early stopping triggered by edge-overlap criterion at "
                        f"epoch {epoch + 1}."
                    )
                    maybe_save_checkpoint(model, optimizer, epoch + 1, args.save_dir)
                    break
            else:
                history.append(epoch_record)
                save_history_snapshot(history, args.save_dir)
        else:
            history.append(epoch_record)
            save_history_snapshot(history, args.save_dir)

        maybe_save_checkpoint(model, optimizer, epoch + 1, args.save_dir)

    save_final_training_artifacts(model, history, args, args.save_dir)
    return model, eval_info, history


def main() -> None:
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
