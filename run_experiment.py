from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from facialgen.early_stopping import (
    connected_link_prediction_split,
    edge_overlap_ratio,
    link_prediction_scores_from_transition_matrix,
)
from facialgen.evaluation import (
    compute_graph_statistics,
    reconstruct_graph_from_transition_matrix,
)
from facialgen.models import FacialGen
from facialgen.sampling import sample_model_transition_counts
from facialgen.train import (
    add_training_args,
    default_face_generation_max_length,
    default_random_walk_generation_max_length,
    resolve_device,
    resolve_run_save_dir,
    seed_everything,
    train_model,
)


def add_run_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_training_args(parser)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--final-generated-walks", type=int, default=500_000)
    parser.add_argument("--final-max-length", type=int, default=None)
    parser.add_argument("--generation-batch-size", type=int, default=256)
    parser.add_argument("--num-generated-graphs", type=int, default=1)
    parser.add_argument("--reconstruction-seed", type=int, default=777)
    parser.add_argument(
        "--save-final-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and run final evaluation for a facialgen experiment."
    )
    return add_run_args(parser).parse_args()


def _load_latest_checkpoint(run_dir: Path) -> tuple[FacialGen, Path]:
    epoch_dirs = sorted(run_dir.glob("epoch_*"))
    if epoch_dirs:
        ckpt_dir = epoch_dirs[-1]
    else:
        ckpt_dir = run_dir / "final"
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"No checkpoint found in {run_dir}. Expected an epoch_* directory or final/."
            )
    return FacialGen.from_pretrained(str(ckpt_dir)), ckpt_dir


def _load_history(run_dir: Path) -> list[dict[str, float]]:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return []
    raw = json.loads(history_path.read_text())
    return raw if isinstance(raw, list) else []


def _edge_scores_from_raw_S(
    S: sp.spmatrix,
    edge_pairs: np.ndarray,
    *,
    darts_per_sequence: int,
) -> np.ndarray:
    total_transition_mass = float(S.sum())
    if edge_pairs.size == 0 or total_transition_mass <= 0:
        return np.empty(0, dtype=float)
    vals = np.asarray(S[edge_pairs[:, 0], edge_pairs[:, 1]]).ravel().astype(float)
    return (float(darts_per_sequence) / total_transition_mass) * vals


def _min_nonzero_gap(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan")
    uniq = np.unique(np.sort(values))
    if uniq.size < 2:
        return float("nan")
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    return float(diffs.min())


def run_final_evaluation(
    args: argparse.Namespace,
    *,
    model: FacialGen,
    eval_info: dict[str, object],
    checkpoint_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    reference_adj = eval_info["reference_adj"]
    overlap_adj = eval_info.get("overlap_adj", reference_adj)
    overlap_name = str(eval_info.get("overlap_name", "reference"))
    reference_labels = eval_info["reference_labels"]
    num_nodes = int(eval_info["num_nodes"])
    num_reference_edges = int(eval_info["num_reference_edges"])
    eval_walk_type = str(eval_info.get("walk_type", getattr(args, "walk_type", "facial")))

    if args.final_max_length is None:
        final_max_length = (
            default_face_generation_max_length(args.vertex_context_size)
            if eval_walk_type == "facial"
            else default_random_walk_generation_max_length(args.vertex_context_size)
        )
    else:
        final_max_length = int(args.final_max_length)

    print(f"final generation max_length = {final_max_length}")

    lp_split = eval_info["link_prediction_split"]
    if lp_split is None:
        lp_split = connected_link_prediction_split(
            reference_adj,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.split_seed,
        )

    reference_stats = compute_graph_statistics(reference_adj, labels=reference_labels)
    generated_results: list[dict[str, float]] = []
    generated_stats_rows: list[dict[str, float | None]] = []
    split_score_tables: list[pd.DataFrame] = []
    darts_per_sequence = max((int(final_max_length) - 1) // 2, 1) if eval_walk_type == "facial" else max(int(final_max_length) - 1, 1)
    progress_mode = str(getattr(args, "progress_mode", "tqdm"))

    for graph_idx in range(int(args.num_generated_graphs)):
        if progress_mode == "log":
            print(
                f"final graph {graph_idx + 1}/{args.num_generated_graphs}: "
                f"sampling {int(args.final_generated_walks)} walks"
            )
        S = sample_model_transition_counts(
            model,
            num_samples=int(args.final_generated_walks),
            max_length=int(final_max_length),
            bos_token_id=int(eval_info["bos_token_id"]),
            num_nodes=num_nodes,
            device=device,
            walk_type=eval_walk_type,
            batch_size=int(args.generation_batch_size),
            show_progress=(progress_mode == "tqdm"),
            progress_desc=f"final sampling graph {graph_idx + 1}/{args.num_generated_graphs}",
            log_every_samples=(
                max(int(args.final_generated_walks) // 10, 1)
                if progress_mode == "log"
                else None
            ),
        )

        if progress_mode == "log":
            print(f"final graph {graph_idx + 1}/{args.num_generated_graphs}: graph reconstruction")
        A_hat = reconstruct_graph_from_transition_matrix(
            S,
            target_num_edges=num_reference_edges,
            seed=int(args.reconstruction_seed) + graph_idx,
            walk_type=eval_walk_type,
            score_symmetrization=eval_info.get("score_symmetrization", args.score_symmetrization),
            show_progress=(
                progress_mode == "tqdm"
                and bool(getattr(args, "debug_graph_reconstruction", False))
            ),
            progress_desc=f"graph reconstruction {graph_idx + 1}/{args.num_generated_graphs}",
            debug=(
                progress_mode == "log"
                or bool(getattr(args, "debug_graph_reconstruction", False))
            ),
        )

        train_upper = sp.triu(eval_info["train_adj"], k=1).tocoo()
        train_edges = (
            np.column_stack((train_upper.row, train_upper.col)).astype(np.int64)
            if train_upper.nnz > 0
            else np.empty((0, 2), dtype=np.int64)
        )
        split_score_rows = []
        for split_name, split_edges in [
            ("train", train_edges),
            ("validation", lp_split["val_edges"]),
            ("test", lp_split["test_edges"]),
        ]:
            split_scores = _edge_scores_from_raw_S(
                S,
                np.asarray(split_edges, dtype=np.int64),
                darts_per_sequence=darts_per_sequence,
            )
            split_score_rows.append(
                {
                    "graph_id": graph_idx,
                    "split": split_name,
                    "num_edges": int(len(split_edges)),
                    "min_8S_over_sumS": float(np.min(split_scores)) if split_scores.size else float("nan"),
                    "min_nonzero_gap": _min_nonzero_gap(split_scores),
                }
            )
        split_score_table = pd.DataFrame(split_score_rows)
        split_score_tables.append(split_score_table)
        print("raw-S edge score diagnostics (8*S[i,j] / sum_uv S[u,v]):")
        print(split_score_table.to_string(index=False))

        val_scores = link_prediction_scores_from_transition_matrix(
            S,
            positive_edges=lp_split["val_edges"],
            negative_edges=lp_split["val_non_edges"],
            walk_type=eval_walk_type,
            score_symmetrization=eval_info.get("score_symmetrization", args.score_symmetrization),
        )
        test_scores = link_prediction_scores_from_transition_matrix(
            S,
            positive_edges=lp_split["test_edges"],
            negative_edges=lp_split["test_non_edges"],
            walk_type=eval_walk_type,
            score_symmetrization=eval_info.get("score_symmetrization", args.score_symmetrization),
        )
        graph_stats = compute_graph_statistics(A_hat, labels=reference_labels)
        overlap = edge_overlap_ratio(A_hat, overlap_adj)

        generated_results.append(
            {
                "graph_id": graph_idx,
                "val_roc_auc": float(val_scores["roc_auc"]),
                "val_ap": float(val_scores["average_precision"]),
                "test_roc_auc": float(test_scores["roc_auc"]),
                "test_ap": float(test_scores["average_precision"]),
                f"edge_overlap[{overlap_name}]": float(overlap),
            }
        )
        generated_stats_rows.append(graph_stats)

    lp_table = pd.DataFrame(generated_results)
    stats_table = pd.DataFrame(generated_stats_rows)
    split_score_table = pd.concat(split_score_tables, ignore_index=True)

    print("\nlink prediction results:")
    print(lp_table.to_string(index=False))

    metric_names = list(reference_stats.keys())
    report_rows = []
    for metric in metric_names:
        row = {"metric": metric, "reference": reference_stats[metric]}
        values = [r.get(metric, np.nan) for r in generated_stats_rows]
        row["generated_mean"] = float(np.nanmean(values)) if values else float("nan")
        row["generated_std"] = float(np.nanstd(values)) if values else float("nan")
        report_rows.append(row)
    graph_report = pd.DataFrame(report_rows)
    print("\ngraph statistics:")
    print(graph_report.to_string(index=False))

    if args.save_final_eval:
        if args.save_dir is None:
            raise RuntimeError("args.save_dir is None; cannot save final evaluation outputs.")
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        lp_table.to_csv(out_dir / "final_eval_link_prediction.csv", index=False)
        graph_report.to_csv(out_dir / "final_eval_graph_stats.csv", index=False)
        split_score_table.to_csv(out_dir / "final_eval_score_diagnostics.csv", index=False)
        meta = {
            "dataset_name": args.dataset_name,
            "walk_type": args.walk_type,
            "score_symmetrization": args.score_symmetrization,
            "edge_overlap_target": args.edge_overlap_target,
            "final_generated_walks": int(args.final_generated_walks),
            "final_max_length": int(final_max_length),
            "generation_batch_size": int(args.generation_batch_size),
            "num_generated_graphs": int(args.num_generated_graphs),
            "reconstruction_seed": int(args.reconstruction_seed),
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        }
        (out_dir / "final_eval_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"\nsaved final evaluation outputs under {out_dir}")

    return lp_table, graph_report, split_score_table


def main() -> None:
    args = parse_args()
    args.save_dir = resolve_run_save_dir(getattr(args, "save_dir", None), args)
    seed_everything(int(getattr(args, "seed", 0)))

    checkpoint_dir: Path | None = None
    if args.skip_train:
        if args.save_dir is None:
            raise RuntimeError("--skip-train requires a save_dir with an existing run.")
        run_dir = Path(args.save_dir)
        model, checkpoint_dir = _load_latest_checkpoint(run_dir)
        history = _load_history(run_dir)
        train_args_path = run_dir / "train_args.json"
        if train_args_path.exists():
            print(f"loaded latest checkpoint from {checkpoint_dir}")
        from facialgen.train import build_training_objects

        _, _, _, eval_info = build_training_objects(args)
    else:
        model, eval_info, history = train_model(args)
        if args.save_dir is not None:
            checkpoint_dir = Path(args.save_dir) / "final"

    print(f"\nhistory rows: {len(history)}")
    run_final_evaluation(
        args,
        model=model,
        eval_info=eval_info,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main()
