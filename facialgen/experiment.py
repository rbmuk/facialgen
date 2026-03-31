from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from .early_stopping import (
    connected_link_prediction_split,
    edge_overlap_ratio,
    link_prediction_scores_from_walks,
    sample_model_walks,
)
from .evaluation import (
    average_rank_from_graph_statistics,
    compute_graph_statistics,
    reconstruct_graph_from_generated_walks,
)
from .models import FacialGen
from .train import add_training_args, build_training_objects, resolve_device, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end facial-walk experiment on CoraML."
    )
    add_training_args(parser)
    parser.set_defaults(use_link_prediction_split=True, early_stop_mode="val")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--final-generated-walks", type=int, default=16384)
    parser.add_argument("--final-max-length", type=int, default=None)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--num-generated-graphs", type=int, default=1)
    parser.add_argument("--reconstruction-seed", type=int, default=777)
    parser.add_argument("--report-average-rank", action="store_true")
    return parser.parse_args()


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf"
    return f"{value:.4f}"


def _print_rows(
    title: str,
    rows: Sequence[dict[str, object]],
    *,
    columns: Sequence[str],
) -> None:
    print(f"\n{title}")
    widths = {
        col: max(len(col), *(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


def run_final_evaluation(
    model: FacialGen,
    *,
    eval_info: dict[str, object],
    args: argparse.Namespace,
    device,
) -> dict[str, object]:
    final_max_length = (
        args.final_max_length if args.final_max_length is not None else args.context_size
    )

    reference_adj = eval_info["reference_adj"]
    reference_labels = eval_info["reference_labels"]
    num_nodes = int(eval_info["num_nodes"])
    num_reference_edges = int(eval_info["num_reference_edges"])

    lp_split = eval_info["link_prediction_split"]
    if lp_split is None:
        lp_split = connected_link_prediction_split(
            reference_adj,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.split_seed,
        )

    reference_stats = compute_graph_statistics(
        reference_adj,
        labels=reference_labels,
    )

    generated_results: list[dict[str, object]] = []
    generated_stats_list: list[dict[str, float]] = []

    for graph_idx in range(args.num_generated_graphs):
        walks = sample_model_walks(
            model,
            num_samples=args.final_generated_walks,
            max_length=final_max_length,
            bos_token_id=int(eval_info["bos_token_id"]),
            eos_token_id=int(eval_info["eos_token_id"]),
            device=device,
            batch_size=args.generation_batch_size,
        )
        A_hat, S = reconstruct_graph_from_generated_walks(
            walks,
            num_nodes=num_nodes,
            target_num_edges=num_reference_edges,
            seed=args.reconstruction_seed + graph_idx,
        )
        val_scores = link_prediction_scores_from_walks(
            walks,
            num_nodes=num_nodes,
            positive_edges=lp_split["val_edges"],
            negative_edges=lp_split["val_non_edges"],
        )
        test_scores = link_prediction_scores_from_walks(
            walks,
            num_nodes=num_nodes,
            positive_edges=lp_split["test_edges"],
            negative_edges=lp_split["test_non_edges"],
        )
        graph_stats = compute_graph_statistics(A_hat, labels=reference_labels)
        overlap = edge_overlap_ratio(A_hat, reference_adj)

        generated_stats_list.append(graph_stats)
        generated_results.append(
            {
                "graph_id": graph_idx,
                "num_walks": args.final_generated_walks,
                "val_roc_auc": float(val_scores["roc_auc"]),
                "val_ap": float(val_scores["average_precision"]),
                "test_roc_auc": float(test_scores["roc_auc"]),
                "test_ap": float(test_scores["average_precision"]),
                "edge_overlap": float(overlap),
                "stats": graph_stats,
            }
        )

    return {
        "reference_stats": reference_stats,
        "generated_results": generated_results,
        "generated_stats_list": generated_stats_list,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.checkpoint_dir is None:
        model, eval_info, history = train_model(args)
    else:
        _, _, _, eval_info = build_training_objects(args)
        model = FacialGen.from_pretrained(args.checkpoint_dir)
        history = []

    model.to(device)

    evaluation = run_final_evaluation(
        model,
        eval_info=eval_info,
        args=args,
        device=device,
    )

    generated_results = evaluation["generated_results"]
    reference_stats = evaluation["reference_stats"]
    generated_stats_list = evaluation["generated_stats_list"]

    lp_rows = []
    for result in generated_results:
        lp_rows.append(
            {
                "graph_id": result["graph_id"],
                "val_roc_auc": _format_float(result["val_roc_auc"]),
                "val_ap": _format_float(result["val_ap"]),
                "test_roc_auc": _format_float(result["test_roc_auc"]),
                "test_ap": _format_float(result["test_ap"]),
                "edge_overlap": _format_float(result["edge_overlap"]),
            }
        )
    _print_rows(
        "Link Prediction And Overlap",
        lp_rows,
        columns=["graph_id", "val_roc_auc", "val_ap", "test_roc_auc", "test_ap", "edge_overlap"],
    )

    metric_names = list(reference_stats.keys())
    stats_rows = []
    generated_means = {
        metric: float(np.nanmean([stats[metric] for stats in generated_stats_list]))
        for metric in metric_names
    }
    for metric in metric_names:
        stats_rows.append(
            {
                "metric": metric,
                "true_coraml": _format_float(float(reference_stats[metric])),
                "generated_mean": _format_float(generated_means[metric]),
                "abs_diff": _format_float(
                    abs(generated_means[metric] - float(reference_stats[metric]))
                ),
            }
        )
    _print_rows(
        "True Vs Generated Graph Statistics",
        stats_rows,
        columns=["metric", "true_coraml", "generated_mean", "abs_diff"],
    )

    if args.report_average_rank and len(generated_stats_list) > 1:
        avg_rank, per_metric_ranks = average_rank_from_graph_statistics(
            reference_stats,
            generated_stats_list,
            metric_names=metric_names,
        )
        rank_rows = []
        for idx, value in enumerate(avg_rank):
            rank_rows.append(
                {
                    "graph_id": idx,
                    "average_rank": _format_float(float(value)),
                }
            )
        _print_rows(
            "Average Rank Across Generated Graphs",
            rank_rows,
            columns=["graph_id", "average_rank"],
        )

    if history:
        history_rows = []
        for record in history:
            row = {"epoch": int(record["epoch"])}
            for key, value in record.items():
                if key == "epoch":
                    continue
                row[key] = _format_float(float(value))
            history_rows.append(row)
        cols = list(history_rows[0].keys())
        _print_rows("Training History", history_rows, columns=cols)


if __name__ == "__main__":
    main()
