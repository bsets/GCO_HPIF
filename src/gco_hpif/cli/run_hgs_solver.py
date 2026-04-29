"""Command-line entry point for the optional HGS solver integration."""

from __future__ import annotations

import argparse
from pathlib import Path

from gco_hpif.solvers.hgs_solver import HGSConfig, run_hgs_experiment


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the optional HGS training/inference wrapper for maximum clique experiments."
    )

    parser.add_argument(
        "--mode",
        choices=["smoke", "train", "infer", "train-and-infer"],
        required=True,
        help="Execution mode. 'smoke' is a small train-and-infer run.",
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--interim-dir", type=Path, required=True)
    parser.add_argument(
        "--raw-graph-store",
        type=Path,
        default=None,
        help="Optional direct graph-store file or directory. Use this when --interim-dir does not contain readable graph artifacts.",
    )
    parser.add_argument(
        "--extra-infer-graph-store",
        action="append",
        default=[],
        help=(
            "Additional all-test graph store for inference, formatted as dataset=path. "
            "Can be supplied multiple times, e.g. collab=... imdb_binary=..."
        ),
    )

    parser.add_argument("--hgs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="twitter")
    parser.add_argument("--limit-per-split", type=int, default=None)

    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--infer-splits", default="test")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--penalty-coeff", type=float, default=2.0)
    parser.add_argument("--moment", type=int, default=1)
    parser.add_argument("--smooth", type=float, default=0.1)
    parser.add_argument("--use-smooth-residual", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-path", type=Path, default=None)

    parser.add_argument("--num-walkers", type=int, default=20)
    parser.add_argument("--sample-length", type=int, default=90)
    parser.add_argument("--time-limit-seconds", type=float, default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    mode = args.mode
    limit_per_split = args.limit_per_split
    epochs = args.epochs
    num_walkers = args.num_walkers

    if mode == "smoke":
        mode = "train-and-infer"
        limit_per_split = 2 if limit_per_split is None else limit_per_split
        epochs = min(args.epochs, 2)
        num_walkers = min(args.num_walkers, 4)

    config = HGSConfig(
        mode=mode,
        split_manifest=args.split_manifest,
        interim_dir=args.interim_dir,
        hgs_root=args.hgs_root,
        output_dir=args.output_dir,
        raw_graph_store=args.raw_graph_store,
        extra_infer_graph_stores=args.extra_infer_graph_store,
        dataset=args.dataset,
        limit_per_split=limit_per_split,
        train_splits=_parse_csv_list(args.train_splits),
        infer_splits=_parse_csv_list(args.infer_splits),
        epochs=epochs,
        train_batch_size=args.train_batch_size,
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        penalty_coeff=args.penalty_coeff,
        moment=args.moment,
        smooth=args.smooth,
        use_smooth_residual=args.use_smooth_residual,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        num_walkers=num_walkers,
        sample_length=args.sample_length,
        time_limit_seconds=args.time_limit_seconds,
    )
    run_hgs_experiment(config)


if __name__ == "__main__":
    main()
