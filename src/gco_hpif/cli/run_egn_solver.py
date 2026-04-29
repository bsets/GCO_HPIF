"""CLI for Part D.4 EGN optional external integration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from gco_hpif.solvers.egn_solver import EGNConfig, run_egn_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Part D.4 EGN training/inference as an optional external integration. "
            "The upstream EGN source is not vendored; pass --egn-root to a local checkout."
        )
    )

    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/manifests/twitter_split_60_20_20.csv"),
        help="Fixed Part B split manifest.",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=Path("artifacts/slice_a_full/interim"),
        help="Part A interim graph store directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/solver_runs/egn_smoke"),
        help="Output directory for EGN artifacts.",
    )
    parser.add_argument(
        "--egn-root",
        type=Path,
        default=Path(os.environ.get("GCO_HPIF_EGN_ROOT", "external/EGN/erdos_neu")),
        help=(
            "Path to local upstream EGN checkout containing models.py, cut_utils.py, "
            "and modules_and_utils.py. Can also be set via GCO_HPIF_EGN_ROOT."
        ),
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

    parser.add_argument("--dataset", default="twitter", help="Dataset to run. Default: twitter.")
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "train-and-infer", "smoke"],
        default="smoke",
        help="EGN workflow mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path to load for inference or write after training.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--infer-batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--hidden-1", type=int, default=64)
    parser.add_argument("--hidden-2", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--penalty-coeff", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--inference-samples", type=int, default=8)
    parser.add_argument("--receptive-field", type=int, default=None)
    parser.add_argument("--effective-volume-range", type=float, default=0.15)
    parser.add_argument("--lr-decay-step-size", type=int, default=5)
    parser.add_argument("--lr-decay-factor", type=float, default=0.95)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Limit rows per split for debugging. Smoke mode defaults to 2 if omitted.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device, e.g. cuda, cuda:0, or cpu. Default auto-detects.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = EGNConfig(
        split_manifest=args.split_manifest,
        interim_dir=args.interim_dir,
        output_dir=args.output_dir,
        egn_root=args.egn_root,
        extra_infer_graph_stores=args.extra_infer_graph_store,
        dataset=args.dataset,
        mode=args.mode,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        infer_batch_size=args.infer_batch_size,
        num_layers=args.num_layers,
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
        learning_rate=args.learning_rate,
        penalty_coeff=args.penalty_coeff,
        seed=args.seed,
        inference_samples=args.inference_samples,
        receptive_field=args.receptive_field,
        effective_volume_range=args.effective_volume_range,
        lr_decay_step_size=args.lr_decay_step_size,
        lr_decay_factor=args.lr_decay_factor,
        warmup_epochs=args.warmup_epochs,
        limit_per_split=args.limit_per_split,
        device=args.device,
    )

    run_egn_pipeline(config)
    print(f"EGN {config.mode} run completed.")
    print(f"Outputs written to: {config.output_dir}")


if __name__ == "__main__":
    main()
