from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gco_hpif.features.networkx_features import (
    DEFAULT_FEATURE_TIMEOUT_SECONDS,
    compute_features_from_part_a_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the 23 NetworkX graph features from Part A graph artifacts."
    )
    parser.add_argument(
        "--graphs-index",
        type=Path,
        default=Path("artifacts/slice_a_full/manifests/graphs_index.csv"),
        help="Path to Part A graphs_index.csv.",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=Path("artifacts/slice_a_full/interim"),
        help="Directory containing Part A *_graphs.pkl.gz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/features"),
        help="Directory where feature CSV outputs will be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["twitter", "collab", "imdb_binary"],
        choices=["twitter", "collab", "imdb_binary"],
        help="Datasets to process.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_FEATURE_TIMEOUT_SECONDS,
        help="Per-graph feature-computation timeout in seconds.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Optional limit for smoke tests. Processes only the first N graphs per dataset.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compute_features_from_part_a_artifacts(
        graphs_index_csv=args.graphs_index,
        interim_dir=args.interim_dir,
        output_dir=args.output_dir,
        datasets=args.datasets,
        timeout_seconds=args.timeout_seconds,
        limit_per_dataset=args.limit_per_dataset,
        show_progress=not args.no_progress,
    )

    print("Part C complete.")
    print(f"Features: {result.features_path}")
    print(f"Failures: {result.failures_path}")
    print(f"Timing summary: {result.summary_path}")
    print(f"Feature manifest: {result.feature_manifest_path}")

    summary = pd.read_csv(result.summary_path)
    if not summary.empty:
        print("\nSummary:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
