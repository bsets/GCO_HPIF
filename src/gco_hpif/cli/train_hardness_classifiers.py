"""CLI for Part F hardness classification.

Example:

python -m gco_hpif.cli.train_hardness_classifiers \
  --features artifacts/features_full/graph_features.csv \
  --labels artifacts/hardness_labels_consensus_runtime/hardness_labels.csv \
  --output-dir artifacts/ml_hardness_part_f
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gco_hpif.ml.hardness_classification import (
    DEFAULT_TARGETS,
    PartFConfig,
    run_part_f,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part F: train tuned ML classifiers for runtime-consensus hardness labels."
    )

    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to artifacts/features_full/graph_features.csv",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to artifacts/hardness_labels_consensus_runtime/hardness_labels.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/ml_hardness_part_f"),
        help="Output directory for Part F artifacts.",
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help=(
            "Binary hardness target columns to model. Defaults to Consensus-5, "
            "Consensus-4, and Majority-3."
        ),
    )

    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--search-iterations",
        type=int,
        default=30,
        help="Maximum randomized-search iterations per model family.",
    )

    parser.add_argument(
        "--first-peak-patience",
        type=int,
        default=2,
        help=(
            "Number of following feature counts that must fail to improve "
            "for the first-peak rule."
        ),
    )

    parser.add_argument(
        "--first-peak-min-delta",
        type=float,
        default=0.002,
        help=(
            "Minimum minority-F1 improvement required to avoid declaring "
            "a plateau/peak."
        ),
    )

    parser.add_argument(
        "--skip-xgb",
        action="store_true",
        help="Skip XGBoost even if xgboost is installed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PartFConfig(
        features_csv=args.features,
        labels_csv=args.labels,
        output_dir=args.output_dir,
        targets=args.targets,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        search_iterations=args.search_iterations,
        first_peak_patience=args.first_peak_patience,
        first_peak_min_delta=args.first_peak_min_delta,
        skip_xgb=args.skip_xgb,
    )

    run_part_f(config)


if __name__ == "__main__":
    main()
