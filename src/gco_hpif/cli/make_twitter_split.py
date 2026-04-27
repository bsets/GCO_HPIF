"""CLI for creating the Part B TWITTER split manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from gco_hpif.data.splits import build_twitter_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Part B TWITTER train/validation/test split manifest "
            "from the Part A graphs_index.csv file."
        )
    )
    parser.add_argument(
        "--graphs-index",
        required=True,
        type=Path,
        help="Path to artifacts/slice_a_full/manifests/graphs_index.csv from Part A.",
    )
    parser.add_argument(
        "--output",
        default=Path("data/manifests/twitter_split_60_20_20.csv"),
        type=Path,
        help="Output CSV path for the committed TWITTER split manifest.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help=(
            "Optional explicit number of training graphs. If omitted, this is "
            "round(train-ratio * number_of_twitter_graphs)."
        ),
    )
    parser.add_argument(
        "--validation-count",
        type=int,
        default=None,
        help=(
            "Optional explicit number of validation graphs. If omitted, this is "
            "round(validation-ratio * number_of_twitter_graphs)."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.60,
        help="Training ratio used when --train-count is omitted. Default: 0.60.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.20,
        help="Validation ratio used when --validation-count is omitted. Default: 0.20.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path, summary = build_twitter_split_manifest(
        graphs_index_csv=args.graphs_index,
        output_csv=args.output,
        train_count=args.train_count,
        validation_count=args.validation_count,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )

    print("Part B complete.")
    print(f"TWITTER split manifest: {output_path}")
    print("Split counts:")
    for key, value in summary.as_dict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
