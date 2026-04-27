from __future__ import annotations

import argparse
from pathlib import Path

from gco_hpif.data.prepare_raw import prepare_raw_graphs


VALID_DATASETS = ["imdb_binary", "collab", "twitter"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare raw graphs for GCO-HPIF Slice A")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root folder where raw/interim/manifests will be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=VALID_DATASETS,
        choices=VALID_DATASETS,
        help="Datasets to prepare.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download dataset archives even if they already exist locally.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-dataset graph limit for smoke tests.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = prepare_raw_graphs(
        output_root=args.output_root,
        datasets=args.datasets,
        force_download=args.force_download,
        limit=args.limit,
    )

    print("Slice A complete.")
    print(f"Graph manifest: {result.manifest_path}")
    print(f"Dataset summary: {result.summary_path}")
    print("Interim graph pickle files:")
    for path in result.interim_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
