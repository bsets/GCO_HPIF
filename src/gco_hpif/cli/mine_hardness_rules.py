"""CLI for Part G FP-Growth hardness association-rule mining."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from gco_hpif.interpretation.association_rules import DEFAULT_TARGETS, PartGConfig, run_part_g


def _split_csv(value: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine Part G FP-Growth association rules for graph-instance hardness."
    )
    parser.add_argument(
        "--features",
        required=True,
        type=Path,
        help="Path to artifacts/features_full/graph_features.csv.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Path to artifacts/hardness_labels_consensus_runtime/hardness_labels.csv.",
    )
    parser.add_argument(
        "--part-f-output-dir",
        required=True,
        type=Path,
        help="Path to artifacts/ml_hardness_part_f containing target folders and saved models.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for Part G results.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="Comma-separated target columns to mine.",
    )
    parser.add_argument(
        "--bin-counts",
        nargs="+",
        type=int,
        default=[3, 4, 5],
        help="Percentile-bin sensitivity settings. Default: 3 4 5.",
    )
    parser.add_argument(
        "--min-support",
        default="auto",
        help="FP-Growth minimum support. Use 'auto' or a fixed float such as 0.03.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Minimum rule confidence.",
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=1.0,
        help="Minimum rule lift.",
    )
    parser.add_argument(
        "--max-rule-antecedents",
        type=int,
        default=3,
        help="Maximum number of feature-bin items in a rule antecedent.",
    )
    parser.add_argument(
        "--max-rules-per-group",
        type=int,
        default=25,
        help="Maximum selected non-overlapping rules per dataset/target/bin/output-source group.",
    )
    parser.add_argument(
        "--min-matched-rows",
        type=int,
        default=10,
        help="Minimum number of rows a selected rule antecedent must cover.",
    )
    parser.add_argument(
        "--evaluation-scope",
        choices=["all", "test"],
        default="all",
        help="Mine rules on all rows or a deterministic held-out-style test subset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Test fraction when --evaluation-scope test.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic test split.",
    )
    parser.add_argument(
        "--model-selection-metric",
        choices=["weighted_f1", "minority_f1"],
        default="weighted_f1",
        help="Metric for choosing the best saved Part F model separately within each dataset.",
    )
    parser.add_argument(
        "--no-model-predictions",
        action="store_true",
        help="Only mine rules for ground-truth labels; skip saved Part F model predictions.",
    )
    parser.add_argument(
        "--max-selected-features-per-target",
        type=int,
        default=10,
        help=(
            "Safety cap on the number of first-peak selected features allowed per target. "
            "Use 10 by default to catch accidental all-23-feature runs while allowing "
            "the expected small Consensus-5/Consensus-4/Majority-3 feature sets."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print output paths at the end.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PartGConfig(
        features_path=args.features,
        labels_path=args.labels,
        part_f_output_dir=args.part_f_output_dir,
        output_dir=args.output_dir,
        targets=_split_csv(args.targets),
        bin_counts=tuple(args.bin_counts),
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        max_rule_antecedents=args.max_rule_antecedents,
        max_rules_per_group=args.max_rules_per_group,
        min_matched_rows=args.min_matched_rows,
        evaluation_scope=args.evaluation_scope,
        test_size=args.test_size,
        random_state=args.random_state,
        include_model_predictions=not args.no_model_predictions,
        model_selection_metric=args.model_selection_metric,
        max_selected_features_per_target=args.max_selected_features_per_target,
        verbose=args.verbose,
    )
    result = run_part_g(config)
    print("\nDone. Part G outputs written to:")
    print(f"  Excel workbook:       {result.workbook_path}")
    print(f"  Selected rules CSV:   {result.selected_rules_path}")
    print(f"  Runtime summary CSV:  {result.runtime_summary_path}")
    print(f"  Model selection CSV:  {result.model_selection_path}")


if __name__ == "__main__":
    main()
