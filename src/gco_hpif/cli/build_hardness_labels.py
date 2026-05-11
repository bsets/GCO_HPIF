"""CLI for Part E: build runtime-consensus hardness labels."""

from __future__ import annotations

import argparse
import pandas as pd

from gco_hpif.labels.hardness import (
    EXPECTED_SOLVERS,
    RuntimeHardnessConfig,
    build_runtime_hardness_labels,
    load_solver_runs,
    write_part_e_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Part E runtime-consensus hardness labels from five solver_runs.csv files."
    )
    parser.add_argument("--solver-runs", nargs="+", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-solvers", nargs="+", default=list(EXPECTED_SOLVERS))
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument(
        "--match-on",
        nargs="+",
        default=["dataset", "graph_id", "source_index"],
        choices=["dataset", "graph_id", "source_index"],
    )
    parser.add_argument("--respect-clique-valid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RuntimeHardnessConfig(
        expected_solvers=tuple(args.expected_solvers),
        primary_quantile=args.quantile,
        match_on=tuple(args.match_on),
        require_all_expected_solvers=True,
        respect_clique_valid=args.respect_clique_valid,
    )

    solver_runs = load_solver_runs(args.solver_runs)
    labels_df, incomplete_df, thresholds_df, coverage_df = build_runtime_hardness_labels(solver_runs, config=config)

    if args.features is not None and not labels_df.empty:
        features = pd.read_csv(args.features)
        features = features.copy()
        features["dataset"] = features["dataset"].astype(str).str.strip().str.lower().str.replace("-", "_")
        features["graph_id"] = features["graph_id"].astype(str).str.strip()
        features["source_index"] = pd.to_numeric(features["source_index"], errors="coerce").astype("Int64")
        features = features.dropna(subset=["dataset", "graph_id", "source_index"]).copy()
        features["source_index"] = features["source_index"].astype(int)
        join_cols = [c for c in config.match_on if c in features.columns and c in labels_df.columns]
        labels_df = labels_df.merge(features[join_cols].drop_duplicates(), on=join_cols, how="inner", validate="one_to_one")

    paths = write_part_e_outputs(labels_df, incomplete_df, thresholds_df, coverage_df, args.output_dir, args.solver_runs, config)

    print("\nPart E runtime-consensus hardness labeling complete.")
    print(f"Complete labeled instances: {len(labels_df)}")
    print(f"Incomplete/excluded instances: {len(incomplete_df)}")
    print("\nOutputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    if not labels_df.empty:
        print("\nConsensus-5 label counts:")
        print(labels_df["consensus5_runtime_hardness_label"].value_counts().to_string())
        print("\nConsensus-4 label counts:")
        print(labels_df["consensus4_runtime_hardness_label"].value_counts().to_string())
        print("\nMajority-3 label counts:")
        print(labels_df["majority3_runtime_hardness_label"].value_counts().to_string())
        print("\nHardness percentage summary:")
        print(pd.read_csv(paths["percentage_summary"]).to_string(index=False))


if __name__ == "__main__":
    main()
