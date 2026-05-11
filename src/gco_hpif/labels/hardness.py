"""Part E hardness-label construction for GCO-HPIF.

Primary label: Consensus-5 runtime hardness.

For graph instances common to all expected solvers:
1. Compute runtime 75th percentile within each dataset and within each algorithm.
2. Mark an instance as top-25 runtime for an algorithm if its runtime is >= that threshold.
3. Consensus-5 Hard: top-25 runtime for all five algorithms.
4. Consensus-4 and Majority-3 labels are also written for sensitivity analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

EXPECTED_SOLVERS = ("gurobi", "clisat", "momc", "egn", "hgs")

REQUIRED_SOLVER_COLUMNS = {
    "dataset",
    "graph_id",
    "source_index",
    "solver_name",
    "runtime_seconds",
    "best_clique_size",
}


@dataclass(frozen=True)
class RuntimeHardnessConfig:
    expected_solvers: tuple[str, ...] = EXPECTED_SOLVERS
    primary_quantile: float = 0.75
    match_on: tuple[str, ...] = ("dataset", "graph_id", "source_index")
    require_all_expected_solvers: bool = True
    respect_clique_valid: bool = False


def normalize_dataset(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def normalize_solver_name(value: object) -> str:
    raw = str(value).strip().lower()
    aliases = {
        "gurobi": "gurobi",
        "gurobi_solver": "gurobi",
        "clisat": "clisat",
        "cli_sat": "clisat",
        "cli-sat": "clisat",
        "momc": "momc",
        "momc_solver": "momc",
        "mixed_order_maximum_clique": "momc",
        "mixed order maximum clique": "momc",
        "egn": "egn",
        "erdos_neu": "egn",
        "hgs": "hgs",
    }
    return aliases.get(raw, raw)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
        .fillna(False)
        .astype(bool)
    )


def load_solver_runs(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Load and concatenate one or more solver_runs.csv files."""
    if not paths:
        raise ValueError("At least one solver_runs.csv file must be supplied.")

    frames: list[pd.DataFrame] = []
    for path_like in paths:
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"Solver-runs file not found: {path}")

        df = pd.read_csv(path)
        missing = REQUIRED_SOLVER_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        df = df.copy()
        df["solver_runs_file"] = str(path)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["dataset"] = out["dataset"].map(normalize_dataset)
    out["graph_id"] = out["graph_id"].astype(str).str.strip()
    out["source_index"] = pd.to_numeric(out["source_index"], errors="coerce")
    out["solver_name"] = out["solver_name"].map(normalize_solver_name)
    out["runtime_seconds"] = pd.to_numeric(out["runtime_seconds"], errors="coerce")
    out["best_clique_size"] = pd.to_numeric(out["best_clique_size"], errors="coerce")

    if "clique_valid" in out.columns:
        out["clique_valid_raw"] = _as_bool(out["clique_valid"])
    else:
        out["clique_valid_raw"] = pd.NA

    out = out.dropna(subset=["dataset", "graph_id", "source_index"]).copy()
    out["source_index"] = out["source_index"].astype(int)
    return out


def _dedupe_solver_rows(df: pd.DataFrame, match_on: Sequence[str]) -> pd.DataFrame:
    """Keep one deterministic row per graph-solver pair."""
    df = df.copy()
    df["_has_runtime"] = df["runtime_seconds"].notna().astype(int)
    df["_has_clique"] = df["best_clique_size"].notna().astype(int)
    df = df.sort_values(
        by=list(match_on) + ["solver_name", "_has_runtime", "_has_clique", "runtime_seconds"],
        ascending=[True] * len(match_on) + [True, False, False, True],
    )
    df = df.drop_duplicates(subset=list(match_on) + ["solver_name"], keep="first")
    return df.drop(columns=["_has_runtime", "_has_clique"])


def _pivot_value(df: pd.DataFrame, value_col: str, prefix: str, match_on: Sequence[str]) -> pd.DataFrame:
    pivot = (
        df.pivot_table(index=list(match_on), columns="solver_name", values=value_col, aggfunc="first")
        .reset_index()
    )
    pivot.columns.name = None
    rename = {solver: f"{prefix}_{solver}" for solver in EXPECTED_SOLVERS if solver in pivot.columns}
    return pivot.rename(columns=rename)


def build_runtime_hardness_labels(
    solver_runs: pd.DataFrame,
    config: RuntimeHardnessConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build consensus runtime-hardness labels.

    Returns labels_df, incomplete_df, thresholds_df, coverage_df.
    """
    config = config or RuntimeHardnessConfig()
    expected = tuple(normalize_solver_name(s) for s in config.expected_solvers)
    expected_set = set(expected)
    match_on = tuple(config.match_on)

    df = solver_runs.copy()
    df["dataset"] = df["dataset"].map(normalize_dataset)
    df["graph_id"] = df["graph_id"].astype(str).str.strip()
    df["source_index"] = pd.to_numeric(df["source_index"], errors="coerce")
    df["solver_name"] = df["solver_name"].map(normalize_solver_name)
    df["runtime_seconds"] = pd.to_numeric(df["runtime_seconds"], errors="coerce")
    df["best_clique_size"] = pd.to_numeric(df["best_clique_size"], errors="coerce")

    df = df[df["solver_name"].isin(expected_set)].copy()

    usable_mask = df["runtime_seconds"].notna() & df["best_clique_size"].notna()
    if config.respect_clique_valid and "clique_valid_raw" in df.columns:
        usable_mask = usable_mask & (df["clique_valid_raw"] == True)  # noqa: E712

    df = df[usable_mask].copy()
    df = df.dropna(subset=["source_index"]).copy()
    df["source_index"] = df["source_index"].astype(int)
    df = _dedupe_solver_rows(df, match_on=match_on)

    coverage_df = (
        df.groupby(list(match_on))["solver_name"]
        .agg(
            solver_names_present=lambda s: ",".join(sorted(set(s))),
            num_solvers_present=lambda s: len(set(s)),
        )
        .reset_index()
    )

    for solver in expected:
        coverage_df[f"has_{solver}"] = coverage_df["solver_names_present"].apply(
            lambda x, solver=solver: int(solver in set(str(x).split(",")))
        )

    complete_keys = coverage_df.loc[
        coverage_df["num_solvers_present"] == len(expected), list(match_on)
    ].copy()
    incomplete_df = coverage_df.loc[coverage_df["num_solvers_present"] != len(expected)].copy()

    if not incomplete_df.empty:
        def missing_solvers(row: pd.Series) -> str:
            present = set(str(row["solver_names_present"]).split(",")) if row["solver_names_present"] else set()
            return ",".join(sorted(expected_set - present))
        incomplete_df["missing_solver_names"] = incomplete_df.apply(missing_solvers, axis=1)
        incomplete_df["exclusion_reason"] = "missing_or_unusable_expected_solver_result"

    if complete_keys.empty:
        return pd.DataFrame(), incomplete_df, pd.DataFrame(), coverage_df

    complete_runs = df.merge(complete_keys, on=list(match_on), how="inner", validate="many_to_one")

    # The key step: thresholds are per dataset and per algorithm, not global.
    complete_runs["runtime_q75_by_dataset_algorithm"] = (
        complete_runs.groupby(["dataset", "solver_name"])["runtime_seconds"]
        .transform(lambda s: s.quantile(config.primary_quantile))
    )
    complete_runs["runtime_top25_for_solver"] = (
        complete_runs["runtime_seconds"] >= complete_runs["runtime_q75_by_dataset_algorithm"]
    ).astype(int)

    thresholds_df = (
        complete_runs.groupby(["dataset", "solver_name"])["runtime_q75_by_dataset_algorithm"]
        .first()
        .reset_index()
        .rename(columns={"runtime_q75_by_dataset_algorithm": "q75_runtime_seconds"})
        .sort_values(["dataset", "solver_name"])
        .reset_index(drop=True)
    )

    labels = _pivot_value(complete_runs, "runtime_seconds", "runtime", match_on)
    labels = labels.merge(_pivot_value(complete_runs, "best_clique_size", "clique", match_on), on=list(match_on), how="inner", validate="one_to_one")
    labels = labels.merge(_pivot_value(complete_runs, "runtime_top25_for_solver", "runtime_top25", match_on), on=list(match_on), how="inner", validate="one_to_one")
    labels = labels.merge(_pivot_value(complete_runs, "runtime_q75_by_dataset_algorithm", "runtime_q75", match_on), on=list(match_on), how="inner", validate="one_to_one")

    flag_cols = [f"runtime_top25_{solver}" for solver in expected]
    runtime_cols = [f"runtime_{solver}" for solver in expected]
    clique_cols = [f"clique_{solver}" for solver in expected]

    for col in flag_cols + runtime_cols + clique_cols:
        if col not in labels.columns:
            raise ValueError(f"Expected column missing from complete labels: {col}")

    labels["num_algorithms_runtime_top25"] = labels[flag_cols].sum(axis=1)

    labels["consensus5_runtime_hardness_label"] = (
        labels["num_algorithms_runtime_top25"] >= 5
    ).map({True: "Hard", False: "Not Hard"})
    labels["consensus5_runtime_hardness_label_binary"] = labels["consensus5_runtime_hardness_label"].map({"Not Hard": 0, "Hard": 1})

    labels["consensus4_runtime_hardness_label"] = (
        labels["num_algorithms_runtime_top25"] >= 4
    ).map({True: "Hard", False: "Not Hard"})
    labels["consensus4_runtime_hardness_label_binary"] = labels["consensus4_runtime_hardness_label"].map({"Not Hard": 0, "Hard": 1})

    labels["majority3_runtime_hardness_label"] = (
        labels["num_algorithms_runtime_top25"] >= 3
    ).map({True: "Hard", False: "Not Hard"})
    labels["majority3_runtime_hardness_label_binary"] = labels["majority3_runtime_hardness_label"].map({"Not Hard": 0, "Hard": 1})

    labels["primary_hardness_label"] = labels["consensus5_runtime_hardness_label"]
    labels["primary_hardness_label_binary"] = labels["consensus5_runtime_hardness_label_binary"]
    labels["primary_hardness_definition"] = (
        "Consensus-5 runtime hardness: Hard if the instance is in the top 25% runtime "
        "within its dataset for all five algorithms."
    )

    labels["best_observed_clique_size"] = labels[clique_cols].max(axis=1)
    labels["num_solvers_matching_best_clique"] = labels[clique_cols].eq(labels["best_observed_clique_size"], axis=0).sum(axis=1)
    labels["all_solvers_match_best_clique"] = (labels["num_solvers_matching_best_clique"] == len(expected)).astype(int)

    def matching_solvers(row: pd.Series) -> str:
        return ",".join([solver for solver in expected if row[f"clique_{solver}"] == row["best_observed_clique_size"]])

    labels["solvers_matching_best_clique"] = labels.apply(matching_solvers, axis=1)
    return labels.sort_values(list(match_on)).reset_index(drop=True), incomplete_df, thresholds_df, coverage_df


def write_part_e_outputs(
    labels_df: pd.DataFrame,
    incomplete_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    output_dir: str | Path,
    solver_run_files: Sequence[str | Path],
    config: RuntimeHardnessConfig,
) -> dict[str, Path]:
    """Write Part E output artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "labels": out / "hardness_labels.csv",
        "summary": out / "hardness_label_summary.csv",
        "thresholds": out / "runtime_thresholds_by_dataset_algorithm.csv",
        "coverage": out / "solver_coverage_by_graph.csv",
        "incomplete": out / "incomplete_solver_diagnostics.csv",
        "percentage_summary": out / "hardness_percentage_summary.csv",
        "manifest": out / "hardness_label_manifest.json",
    }

    labels_df.to_csv(paths["labels"], index=False)
    incomplete_df.to_csv(paths["incomplete"], index=False)
    thresholds_df.to_csv(paths["thresholds"], index=False)
    coverage_df.to_csv(paths["coverage"], index=False)

    if labels_df.empty:
        summary = pd.DataFrame(columns=["label_type", "dataset", "label", "count"])
        percentage_summary = pd.DataFrame(columns=["method", "total_instances", "hard_instances", "not_hard_instances", "hard_percentage"])
    else:
        summary_frames = []
        for label_col in ["consensus5_runtime_hardness_label", "consensus4_runtime_hardness_label", "majority3_runtime_hardness_label"]:
            temp = (
                labels_df.groupby(["dataset", label_col])
                .size()
                .reset_index(name="count")
                .rename(columns={label_col: "label"})
            )
            temp.insert(0, "label_type", label_col)
            summary_frames.append(temp)
        summary = pd.concat(summary_frames, ignore_index=True)

        rows = []
        methods = {
            "Consensus-5": "consensus5_runtime_hardness_label",
            "Consensus-4": "consensus4_runtime_hardness_label",
            "Majority-3": "majority3_runtime_hardness_label",
        }
        total = len(labels_df)
        for method, col in methods.items():
            hard = int((labels_df[col] == "Hard").sum())
            not_hard = int((labels_df[col] == "Not Hard").sum())
            rows.append({
                "method": method,
                "total_instances": total,
                "hard_instances": hard,
                "not_hard_instances": not_hard,
                "hard_percentage": 100.0 * hard / total if total else 0.0,
            })
        percentage_summary = pd.DataFrame(rows)

    summary.to_csv(paths["summary"], index=False)
    percentage_summary.to_csv(paths["percentage_summary"], index=False)

    manifest = {
        "stage": "Part E",
        "description": "Runtime-consensus hardness labels for common five-solver MCP instances.",
        "primary_label": "consensus5_runtime_hardness_label_binary",
        "primary_definition": "Hard if the instance is in the top 25% runtime group within its dataset for all five algorithms: Gurobi, CliSAT, MOMC, EGN, HGS.",
        "sensitivity_labels": ["consensus4_runtime_hardness_label_binary", "majority3_runtime_hardness_label_binary"],
        "expected_solvers": list(config.expected_solvers),
        "match_on": list(config.match_on),
        "primary_quantile": config.primary_quantile,
        "respect_clique_valid": config.respect_clique_valid,
        "solver_run_files": [str(p) for p in solver_run_files],
        "num_labeled_complete_instances": int(len(labels_df)),
        "num_incomplete_instances": int(len(incomplete_df)),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
