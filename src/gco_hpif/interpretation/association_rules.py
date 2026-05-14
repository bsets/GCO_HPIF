"""Part G: FP-Growth association-rule mining for hardness interpretation.

This module turns the Part F selected graph features into percentile-bin
transactions and mines rules of the form:

    feature-bin(s) -> Hard
    feature-bin(s) -> Not Hard

It supports both empirical ground-truth labels and predictions from the best
saved Part F model selected separately for each dataset.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

try:  # joblib is an optional dependency only needed for model-prediction rules.
    import joblib
except Exception:  # pragma: no cover - handled at runtime.
    joblib = None

JOIN_KEYS: Tuple[str, str, str] = ("dataset", "graph_id", "source_index")
DEFAULT_TARGETS: Tuple[str, str, str] = (
    "consensus5_runtime_hardness_label_binary",
    "consensus4_runtime_hardness_label_binary",
    "majority3_runtime_hardness_label_binary",
)
OUTCOME_HARD = "OUTCOME::Hard"
OUTCOME_NOT_HARD = "OUTCOME::Not_Hard"
OUTCOME_ITEMS = {OUTCOME_HARD, OUTCOME_NOT_HARD}
BIN_ITEM_RE = re.compile(r"^BIN::(?P<feature>feature_.+?)::P(?P<lo>\d+)_(?P<hi>\d+)$")


@dataclass(frozen=True)
class PartGConfig:
    """Configuration for Part G rule mining."""

    features_path: Path
    labels_path: Path
    part_f_output_dir: Path
    output_dir: Path
    targets: Tuple[str, ...] = DEFAULT_TARGETS
    bin_counts: Tuple[int, ...] = (3, 4, 5)
    min_support: str = "auto"
    min_confidence: float = 0.60
    min_lift: float = 1.0
    max_rule_antecedents: int = 3
    max_rules_per_group: int = 25
    min_matched_rows: int = 10
    evaluation_scope: str = "all"  # all or test
    test_size: float = 0.20
    random_state: int = 42
    include_model_predictions: bool = True
    model_selection_metric: str = "weighted_f1"  # weighted_f1 or minority_f1
    selected_features_file: str = "selected_features_anova_first_peak.csv"
    max_selected_features_per_target: int = 10
    verbose: bool = False


@dataclass
class PartGResult:
    """Paths and tables produced by Part G."""

    workbook_path: Path
    selected_rules_path: Path
    runtime_summary_path: Path
    model_selection_path: Path
    selected_rules: pd.DataFrame = field(repr=False)
    runtime_summary: pd.DataFrame = field(repr=False)
    model_selection: pd.DataFrame = field(repr=False)


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_and_join_features_labels(features_path: Path, labels_path: Path) -> pd.DataFrame:
    """Load graph features and hardness labels, then join on canonical keys."""
    features_path = Path(features_path)
    labels_path = Path(labels_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    features = _normalize_columns(pd.read_csv(features_path))
    labels = _normalize_columns(pd.read_csv(labels_path))

    missing_features = [c for c in JOIN_KEYS if c not in features.columns]
    missing_labels = [c for c in JOIN_KEYS if c not in labels.columns]
    if missing_features:
        raise ValueError(f"Features file is missing join keys: {missing_features}")
    if missing_labels:
        raise ValueError(f"Labels file is missing join keys: {missing_labels}")

    label_cols = [c for c in labels.columns if c not in JOIN_KEYS]
    merged = features.merge(labels[list(JOIN_KEYS) + label_cols], on=list(JOIN_KEYS), how="inner")
    if merged.empty:
        raise ValueError("The features and labels join produced zero rows.")
    return merged


def to_hard_binary(series: pd.Series) -> pd.Series:
    """Convert common binary/string label encodings into integer 0/1."""
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=series.index, dtype="Int64")

    numeric_mask = numeric.notna()
    out.loc[numeric_mask] = (numeric.loc[numeric_mask].astype(float) > 0).astype(int)

    if (~numeric_mask).any():
        s = series.loc[~numeric_mask].astype(str).str.strip().str.lower()
        hard_values = {"1", "hard", "true", "yes", "y", "positive", "difficult"}
        not_hard_values = {"0", "not hard", "not-hard", "false", "no", "n", "negative", "easy"}
        converted = []
        for value in s:
            if value in hard_values:
                converted.append(1)
            elif value in not_hard_values:
                converted.append(0)
            else:
                converted.append(pd.NA)
        out.loc[~numeric_mask] = converted

    if out.isna().any():
        bad = series.loc[out.isna()].head(10).tolist()
        raise ValueError(f"Could not convert hardness labels to binary. Examples: {bad}")
    return out.astype(int)


def discover_selected_features_path(part_f_output_dir: Path, target: str, filename: str) -> Path:
    """Find the target-specific Part F selected-feature CSV.

    This is intentionally strict. Part G should not fall back to all feature_*
    columns, and it should not silently reuse the Consensus-5 feature set for
    Consensus-4 or Majority-3. Each target must use its own first-peak feature
    file from Part F:

        <part_f_output_dir>/<target>/tables/selected_features_anova_first_peak.csv

    Passing a target-specific Part F directory directly is also allowed, but
    only when that directory contains tables/<filename>.
    """
    part_f_output_dir = Path(part_f_output_dir)

    target_specific = part_f_output_dir / target / "tables" / filename
    if target_specific.exists():
        return target_specific

    # Allow a direct target-specific directory such as
    # artifacts/ml_hardness_part_f/consensus5_runtime_hardness_label_binary
    direct_target_dir = part_f_output_dir / "tables" / filename
    if direct_target_dir.exists() and part_f_output_dir.name == target:
        return direct_target_dir

    raise FileNotFoundError(
        "Target-specific selected-feature file was not found. Part G is strict "
        "about using only the first-peak selected features from Part F and will "
        "not fall back to all feature_* columns or to another target's selected "
        "features. Expected one of these paths:\n"
        f"  1) {target_specific}\n"
        f"  2) {direct_target_dir}  (only valid when --part-f-output-dir is the target directory)"
    )


def read_selected_features(selected_features_path: Path) -> List[str]:
    """Read feature_* names from Part F's selected_features_anova_first_peak.csv."""
    selected_features_path = Path(selected_features_path)
    if not selected_features_path.exists():
        raise FileNotFoundError(f"Selected features file not found: {selected_features_path}")

    df = _normalize_columns(pd.read_csv(selected_features_path))
    candidate_columns = [
        "feature",
        "features",
        "feature_name",
        "selected_feature",
        "selected_features",
        "Selected Feature",
        "Feature",
    ]
    for col in candidate_columns:
        if col in df.columns:
            values = df[col].dropna().astype(str).str.strip().tolist()
            features = [v for v in values if v.startswith("feature_")]
            if features:
                return list(dict.fromkeys(features))

    # Fallback: scan every cell for values that look like graph-feature names.
    # This supports slightly different CSV schemas, but still only accepts
    # explicit feature_* values from Part F's first-peak selection output.
    found: List[str] = []
    for value in df.astype(str).to_numpy().ravel().tolist():
        value = value.strip()
        if value.startswith("feature_"):
            found.append(value)
    found = list(dict.fromkeys(found))
    if not found:
        raise ValueError(
            f"No feature_* values found in selected feature file: {selected_features_path}"
        )
    return found


def validate_selected_features(
    selected_features: Sequence[str],
    selected_features_path: Path,
    max_selected_features_per_target: int,
) -> List[str]:
    """Validate that Part G is using a small Part F first-peak feature set.

    This protects against accidentally passing a full feature matrix or a file
    containing all 23 graph features. The default cap is intentionally generous
    relative to the expected 3/4/5 selected features, but low enough to catch
    accidental all-feature runs.
    """
    features = list(dict.fromkeys(str(f).strip() for f in selected_features if str(f).strip()))
    invalid = [f for f in features if not f.startswith("feature_")]
    if invalid:
        raise ValueError(
            "Selected feature file contains non-feature columns. Part G accepts "
            f"only feature_* predictors. Invalid values from {selected_features_path}: {invalid}"
        )
    if not features:
        raise ValueError(f"No selected feature_* values found in: {selected_features_path}")
    if max_selected_features_per_target > 0 and len(features) > max_selected_features_per_target:
        raise ValueError(
            "Too many selected features were loaded for Part G. This likely means "
            "the wrong file was supplied, or all graph features were passed instead "
            "of Part F's first-peak selected features.\n"
            f"File: {selected_features_path}\n"
            f"Loaded feature count: {len(features)}\n"
            f"Maximum allowed by --max-selected-features-per-target: {max_selected_features_per_target}\n"
            f"Loaded features: {features}"
        )
    return features


# ---------------------------------------------------------------------------
# Percentile binning and transaction construction
# ---------------------------------------------------------------------------


def percentile_edges_for_bin_count(n_bins: int) -> List[int]:
    """Return the requested percentile labels for 3, 4, or 5 bins."""
    if n_bins == 3:
        return [0, 33, 66, 100]
    if n_bins == 4:
        return [0, 25, 50, 75, 100]
    if n_bins == 5:
        return [0, 20, 40, 60, 80, 100]
    return [int(round(i * 100 / n_bins)) for i in range(n_bins)] + [100]


def bin_item(feature: str, lo: int, hi: int) -> str:
    return f"BIN::{feature}::P{lo}_{hi}"


def outcome_item(value: int) -> str:
    return OUTCOME_HARD if int(value) == 1 else OUTCOME_NOT_HARD


def pretty_feature_name(feature: str) -> str:
    return feature.replace("feature_", "").replace("_", " ")


def pretty_item(item: str) -> str:
    if item == OUTCOME_HARD:
        return "Hard"
    if item == OUTCOME_NOT_HARD:
        return "Not Hard"
    match = BIN_ITEM_RE.match(item)
    if match:
        feature = pretty_feature_name(match.group("feature"))
        return f"{feature} in [{match.group('lo')}%, {match.group('hi')}%]"
    return str(item)


def pretty_items(items: Iterable[str]) -> str:
    return " AND ".join(pretty_item(item) for item in sorted(items))


def bin_features_by_dataset_percentile(
    df: pd.DataFrame,
    selected_features: Sequence[str],
    n_bins: int,
    dataset_col: str = "dataset",
) -> Tuple[pd.DataFrame, List[str]]:
    """Create boolean dataset-wise percentile-bin columns for selected features.

    Percentiles are computed separately inside each dataset, using percentile
    ranks. This avoids leakage across datasets and handles tied feature values
    more robustly than strict quantile edges.
    """
    if dataset_col not in df.columns:
        raise ValueError(f"Missing dataset column: {dataset_col}")

    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        raise ValueError(f"Selected features are missing from joined dataset: {missing}")

    edges = percentile_edges_for_bin_count(n_bins)
    interval_pairs = list(zip(edges[:-1], edges[1:]))
    out = pd.DataFrame(index=df.index)
    created_items: List[str] = []

    for feature in selected_features:
        for lo, hi in interval_pairs:
            item = bin_item(feature, lo, hi)
            out[item] = False
            created_items.append(item)

        numeric = pd.to_numeric(df[feature], errors="coerce")
        for _, group_idx in df.groupby(dataset_col, sort=False).groups.items():
            idx = pd.Index(group_idx)
            values = numeric.loc[idx]
            valid = values.notna()
            if not valid.any():
                continue

            # Percentile rank within this dataset on a 0--100 scale.
            # Using (rank - 1) / (n - 1) puts the minimum at 0 and maximum
            # at 100, matching percentile interval labels such as [0, 33].
            ranks = values.loc[valid].rank(method="average")
            n_valid = int(valid.sum())
            if n_valid <= 1:
                ranks_pct = pd.Series(50.0, index=ranks.index)
            else:
                ranks_pct = (ranks - 1.0) / (n_valid - 1.0) * 100.0
            upper_bounds = np.array(edges[1:], dtype=float)
            bin_indices = np.searchsorted(upper_bounds, ranks_pct.to_numpy(), side="left")
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            for local_pos, bin_idx in zip(ranks_pct.index, bin_indices):
                lo, hi = interval_pairs[int(bin_idx)]
                out.loc[local_pos, bin_item(feature, lo, hi)] = True

    # Remove duplicate created_items if custom n_bins creates repeated rounded labels.
    created_items = list(dict.fromkeys(created_items))
    return out[created_items].astype(bool), created_items


def make_transaction_matrix(
    bin_matrix: pd.DataFrame,
    outcome_values: pd.Series,
) -> pd.DataFrame:
    """Append one-hot Hard/Not-Hard outcome items to the feature-bin matrix."""
    tx = bin_matrix.astype(bool).copy()
    y = to_hard_binary(outcome_values)
    tx[OUTCOME_HARD] = y.eq(1).to_numpy()
    tx[OUTCOME_NOT_HARD] = y.eq(0).to_numpy()
    return tx.astype(bool)


# ---------------------------------------------------------------------------
# Rule mining and rule metrics
# ---------------------------------------------------------------------------


def resolve_min_support(value: str, y: pd.Series) -> float:
    """Resolve fixed or adaptive minimum support.

    The adaptive rule keeps support high enough to avoid very tiny slices, while
    ensuring the minority class can still appear in frequent itemsets.
    """
    if str(value).strip().lower() != "auto":
        fixed = float(value)
        if fixed <= 0 or fixed > 1:
            raise ValueError("--min-support must be in (0, 1] or 'auto'.")
        return fixed

    y_bin = to_hard_binary(y)
    prevalence_hard = float(y_bin.mean())
    minority_prevalence = min(prevalence_hard, 1.0 - prevalence_hard)
    if minority_prevalence <= 0:
        return 1.0

    adaptive = max(0.02, 0.50 * minority_prevalence)
    adaptive = min(0.10, adaptive, 0.90 * minority_prevalence)
    return max(0.001, float(adaptive))


def _association_rules_compat(frequent_itemsets: pd.DataFrame, metric: str, min_threshold: float) -> pd.DataFrame:
    """Call mlxtend.association_rules across mlxtend versions."""
    from mlxtend.frequent_patterns import association_rules

    try:
        return association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    except TypeError:
        # Newer mlxtend versions may require num_itemsets.
        return association_rules(
            frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold,
            num_itemsets=len(frequent_itemsets),
        )


def mine_rules_from_transactions(
    tx: pd.DataFrame,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    max_rule_antecedents: int,
) -> pd.DataFrame:
    """Run FP-Growth and keep only feature-bin(s) -> Hard/Not-Hard rules."""
    from mlxtend.frequent_patterns import fpgrowth

    max_len = max_rule_antecedents + 1  # +1 for the consequent item.
    frequent_itemsets = fpgrowth(
        tx,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = _association_rules_compat(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )
    if rules.empty:
        return pd.DataFrame()

    rows = []
    for _, row in rules.iterrows():
        ants = set(row["antecedents"])
        cons = set(row["consequents"])
        if cons not in ({OUTCOME_HARD}, {OUTCOME_NOT_HARD}):
            continue
        if any(item in OUTCOME_ITEMS for item in ants):
            continue
        if len(ants) == 0 or len(ants) > max_rule_antecedents:
            continue
        if float(row.get("lift", np.nan)) < min_lift:
            continue
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def evaluate_rule_as_classifier(
    antecedent_items: Sequence[str],
    consequent_item: str,
    bin_matrix: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    """Evaluate a rule as an if-then-else binary classifier."""
    y_true = to_hard_binary(y).astype(int)
    if antecedent_items:
        match = bin_matrix[list(antecedent_items)].all(axis=1)
    else:
        match = pd.Series(True, index=bin_matrix.index)

    if consequent_item == OUTCOME_HARD:
        y_pred = match.astype(int)
    elif consequent_item == OUTCOME_NOT_HARD:
        y_pred = (~match).astype(int)
    else:
        raise ValueError(f"Invalid consequent item: {consequent_item}")

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    f1_hard = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_not_hard = f1_score(1 - y_true, 1 - y_pred, pos_label=1, zero_division=0)
    support_hard = int((y_true == 1).sum())
    support_not_hard = int((y_true == 0).sum())
    total = support_hard + support_not_hard
    weighted_f1 = (
        (support_hard * f1_hard + support_not_hard * f1_not_hard) / total
        if total else np.nan
    )
    minority_class = "Hard" if support_hard <= support_not_hard else "Not Hard"
    minority_f1 = f1_hard if minority_class == "Hard" else f1_not_hard

    return {
        "matched_antecedent_rows": int(match.sum()),
        "not_matched_antecedent_rows": int((~match).sum()),
        "satisfy_rule_and_hard": int((match & (y_true == 1)).sum()),
        "satisfy_rule_and_not_hard": int((match & (y_true == 0)).sum()),
        "not_satisfy_rule_and_hard": int(((~match) & (y_true == 1)).sum()),
        "not_satisfy_rule_and_not_hard": int(((~match) & (y_true == 0)).sum()),
        "tp_hard": tp,
        "fp_hard": fp,
        "tn_hard": tn,
        "fn_hard": fn,
        "precision_hard": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_hard": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_hard": f1_hard,
        "precision_not_hard": precision_score(1 - y_true, 1 - y_pred, pos_label=1, zero_division=0),
        "recall_not_hard": recall_score(1 - y_true, 1 - y_pred, pos_label=1, zero_division=0),
        "f1_not_hard": f1_not_hard,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "minority_class": minority_class,
        "minority_class_f1": minority_f1,
        "support_hard_rows": support_hard,
        "support_not_hard_rows": support_not_hard,
    }


def _mask_for_rule(bin_matrix: pd.DataFrame, antecedent_items: Sequence[str]) -> pd.Series:
    if not antecedent_items:
        return pd.Series(True, index=bin_matrix.index)
    return bin_matrix[list(antecedent_items)].all(axis=1)


def select_non_overlapping_rules(
    rules: pd.DataFrame,
    bin_matrix: pd.DataFrame,
    max_rules: int,
    min_matched_rows: int,
) -> pd.DataFrame:
    """Greedily keep rules whose covered rows do not overlap previously kept rules."""
    if rules.empty:
        return rules

    sorted_rules = rules.sort_values(
        by=["support", "lift", "confidence"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    used_mask = pd.Series(False, index=bin_matrix.index)
    kept = []
    for _, row in sorted_rules.iterrows():
        ants = sorted(set(row["antecedents"]))
        mask = _mask_for_rule(bin_matrix, ants)
        matched_rows = int(mask.sum())
        if matched_rows < min_matched_rows:
            continue
        overlap_rows = int((used_mask & mask).sum())
        if overlap_rows > 0:
            continue
        row = row.copy()
        row["selected_rule_overlap_rows_with_prior"] = overlap_rows
        kept.append(row)
        used_mask = used_mask | mask
        if len(kept) >= max_rules:
            break

    if not kept:
        return pd.DataFrame()
    return pd.DataFrame(kept).reset_index(drop=True)


def finalize_rules(
    rules: pd.DataFrame,
    bin_matrix: pd.DataFrame,
    y: pd.Series,
    metadata: Mapping[str, object],
) -> pd.DataFrame:
    """Add readable rule text and classifier-style metrics."""
    rows: List[Dict[str, object]] = []
    for position, (_, row) in enumerate(rules.iterrows(), start=1):
        ants = sorted(set(row["antecedents"]))
        cons = sorted(set(row["consequents"]))
        if len(cons) != 1:
            continue
        consequent = cons[0]
        eval_stats = evaluate_rule_as_classifier(ants, consequent, bin_matrix, y)
        out: Dict[str, object] = dict(metadata)
        out.update(
            {
                "rule_rank_within_group": position,
                "antecedents": pretty_items(ants),
                "consequents": pretty_items(cons),
                "rule": f"IF {pretty_items(ants)} THEN {pretty_items(cons)}",
                "antecedents_raw": " | ".join(ants),
                "consequents_raw": " | ".join(cons),
                "antecedent_item_count": len(ants),
                "association_rule_mining_algorithm": "FP-Growth",
            }
        )
        for metric in [
            "antecedent support",
            "consequent support",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
            "zhangs_metric",
            "jaccard",
            "certainty",
            "kulczynski",
        ]:
            if metric in row.index:
                out[metric.replace(" ", "_")] = row[metric]
        if "selected_rule_overlap_rows_with_prior" in row.index:
            out["selected_rule_overlap_rows_with_prior"] = row["selected_rule_overlap_rows_with_prior"]
        out.update(eval_stats)
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model prediction handling
# ---------------------------------------------------------------------------


def discover_part_f_models(part_f_output_dir: Path, target: str) -> Dict[str, Path]:
    """Find saved Part F models for a target."""
    model_dir = Path(part_f_output_dir) / target / "models"
    if not model_dir.exists():
        # Allow directly passing a target-specific Part F directory.
        model_dir = Path(part_f_output_dir) / "models"
    if not model_dir.exists():
        return {}

    model_paths: Dict[str, Path] = {}
    for path in sorted(model_dir.glob("best_*_selected_features.joblib")):
        match = re.match(r"best_(.+?)_selected_features\.joblib", path.name)
        model_name = match.group(1) if match else path.stem
        model_paths[model_name] = path
    return model_paths


def predict_with_model(model, X: pd.DataFrame) -> np.ndarray:
    """Predict robustly with either pandas feature names or float32 arrays."""
    try:
        pred = model.predict(X)
    except Exception:
        pred = model.predict(X.to_numpy(dtype=np.float32))
    return np.asarray(pred).astype(int)


def evaluate_model_predictions(y_true: pd.Series, y_pred: Sequence[int]) -> Dict[str, float]:
    y = to_hard_binary(y_true).astype(int).to_numpy()
    pred = np.asarray(y_pred).astype(int)
    support_hard = int((y == 1).sum())
    support_not_hard = int((y == 0).sum())
    minority_class = "Hard" if support_hard <= support_not_hard else "Not Hard"
    f1_hard = f1_score(y, pred, pos_label=1, zero_division=0)
    f1_not_hard = f1_score(1 - y, 1 - pred, pos_label=1, zero_division=0)
    return {
        "model_weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "model_minority_class": minority_class,
        "model_minority_class_f1": f1_hard if minority_class == "Hard" else f1_not_hard,
        "model_f1_hard": f1_hard,
        "model_f1_not_hard": f1_not_hard,
        "model_balanced_accuracy": balanced_accuracy_score(y, pred),
        "model_support_hard_rows": support_hard,
        "model_support_not_hard_rows": support_not_hard,
    }


def choose_best_models_by_dataset(
    df: pd.DataFrame,
    target: str,
    selected_features: Sequence[str],
    part_f_output_dir: Path,
    metric: str = "weighted_f1",
) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame]:
    """Select the best saved Part F model separately for each dataset."""
    model_paths = discover_part_f_models(part_f_output_dir, target)
    rows: List[Dict[str, object]] = []
    best_by_dataset: Dict[str, Dict[str, object]] = {}

    if not model_paths:
        return best_by_dataset, pd.DataFrame(
            [{"target": target, "note": "No saved Part F models found for this target."}]
        )
    if joblib is None:
        return best_by_dataset, pd.DataFrame(
            [{"target": target, "note": "joblib is not available; model predictions were skipped."}]
        )

    X = df[list(selected_features)].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    y = to_hard_binary(df[target])

    loaded_models = {}
    for model_name, path in model_paths.items():
        try:
            loaded_models[model_name] = joblib.load(path)
        except Exception as exc:
            rows.append(
                {
                    "target": target,
                    "model_name": model_name,
                    "model_path": str(path),
                    "note": f"Could not load model: {exc}",
                }
            )

    for dataset, idx in df.groupby("dataset", sort=True).groups.items():
        dataset_rows: List[Dict[str, object]] = []
        for model_name, model in loaded_models.items():
            try:
                pred = predict_with_model(model, X.loc[idx])
                metrics = evaluate_model_predictions(y.loc[idx], pred)
                row = {
                    "target": target,
                    "dataset": dataset,
                    "model_name": model_name,
                    "model_path": str(model_paths[model_name]),
                    "n_rows": len(idx),
                    "selected_for_dataset": False,
                }
                row.update(metrics)
                dataset_rows.append(row)
            except Exception as exc:
                dataset_rows.append(
                    {
                        "target": target,
                        "dataset": dataset,
                        "model_name": model_name,
                        "model_path": str(model_paths[model_name]),
                        "n_rows": len(idx),
                        "selected_for_dataset": False,
                        "note": f"Prediction failed: {exc}",
                    }
                )
        if not dataset_rows:
            continue
        valid = pd.DataFrame(dataset_rows)
        score_col = "model_minority_class_f1" if metric == "minority_f1" else "model_weighted_f1"
        if score_col not in valid.columns or valid[score_col].notna().sum() == 0:
            rows.extend(dataset_rows)
            continue
        valid_sorted = valid.sort_values(
            by=[score_col, "model_minority_class_f1", "model_weighted_f1"],
            ascending=[False, False, False],
        )
        best = valid_sorted.iloc[0].to_dict()
        best["selected_for_dataset"] = True
        rows.extend(
            [{**r, "selected_for_dataset": bool(r["model_name"] == best["model_name"])} for r in dataset_rows]
        )
        best_by_dataset[str(dataset)] = {
            "model_name": best["model_name"],
            "model_path": best["model_path"],
            "model": loaded_models[best["model_name"]],
            "metrics": best,
        }

    return best_by_dataset, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration and outputs
# ---------------------------------------------------------------------------


def make_scope_mask(df: pd.DataFrame, target: str, config: PartGConfig) -> pd.Series:
    """Return rows used for rule mining: all rows or a deterministic test subset."""
    if config.evaluation_scope == "all":
        return pd.Series(True, index=df.index)
    if config.evaluation_scope != "test":
        raise ValueError("evaluation_scope must be 'all' or 'test'.")

    y = to_hard_binary(df[target])
    stratify = y if y.value_counts().min() >= 2 else None
    _, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )
    return df.index.isin(test_idx)


def mine_one_group(
    group_df: pd.DataFrame,
    selected_features: Sequence[str],
    outcome_values: pd.Series,
    metadata: Mapping[str, object],
    config: PartGConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Mine rules for one dataset-target-bin-output group."""
    n_bins = int(metadata["bin_count"])
    start_time = time.perf_counter()
    bin_matrix, bin_cols = bin_features_by_dataset_percentile(group_df, selected_features, n_bins)
    min_support = resolve_min_support(config.min_support, outcome_values)
    tx = make_transaction_matrix(bin_matrix, outcome_values)

    frequent_rule_count = 0
    selected_rule_count = 0
    raw_rule_count = 0
    try:
        rules = mine_rules_from_transactions(
            tx,
            min_support=min_support,
            min_confidence=config.min_confidence,
            min_lift=config.min_lift,
            max_rule_antecedents=config.max_rule_antecedents,
        )
        raw_rule_count = len(rules)
        selected = select_non_overlapping_rules(
            rules,
            bin_matrix=bin_matrix,
            max_rules=config.max_rules_per_group,
            min_matched_rows=config.min_matched_rows,
        )
        selected_rule_count = len(selected)
        final = finalize_rules(selected, bin_matrix, outcome_values, metadata)
        frequent_rule_count = raw_rule_count
    except ImportError as exc:
        raise ImportError("Part G requires mlxtend. Install requirements_part_g.txt.") from exc

    elapsed = time.perf_counter() - start_time
    runtime = dict(metadata)
    runtime.update(
        {
            "n_rows": len(group_df),
            "n_selected_features": len(selected_features),
            "n_bin_items": len(bin_cols),
            "resolved_min_support": min_support,
            "min_confidence": config.min_confidence,
            "min_lift": config.min_lift,
            "raw_rule_count_after_filters": raw_rule_count,
            "selected_non_overlapping_rule_count": selected_rule_count,
            "runtime_seconds": elapsed,
        }
    )
    return final, runtime


def order_rule_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "dataset",
        "target",
        "bin_count",
        "output_source",
        "selected_model_name",
        "selected_model_weighted_f1",
        "selected_model_minority_class_f1",
        "selected_model_balanced_accuracy",
        "rule_rank_within_group",
        "association_rule_mining_algorithm",
        "rule",
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
        "conviction",
        "leverage",
        "antecedent_support",
        "consequent_support",
        "matched_antecedent_rows",
        "weighted_f1",
        "minority_class",
        "minority_class_f1",
        "balanced_accuracy",
        "precision_hard",
        "recall_hard",
        "f1_hard",
        "precision_not_hard",
        "recall_not_hard",
        "f1_not_hard",
        "satisfy_rule_and_hard",
        "satisfy_rule_and_not_hard",
        "not_satisfy_rule_and_hard",
        "not_satisfy_rule_and_not_hard",
        "tp_hard",
        "fp_hard",
        "tn_hard",
        "fn_hard",
        "support_hard_rows",
        "support_not_hard_rows",
        "antecedent_item_count",
        "antecedents_raw",
        "consequents_raw",
        "selected_rule_overlap_rows_with_prior",
    ]
    existing = [c for c in preferred if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df[existing + remaining]


def write_outputs(
    selected_rules: pd.DataFrame,
    runtime_summary: pd.DataFrame,
    model_selection: pd.DataFrame,
    config: PartGConfig,
) -> PartGResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rules = order_rule_columns(selected_rules) if not selected_rules.empty else selected_rules
    selected_rules_path = output_dir / "selected_association_rules_part_g.csv"
    runtime_summary_path = output_dir / "association_rule_runtime_summary_part_g.csv"
    model_selection_path = output_dir / "model_selection_by_dataset_part_g.csv"
    workbook_path = output_dir / "part_g_association_rules.xlsx"

    selected_rules.to_csv(selected_rules_path, index=False)
    runtime_summary.to_csv(runtime_summary_path, index=False)
    model_selection.to_csv(model_selection_path, index=False)

    config_summary = pd.DataFrame(
        [
            {
                "parameter": k,
                "value": json.dumps(v) if isinstance(v, (list, tuple, dict)) else str(v),
            }
            for k, v in {
                "features_path": config.features_path,
                "labels_path": config.labels_path,
                "part_f_output_dir": config.part_f_output_dir,
                "targets": list(config.targets),
                "bin_counts": list(config.bin_counts),
                "min_support": config.min_support,
                "min_confidence": config.min_confidence,
                "min_lift": config.min_lift,
                "max_rule_antecedents": config.max_rule_antecedents,
                "max_rules_per_group": config.max_rules_per_group,
                "min_matched_rows": config.min_matched_rows,
                "evaluation_scope": config.evaluation_scope,
                "test_size": config.test_size,
                "random_state": config.random_state,
                "include_model_predictions": config.include_model_predictions,
                "model_selection_metric": config.model_selection_metric,
                "selected_features_file": config.selected_features_file,
                "max_selected_features_per_target": config.max_selected_features_per_target,
            }.items()
        ]
    )

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        config_summary.to_excel(writer, sheet_name="Config", index=False)
        runtime_summary.to_excel(writer, sheet_name="Runtime", index=False)
        model_selection.to_excel(writer, sheet_name="Model Selection", index=False)
        selected_rules.to_excel(writer, sheet_name="All Rules", index=False)
        if "dataset" in selected_rules.columns:
            for dataset, dataset_df in selected_rules.groupby("dataset", sort=True):
                sheet = re.sub(r"[^A-Za-z0-9 _-]", "_", str(dataset))[:31]
                dataset_df.to_excel(writer, sheet_name=sheet, index=False)

    return PartGResult(
        workbook_path=workbook_path,
        selected_rules_path=selected_rules_path,
        runtime_summary_path=runtime_summary_path,
        model_selection_path=model_selection_path,
        selected_rules=selected_rules,
        runtime_summary=runtime_summary,
        model_selection=model_selection,
    )


def run_part_g(config: PartGConfig) -> PartGResult:
    """Run the complete Part G pipeline."""
    merged = load_and_join_features_labels(config.features_path, config.labels_path)
    all_rules: List[pd.DataFrame] = []
    all_runtime: List[Dict[str, object]] = []
    all_model_selection: List[pd.DataFrame] = []

    for target in config.targets:
        if target not in merged.columns:
            raise ValueError(f"Target column not found in joined data: {target}")

        selected_path = discover_selected_features_path(
            config.part_f_output_dir,
            target,
            config.selected_features_file,
        )
        selected_features = validate_selected_features(
            read_selected_features(selected_path),
            selected_features_path=selected_path,
            max_selected_features_per_target=config.max_selected_features_per_target,
        )
        missing = [f for f in selected_features if f not in merged.columns]
        if missing:
            raise ValueError(
                f"Selected features from {selected_path} are not in features file: {missing}"
            )

        target_df = merged.loc[make_scope_mask(merged, target, config)].copy()
        target_df[target] = to_hard_binary(target_df[target])

        best_models: Dict[str, Dict[str, object]] = {}
        if config.include_model_predictions:
            best_models, model_selection = choose_best_models_by_dataset(
                target_df,
                target=target,
                selected_features=selected_features,
                part_f_output_dir=config.part_f_output_dir,
                metric=config.model_selection_metric,
            )
            model_selection["selected_features_path"] = str(selected_path)
            model_selection["n_selected_features"] = len(selected_features)
            all_model_selection.append(model_selection)

        for dataset, dataset_df in target_df.groupby("dataset", sort=True):
            dataset_df = dataset_df.copy()
            for n_bins in config.bin_counts:
                base_meta: Dict[str, object] = {
                    "dataset": dataset,
                    "target": target,
                    "bin_count": int(n_bins),
                    "selected_features_path": str(selected_path),
                    "selected_features": ", ".join(selected_features),
                    "n_selected_features": len(selected_features),
                    "evaluation_scope": config.evaluation_scope,
                }

                # Rules against empirical ground truth.
                meta = dict(base_meta)
                meta.update(
                    {
                        "output_source": "ground_truth",
                        "selected_model_name": "",
                        "selected_model_weighted_f1": np.nan,
                        "selected_model_minority_class_f1": np.nan,
                        "selected_model_balanced_accuracy": np.nan,
                    }
                )
                rules, runtime = mine_one_group(
                    dataset_df,
                    selected_features=selected_features,
                    outcome_values=dataset_df[target],
                    metadata=meta,
                    config=config,
                )
                if not rules.empty:
                    all_rules.append(rules)
                all_runtime.append(runtime)

                # Rules against predictions from this dataset's selected best model.
                model_info = best_models.get(str(dataset))
                if model_info is not None:
                    X = dataset_df[list(selected_features)].apply(pd.to_numeric, errors="coerce")
                    X = X.replace([np.inf, -np.inf], np.nan)
                    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
                    pred = predict_with_model(model_info["model"], X)
                    metrics = model_info.get("metrics", {})
                    meta = dict(base_meta)
                    meta.update(
                        {
                            "output_source": "model_prediction",
                            "selected_model_name": model_info.get("model_name", ""),
                            "selected_model_path": model_info.get("model_path", ""),
                            "selected_model_weighted_f1": metrics.get("model_weighted_f1", np.nan),
                            "selected_model_minority_class_f1": metrics.get("model_minority_class_f1", np.nan),
                            "selected_model_balanced_accuracy": metrics.get("model_balanced_accuracy", np.nan),
                        }
                    )
                    rules, runtime = mine_one_group(
                        dataset_df,
                        selected_features=selected_features,
                        outcome_values=pd.Series(pred, index=dataset_df.index),
                        metadata=meta,
                        config=config,
                    )
                    if not rules.empty:
                        all_rules.append(rules)
                    all_runtime.append(runtime)

    selected_rules = pd.concat(all_rules, ignore_index=True) if all_rules else pd.DataFrame()
    if not selected_rules.empty:
        selected_rules = selected_rules.sort_values(
            by=["dataset", "target", "bin_count", "output_source", "support", "lift"],
            ascending=[True, True, True, True, False, False],
        ).reset_index(drop=True)

    runtime_summary = pd.DataFrame(all_runtime)
    model_selection = (
        pd.concat(all_model_selection, ignore_index=True)
        if all_model_selection else pd.DataFrame()
    )
    return write_outputs(selected_rules, runtime_summary, model_selection, config)
