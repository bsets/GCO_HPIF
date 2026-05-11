"""Part F: ML hardness classification from graph features.

This module joins graph features with Part E runtime-consensus hardness labels,
selects a compact feature subset using ANOVA F-test + first minority-F1 peak,
tunes multiple classifiers, evaluates them on a stratified holdout split, and
saves models/metrics/plots for downstream paper and README use.

Important leakage-control design choice
---------------------------------------
Only columns from ``artifacts/features_full/graph_features.csv`` are eligible as
ML predictors. Runtime columns from ``hardness_labels.csv`` are used only for the
label/target and are not used as model inputs.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - xgboost is optional
    XGBClassifier = None


JOIN_KEYS = ["dataset", "graph_id", "source_index"]

DEFAULT_TARGETS = [
    "consensus5_runtime_hardness_label_binary",
    "consensus4_runtime_hardness_label_binary",
    "majority3_runtime_hardness_label_binary",
]


@dataclass(frozen=True)
class PartFConfig:
    """Configuration for Part F ML hardness classification."""

    features_csv: Path
    labels_csv: Path
    output_dir: Path
    targets: Sequence[str] = tuple(DEFAULT_TARGETS)
    test_size: float = 0.20
    cv_folds: int = 5
    random_state: int = 42
    search_iterations: int = 30
    first_peak_patience: int = 2
    first_peak_min_delta: float = 0.002
    skip_xgb: bool = False


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _ensure_dirs(base: Path) -> Dict[str, Path]:
    paths = {
        "base": base,
        "models": base / "models",
        "plots": base / "plots",
        "tables": base / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _normalize_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    missing = [col for col in JOIN_KEYS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing join key column(s): {missing}")

    df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
    df["graph_id"] = df["graph_id"].astype(str).str.strip()
    df["source_index"] = pd.to_numeric(df["source_index"], errors="raise").astype("int64")

    return df


def _coerce_binary_target(series: pd.Series, target_name: str) -> pd.Series:
    """Coerce a target column to integer 0/1 values."""

    if pd.api.types.is_numeric_dtype(series):
        values = series.astype(int)
    else:
        mapped = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "hard": 1,
                    "not hard": 0,
                    "not_hard": 0,
                    "false": 0,
                    "true": 1,
                    "0": 0,
                    "1": 1,
                }
            )
        )
        if mapped.isna().any():
            bad_values = sorted(series[mapped.isna()].astype(str).unique().tolist())[:10]
            raise ValueError(
                f"Could not coerce target {target_name!r} to binary 0/1. "
                f"Examples: {bad_values}"
            )
        values = mapped.astype(int)

    unique_values = sorted(values.dropna().unique().tolist())
    if unique_values != [0, 1]:
        raise ValueError(
            f"Target {target_name!r} must contain both classes 0 and 1; "
            f"found {unique_values}"
        )

    return values


def build_ml_dataset(
    features_csv: Path,
    labels_csv: Path,
    target_cols: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Join graph features with Part E target columns.

    Only columns from graph_features.csv are eligible as ML predictors. This
    avoids leakage from Part E runtime columns such as runtime_gurobi,
    runtime_top25_gurobi, etc.
    """

    features = _normalize_join_keys(pd.read_csv(features_csv))
    labels = _normalize_join_keys(pd.read_csv(labels_csv))

    missing_targets = [target for target in target_cols if target not in labels.columns]
    if missing_targets:
        raise ValueError(f"Missing target column(s) in labels CSV: {missing_targets}")

    labels_small = labels[JOIN_KEYS + list(target_cols)].drop_duplicates(subset=JOIN_KEYS)

    data = features.merge(
        labels_small,
        on=JOIN_KEYS,
        how="inner",
        validate="one_to_one",
    )

    if data.empty:
        raise ValueError(
            "The feature/label join produced zero rows. "
            "Check dataset, graph_id, and source_index keys."
        )

    # Part F predictor rule:
    # Only canonical graph feature columns are eligible ML predictors.
    # In artifacts/features_full/graph_features.csv, actual graph features are
    # the columns whose names begin with "feature_", e.g.,
    # "feature_Number_of_Nodes". Columns such as "source_index", "n_nodes",
    # "n_edges", and "Time_feature_Number_of_Nodes" are metadata or timing
    # columns and must not be used as predictors.
    feature_cols = [
        col for col in data.columns
        if isinstance(col, str) and col.startswith("feature_")
    ]

    if not feature_cols:
        raise ValueError(
            "No graph feature columns were found. Part F expects predictor "
            "columns in graph_features.csv to have names beginning with "
            "'feature_'."
        )

    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan)

    usable_feature_cols: List[str] = []
    dropped: Dict[str, str] = {}

    for col in feature_cols:
        non_missing = data[col].dropna()

        if non_missing.empty:
            dropped[col] = "all_missing"
            continue

        if non_missing.nunique() <= 1:
            dropped[col] = "constant"
            continue

        usable_feature_cols.append(col)

    if not usable_feature_cols:
        raise ValueError(
            "No usable numeric graph-feature columns were found after excluding "
            "join keys and targets."
        )

    data.attrs["dropped_feature_columns"] = dropped

    return data, usable_feature_cols


def _make_stratify_vector(y: pd.Series, dataset: pd.Series) -> pd.Series:
    """Prefer stratifying by dataset and class; fallback to class only."""

    combo = dataset.astype(str) + "__" + y.astype(str)

    if combo.value_counts().min() >= 2:
        return combo

    return y


def _scoring() -> Dict[str, Any]:
    from sklearn.metrics import make_scorer

    return {
        "minority_f1": make_scorer(f1_score, pos_label=1, zero_division=0),
        "weighted_f1": make_scorer(f1_score, average="weighted", zero_division=0),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }


def make_preselect_pipeline(k: int, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )


def choose_first_peak_k(
    scores: Sequence[float],
    min_delta: float = 0.002,
    patience: int = 2,
    min_k: int = 2,
) -> int:
    """Return the first sustained local peak in the feature-count curve.

    A point is treated as the first peak when it is at least as good as all
    earlier points and the next ``patience`` points do not improve by more than
    ``min_delta``. If no such point exists, the global-best feature count is
    returned.
    """

    if not scores:
        raise ValueError("Cannot select k from an empty score sequence.")

    values = np.asarray(scores, dtype=float)
    n = len(values)
    min_k = max(1, min(min_k, n))

    for idx in range(min_k - 1, n):
        current = values[idx]
        historical_best = np.nanmax(values[: idx + 1])

        if current + min_delta < historical_best:
            continue

        future = values[idx + 1 : idx + 1 + patience]

        if len(future) < patience:
            continue

        if np.all(future <= current + min_delta):
            return idx + 1

    return int(np.nanargmax(values) + 1)


def run_feature_count_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: Sequence[str],
    cv_folds: int,
    random_state: int,
    first_peak_min_delta: float,
    first_peak_patience: int,
) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
    """Evaluate k = 1..p features and select k using first minority-F1 peak."""

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    rows: List[Dict[str, Any]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        warnings.simplefilter("ignore", RuntimeWarning)

        for k in range(1, len(feature_cols) + 1):
            pipe = make_preselect_pipeline(k=k, random_state=random_state)

            scores = cross_validate(
                pipe,
                X_train[feature_cols],
                y_train,
                cv=cv,
                scoring=_scoring(),
                n_jobs=-1,
                error_score="raise",
            )

            rows.append(
                {
                    "num_features": k,
                    "minority_class_f1_mean": float(np.mean(scores["test_minority_f1"])),
                    "minority_class_f1_std": float(np.std(scores["test_minority_f1"])),
                    "weighted_f1_mean": float(np.mean(scores["test_weighted_f1"])),
                    "weighted_f1_std": float(np.std(scores["test_weighted_f1"])),
                    "balanced_accuracy_mean": float(
                        np.mean(scores["test_balanced_accuracy"])
                    ),
                    "balanced_accuracy_std": float(
                        np.std(scores["test_balanced_accuracy"])
                    ),
                }
            )

    curve = pd.DataFrame(rows)

    selected_k = choose_first_peak_k(
        curve["minority_class_f1_mean"].tolist(),
        min_delta=first_peak_min_delta,
        patience=first_peak_patience,
        min_k=2,
    )

    selector_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=selected_k)),
        ]
    )

    selector_pipe.fit(X_train[feature_cols], y_train)

    selector = selector_pipe.named_steps["select"]
    selected_features = np.asarray(feature_cols)[selector.get_support()]

    selected_features_df = pd.DataFrame(
        {
            "feature": selected_features,
            "anova_f_score": selector.scores_[selector.get_support()],
            "anova_p_value": selector.pvalues_[selector.get_support()],
        }
    ).sort_values("anova_f_score", ascending=False)

    return selected_k, curve, selected_features_df


def plot_feature_selection_curve(
    curve: pd.DataFrame,
    selected_k: int,
    output_png: Path,
) -> None:
    long_df = curve.melt(
        id_vars="num_features",
        value_vars=[
            "minority_class_f1_mean",
            "weighted_f1_mean",
            "balanced_accuracy_mean",
        ],
        var_name="metric",
        value_name="score",
    )

    label_map = {
        "minority_class_f1_mean": "Minority class F1",
        "weighted_f1_mean": "Weighted F1",
        "balanced_accuracy_mean": "Balanced accuracy",
    }

    long_df["metric"] = long_df["metric"].map(label_map)

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=long_df,
        x="num_features",
        y="score",
        hue="metric",
        marker="o",
    )
    plt.axvline(selected_k, linestyle="--", linewidth=1.5)
    plt.xlabel("Number of selected graph features")
    plt.ylabel("Cross-validated score")
    plt.title("ANOVA feature-count selection curve")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def _get_xgb_scale_pos_weight(y: pd.Series) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    if positives == 0:
        return 1.0

    return float(negatives / positives)


def _model_spaces(
    random_state: int,
    y_train: pd.Series,
    skip_xgb: bool,
) -> Dict[str, Tuple[BaseEstimator, Mapping[str, Sequence[Any]], int]]:
    spaces: Dict[str, Tuple[BaseEstimator, Mapping[str, Sequence[Any]], int]] = {
        "rf": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [300, 600, 900],
                "model__max_depth": [None, 5, 10, 20, 40],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__max_features": ["sqrt", "log2", None],
            },
            35,
        ),
        "lr": (
            LogisticRegression(
                class_weight="balanced",
                max_iter=8000,
                solver="liblinear",
                random_state=random_state,
            ),
            {
                "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "model__penalty": ["l1", "l2"],
            },
            12,
        ),
        "svc": (
            SVC(
                class_weight="balanced",
                random_state=random_state,
            ),
            {
                "model__C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
                "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
                "model__kernel": ["rbf"],
            },
            30,
        ),
    }

    if not skip_xgb and XGBClassifier is not None:
        spaces["xgb"] = (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
                scale_pos_weight=_get_xgb_scale_pos_weight(y_train),
            ),
            {
                "model__n_estimators": [200, 400, 700, 1000],
                "model__max_depth": [2, 3, 4, 5, 7],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__min_child_weight": [1, 3, 5, 10],
                "model__reg_lambda": [0.5, 1.0, 3.0, 10.0],
                "model__reg_alpha": [0.0, 0.01, 0.1, 1.0],
            },
            45,
        )

    return spaces


def make_model_pipeline(
    estimator: BaseEstimator,
    selected_k: Optional[int],
) -> Pipeline:
    steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    if selected_k is not None:
        steps.append(("select", SelectKBest(score_func=f_classif, k=selected_k)))

    steps.append(("model", estimator))

    return Pipeline(steps=steps)


def tune_model(
    model_name: str,
    estimator: BaseEstimator,
    param_space: Mapping[str, Sequence[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_k: Optional[int],
    cv_folds: int,
    random_state: int,
    search_iterations: int,
    default_iterations: int,
) -> RandomizedSearchCV:
    pipe = make_model_pipeline(estimator=estimator, selected_k=selected_k)

    n_iter = min(search_iterations, default_iterations)

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring=_scoring(),
        refit="minority_f1",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        search.fit(X_train, y_train)

    return search


def predict_scores(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]

    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        return np.asarray(decision, dtype=float)

    return np.asarray(model.predict(X), dtype=float)


def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_score = predict_scores(model, X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[1],
        average=None,
        zero_division=0,
    )

    metrics = {
        "accuracy": float((y_pred == y_test).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "hard_precision": float(precision[0]),
        "hard_recall": float(recall[0]),
        "hard_f1": float(f1[0]),
        "weighted_f1": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(average_precision_score(y_test, y_score))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    return metrics


def plot_roc_curves(
    models: Mapping[str, BaseEstimator],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_png: Path,
) -> None:
    plt.figure(figsize=(8, 7))

    for model_name, model in models.items():
        y_score = predict_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name.upper()} ROC-AUC = {roc_auc:.3f}",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves for tuned models after feature selection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def plot_pr_curves(
    models: Mapping[str, BaseEstimator],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_png: Path,
) -> None:
    plt.figure(figsize=(8, 7))

    for model_name, model in models.items():
        y_score = predict_scores(model, X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        pr_auc = average_precision_score(y_test, y_score)

        plt.plot(
            recall,
            precision,
            label=f"{model_name.upper()} PR-AUC = {pr_auc:.3f}",
        )

    baseline = float(np.mean(y_test == 1))
    plt.axhline(
        baseline,
        linestyle="--",
        linewidth=1,
        label=f"Hard-class baseline = {baseline:.3f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves for tuned models after feature selection")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def plot_confusion_matrix_2x2(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_png: Path,
    output_csv: Path,
) -> None:
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    cm_df = pd.DataFrame(
        cm,
        index=["actual_not_hard", "actual_hard"],
        columns=["pred_not_hard", "pred_hard"],
    )

    cm_df.to_csv(output_csv)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("2 by 2 confusion matrix for best model")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def _extract_feature_importance(
    model: Pipeline,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        raise ValueError(
            f"Estimator {type(estimator).__name__} does not expose "
            "feature_importances_."
        )

    importances = np.asarray(estimator.feature_importances_, dtype=float)

    if len(importances) != len(feature_cols):
        raise ValueError(
            f"Feature-importance length mismatch: got {len(importances)} "
            f"importances for {len(feature_cols)} features."
        )

    return pd.DataFrame(
        {
            "feature": list(feature_cols),
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_png: Path,
    title: str,
    max_features: int = 30,
) -> None:
    plot_df = importance_df.head(max_features).iloc[::-1]

    height = max(5, 0.28 * len(plot_df) + 2)

    plt.figure(figsize=(10, height))
    sns.barplot(
        data=plot_df,
        x="importance",
        y="feature",
        orient="h",
    )
    plt.xlabel("Model feature importance")
    plt.ylabel("Graph feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def _rank_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df.sort_values(
        by=["hard_f1", "balanced_accuracy", "pr_auc", "roc_auc"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def run_for_target(config: PartFConfig, target: str) -> Dict[str, Any]:
    target_dir = config.output_dir / _safe_name(target)
    paths = _ensure_dirs(target_dir)

    data, feature_cols = build_ml_dataset(
        config.features_csv,
        config.labels_csv,
        config.targets,
    )

    data[target] = _coerce_binary_target(data[target], target)

    data.to_csv(
        paths["tables"] / f"ml_dataset_{_safe_name(target)}.csv",
        index=False,
    )

    y = data[target].astype(int)
    X = data[feature_cols]

    stratify_vector = _make_stratify_vector(y, data["dataset"])

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        data.index,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_vector,
    )

    split_train = data.loc[train_idx, JOIN_KEYS + [target]].copy()
    split_train["split"] = "train"
    split_test = data.loc[test_idx, JOIN_KEYS + [target]].copy()
    split_test["split"] = "test"
    split_df = pd.concat([split_train, split_test], ignore_index=True)
    split_df.to_csv(paths["tables"] / "train_test_split_graph_ids.csv", index=False)

    selected_k, selection_curve, selected_features_df = run_feature_count_selection(
        X_train=X_train,
        y_train=y_train,
        feature_cols=feature_cols,
        cv_folds=config.cv_folds,
        random_state=config.random_state,
        first_peak_min_delta=config.first_peak_min_delta,
        first_peak_patience=config.first_peak_patience,
    )

    selection_curve.to_csv(
        paths["tables"] / "feature_selection_curve.csv",
        index=False,
    )

    selected_features_df.to_csv(
        paths["tables"] / "selected_features_anova_first_peak.csv",
        index=False,
    )

    plot_feature_selection_curve(
        selection_curve,
        selected_k,
        paths["plots"] / "feature_selection_curve_seaborn.png",
    )

    model_spaces = _model_spaces(
        config.random_state,
        y_train,
        config.skip_xgb,
    )

    selected_models: Dict[str, BaseEstimator] = {}
    metrics_rows: List[Dict[str, Any]] = []
    cv_rows: List[pd.DataFrame] = []

    for model_name, (estimator, param_space, default_n_iter) in model_spaces.items():
        search = tune_model(
            model_name=model_name,
            estimator=estimator,
            param_space=param_space,
            X_train=X_train,
            y_train=y_train,
            selected_k=selected_k,
            cv_folds=config.cv_folds,
            random_state=config.random_state,
            search_iterations=config.search_iterations,
            default_iterations=default_n_iter,
        )

        best_model = search.best_estimator_
        selected_models[model_name] = best_model

        joblib.dump(
            best_model,
            paths["models"] / f"best_{model_name}_selected_features.joblib",
        )

        row = {
            "model": model_name,
            "selected_k": selected_k,
            "cv_best_minority_f1": float(search.best_score_),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        }

        row.update(evaluate_model(best_model, X_test, y_test))
        metrics_rows.append(row)

        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.insert(0, "model", model_name)
        cv_rows.append(cv_df)

    metrics_df = _rank_models(pd.DataFrame(metrics_rows))

    metrics_df.to_csv(
        paths["tables"] / "metrics_by_model_selected_features.csv",
        index=False,
    )

    if cv_rows:
        pd.concat(cv_rows, ignore_index=True).to_csv(
            paths["tables"] / "hyperparameter_cv_results_selected_features.csv",
            index=False,
        )

    plot_roc_curves(
        selected_models,
        X_test,
        y_test,
        paths["plots"] / "roc_curves_tuned_selected_models.png",
    )

    plot_pr_curves(
        selected_models,
        X_test,
        y_test,
        paths["plots"] / "pr_curves_tuned_selected_models.png",
    )

    best_model_name = str(metrics_df.iloc[0]["model"])
    best_model = selected_models[best_model_name]

    plot_confusion_matrix_2x2(
        best_model,
        X_test,
        y_test,
        paths["plots"] / f"confusion_matrix_2x2_best_{best_model_name}.png",
        paths["tables"] / f"confusion_matrix_2x2_best_{best_model_name}.csv",
    )

    all_feature_importance_outputs: Dict[str, str] = {}

    for model_name in ["rf", "xgb"]:
        if model_name not in model_spaces:
            continue

        estimator, param_space, default_n_iter = model_spaces[model_name]

        search = tune_model(
            model_name=f"{model_name}_all_features",
            estimator=estimator,
            param_space=param_space,
            X_train=X_train,
            y_train=y_train,
            selected_k=None,
            cv_folds=config.cv_folds,
            random_state=config.random_state,
            search_iterations=config.search_iterations,
            default_iterations=default_n_iter,
        )

        best_all_feature_model = search.best_estimator_

        joblib.dump(
            best_all_feature_model,
            paths["models"] / f"best_{model_name}_all_features_before_feature_selection.joblib",
        )

        importance_df = _extract_feature_importance(
            best_all_feature_model,
            feature_cols,
        )

        importance_csv = (
            paths["tables"]
            / f"{model_name}_feature_importance_all_features_before_selection.csv"
        )

        importance_png = (
            paths["plots"]
            / f"{model_name}_feature_importance_all_features_before_selection.png"
        )

        importance_df.to_csv(importance_csv, index=False)

        plot_feature_importance(
            importance_df,
            importance_png,
            title=f"{model_name.upper()} feature importance before feature selection",
            max_features=len(feature_cols),
        )

        all_feature_importance_outputs[model_name] = str(importance_png)

    manifest = {
        "target": target,
        "n_rows_joined": int(len(data)),
        "n_features_available": int(len(feature_cols)),
        "feature_column_selection_rule": "Only columns whose names start with 'feature_' are used as graph-feature predictors.",
        "feature_columns": list(feature_cols),
        "dropped_feature_columns": data.attrs.get("dropped_feature_columns", {}),
        "class_counts_total": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "class_counts_train": {
            str(k): int(v) for k, v in y_train.value_counts().sort_index().items()
        },
        "class_counts_test": {
            str(k): int(v) for k, v in y_test.value_counts().sort_index().items()
        },
        "selected_k_first_peak": int(selected_k),
        "selected_features_csv": str(
            paths["tables"] / "selected_features_anova_first_peak.csv"
        ),
        "best_model_by_hard_f1_then_balanced_accuracy": best_model_name,
        "metrics_csv": str(paths["tables"] / "metrics_by_model_selected_features.csv"),
        "roc_plot": str(paths["plots"] / "roc_curves_tuned_selected_models.png"),
        "pr_plot": str(paths["plots"] / "pr_curves_tuned_selected_models.png"),
        "feature_selection_plot": str(
            paths["plots"] / "feature_selection_curve_seaborn.png"
        ),
        "all_feature_importance_plots": all_feature_importance_outputs,
    }

    with open(paths["base"] / "part_f_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def run_part_f(config: PartFConfig) -> List[Dict[str, Any]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    manifests: List[Dict[str, Any]] = []

    for target in config.targets:
        print(f"\n=== Part F target: {target} ===")
        manifests.append(run_for_target(config, target))

    summary_df = pd.DataFrame(manifests)
    summary_df.to_csv(config.output_dir / "part_f_target_summary.csv", index=False)

    with open(
        config.output_dir / "part_f_manifest_all_targets.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifests, f, indent=2)

    return manifests
