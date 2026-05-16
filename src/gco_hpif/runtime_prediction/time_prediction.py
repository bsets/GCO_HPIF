"""Part H: maximum-clique runtime prediction from graph features.

This module builds a runtime-prediction dataset from the Part C graph features and
standardized Part D/E solver output files, then trains one runtime-regression
problem per solver/algorithm.

Design choices
--------------
* Predictors are the Part C ``feature_`` columns only, by default all 23 graph
  features.
* The model target is runtime in seconds, but all estimators are wrapped in a
  ``TransformedTargetRegressor`` so they are trained on ``log(runtime_seconds)``
  and predict back on the original seconds scale.
* Hyperparameter tuning uses cross-validated MAPE on the original seconds scale.
* The winning model for each solver is selected by test-set MAPE, with RMSE and
  R2 reported as secondary diagnostics.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Conservative defaults reduce BLAS / joblib nested parallelism crashes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

try:  # XGBoost is optional in some environments.
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - depends on local environment
    XGBRegressor = None

JOIN_KEYS = ["dataset", "graph_id", "source_index"]
RUNTIME_COL = "runtime_seconds"
ALGORITHM_COL = "algorithm"

CANONICAL_ALGORITHM_ORDER = ["Gurobi", "CliSAT", "MOMC", "EGN", "HGS"]

DEFAULT_SOLVER_RUNS = [
    "artifacts/solver_runs/gurobi_full/solver_runs.csv",
    "artifacts/solver_runs/clisat_full/solver_runs.csv",
    "artifacts/solver_runs/momc_full/solver_runs.csv",
    "artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv",
    "artifacts/solver_runs/hgs_full_all_test_graphs/solver_runs.csv",
]


@dataclass
class RuntimePredictionConfig:
    """Configuration for Part H runtime prediction."""

    features_path: Path
    solver_run_paths: List[Path]
    output_dir: Path
    feature_columns: Optional[List[str]] = None
    algorithms: List[str] = field(default_factory=lambda: CANONICAL_ALGORITHM_ORDER.copy())
    models: List[str] = field(default_factory=lambda: ["XGB", "RF", "SVR", "LR"])
    target_transform: str = "log"
    winner_metric: str = "mape"
    test_size: float = 0.20
    cv_folds: int = 5
    random_state: int = 42
    grid_size: str = "compact"
    grid_n_jobs: int = 1
    model_n_jobs: int = 1
    pre_dispatch: str = "1*n_jobs"
    verbose: int = 1
    common_instance_only: bool = True
    min_runtime_seconds: float = 1e-9
    max_rows_per_algorithm: Optional[int] = None
    top_n_features_plot: int = 10
    include_dataset_onehot: bool = False
    save_models: bool = True

    def __post_init__(self) -> None:
        self.features_path = Path(self.features_path)
        self.solver_run_paths = [Path(p) for p in self.solver_run_paths]
        self.output_dir = Path(self.output_dir)
        self.models = normalize_model_names(self.models)
        self.algorithms = [canonical_algorithm_name(a) for a in self.algorithms]
        if self.target_transform != "log":
            raise ValueError(
                "Part H is designed to model log(runtime_seconds). "
                "Set target_transform='log'."
            )
        if self.winner_metric.lower() != "mape":
            raise ValueError("Part H selects the best model by MAPE; use winner_metric='mape'.")
        if self.grid_size not in {"tiny", "compact", "standard"}:
            raise ValueError("grid_size must be one of: tiny, compact, standard")


class ToFloat32Array(BaseEstimator, TransformerMixin):
    """Convert tabular data to a plain float32 NumPy array.

    This avoids fragile pandas feature-name handling in some XGBoost versions
    while still allowing the rest of the pipeline to receive pandas data.
    """

    def fit(self, X, y=None):  # noqa: D401
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


def canonical_algorithm_name(value: str) -> str:
    """Map path/solver-name variants to canonical algorithm names."""
    text = str(value).strip()
    lower = text.lower()
    if "gurobi" in lower:
        return "Gurobi"
    if "clisat" in lower or "cli-sat" in lower:
        return "CliSAT"
    if "momc" in lower:
        return "MOMC"
    if re.search(r"(^|[^a-z])egn([^a-z]|$)", lower) or "erdos" in lower:
        return "EGN"
    if "hgs" in lower or "geometric" in lower:
        return "HGS"
    return text


def infer_algorithm_from_path(path: Path) -> str:
    return canonical_algorithm_name(str(path))


def normalize_model_names(models: Sequence[str]) -> List[str]:
    """Normalize CLI model names.

    ``SVC`` is accepted as an alias but mapped to ``SVR`` because runtime
    prediction is a regression problem.
    """
    out: List[str] = []
    aliases = {"SVC": "SVR", "SVM": "SVR", "RIDGE": "LR", "LINEAR": "LR"}
    valid = {"XGB", "RF", "SVR", "LR"}
    for model in models:
        m = str(model).strip().upper()
        if not m:
            continue
        m = aliases.get(m, m)
        if m not in valid:
            raise ValueError(f"Unsupported model '{model}'. Allowed: XGB, RF, SVR/SVC, LR.")
        if m not in out:
            out.append(m)
    if not out:
        raise ValueError("No valid models were provided.")
    return out


def parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def get_feature_columns(features_df: pd.DataFrame, requested: Optional[Sequence[str]] = None) -> List[str]:
    if requested:
        missing = [c for c in requested if c not in features_df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from features file: {missing}")
        bad = [c for c in requested if not c.startswith("feature_")]
        if bad:
            raise ValueError(f"Part H predictors must be Part C feature_ columns only. Invalid: {bad}")
        return list(requested)

    feature_cols = [c for c in features_df.columns if c.startswith("feature_")]
    if not feature_cols:
        raise ValueError("No feature_ columns found in the features file.")
    return feature_cols


def validate_join_keys(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in JOIN_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing join keys {missing} in {path}")


def load_features(path: Path, requested_feature_columns: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    validate_join_keys(df, Path(path))
    feature_cols = get_feature_columns(df, requested_feature_columns)
    keep = JOIN_KEYS + feature_cols
    return df[keep].copy(), feature_cols


def load_solver_runtime_long(solver_run_paths: Sequence[Path]) -> pd.DataFrame:
    """Load standardized solver_runs.csv files into long runtime format."""
    rows = []
    for path in solver_run_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Solver run file not found: {path}")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        validate_join_keys(df, path)
        if RUNTIME_COL not in df.columns:
            raise ValueError(f"Missing '{RUNTIME_COL}' in {path}")

        if "solver_name" in df.columns:
            solver_values = df["solver_name"].dropna().astype(str).unique().tolist()
            if solver_values:
                algorithm = canonical_algorithm_name(solver_values[0])
            else:
                algorithm = infer_algorithm_from_path(path)
        else:
            algorithm = infer_algorithm_from_path(path)

        tmp = df[JOIN_KEYS + [RUNTIME_COL]].copy()
        tmp[ALGORITHM_COL] = algorithm
        tmp["solver_run_file"] = str(path)
        rows.append(tmp)

    if not rows:
        raise ValueError("No solver run files were provided.")

    out = pd.concat(rows, ignore_index=True)
    out[RUNTIME_COL] = pd.to_numeric(out[RUNTIME_COL], errors="coerce")
    out["dataset"] = out["dataset"].astype(str).str.strip()
    return out


def restrict_to_common_instances(runtime_long: pd.DataFrame, algorithms: Sequence[str]) -> pd.DataFrame:
    """Keep only instances that have runtime rows for all requested algorithms."""
    algorithms = [canonical_algorithm_name(a) for a in algorithms]
    key_algo = runtime_long[JOIN_KEYS + [ALGORITHM_COL]].drop_duplicates()
    counts = key_algo.groupby(JOIN_KEYS)[ALGORITHM_COL].nunique().reset_index(name="n_algorithms")
    common_keys = counts[counts["n_algorithms"] >= len(set(algorithms))][JOIN_KEYS]
    return runtime_long.merge(common_keys, on=JOIN_KEYS, how="inner")


def build_runtime_prediction_dataset(config: RuntimePredictionConfig) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Build long and wide runtime-prediction datasets.

    Returns
    -------
    long_df:
        One row per instance-algorithm pair.
    wide_df:
        One row per instance with separate runtime columns for each algorithm.
    feature_cols:
        Predictor columns used by the models.
    """
    features_df, feature_cols = load_features(config.features_path, config.feature_columns)
    runtime_long = load_solver_runtime_long(config.solver_run_paths)

    requested_algorithms = [canonical_algorithm_name(a) for a in config.algorithms]
    runtime_long = runtime_long[runtime_long[ALGORITHM_COL].isin(requested_algorithms)].copy()
    if runtime_long.empty:
        raise ValueError(f"No runtime rows remain for requested algorithms: {requested_algorithms}")

    if config.common_instance_only:
        runtime_long = restrict_to_common_instances(runtime_long, requested_algorithms)

    runtime_long = runtime_long.dropna(subset=[RUNTIME_COL]).copy()
    runtime_long = runtime_long[runtime_long[RUNTIME_COL] > config.min_runtime_seconds].copy()
    if runtime_long.empty:
        raise ValueError("No positive runtime rows are available after filtering.")

    long_df = runtime_long.merge(features_df, on=JOIN_KEYS, how="inner")
    if long_df.empty:
        raise ValueError("Merging features with solver runtimes produced zero rows.")

    # Ensure deterministic order and useful metadata.
    long_df[ALGORITHM_COL] = pd.Categorical(
        long_df[ALGORITHM_COL],
        categories=[a for a in CANONICAL_ALGORITHM_ORDER if a in requested_algorithms],
        ordered=True,
    )
    long_df = long_df.sort_values([ALGORITHM_COL, "dataset", "graph_id", "source_index"]).reset_index(drop=True)
    long_df["log_runtime_seconds"] = np.log(long_df[RUNTIME_COL].astype(float))

    wide_runtime = runtime_long.pivot_table(
        index=JOIN_KEYS,
        columns=ALGORITHM_COL,
        values=RUNTIME_COL,
        aggfunc="first",
    ).reset_index()
    wide_runtime.columns = [
        f"runtime_seconds_{c}" if c not in JOIN_KEYS else c for c in wide_runtime.columns
    ]
    wide_df = features_df.merge(wide_runtime, on=JOIN_KEYS, how="inner")

    return long_df, wide_df, feature_cols


def safe_mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if not mask.any():
        return float("nan")
    return float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))


def safe_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def make_log_target_regressor(pipeline: Pipeline) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log,
        inverse_func=np.exp,
        check_inverse=False,
    )


def build_model_and_grid(model_name: str, random_state: int, model_n_jobs: int, grid_size: str) -> Tuple[TransformedTargetRegressor, Dict[str, List]]:
    """Return a log-target regressor and grid for a model name."""
    model_name = normalize_model_names([model_name])[0]

    if model_name == "XGB":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed. Install it with: pip install xgboost")
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=model_n_jobs,
            tree_method="hist",
            verbosity=0,
        )
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
            ("to_float32", ToFloat32Array()),
            ("model", model),
        ])
        if grid_size == "tiny":
            grid = {
                "regressor__model__n_estimators": [50],
                "regressor__model__max_depth": [2],
                "regressor__model__learning_rate": [0.1],
            }
        elif grid_size == "compact":
            grid = {
                "regressor__model__n_estimators": [100, 300],
                "regressor__model__max_depth": [3, 5],
                "regressor__model__learning_rate": [0.03, 0.1],
                "regressor__model__subsample": [0.8, 1.0],
                "regressor__model__colsample_bytree": [0.8, 1.0],
            }
        else:
            grid = {
                "regressor__model__n_estimators": [100, 300, 500],
                "regressor__model__max_depth": [3, 5, 7],
                "regressor__model__learning_rate": [0.01, 0.03, 0.1],
                "regressor__model__subsample": [0.8, 1.0],
                "regressor__model__colsample_bytree": [0.6, 0.8, 1.0],
            }
        return make_log_target_regressor(pipe), grid

    if model_name == "RF":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
            ("model", RandomForestRegressor(random_state=random_state, n_jobs=model_n_jobs)),
        ])
        if grid_size == "tiny":
            grid = {
                "regressor__model__n_estimators": [30],
                "regressor__model__max_depth": [None],
                "regressor__model__min_samples_leaf": [1],
                "regressor__model__max_features": ["sqrt"],
            }
        elif grid_size == "compact":
            grid = {
                "regressor__model__n_estimators": [200, 500],
                "regressor__model__max_depth": [None, 10, 20],
                "regressor__model__min_samples_leaf": [1, 2],
                "regressor__model__max_features": ["sqrt"],
            }
        else:
            grid = {
                "regressor__model__n_estimators": [200, 500, 800],
                "regressor__model__max_depth": [None, 10, 20, 40],
                "regressor__model__min_samples_split": [2, 5, 10],
                "regressor__model__min_samples_leaf": [1, 2, 4],
                "regressor__model__max_features": ["sqrt", "log2"],
            }
        return make_log_target_regressor(pipe), grid

    if model_name == "SVR":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
            ("model", SVR()),
        ])
        if grid_size == "tiny":
            grid = {
                "regressor__model__C": [1.0],
                "regressor__model__gamma": ["scale"],
                "regressor__model__epsilon": [0.1],
                "regressor__model__kernel": ["rbf"],
            }
        elif grid_size == "compact":
            grid = {
                "regressor__model__C": [1, 10, 100],
                "regressor__model__gamma": ["scale", 0.01],
                "regressor__model__epsilon": [0.01, 0.1],
                "regressor__model__kernel": ["rbf", "linear"],
            }
        else:
            grid = {
                "regressor__model__C": [0.1, 1, 10, 100],
                "regressor__model__gamma": ["scale", 0.1, 0.01, 0.001],
                "regressor__model__epsilon": [0.01, 0.1, 0.5],
                "regressor__model__kernel": ["rbf", "linear"],
            }
        return make_log_target_regressor(pipe), grid

    if model_name == "LR":
        # Tuned regularized linear regression; reported as LR for continuity
        # with the earlier experiment notes.
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
            ("model", Ridge(random_state=random_state)),
        ])
        if grid_size == "tiny":
            grid = {"regressor__model__alpha": [1.0]}
        elif grid_size == "compact":
            grid = {"regressor__model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        else:
            grid = {"regressor__model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
        return make_log_target_regressor(pipe), grid

    raise ValueError(f"Unsupported model: {model_name}")


def choose_stratify_labels(df: pd.DataFrame, test_size: float) -> Optional[pd.Series]:
    """Use dataset stratification when each dataset has enough rows."""
    if "dataset" not in df.columns:
        return None
    counts = df["dataset"].value_counts()
    if counts.empty or counts.min() < 2:
        return None
    # Need at least one test sample per class.
    if int(math.ceil(len(df) * test_size)) < len(counts):
        return None
    return df["dataset"]


def train_one_algorithm(
    algorithm_df: pd.DataFrame,
    algorithm: str,
    feature_cols: Sequence[str],
    config: RuntimePredictionConfig,
) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    """Train/tune all configured models for one algorithm."""
    working = algorithm_df.dropna(subset=[RUNTIME_COL]).copy()
    working = working[working[RUNTIME_COL] > config.min_runtime_seconds].copy()
    if config.max_rows_per_algorithm and len(working) > config.max_rows_per_algorithm:
        working = working.sample(n=config.max_rows_per_algorithm, random_state=config.random_state).reset_index(drop=True)

    if len(working) < max(10, config.cv_folds * 2):
        raise ValueError(
            f"Not enough rows for {algorithm}: {len(working)} rows available. "
            "Reduce cv_folds/test_size or check the solver outputs."
        )

    X = working[list(feature_cols)].copy()
    if config.include_dataset_onehot:
        # Optional convenience for sensitivity analysis; disabled by default
        # because the requested predictors are the 23 graph features.
        X = pd.concat([X, pd.get_dummies(working["dataset"], prefix="dataset")], axis=1)

    y = working[RUNTIME_COL].astype(float).to_numpy()
    metadata_cols = JOIN_KEYS + [ALGORITHM_COL, RUNTIME_COL, "log_runtime_seconds"]
    metadata = working[[c for c in metadata_cols if c in working.columns]].copy()

    stratify = choose_stratify_labels(working, config.test_size)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        metadata,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    model_rows: List[Dict] = []
    prediction_frames: List[pd.DataFrame] = []
    best_estimators: Dict[str, object] = {}

    for model_name in config.models:
        if model_name == "XGB" and XGBRegressor is None:
            warnings.warn("Skipping XGB because xgboost is not installed.", RuntimeWarning)
            continue

        print(f"Training {model_name} runtime predictor for {algorithm} ...")
        estimator, param_grid = build_model_and_grid(
            model_name=model_name,
            random_state=config.random_state,
            model_n_jobs=config.model_n_jobs,
            grid_size=config.grid_size,
        )

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="neg_mean_absolute_percentage_error",
            cv=config.cv_folds,
            n_jobs=config.grid_n_jobs,
            pre_dispatch=config.pre_dispatch,
            verbose=config.verbose,
            return_train_score=False,
            error_score="raise",
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        best_estimators[model_name] = best

        y_pred = np.asarray(best.predict(X_test), dtype=float)
        y_pred = np.maximum(y_pred, config.min_runtime_seconds)
        log_y_test = np.log(y_test)
        log_y_pred = np.log(y_pred)

        row = {
            "Algorithm": algorithm,
            "Model": model_name,
            "Target Transform": config.target_transform,
            "CV Optimize Metric": "MAPE_on_original_runtime_seconds",
            "Best CV MAPE": float(-grid.best_score_),
            "Best Hyperparameters": json.dumps(grid.best_params_, sort_keys=True),
            "Test MAPE": safe_mape(y_test, y_pred),
            "Test MAE": float(mean_absolute_error(y_test, y_pred)),
            "Test RMSE": safe_rmse(y_test, y_pred),
            "Test R2": float(r2_score(y_test, y_pred)),
            "Test Log MAE": float(mean_absolute_error(log_y_test, log_y_pred)),
            "Test Log RMSE": safe_rmse(log_y_test, log_y_pred),
            "Train Rows": int(len(X_train)),
            "Test Rows": int(len(X_test)),
            "Feature Count": int(len(X.columns)),
        }

        model_filename = f"{algorithm}_{model_name}_best_log_runtime_regressor.joblib"
        if config.save_models:
            model_path = config.output_dir / "models" / model_filename
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best, model_path)
            row["Saved Model File"] = str(model_path.relative_to(config.output_dir))
        else:
            row["Saved Model File"] = ""

        model_rows.append(row)

        pred_df = meta_test.reset_index(drop=True).copy()
        pred_df["Algorithm"] = algorithm
        pred_df["Model"] = model_name
        pred_df["Actual Runtime Seconds"] = y_test
        pred_df["Predicted Runtime Seconds"] = y_pred
        pred_df["Actual Log Runtime"] = log_y_test
        pred_df["Predicted Log Runtime"] = log_y_pred
        pred_df["Absolute Percentage Error"] = np.abs((y_test - y_pred) / y_test)
        prediction_frames.append(pred_df)

    if not model_rows:
        raise RuntimeError(f"No models were successfully trained for {algorithm}.")

    metrics_df = pd.DataFrame(model_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    # Return best estimators in a hidden attribute-like dataframe for saving feature importances.
    best_estimator_rows = [
        {"Algorithm": algorithm, "Model": model, "Estimator": estimator}
        for model, estimator in best_estimators.items()
    ]
    estimators_df = pd.DataFrame(best_estimator_rows)
    return model_rows, predictions_df, estimators_df


def get_underlying_pipeline(estimator) -> Optional[Pipeline]:
    if hasattr(estimator, "regressor_"):
        reg = estimator.regressor_
    elif hasattr(estimator, "regressor"):
        reg = estimator.regressor
    else:
        reg = estimator
    if isinstance(reg, Pipeline):
        return reg
    if hasattr(reg, "named_steps"):
        return reg
    return None


def extract_feature_importance(estimator, feature_names: Sequence[str]) -> Optional[pd.DataFrame]:
    """Extract feature importance or coefficient magnitudes where available."""
    pipeline = get_underlying_pipeline(estimator)
    if pipeline is None or "model" not in pipeline.named_steps:
        return None
    model = pipeline.named_steps["model"]

    importance_type = None
    values = None
    signed_values = None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        signed_values = values.copy()
        importance_type = "feature_importances_"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float).reshape(-1)
        values = np.abs(coef)
        signed_values = coef
        importance_type = "absolute_coefficient"
    else:
        return None

    if len(values) != len(feature_names):
        # This can happen only when optional dataset one-hot features are used;
        # in that case, skip to avoid misleading labels.
        return None

    out = pd.DataFrame({
        "Feature": list(feature_names),
        "Importance": values,
        "Signed Value": signed_values,
        "Importance Type": importance_type,
    })
    out["Display Feature"] = out["Feature"].str.replace("feature_", "", regex=False).str.replace("_", " ", regex=False)
    out = out.sort_values("Importance", ascending=False).reset_index(drop=True)
    return out


def make_actual_vs_predicted_plot(plot_df: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    x = plot_df["Actual Runtime Seconds"].astype(float)
    y = plot_df["Predicted Runtime Seconds"].astype(float)
    ax.scatter(x, y, alpha=0.75, s=28)
    min_val = max(min(x.min(), y.min()), 1e-12)
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual runtime (seconds, log scale)")
    ax.set_ylabel("Predicted runtime (seconds, log scale)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_feature_importance_plot(fi_df: pd.DataFrame, output_path: Path, title: str, top_n: int = 10) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = fi_df.head(top_n).sort_values("Importance", ascending=True)
    fig_height = max(4.0, 0.38 * len(top) + 1.5)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    ax.barh(top["Display Feature"], top["Importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_combined_actual_vs_predicted_plot(pred_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = [(algo, g.copy()) for algo, g in pred_df.groupby("Algorithm", sort=False)]
    n = len(groups)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.5 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, (algo, group) in zip(axes, groups):
        x = group["Actual Runtime Seconds"].astype(float)
        y = group["Predicted Runtime Seconds"].astype(float)
        ax.scatter(x, y, alpha=0.75, s=22)
        min_val = max(min(x.min(), y.min()), 1e-12)
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        model_name = str(group["Model"].iloc[0])
        ax.set_title(f"{algo} ({model_name})")
        ax.set_xlabel("Actual runtime (s)")
        ax.set_ylabel("Predicted runtime (s)")

    for ax in axes[len(groups):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_combined_feature_importance_plot(fi_all: pd.DataFrame, output_path: Path, top_n: int = 8) -> None:
    if fi_all.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = [(algo, g.copy()) for algo, g in fi_all.groupby("Algorithm", sort=False)]
    n = len(groups)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.5 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, (algo, group) in zip(axes, groups):
        top = group.head(top_n).sort_values("Importance", ascending=True)
        ax.barh(top["Display Feature"], top["Importance"])
        model_name = str(top["Model"].iloc[0]) if "Model" in top.columns and not top.empty else ""
        ax.set_title(f"{algo} ({model_name})")
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
    for ax in axes[len(groups):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def run_part_h(config: RuntimePredictionConfig) -> Dict[str, Path]:
    """Run the full Part H runtime-prediction workflow."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "tables").mkdir(exist_ok=True)
    (config.output_dir / "plots").mkdir(exist_ok=True)
    (config.output_dir / "models").mkdir(exist_ok=True)

    long_df, wide_df, feature_cols = build_runtime_prediction_dataset(config)

    dataset_long_path = config.output_dir / "tables" / "runtime_prediction_dataset_long.csv"
    dataset_wide_path = config.output_dir / "tables" / "runtime_prediction_dataset_wide.csv"
    long_df.to_csv(dataset_long_path, index=False)
    wide_df.to_csv(dataset_wide_path, index=False)

    all_metrics_rows: List[Dict] = []
    all_prediction_frames: List[pd.DataFrame] = []
    estimator_frames: List[pd.DataFrame] = []

    for algorithm in config.algorithms:
        algo_df = long_df[long_df[ALGORITHM_COL].astype(str) == algorithm].copy()
        if algo_df.empty:
            warnings.warn(f"Skipping {algorithm}; no rows found in runtime dataset.", RuntimeWarning)
            continue
        rows, pred_df, estimators_df = train_one_algorithm(algo_df, algorithm, feature_cols, config)
        all_metrics_rows.extend(rows)
        all_prediction_frames.append(pred_df)
        estimator_frames.append(estimators_df)

    metrics_df = pd.DataFrame(all_metrics_rows)
    if metrics_df.empty:
        raise RuntimeError("No runtime-prediction models were trained.")
    predictions_df = pd.concat(all_prediction_frames, ignore_index=True)
    estimators_df = pd.concat(estimator_frames, ignore_index=True) if estimator_frames else pd.DataFrame()

    metrics_df = metrics_df.sort_values(["Algorithm", "Test MAPE", "Test RMSE"], ascending=[True, True, True]).reset_index(drop=True)
    best_df = metrics_df.groupby("Algorithm", as_index=False).head(1).reset_index(drop=True)
    best_keys = set(zip(best_df["Algorithm"], best_df["Model"]))
    winner_predictions_df = predictions_df[
        predictions_df[["Algorithm", "Model"]].apply(lambda r: (r["Algorithm"], r["Model"]) in best_keys, axis=1)
    ].copy()

    all_metrics_path = config.output_dir / "tables" / "runtime_prediction_all_model_results.csv"
    best_path = config.output_dir / "tables" / "runtime_prediction_best_models_by_mape.csv"
    predictions_path = config.output_dir / "tables" / "runtime_prediction_winner_predictions.csv"
    config_path = config.output_dir / "runtime_prediction_part_h_config.json"
    excel_path = config.output_dir / "part_h_runtime_prediction_results.xlsx"

    metrics_df.to_csv(all_metrics_path, index=False)
    best_df.to_csv(best_path, index=False)
    winner_predictions_df.to_csv(predictions_path, index=False)

    # Save plots and feature importances for winners.
    feature_importance_frames: List[pd.DataFrame] = []
    for _, row in best_df.iterrows():
        algorithm = row["Algorithm"]
        model_name = row["Model"]
        pred_group = winner_predictions_df[
            (winner_predictions_df["Algorithm"] == algorithm)
            & (winner_predictions_df["Model"] == model_name)
        ].copy()
        make_actual_vs_predicted_plot(
            pred_group,
            config.output_dir / "plots" / f"{algorithm}_best_by_mape_actual_vs_predicted_runtime.png",
            title=f"{algorithm}: actual vs predicted runtime ({model_name}, best by MAPE)",
        )

        est_row = estimators_df[(estimators_df["Algorithm"] == algorithm) & (estimators_df["Model"] == model_name)]
        if not est_row.empty and not config.include_dataset_onehot:
            fi_df = extract_feature_importance(est_row["Estimator"].iloc[0], feature_cols)
            if fi_df is not None and not fi_df.empty:
                fi_df.insert(0, "Algorithm", algorithm)
                fi_df.insert(1, "Model", model_name)
                fi_path = config.output_dir / "tables" / f"{algorithm}_{model_name}_feature_importance.csv"
                fi_df.to_csv(fi_path, index=False)
                make_feature_importance_plot(
                    fi_df,
                    config.output_dir / "plots" / f"{algorithm}_{model_name}_feature_importance.png",
                    title=f"{algorithm}: feature importance ({model_name})",
                    top_n=config.top_n_features_plot,
                )
                feature_importance_frames.append(fi_df)

    make_combined_actual_vs_predicted_plot(
        winner_predictions_df,
        config.output_dir / "plots" / "runtime_prediction_actual_vs_predicted_best_by_mape_combined.png",
    )

    if feature_importance_frames:
        fi_all = pd.concat(feature_importance_frames, ignore_index=True)
        fi_all_path = config.output_dir / "tables" / "runtime_prediction_winner_feature_importances.csv"
        fi_all.to_csv(fi_all_path, index=False)
        make_combined_feature_importance_plot(
            fi_all,
            config.output_dir / "plots" / "runtime_prediction_feature_importances_best_by_mape_combined.png",
            top_n=min(config.top_n_features_plot, 8),
        )
    else:
        fi_all_path = config.output_dir / "tables" / "runtime_prediction_winner_feature_importances.csv"
        pd.DataFrame(columns=["Algorithm", "Model", "Feature", "Importance"]).to_csv(fi_all_path, index=False)

    save_json(
        {
            **asdict(config),
            "feature_columns": feature_cols,
            "n_feature_columns": len(feature_cols),
            "n_long_rows": len(long_df),
            "n_wide_rows": len(wide_df),
        },
        config_path,
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="All Tuned Models", index=False)
        best_df.to_excel(writer, sheet_name="Best by MAPE", index=False)
        winner_predictions_df.to_excel(writer, sheet_name="Winner Predictions", index=False)
        long_df.head(50000).to_excel(writer, sheet_name="Runtime Dataset Long", index=False)
        if feature_importance_frames:
            pd.concat(feature_importance_frames, ignore_index=True).to_excel(writer, sheet_name="Winner Feature Importance", index=False)

    print("\nPart H runtime prediction complete.")
    print(f"All model results: {all_metrics_path}")
    print(f"Best models by MAPE: {best_path}")
    print(f"Winner predictions: {predictions_path}")
    print(f"Excel workbook: {excel_path}")
    print(f"Plots directory: {config.output_dir / 'plots'}")

    return {
        "dataset_long": dataset_long_path,
        "dataset_wide": dataset_wide_path,
        "all_model_results": all_metrics_path,
        "best_models": best_path,
        "winner_predictions": predictions_path,
        "feature_importances": fi_all_path,
        "excel_workbook": excel_path,
        "config": config_path,
        "plots_dir": config.output_dir / "plots",
    }
