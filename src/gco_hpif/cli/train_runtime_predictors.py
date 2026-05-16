"""CLI for Part H runtime prediction.

Example
-------
PYTHONPATH=$PWD/src python -m gco_hpif.cli.train_runtime_predictors \
  --features artifacts/features_full/graph_features.csv \
  --solver-runs \
    artifacts/solver_runs/gurobi_full/solver_runs.csv \
    artifacts/solver_runs/clisat_full/solver_runs.csv \
    artifacts/solver_runs/momc_full/solver_runs.csv \
    artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv \
    artifacts/solver_runs/hgs_full_all_test_graphs/solver_runs.csv \
  --output-dir artifacts/runtime_prediction_part_h \
  --models XGB,RF,SVR,LR
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import List, Optional

from gco_hpif.runtime_prediction.time_prediction import (
    CANONICAL_ALGORITHM_ORDER,
    DEFAULT_SOLVER_RUNS,
    RuntimePredictionConfig,
    normalize_model_names,
    parse_csv_arg,
    run_part_h,
)


def parse_feature_columns(value: Optional[str]) -> Optional[List[str]]:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return parse_csv_arg(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Part H: train maximum-clique runtime-prediction models from Part C graph "
            "features and the five standardized solver_runs.csv files. Models are trained "
            "on log(runtime_seconds) and selected by MAPE on original runtime seconds."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--features",
        required=True,
        help="Part C graph_features.csv file.",
    )
    parser.add_argument(
        "--solver-runs",
        nargs="+",
        default=DEFAULT_SOLVER_RUNS,
        help="Five standardized solver_runs.csv files for Gurobi, CliSAT, MOMC, EGN, and HGS.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/runtime_prediction_part_h",
        help="Output directory for Part H tables, models, and plots.",
    )
    parser.add_argument(
        "--feature-columns",
        default=None,
        help=(
            "Optional feature list. Accepts a Python list string or comma-separated names. "
            "If omitted, all columns beginning with feature_ are used."
        ),
    )
    parser.add_argument(
        "--algorithms",
        default=",".join(CANONICAL_ALGORITHM_ORDER),
        help="Comma-separated algorithms to model.",
    )
    parser.add_argument(
        "--models",
        default="XGB,RF,SVR,LR",
        help="Comma-separated models. SVC is accepted as an alias for SVR.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["log"],
        default="log",
        help="Part H always trains models on log(runtime_seconds).",
    )
    parser.add_argument(
        "--winner-metric",
        choices=["mape"],
        default="mape",
        help="Metric used to select the best model per algorithm.",
    )
    parser.add_argument("--test-size", type=float, default=0.20, help="Held-out test-set fraction.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds for GridSearchCV.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--grid-size",
        choices=["tiny", "compact", "standard"],
        default="compact",
        help="Hyperparameter grid size. Use tiny for a smoke test.",
    )
    parser.add_argument("--grid-n-jobs", type=int, default=1, help="Parallel jobs for GridSearchCV.")
    parser.add_argument("--model-n-jobs", type=int, default=1, help="Threads used inside RF/XGB models.")
    parser.add_argument("--pre-dispatch", default="1*n_jobs", help="GridSearchCV pre_dispatch setting.")
    parser.add_argument("--verbose", type=int, default=1, help="GridSearchCV verbosity.")
    parser.add_argument(
        "--use-union-instances",
        action="store_true",
        help="Use any instance with a runtime for each algorithm. By default, Part H keeps only the common five-solver instance set.",
    )
    parser.add_argument(
        "--min-runtime-seconds",
        type=float,
        default=1e-9,
        help="Drop runtime values at or below this threshold before log transformation.",
    )
    parser.add_argument(
        "--max-rows-per-algorithm",
        type=int,
        default=None,
        help="Optional row cap per algorithm for smoke tests/debugging.",
    )
    parser.add_argument(
        "--top-n-features-plot",
        type=int,
        default=10,
        help="Top N feature importances to display in each plot.",
    )
    parser.add_argument(
        "--include-dataset-onehot",
        action="store_true",
        help="Optional sensitivity analysis: add dataset one-hot indicators to the 23 graph features.",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Do not write joblib model artifacts.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    feature_columns = parse_feature_columns(args.feature_columns)
    algorithms = parse_csv_arg(args.algorithms)
    models = normalize_model_names(parse_csv_arg(args.models))

    config = RuntimePredictionConfig(
        features_path=Path(args.features),
        solver_run_paths=[Path(p) for p in args.solver_runs],
        output_dir=Path(args.output_dir),
        feature_columns=feature_columns,
        algorithms=algorithms,
        models=models,
        target_transform=args.target_transform,
        winner_metric=args.winner_metric,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        grid_size=args.grid_size,
        grid_n_jobs=args.grid_n_jobs,
        model_n_jobs=args.model_n_jobs,
        pre_dispatch=args.pre_dispatch,
        verbose=args.verbose,
        common_instance_only=not args.use_union_instances,
        min_runtime_seconds=args.min_runtime_seconds,
        max_rows_per_algorithm=args.max_rows_per_algorithm,
        top_n_features_plot=args.top_n_features_plot,
        include_dataset_onehot=args.include_dataset_onehot,
        save_models=not args.no_save_models,
    )
    run_part_h(config)


if __name__ == "__main__":
    main()
