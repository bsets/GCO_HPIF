from pathlib import Path

import numpy as np
import pandas as pd

from gco_hpif.runtime_prediction.time_prediction import (
    RuntimePredictionConfig,
    build_runtime_prediction_dataset,
    normalize_model_names,
    run_part_h,
)


def make_synthetic_inputs(tmp_path: Path, n_per_dataset: int = 8):
    datasets = ["TWITTER", "COLLAB", "IMDB-BINARY"]
    rows = []
    for dataset in datasets:
        for i in range(n_per_dataset):
            n = 20 + i + (10 if dataset == "COLLAB" else 0) + (5 if dataset == "IMDB-BINARY" else 0)
            rows.append(
                {
                    "dataset": dataset,
                    "graph_id": f"{dataset}_{i}",
                    "source_index": i,
                    "feature_Number_of_Nodes": float(n),
                    "feature_Number_of_Edges": float(n * (i + 2)),
                    "feature_Density": float((i + 1) / (n + 1)),
                }
            )
    features = pd.DataFrame(rows)
    features_path = tmp_path / "graph_features.csv"
    features.to_csv(features_path, index=False)

    solver_paths = []
    algorithms = ["Gurobi", "CliSAT", "MOMC", "EGN", "HGS"]
    algo_factor = {"Gurobi": 1.0, "CliSAT": 0.7, "MOMC": 0.8, "EGN": 0.3, "HGS": 0.4}
    for algo in algorithms:
        srows = []
        for _, row in features.iterrows():
            runtime = algo_factor[algo] * (0.05 * row["feature_Number_of_Nodes"] + 0.001 * row["feature_Number_of_Edges"] + 1.0)
            srows.append(
                {
                    "dataset": row["dataset"],
                    "graph_id": row["graph_id"],
                    "source_index": row["source_index"],
                    "solver_name": algo.lower(),
                    "runtime_seconds": runtime,
                    "status": "ok",
                }
            )
        path = tmp_path / f"{algo.lower()}_solver_runs.csv"
        pd.DataFrame(srows).to_csv(path, index=False)
        solver_paths.append(path)
    return features_path, solver_paths


def test_normalize_model_names_accepts_svc_alias():
    assert normalize_model_names(["XGB", "RF", "SVC", "LR"]) == ["XGB", "RF", "SVR", "LR"]


def test_build_runtime_prediction_dataset_uses_common_five_solver_instances(tmp_path):
    features_path, solver_paths = make_synthetic_inputs(tmp_path, n_per_dataset=4)
    config = RuntimePredictionConfig(
        features_path=features_path,
        solver_run_paths=solver_paths,
        output_dir=tmp_path / "out",
        models=["RF"],
        grid_size="tiny",
        cv_folds=2,
    )
    long_df, wide_df, feature_cols = build_runtime_prediction_dataset(config)
    assert set(feature_cols) == {
        "feature_Number_of_Nodes",
        "feature_Number_of_Edges",
        "feature_Density",
    }
    assert set(long_df["algorithm"].astype(str)) == {"Gurobi", "CliSAT", "MOMC", "EGN", "HGS"}
    assert len(wide_df) == 12
    assert len(long_df) == 12 * 5
    assert "log_runtime_seconds" in long_df.columns


def test_run_part_h_smoke_with_rf_and_lr(tmp_path):
    features_path, solver_paths = make_synthetic_inputs(tmp_path, n_per_dataset=8)
    out = tmp_path / "runtime_prediction_part_h"
    config = RuntimePredictionConfig(
        features_path=features_path,
        solver_run_paths=solver_paths,
        output_dir=out,
        models=["RF", "LR"],
        algorithms=["Gurobi", "CliSAT"],
        grid_size="tiny",
        cv_folds=2,
        test_size=0.25,
        verbose=0,
        save_models=False,
        top_n_features_plot=3,
    )
    result = run_part_h(config)
    assert result["all_model_results"].exists()
    assert result["best_models"].exists()
    assert result["winner_predictions"].exists()
    assert result["excel_workbook"].exists()

    best = pd.read_csv(result["best_models"])
    assert set(best["Algorithm"]) == {"Gurobi", "CliSAT"}
    assert "Test MAPE" in best.columns
    assert np.isfinite(best["Test MAPE"]).all()

    preds = pd.read_csv(result["winner_predictions"])
    assert {"Actual Runtime Seconds", "Predicted Runtime Seconds", "Actual Log Runtime", "Predicted Log Runtime"}.issubset(preds.columns)
    assert (preds["Predicted Runtime Seconds"] > 0).all()
