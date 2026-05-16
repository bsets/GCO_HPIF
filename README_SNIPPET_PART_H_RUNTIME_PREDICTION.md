## Part H — Runtime prediction from graph features

Part H trains machine-learning models to predict maximum-clique solver runtime from the Part C graph features.

The input data are built by joining:

```text
artifacts/features_full/graph_features.csv
```

with the five standardized solver-output files used in the Part E common instance set:

```text
artifacts/solver_runs/gurobi_full/solver_runs.csv
artifacts/solver_runs/clisat_full/solver_runs.csv
artifacts/solver_runs/momc_full/solver_runs.csv
artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv
artifacts/solver_runs/hgs_full_all_test_graphs/solver_runs.csv
```

The predictors are the 23 Part C graph features, identified as columns beginning with `feature_`. The model target is runtime in seconds, but each regressor is trained internally on `log(runtime_seconds)` because runtime varies by orders of magnitude across graph instances and algorithms. Predictions are converted back to seconds for evaluation.

Part H trains and tunes:

- XGBoost regressor, if installed
- Random Forest regressor
- Support Vector Regressor
- Ridge regression as the linear-regression baseline

The best predictor for each algorithm is selected by held-out test-set MAPE. Secondary metrics include MAE, RMSE, R2, log-MAE, and log-RMSE.

Example command:

```bash
PYTHONPATH=$PWD/src python -m gco_hpif.cli.train_runtime_predictors \
  --features artifacts/features_full/graph_features.csv \
  --solver-runs \
    artifacts/solver_runs/gurobi_full/solver_runs.csv \
    artifacts/solver_runs/clisat_full/solver_runs.csv \
    artifacts/solver_runs/momc_full/solver_runs.csv \
    artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv \
    artifacts/solver_runs/hgs_full_all_test_graphs/solver_runs.csv \
  --output-dir artifacts/runtime_prediction_part_h \
  --models XGB,RF,SVR,LR \
  --grid-size compact \
  --cv-folds 5 \
  --test-size 0.20
```

Main outputs:

```text
artifacts/runtime_prediction_part_h/
├── part_h_runtime_prediction_results.xlsx
├── tables/
│   ├── runtime_prediction_dataset_long.csv
│   ├── runtime_prediction_dataset_wide.csv
│   ├── runtime_prediction_all_model_results.csv
│   ├── runtime_prediction_best_models_by_mape.csv
│   ├── runtime_prediction_winner_predictions.csv
│   └── runtime_prediction_winner_feature_importances.csv
├── plots/
│   ├── runtime_prediction_actual_vs_predicted_best_by_mape_combined.png
│   └── runtime_prediction_feature_importances_best_by_mape_combined.png
└── models/
    └── <Algorithm>_<Model>_best_log_runtime_regressor.joblib
```
