# Part H — Runtime Prediction from Graph Features

Part H trains maximum-clique runtime-prediction models for the five algorithms used in the GCO-HPIF empirical hardness pipeline:

- Gurobi
- CliSAT
- MOMC
- EGN
- HGS

The predictors are the Part C graph features. By default, the CLI uses every column in `graph_features.csv` whose name begins with `feature_`, which should correspond to the 23 graph features used in the earlier parts of the repository.

## Why log-runtime is modelled

Runtime varies by orders of magnitude across graph instances and algorithms. Part H therefore uses `log(runtime_seconds)` internally as the target transformation. The models are wrapped so that predictions are converted back to runtime seconds before evaluation.

The primary selection metric is MAPE on the original runtime scale. RMSE, MAE, R2, log-MAE, and log-RMSE are also reported as diagnostics.

## Inputs

Required inputs:

```text
artifacts/features_full/graph_features.csv
artifacts/solver_runs/gurobi_full/solver_runs.csv
artifacts/solver_runs/clisat_full/solver_runs.csv
artifacts/solver_runs/momc_full/solver_runs.csv
artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv
artifacts/solver_runs/hgs_full_all_test_graphs/solver_runs.csv
```

The solver run files must follow the standardized solver-output contract used in Parts D and E and include:

```text
dataset
graph_id
source_index
solver_name
runtime_seconds
```

The feature file must include:

```text
dataset
graph_id
source_index
feature_*
```

## Models

Part H supports:

- `XGB` — XGBoost regressor, if `xgboost` is installed
- `RF` — Random Forest regressor
- `SVR` — support-vector regression
- `LR` — tuned Ridge regression, used as the linear-regression baseline

The CLI also accepts `SVC` as an alias for `SVR`, because runtime prediction is a regression task rather than a classification task.

## Example run

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
  --test-size 0.20 \
  --grid-n-jobs 1 \
  --model-n-jobs 1
```

For a quick smoke test, use:

```bash
PYTHONPATH=$PWD/src python -m gco_hpif.cli.train_runtime_predictors \
  --features artifacts/features_full/graph_features.csv \
  --solver-runs \
    artifacts/solver_runs/gurobi_full/solver_runs.csv \
    artifacts/solver_runs/clisat_full/solver_runs.csv \
    artifacts/solver_runs/momc_full/solver_runs.csv \
    artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv \
    artifacts/solver_runs/hgs_full_all_test_graphs/solver_runs.csv \
  --output-dir artifacts/runtime_prediction_part_h_smoke \
  --models RF,LR \
  --grid-size tiny \
  --cv-folds 2 \
  --max-rows-per-algorithm 100
```

## Outputs

Part H writes outputs under the requested output directory, for example:

```text
artifacts/runtime_prediction_part_h/
├── runtime_prediction_part_h_config.json
├── part_h_runtime_prediction_results.xlsx
├── tables/
│   ├── runtime_prediction_dataset_long.csv
│   ├── runtime_prediction_dataset_wide.csv
│   ├── runtime_prediction_all_model_results.csv
│   ├── runtime_prediction_best_models_by_mape.csv
│   ├── runtime_prediction_winner_predictions.csv
│   └── runtime_prediction_winner_feature_importances.csv
├── plots/
│   ├── <Algorithm>_best_by_mape_actual_vs_predicted_runtime.png
│   ├── <Algorithm>_<Model>_feature_importance.png
│   ├── runtime_prediction_actual_vs_predicted_best_by_mape_combined.png
│   └── runtime_prediction_feature_importances_best_by_mape_combined.png
└── models/
    └── <Algorithm>_<Model>_best_log_runtime_regressor.joblib
```

## Notes

- The default setting keeps only the common five-solver instance set, matching the Part E design.
- The winner for each algorithm is selected by held-out test-set MAPE.
- Actual-vs-predicted plots are saved on log-log axes because runtime spans orders of magnitude.
- Feature-importance plots are saved for winning models when the model exposes feature importances or coefficients.
