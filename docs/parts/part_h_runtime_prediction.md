# Part H — Runtime prediction

Part H predicts solver runtime from graph features.

## Inputs

```text
artifacts/features_full/graph_features.csv
artifacts/solver_runs/gurobi_full/solver_runs.csv
artifacts/solver_runs/clisat_full/solver_runs.csv
artifacts/solver_runs/momc_full/solver_runs.csv
artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv
artifacts/solver_runs/hgs_full_all_test_graphs/solver_runs.csv
```

## Predictors

Use valid `feature_` columns from Part C.

## Target

```text
log(runtime_seconds)
```

Runtime spans orders of magnitude, so the model learns log-runtime and transforms predictions back to seconds for evaluation.

## Models

- XGBoost Regressor
- Random Forest Regressor
- Support Vector Regressor
- Linear/Ridge Regression baseline

For runtime prediction, the correct support-vector model is `SVR`, not `SVC`.

## Full command

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

## Main outputs

```text
artifacts/runtime_prediction_part_h/
├── runtime_prediction_part_h_config.json
├── part_h_runtime_prediction_results.xlsx
├── tables/
├── plots/
└── models/
```
