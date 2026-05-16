# GCO-HPIF

**GCO-HPIF** stands for **Graph-Based Combinatorial Optimization — Hardness Prediction and Interpretation Framework**.

This repository contains a modular, reproducible pipeline for studying graph-instance hardness for the Maximum Clique Problem (MCP). The completed stages cover raw graph ingestion, fixed TWITTER train/validation/test splitting, graph-feature computation, solver wrappers for exact, heuristic, and learned maximum-clique solvers, runtime-consensus hardness label construction, ML hardness classification from graph features, association-rule-based interpretation of hardness patterns, and solver-specific runtime prediction from graph features.

The current planned numbered pipeline is complete through **Part H**. No additional numbered pipeline parts are currently planned.

## Completed pipeline status

| Part | Status | Description |
|---|---|---|
| Part A | Complete | Raw graph ingestion and graph manifests |
| Part B | Complete | Fixed TWITTER 60/20/20 train/validation/test split manifest |
| Part C | Complete | 23 NetworkX graph features with 60-second timeout logging |
| Part D.1 | Complete | Gurobi maximum-clique solver wrapper |
| Part D.2 | Complete | CliSAT maximum-clique solver wrapper |
| Part D.3 | Complete | MoMC maximum-clique solver wrapper |
| Part D.4 | Complete | Optional EGN training/inference wrapper |
| Part D.5 | Complete | Optional HGS training/inference wrapper |
| Part E | Complete | Runtime-consensus hardness label construction from five solver outputs |
| Part F | Complete | ML hardness classification from graph features using feature selection and tuned classifiers |
| Part G | Complete | FP-Growth association-rule mining for interpretable hardness-pattern discovery |
| Part H | Complete | Runtime prediction from 23 graph features and five solver runtime outputs |

---

## Datasets

The current pipeline uses:

* `TWITTER` ego-network graphs from SNAP.
* `COLLAB` from TU-format graph datasets.
* `IMDB-BINARY` from TU-format graph datasets.

Part A writes graph stores under:

```text
artifacts/slice_a_full/interim/
├── twitter_graphs.pkl.gz
├── collab_graphs.pkl.gz
└── imdb_binary_graphs.pkl.gz
```

Generated data and solver outputs are written under `artifacts/` and are not committed to GitHub by default.

---

## Installation

Create and activate a Python virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -U pip
pip install -e .[dev]
```

Run tests:

```bash
pytest -q
```

---

## Part A — Raw graph ingestion

Smoke test:

```bash
python -m gco_hpif.cli.prepare_raw_graphs \
  --output-root artifacts/slice_a_smoke \
  --datasets imdb_binary collab twitter \
  --limit 3
```

Full run:

```bash
python -m gco_hpif.cli.prepare_raw_graphs \
  --output-root artifacts/slice_a_full \
  --datasets imdb_binary collab twitter
```

Expected full graph counts:

| Dataset     | Count |
| ----------- | ----- |
| TWITTER     | 973   |
| COLLAB      | 5000  |
| IMDB-BINARY | 1000  |

Main outputs:

```text
artifacts/slice_a_full/
├── raw/
├── interim/
│   ├── twitter_graphs.pkl.gz
│   ├── collab_graphs.pkl.gz
│   └── imdb_binary_graphs.pkl.gz
└── manifests/
    ├── graphs_index.csv
    └── dataset_summary.csv
```

---

## Part B — TWITTER split manifest

Create the fixed 60/20/20 TWITTER split:

```bash
python -m gco_hpif.cli.make_twitter_split \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --output data/manifests/twitter_split_60_20_20.csv
```

Expected split counts:

| Split      | Count |
| ---------- | ----- |
| train      | 584   |
| validation | 195   |
| test       | 194   |

The split manifest is committed because it defines the reproducible TWITTER split used by EGN and HGS.

---

## Part C — NetworkX graph features

Part C computes 23 graph-level features using NetworkX/NumPy. Graphs whose features fail or exceed the timeout are logged explicitly.

Smoke test:

```bash
python -m gco_hpif.cli.compute_graph_features \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/features_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 3 \
  --timeout-seconds 60
```

Full run:

```bash
python -m gco_hpif.cli.compute_graph_features \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/features_full \
  --datasets twitter collab imdb_binary \
  --timeout-seconds 60
```

Main outputs:

```text
artifacts/features_full/
├── graph_features.csv
├── feature_failures.csv
├── feature_timing_summary.csv
└── feature_column_manifest.csv
```

---

## Common solver-output contract

All solver wrappers write standardized outputs:

```text
solver_runs.csv
solver_errors.csv
run_summary.csv
```

Common columns include:

```text
dataset
graph_id
source_index
solver_name
run_type
time_limit_seconds
status
optimality_status
runtime_seconds
best_clique_size
best_clique_nodes
clique_valid
num_nodes
num_edges
seed
threads
mip_gap
objective_bound
error_message
```

---

## Part D.1 — Gurobi solver wrapper

Gurobi is an optional exact solver backend. You need:

* `gurobipy`
* a valid Gurobi license
* `GRB_LICENSE_FILE` configured if the license is not in the default location

Example:

```bash
export GRB_LICENSE_FILE=/home/bharat/opt/gurobi1301/gurobi.lic
```

Smoke test:

```bash
python -m gco_hpif.cli.run_gurobi_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/gurobi_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 2 \
  --time-limit-seconds 30 \
  --threads 1
```

Full run:

```bash
python -m gco_hpif.cli.run_gurobi_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/gurobi_full \
  --datasets twitter collab imdb_binary \
  --time-limit-seconds 1800 \
  --threads 1
```

---

## Part D.2 — CliSAT solver wrapper

CliSAT runs on graphs that passed Part C feature computation. The wrapper writes DIMACS input files and standardized solver logs.

Smoke test:

```bash
python -m gco_hpif.cli.run_clisat_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/clisat_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 2 \
  --time-limit-seconds 30 \
  --threads 1
```

Full run:

```bash
python -m gco_hpif.cli.run_clisat_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/clisat_full \
  --datasets twitter collab imdb_binary \
  --time-limit-seconds 1800 \
  --threads 1
```

Default executable path:

```text
external/CliSAT/bin/CliSAT
```

---

## Part D.3 — MoMC solver wrapper

MoMC runs on graphs that passed Part C feature computation. If the executable is missing, the CLI can compile it from the bundled C source unless `--no-compile` is used.

Smoke test:

```bash
python -m gco_hpif.cli.run_momc_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/momc_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 2 \
  --time-limit-seconds 30 \
  --threads 1
```

Full run:

```bash
python -m gco_hpif.cli.run_momc_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/momc_full \
  --datasets twitter collab imdb_binary \
  --time-limit-seconds 1800 \
  --threads 1
```

Default source/executable locations:

```text
external/MOMC/src/
external/MOMC/bin/MoMC
```

---

## Part D.4 — EGN training and all-test inference

EGN is treated as an optional external integration. The upstream EGN source is not vendored into this repository; clone it locally under:

```text
external/EGN/erdos_neu
```

Smoke test:

```bash
python -m gco_hpif.cli.run_egn_solver \
  --mode smoke \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --egn-root external/EGN/erdos_neu \
  --output-dir artifacts/solver_runs/egn_smoke \
  --dataset twitter \
  --limit-per-split 2 \
  --epochs 2 \
  --inference-samples 2
```

Train on the TWITTER training split and infer on the TWITTER test split:

```bash
python -m gco_hpif.cli.run_egn_solver \
  --mode train-and-infer \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --egn-root external/EGN/erdos_neu \
  --output-dir artifacts/solver_runs/egn_full \
  --dataset twitter \
  --epochs 100 \
  --train-batch-size 4 \
  --infer-batch-size 1 \
  --num-layers 5 \
  --hidden-1 64 \
  --hidden-2 1 \
  --learning-rate 0.001 \
  --penalty-coeff 4.0 \
  --seed 66 \
  --inference-samples 8
```

Run all-test inference using the trained EGN checkpoint:

```bash
EGN_CKPT="artifacts/solver_runs/egn_full/trained_egn_model.pt"

python -m gco_hpif.cli.run_egn_solver \
  --mode infer \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --egn-root external/EGN/erdos_neu \
  --checkpoint "$EGN_CKPT" \
  --output-dir artifacts/solver_runs/egn_full_all_test_graphs \
  --dataset twitter \
  --infer-batch-size 1 \
  --inference-samples 8 \
  --extra-infer-graph-store collab=artifacts/slice_a_full/interim/collab_graphs.pkl.gz \
  --extra-infer-graph-store imdb_binary=artifacts/slice_a_full/interim/imdb_binary_graphs.pkl.gz
```

Expected all-test inference count:

| Dataset      | Inference graphs |
| ------------ | ---------------- |
| TWITTER test | 194              |
| COLLAB       | 5000             |
| IMDB-BINARY  | 1000             |

---

## Part D.5 — HGS training and all-test inference

HGS is treated as an optional external integration. The upstream HGS source is not vendored into this repository; clone it locally under:

```text
external/HGS/GeometricScatteringMaximalClique
```

Example local setup:

```bash
mkdir -p external/HGS

git clone https://github.com/yimengmin/GeometricScatteringMaximalClique.git \
  external/HGS/GeometricScatteringMaximalClique
```

Smoke test:

```bash
python -m gco_hpif.cli.run_hgs_solver \
  --mode smoke \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --raw-graph-store artifacts/slice_a_full/interim/twitter_graphs.pkl.gz \
  --hgs-root external/HGS/GeometricScatteringMaximalClique \
  --output-dir artifacts/solver_runs/hgs_smoke \
  --dataset twitter \
  --limit-per-split 2 \
  --epochs 2 \
  --infer-splits test \
  --num-walkers 4
```

Train on the TWITTER training split and run all-test inference:

```bash
python -m gco_hpif.cli.run_hgs_solver \
  --mode train-and-infer \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --raw-graph-store artifacts/slice_a_full/interim/twitter_graphs.pkl.gz \
  --hgs-root external/HGS/GeometricScatteringMaximalClique \
  --output-dir artifacts/solver_runs/hgs_full_all_test_graphs \
  --dataset twitter \
  --epochs 20 \
  --train-batch-size 1 \
  --infer-splits test \
  --hidden 8 \
  --num-layers 4 \
  --learning-rate 0.001 \
  --penalty-coeff 2.0 \
  --seed 42 \
  --num-walkers 20 \
  --sample-length 90 \
  --extra-infer-graph-store collab=artifacts/slice_a_full/interim/collab_graphs.pkl.gz \
  --extra-infer-graph-store imdb_binary=artifacts/slice_a_full/interim/imdb_binary_graphs.pkl.gz
```

Run all-test inference using an existing HGS checkpoint:

```bash
python -m gco_hpif.cli.run_hgs_solver \
  --mode infer \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --raw-graph-store artifacts/slice_a_full/interim/twitter_graphs.pkl.gz \
  --hgs-root external/HGS/GeometricScatteringMaximalClique \
  --checkpoint-path artifacts/solver_runs/hgs_full/hgs_checkpoint.pt \
  --output-dir artifacts/solver_runs/hgs_full_all_test_graphs \
  --dataset twitter \
  --infer-splits test \
  --extra-infer-graph-store collab=artifacts/slice_a_full/interim/collab_graphs.pkl.gz \
  --extra-infer-graph-store imdb_binary=artifacts/slice_a_full/interim/imdb_binary_graphs.pkl.gz \
  --num-walkers 20 \
  --sample-length 90
```

Expected all-test inference count:

| Dataset      | Inference graphs |
| ------------ | ---------------- |
| TWITTER test | 194              |
| COLLAB       | 5000             |
| IMDB-BINARY  | 1000             |

---

## Part E — Runtime-consensus hardness labels

Part E constructs empirical hardness labels from the common five-solver instance set using solver runtime behaviour. The five solver outputs used are Gurobi, CliSAT, MoMC, EGN, and HGS.

The final Part E design uses dataset-specific and solver-specific runtime quartiles. For each dataset and solver, an instance is marked as top-runtime-quartile if its runtime is greater than or equal to that solver’s 75th percentile runtime within the same dataset.

The primary hardness label is:

```text
consensus5_runtime_hardness_label_binary
```

This marks an instance as `Hard` only if it falls in the top runtime quartile for all five solvers.

Sensitivity-analysis labels are also created:

```text
consensus4_runtime_hardness_label_binary
majority3_runtime_hardness_label_binary
```

The common five-solver instance set contains:

| Dataset | Instances |
|---|---:|
| COLLAB | 5000 |
| IMDB-BINARY | 1000 |
| TWITTER | 194 |
| Total | 6194 |

The final label distributions are:

| Label | Not Hard | Hard |
|---|---:|---:|
| Consensus-5 | 5817 | 377 |
| Consensus-4 | 5301 | 893 |
| Majority-3 | 4819 | 1375 |

Main output:

```text
artifacts/hardness_labels_consensus_runtime/hardness_labels.csv
```

Example command:

```bash
python -m gco_hpif.cli.build_hardness_labels \
  --solver-runs \
    artifacts/solver_runs/gurobi_full/solver_runs.csv \
    artifacts/solver_runs/clisat_full/solver_runs.csv \
    artifacts/solver_runs/momc_full/solver_runs.csv \
    artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv \
    artifacts/solver_runs/hgs_full_all_test_graphs/solver_runs.csv \
  --features artifacts/features_full/graph_features.csv \
  --output-dir artifacts/hardness_labels_consensus_runtime
```

Main outputs:

```text
artifacts/hardness_labels_consensus_runtime/
├── hardness_labels.csv
├── hardness_label_summary.csv
└── runtime_thresholds_by_dataset_solver.csv
```

---

## Part F — ML hardness classification from graph features

Part F trains machine-learning classifiers to predict runtime-consensus hardness labels from graph features.

The ML dataset is built by joining:

```text
artifacts/features_full/graph_features.csv
artifacts/hardness_labels_consensus_runtime/hardness_labels.csv
```

using the keys:

```text
dataset
graph_id
source_index
```

Only columns beginning with `feature_` are used as predictors. Runtime columns, solver-derived columns, label columns, and metadata columns are excluded from the predictor matrix.

Part F supports the primary Consensus-5 target and the two sensitivity-analysis targets:

```text
consensus5_runtime_hardness_label_binary
consensus4_runtime_hardness_label_binary
majority3_runtime_hardness_label_binary
```

The classification pipeline trains and tunes:

- Random Forest
- Logistic Regression
- SVC
- XGBoost, if installed

Class imbalance is handled using stratified splits, class weighting where available, and XGBoost `scale_pos_weight`. Evaluation focuses on minority-class F1, weighted F1, balanced accuracy, ROC-AUC, PR-AUC, and confusion matrices.

Feature selection is performed using ANOVA `SelectKBest`. The selected feature set is chosen at the first peak or early plateau of minority-class F1. These selected first-peak features are later used by Part G for association-rule mining.

Example command:

```bash
python -m gco_hpif.cli.train_hardness_classifiers \
  --features artifacts/features_full/graph_features.csv \
  --labels artifacts/hardness_labels_consensus_runtime/hardness_labels.csv \
  --output-dir artifacts/ml_hardness_part_f_feature_prefix_fixed \
  --targets \
    consensus5_runtime_hardness_label_binary \
    consensus4_runtime_hardness_label_binary \
    majority3_runtime_hardness_label_binary
```

Typical outputs for each target:

```text
artifacts/ml_hardness_part_f_feature_prefix_fixed/<target_name>/
├── tables/
│   ├── ml_dataset_<target>.csv
│   ├── feature_selection_curve.csv
│   ├── selected_features_anova_first_peak.csv
│   ├── metrics_by_model_selected_features.csv
│   └── hyperparameter_cv_results_selected_features.csv
├── plots/
│   ├── feature_selection_curve_seaborn.png
│   ├── roc_curves_tuned_selected_models.png
│   ├── pr_curves_tuned_selected_models.png
│   └── confusion_matrix_2x2_best_<model>.png
└── models/
    ├── best_rf_selected_features.joblib
    ├── best_lr_selected_features.joblib
    ├── best_svc_selected_features.joblib
    └── best_xgb_selected_features.joblib
```

---

## Part G — Association-rule mining for hardness interpretation

Part G adds an interpretable rule-mining layer on top of the Part E hardness labels and Part F selected features.

The goal is to identify compact combinations of graph-feature percentile ranges that are associated with `Hard` and `Not Hard` instances. Part G uses FP-Growth association rule mining and restricts consequents to standalone hardness labels only.

Part G intentionally does not use all 23 graph features. Instead, for each hardness target, it uses only the Part F first-peak selected features saved in:

```text
artifacts/ml_hardness_part_f_feature_prefix_fixed/<target_name>/tables/selected_features_anova_first_peak.csv
```

This avoids unnecessary combinatorial explosion and keeps the resulting rules more interpretable.

The supported hardness targets are:

```text
consensus5_runtime_hardness_label_binary
consensus4_runtime_hardness_label_binary
majority3_runtime_hardness_label_binary
```

Selected graph features are discretized within each dataset using percentile bins:

- 3 bins: 0–33, 33–66, 66–100
- 4 bins: 0–25, 25–50, 50–75, 75–100
- 5 bins: 0–20, 20–40, 40–60, 60–80, 80–100

Rules are evaluated using association-rule metrics and classification-style diagnostics, including support, confidence, lift, conviction, matched rows, precision, recall, F1, minority-class F1, weighted F1, balanced accuracy, and confusion-matrix counts.

Example command:

```bash
PYTHONPATH=$PWD/src python -m gco_hpif.cli.mine_hardness_rules \
  --features artifacts/features_full/graph_features.csv \
  --labels artifacts/hardness_labels_consensus_runtime/hardness_labels.csv \
  --part-f-output-dir artifacts/ml_hardness_part_f_feature_prefix_fixed \
  --output-dir artifacts/association_rules_part_g \
  --bin-counts 3 4 5 \
  --min-support auto \
  --min-confidence 0.60 \
  --min-lift 1.0 \
  --max-rule-antecedents 3 \
  --max-rules-per-group 25 \
  --min-matched-rows 10 \
  --max-selected-features-per-target 10 \
  --evaluation-scope all
```

Main outputs:

```text
artifacts/association_rules_part_g/
├── selected_association_rules_part_g.csv
├── association_rule_runtime_summary_part_g.csv
├── model_selection_by_dataset_part_g.csv
└── part_g_association_rules.xlsx
```

Part G also includes strict safeguards to prevent accidental use of all `feature_` columns or the wrong target’s selected-feature file.

---

## Part H — Runtime prediction from graph features

Part H adds a runtime-prediction layer to the GCO-HPIF pipeline. It uses the 23 graph features computed in Part C and the runtime outputs from the five maximum-clique solvers used in Part E to predict solver runtime for graph instances from TWITTER, COLLAB, and IMDB-BINARY.

The runtime-prediction dataset is built by joining:

```text
artifacts/features_full/graph_features.csv
```

with the five standardized solver-run files:

```text
artifacts/solver_runs/gurobi_full/solver_runs.csv
artifacts/solver_runs/clisat_full/solver_runs.csv
artifacts/solver_runs/momc_full/solver_runs.csv
artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv
artifacts/solver_runs/hgs_full_all_test_graphs/solver_runs.csv
```

Part H uses all valid `feature_` columns from Part C as predictors. The prediction target is:

```text
log(runtime_seconds)
```

The logarithmic runtime target is used because solver runtimes vary by orders of magnitude across instances, datasets, and algorithms. Model predictions are transformed back to runtime seconds before evaluation.

The pipeline trains and tunes the following regressors:

- XGBoost Regressor
- Random Forest Regressor
- Support Vector Regressor
- Linear/Ridge Regression baseline

The best model for each solver is selected using held-out MAPE on original runtime seconds. Additional metrics include RMSE, MAE, and R2.

Example smoke-test command:

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
  --max-rows-per-algorithm 100 \
  --verbose 0
```

Example full command:

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

Main outputs:

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

Part H completes the current GCO-HPIF pipeline by extending the framework from hardness labelling, hardness prediction, and hardness interpretation to solver-specific runtime prediction.

---

## Checking solver outputs

After any solver run, inspect the output folder:

```bash
ls -lh artifacts/solver_runs/<run_name>
cat artifacts/solver_runs/<run_name>/run_summary.csv
head artifacts/solver_runs/<run_name>/solver_runs.csv
cat artifacts/solver_runs/<run_name>/solver_errors.csv
```

For all-test EGN/HGS runs, check dataset-level counts:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("artifacts/solver_runs/<run_name>/solver_runs.csv")
df = pd.read_csv(path)

print("Total rows:", len(df))
print(df.groupby("dataset").size())
print()
print("Invalid clique counts:")
print(df.groupby("dataset")["clique_valid"].apply(lambda s: (~s.astype(bool)).sum()))
PY
```

---

## Generated artifacts

The following are generated locally and should not be committed by default:

```text
artifacts/
data/raw/
data/interim/
external/EGN/erdos_neu/
external/HGS/GeometricScatteringMaximalClique/
*.pt
*.pth
*.ckpt
*.joblib
__pycache__/
.pytest_cache/
```

The repository commits source code, tests, documentation, small fixed manifests, external-dependency placeholders, and selectively committed small summary outputs when useful for documenting reproducible pipeline results. Large generated artifacts, trained model files, raw data, checkpoints, and local caches should not be committed.

---

## Development workflow

Start a new branch:

```bash
git checkout main
git pull origin main
git checkout -b <branch-name>
```

Run checks:

```bash
pytest -q
python -m py_compile src/gco_hpif/cli/*.py
```

Commit and push:

```bash
git status
git add <relevant source/docs/tests only>
git commit -m "Descriptive commit message"
git push -u origin <branch-name>
```

Open a pull request into `main`, merge online, then sync locally:

```bash
git checkout main
git pull origin main
git branch -d <branch-name>
git status
```

---

## License

This repository is released under the MIT License.
