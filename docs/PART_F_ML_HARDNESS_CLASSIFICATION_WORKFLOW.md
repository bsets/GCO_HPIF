# Part F — ML Classification for Runtime-Consensus Hardness

Part F trains graph-feature-based classifiers to predict empirical hardness labels produced in Part E.

## Inputs

```text
artifacts/features_full/graph_features.csv
artifacts/hardness_labels_consensus_runtime/hardness_labels.csv
```

The join keys are:

```text
dataset
graph_id
source_index
```

## Targets

Primary target:

```text
consensus5_runtime_hardness_label_binary
```

Sensitivity-analysis targets:

```text
consensus4_runtime_hardness_label_binary
majority3_runtime_hardness_label_binary
```

## Leakage control

The model uses only numeric graph-feature columns from `graph_features.csv`. It does not use solver runtime columns, runtime-top-25 flags, runtime thresholds, best clique size, or solver-matching columns as predictors.

## Feature selection

For each target, the pipeline evaluates `k = 1..p` graph features using:

- ANOVA F-test feature scoring via `SelectKBest(f_classif)`
- stratified cross-validation
- minority-class F1 score as the main selection criterion

The selected number of features is the first sustained peak/plateau in minority-class F1. The chosen features are saved to:

```text
tables/selected_features_anova_first_peak.csv
```

The feature-count curve is saved to:

```text
plots/feature_selection_curve_seaborn.png
```

## Models

The pipeline tunes and evaluates:

- Random Forest
- Logistic Regression
- SVC
- XGBoost, if installed

The models use class imbalance handling where available:

- `class_weight="balanced"` for RF, LR, and SVC
- `scale_pos_weight` for XGBoost

## Metrics

The pipeline reports:

- balanced accuracy
- hard-class precision
- hard-class recall
- hard-class F1
- weighted F1
- ROC-AUC
- PR-AUC

Plain accuracy is included but should not be used as the main metric because the Consensus-5 target is imbalanced.

## Run command

From the repo root:

```bash
python -m gco_hpif.cli.train_hardness_classifiers \
  --features artifacts/features_full/graph_features.csv \
  --labels artifacts/hardness_labels_consensus_runtime/hardness_labels.csv \
  --output-dir artifacts/ml_hardness_part_f
```

For a quick smoke test:

```bash
python -m gco_hpif.cli.train_hardness_classifiers \
  --features artifacts/features_full/graph_features.csv \
  --labels artifacts/hardness_labels_consensus_runtime/hardness_labels.csv \
  --output-dir artifacts/ml_hardness_part_f_quick \
  --search-iterations 5
```

To run only the primary Consensus-5 target:

```bash
python -m gco_hpif.cli.train_hardness_classifiers \
  --features artifacts/features_full/graph_features.csv \
  --labels artifacts/hardness_labels_consensus_runtime/hardness_labels.csv \
  --output-dir artifacts/ml_hardness_part_f_consensus5_only \
  --targets consensus5_runtime_hardness_label_binary
```

## Main outputs

For Consensus-5, outputs are written under:

```text
artifacts/ml_hardness_part_f/consensus5_runtime_hardness_label_binary/
```

Important files:

```text
tables/ml_dataset_consensus5_runtime_hardness_label_binary.csv
tables/feature_selection_curve.csv
tables/selected_features_anova_first_peak.csv
tables/metrics_by_model_selected_features.csv
tables/hyperparameter_cv_results_selected_features.csv
tables/confusion_matrix_2x2_best_<model>.csv

plots/feature_selection_curve_seaborn.png
plots/roc_curves_tuned_selected_models.png
plots/pr_curves_tuned_selected_models.png
plots/confusion_matrix_2x2_best_<model>.png
plots/rf_feature_importance_all_features_before_selection.png
plots/xgb_feature_importance_all_features_before_selection.png

models/best_rf_selected_features.joblib
models/best_lr_selected_features.joblib
models/best_svc_selected_features.joblib
models/best_xgb_selected_features.joblib
models/best_rf_all_features_before_feature_selection.joblib
models/best_xgb_all_features_before_feature_selection.joblib

part_f_manifest.json
```

Cross-target summary:

```text
artifacts/ml_hardness_part_f/part_f_target_summary.csv
artifacts/ml_hardness_part_f/part_f_manifest_all_targets.json
```

## Suggested paper/README wording

We train graph-feature-based classifiers to predict runtime-consensus hardness labels. To reduce feature dimensionality for downstream interpretability and association-rule mining, we apply ANOVA F-test feature ranking and select the first sustained peak in cross-validated minority-class F1 as the final feature count. Class imbalance is handled through balanced class weights or positive-class scaling, and performance is evaluated using balanced accuracy, hard-class precision/recall/F1, ROC-AUC, and PR-AUC rather than plain accuracy.
