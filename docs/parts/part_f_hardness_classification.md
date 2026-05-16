# Part F — ML hardness classification

Part F trains graph-feature-based classifiers to predict runtime-consensus hardness labels.

## Inputs

```text
artifacts/features_full/graph_features.csv
artifacts/hardness_labels_consensus_runtime/hardness_labels.csv
```

## Join keys

```text
dataset
graph_id
source_index
```

## Predictor rule

Only columns beginning with `feature_` are valid predictors.

Do not use solver runtimes, top-runtime flags, label columns, best clique sizes, or other solver-derived columns as graph-feature predictors.

## Models

- Random Forest
- Logistic Regression
- SVC
- XGBoost, if installed

## Feature selection

Feature selection uses ANOVA `SelectKBest` and selects the first peak or early plateau of minority-class F1.

## Main output directory

```text
artifacts/ml_hardness_part_f_feature_prefix_fixed/
```

## Important selected-feature files

```text
artifacts/ml_hardness_part_f_feature_prefix_fixed/consensus5_runtime_hardness_label_binary/tables/selected_features_anova_first_peak.csv
artifacts/ml_hardness_part_f_feature_prefix_fixed/consensus4_runtime_hardness_label_binary/tables/selected_features_anova_first_peak.csv
artifacts/ml_hardness_part_f_feature_prefix_fixed/majority3_runtime_hardness_label_binary/tables/selected_features_anova_first_peak.csv
```
