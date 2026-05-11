## Part F — ML classification for hardness prediction

Part F builds a downstream ML dataset by joining:

```text
artifacts/features_full/graph_features.csv
artifacts/hardness_labels_consensus_runtime/hardness_labels.csv
```

on:

```text
dataset, graph_id, source_index
```

The primary prediction target is:

```text
consensus5_runtime_hardness_label_binary
```

Sensitivity-analysis targets are:

```text
consensus4_runtime_hardness_label_binary
majority3_runtime_hardness_label_binary
```

Run:

```bash
python -m gco_hpif.cli.train_hardness_classifiers \
  --features artifacts/features_full/graph_features.csv \
  --labels artifacts/hardness_labels_consensus_runtime/hardness_labels.csv \
  --output-dir artifacts/ml_hardness_part_f
```

The Part F pipeline performs ANOVA-based feature selection, chooses the first sustained peak in minority-class F1, tunes RF/LR/SVC/XGB classifiers, and saves model artifacts, ROC/PR plots, a seaborn feature-selection curve, feature-importance plots for RF/XGB before feature selection, and a 2 by 2 confusion matrix for the best model.
