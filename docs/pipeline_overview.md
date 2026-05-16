# Pipeline overview

| Part | Stage | Main output |
|---|---|---|
| A | Raw graph ingestion | `artifacts/slice_a_full/` |
| B | TWITTER split | `data/manifests/twitter_split_60_20_20.csv` |
| C | Graph features | `artifacts/features_full/graph_features.csv` |
| D | Solver wrappers | `artifacts/solver_runs/*/solver_runs.csv` |
| E | Hardness labels | `artifacts/hardness_labels_consensus_runtime/hardness_labels.csv` |
| F | Hardness classification | `artifacts/ml_hardness_part_f_feature_prefix_fixed/` |
| G | Association rules | `artifacts/association_rules_part_g/` |
| H | Runtime prediction | `artifacts/runtime_prediction_part_h/` |
