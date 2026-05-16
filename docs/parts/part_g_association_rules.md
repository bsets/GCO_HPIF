# Part G — Association-rule mining

Part G mines FP-Growth association rules for hardness interpretation.

## Design choice

Do not mine over all 23 graph features. Use only the target-specific Part F first-peak selected features.

This keeps the rule-mining problem smaller and the resulting rules more interpretable.

## Targets

```text
consensus5_runtime_hardness_label_binary
consensus4_runtime_hardness_label_binary
majority3_runtime_hardness_label_binary
```

## Feature bins

```text
3 bins: 0–33, 33–66, 66–100
4 bins: 0–25, 25–50, 50–75, 75–100
5 bins: 0–20, 20–40, 40–60, 60–80, 80–100
```

## Consequents

Consequents are restricted to standalone labels:

```text
Hard
Not Hard
```

## Main command

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

## Main outputs

```text
artifacts/association_rules_part_g/
├── selected_association_rules_part_g.csv
├── association_rule_runtime_summary_part_g.csv
├── model_selection_by_dataset_part_g.csv
└── part_g_association_rules.xlsx
```
