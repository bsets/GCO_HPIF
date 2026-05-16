## Part G — FP-Growth Association-Rule Mining for Hardness Interpretation

Part G mines interpretable FP-Growth association rules over the selected Part F graph features. It compares rules for empirical hardness labels against rules for predictions from the best saved Part F model selected separately for each dataset.

Run from the repository root:

```bash
pip install -r requirements_part_g.txt

python -m gco_hpif.cli.mine_hardness_rules \
  --features artifacts/features_full/graph_features.csv \
  --labels artifacts/hardness_labels_consensus_runtime/hardness_labels.csv \
  --part-f-output-dir artifacts/ml_hardness_part_f \
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
artifacts/association_rules_part_g/part_g_association_rules.xlsx
artifacts/association_rules_part_g/selected_association_rules_part_g.csv
artifacts/association_rules_part_g/association_rule_runtime_summary_part_g.csv
artifacts/association_rules_part_g/model_selection_by_dataset_part_g.csv
```

Part G uses only the target-specific `feature_` columns selected at the Part F minority-class F1 first peak. It now refuses to fall back to all 23 graph features or to another target's selected-feature file. It bins the selected features dataset-wise into 3/4/5 percentile schemes, keeps only rules with standalone `Hard` or `Not Hard` consequents, and sorts selected non-overlapping rules by support and lift.
