# Part G — FP-Growth Association-Rule Mining for Hardness Interpretation

Part G explains the empirical hardness labels and the Part F model predictions using association rules over percentile-binned graph features.

## Purpose

Part F predicts whether a graph instance is Hard or Not Hard from graph features. Part G converts the selected Part F features into interpretable percentile-bin items and mines rules such as:

```text
IF Number of Nodes in [75%, 100%] AND Density in [0%, 25%] THEN Hard
```

The goal is not to maximize predictive accuracy with a complex rule set. The goal is to identify simple, human-readable feature-space regions associated with Hard or Not-Hard instances.

## Design decisions

1. **Use only target-specific Part F first-peak selected features.**
   Part G reads `selected_features_anova_first_peak.csv` separately from each Part F target folder. It does **not** fall back to all `feature_` columns and it does **not** reuse the Consensus-5 feature list for Consensus-4 or Majority-3. This avoids combinatorial explosion and keeps the rule set interpretable.

   Expected locations are:

   ```text
   artifacts/ml_hardness_part_f/consensus5_runtime_hardness_label_binary/tables/selected_features_anova_first_peak.csv
   artifacts/ml_hardness_part_f/consensus4_runtime_hardness_label_binary/tables/selected_features_anova_first_peak.csv
   artifacts/ml_hardness_part_f/majority3_runtime_hardness_label_binary/tables/selected_features_anova_first_peak.csv
   ```

   The CLI also has a safety check: `--max-selected-features-per-target 10` by default. If a file accidentally contains all 23 graph features, Part G stops with an error rather than running FP-Growth over the full feature set.

2. **Run sensitivity over 3, 4, and 5 percentile bins.**
   The default bins are:

   - 3 bins: `[0,33]`, `[33,66]`, `[66,100]`
   - 4 bins: `[0,25]`, `[25,50]`, `[50,75]`, `[75,100]`
   - 5 bins: `[0,20]`, `[20,40]`, `[40,60]`, `[60,80]`, `[80,100]`

   Binning is done **separately within each dataset**, because the hardness labels were defined dataset-wise.

3. **Ground truth and model predictions are both interpreted.**
   For each target and dataset, Part G mines rules for:

   - the empirical hardness label itself (`ground_truth`), and
   - predictions from the best saved Part F model selected separately within that dataset (`model_prediction`).

   The default model-selection criterion is weighted F1, matching the requirement to pick the best model per dataset. You can switch to minority-class F1 with `--model-selection-metric minority_f1`.

4. **Consequents are restricted to standalone labels.**
   Rules are kept only when the consequent is exactly one item:

   - `Hard`, or
   - `Not Hard`.

   Feature bins are never allowed in consequents.

5. **Minimum support uses an adaptive default.**
   The default `--min-support auto` uses the minority-class prevalence to avoid thresholds that are impossible for rare Hard classes, while still avoiding extremely tiny slices. You can override it with a fixed value such as `--min-support 0.03`.

6. **Rules are sorted and selected for non-overlap.**
   Candidate rules are sorted by decreasing support, then decreasing lift, then decreasing confidence. Selected rules are greedily filtered so their antecedents do not cover the same rows in the current dataset/target/bin/output group.

7. **Rules are evaluated as simple if-then-else classifiers.**
   Each rule is evaluated as:

   - if the antecedent is satisfied, predict the consequent class;
   - otherwise, predict the opposite class.

   The output includes weighted F1, minority-class F1, precision/recall/F1 for Hard and Not Hard, balanced accuracy, and the four requested cases:

   - antecedent satisfied and actual Hard,
   - antecedent satisfied and actual Not Hard,
   - antecedent not satisfied and actual Hard,
   - antecedent not satisfied and actual Not Hard.

## Inputs

Default expected files:

```text
artifacts/features_full/graph_features.csv
artifacts/hardness_labels_consensus_runtime/hardness_labels.csv
artifacts/ml_hardness_part_f/<target>/tables/selected_features_anova_first_peak.csv
artifacts/ml_hardness_part_f/<target>/models/best_*_selected_features.joblib
```

Targets used by default:

```text
consensus5_runtime_hardness_label_binary
consensus4_runtime_hardness_label_binary
majority3_runtime_hardness_label_binary
```

## Run command

From the repository root:

```bash
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

## Recommended paper workflow

Run the default `--evaluation-scope all` first. This gives the clearest descriptive interpretation of the whole instance space and is appropriate when the question is: “What regions of the dataset are associated with hard instances?”

Then run a robustness version using:

```bash
--evaluation-scope test
```

Use the test-scope run when the paper needs to emphasize generalization of the learned model’s predictions. The all-scope run is stronger for descriptive instance-space interpretation; the test-scope run is stronger for avoiding any appearance of training-set interpretation bias.

## Outputs

Part G writes:

```text
artifacts/association_rules_part_g/part_g_association_rules.xlsx
artifacts/association_rules_part_g/selected_association_rules_part_g.csv
artifacts/association_rules_part_g/association_rule_runtime_summary_part_g.csv
artifacts/association_rules_part_g/model_selection_by_dataset_part_g.csv
```

The Excel workbook includes:

- `Config`
- `Runtime`
- `Model Selection`
- `All Rules`
- one sheet per dataset, such as `COLLAB`, `IMDB-BINARY`, and `TWITTER`

## Dependency installation

```bash
pip install -r requirements_part_g.txt
```

## Test command

```bash
pytest tests/test_part_g_association_rules.py -q
```
