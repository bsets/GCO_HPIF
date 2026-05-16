# Part C — NetworkX graph features

Part C computes graph-level NetworkX features for every graph instance.

## Purpose

These features are the main explanatory and predictive variables used in the hardness-classification and runtime-prediction stages.

## Smoke test

```bash
python -m gco_hpif.cli.compute_graph_features \
  --graphs-index artifacts/slice_a_smoke/manifests/graphs_index.csv \
  --interim-dir artifacts/slice_a_smoke/interim \
  --output-dir artifacts/features_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 3 \
  --timeout-seconds 60
```

## Full run

```bash
python -m gco_hpif.cli.compute_graph_features \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/features_full \
  --datasets twitter collab imdb_binary \
  --timeout-seconds 60
```

## Main outputs

```text
artifacts/features_full/
├── graph_features.csv
├── feature_failures.csv
├── feature_timing_summary.csv
└── feature_column_manifest.csv
```

## Predictor-column rule

Only columns beginning with `feature_` should be treated as graph-feature predictors in later ML stages.
