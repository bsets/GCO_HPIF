## Part C: NetworkX graph-feature computation

Part C computes the 23 NetworkX graph features used in the GCO-HPIF pipeline.
It uses the graph pickle files and graph manifest created by Part A.

This step applies a per-graph timeout and writes failures explicitly instead of
hard-coding exclusions.

### Adjacency eigenvalue definitions

The adjacency eigenvalue features use algebraic ascending order:

```text
feature_Smallest_Eigenvalue_Adjacency = lambda_1(A)
feature_Second_Smallest_Eigenvalue_Adjacency = lambda_2(A)
```

That is, `feature_Smallest_Eigenvalue_Adjacency` is the most negative adjacency
eigenvalue when negative eigenvalues exist. It is **not** the nonzero adjacency
eigenvalue closest to zero.

Laplacian nonzero eigenvalue features remain explicitly nonzero:

```text
feature_Smallest_NonZero_Eigenvalue_Laplacian
feature_Second_Smallest_NonZero_Eigenvalue_Laplacian
```

### Smoke test

```bash
python -m gco_hpif.cli.compute_graph_features \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/features_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 3 \
  --timeout-seconds 60
```

Expected smoke-test outputs:

```text
artifacts/features_smoke/
├── graph_features.csv
├── feature_failures.csv
├── feature_timing_summary.csv
└── feature_column_manifest.csv
```

### Full run

```bash
python -m gco_hpif.cli.compute_graph_features \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/features_full \
  --datasets twitter collab imdb_binary \
  --timeout-seconds 60
```

The successful graph rows are written to:

```text
artifacts/features_full/graph_features.csv
```

Timeouts and other failures are written to:

```text
artifacts/features_full/feature_failures.csv
```

The failure file contains the following fields:

```text
dataset,graph_id,source_index,failure_reason,exception_type,error_message,timeout_seconds,elapsed_seconds
```

The full feature outputs are generated artifacts and should not be committed to GitHub.
Only the source code, tests, documentation, and small committed manifests should be versioned.
