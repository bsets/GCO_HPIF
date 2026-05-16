# Part A — Raw graph ingestion

Part A prepares the raw graph datasets used throughout the GCO-HPIF pipeline.

## Purpose

This stage downloads or reads the source graph datasets, normalizes them into a common internal representation, and writes graph manifests that later stages can consume.

## Datasets

- `TWITTER`
- `COLLAB`
- `IMDB-BINARY`

## Smoke test

```bash
python -m gco_hpif.cli.prepare_raw_graphs \
  --output-root artifacts/slice_a_smoke \
  --datasets imdb_binary collab twitter \
  --limit 3
```

## Full run

```bash
python -m gco_hpif.cli.prepare_raw_graphs \
  --output-root artifacts/slice_a_full \
  --datasets imdb_binary collab twitter
```

## Main outputs

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

## Notes

- Generated artifacts should not be committed to Git.
- Later stages use the manifest files and serialized graph objects written by this part.
