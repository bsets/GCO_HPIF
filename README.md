# GCO-HPIF Slice A Starter

This starter implements **Part A** of the GCO-HPIF reproducible pipeline:

1. download/load the raw graph datasets
2. normalize them into simple undirected NetworkX graphs
3. assign stable `graph_id` values
4. save a manifest and pickled graph collections for later parts

This part **does not compute features** and **does not exclude any graphs**.
Dynamic timeout-based exclusions belong in **Part C**.

## Datasets

- **IMDB-BINARY** and **COLLAB** are fetched from the TU-format source used by PyG.
- **TWITTER** is fetched from SNAP `twitter.tar.gz` and parsed as one ego-network per `.edges` file.
- Twitter graphs are converted to **simple undirected graphs** for consistency with the later MCP / NetworkX pipeline.
- The ego node is added back and connected to every node that appears in the `.edges` file.

## Output layout

Running the CLI creates:

```text
<output-root>/
├── raw/
│   ├── tu/
│   │   ├── IMDB-BINARY.zip
│   │   └── COLLAB.zip
│   └── twitter/
│       └── twitter.tar.gz
├── interim/
│   ├── imdb_binary_graphs.pkl.gz
│   ├── collab_graphs.pkl.gz
│   └── twitter_graphs.pkl.gz
└── manifests/
    ├── graphs_index.csv
    └── dataset_summary.csv
```

## Installation

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Smoke test

Run a tiny end-to-end smoke test first:

```bash
python -m gco_hpif.cli.prepare_raw_graphs \
  --output-root ./artifacts/slice_a_smoke \
  --datasets imdb_binary collab twitter \
  --limit 3
```

This should create:
- `graphs_index.csv` with **9 rows**
- `dataset_summary.csv` with **3 rows**
- three compressed pickle files in `interim/`

## Full run

```bash
python -m gco_hpif.cli.prepare_raw_graphs \
  --output-root ./artifacts/slice_a_full \
  --datasets imdb_binary collab twitter
```

Expected graph counts when fully loaded:
- `imdb_binary`: **1000**
- `collab`: **5000**
- `twitter`: **973**

## Quick validation checks

After the smoke test, use the following lines to check the output:

```bash
python - <<'PY'
import pandas as pd

idx = pd.read_csv('artifacts/slice_a_smoke/manifests/graphs_index.csv')
summary = pd.read_csv('artifacts/slice_a_smoke/manifests/dataset_summary.csv')

print(idx.groupby('dataset').size())
print(summary[['dataset', 'graph_count']])
print(idx.head())
PY
```

For the full run, the `graph_count` values should be 1000, 5000, and 973.

## Next parts

- **Part B:** commit the TWITTER train/validation/test split manifest
- **Part C:** compute the 23 NetworkX features with a 60 second per-graph timeout and log failures dynamically
- **Part D:** solver wrappers and per-instance result logging

