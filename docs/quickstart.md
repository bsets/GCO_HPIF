# Quickstart

Install dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements_part_f.txt
pip install -r requirements_part_g.txt
```

Run a small raw-graph smoke test:

```bash
python -m gco_hpif.cli.prepare_raw_graphs \
  --output-root artifacts/slice_a_smoke \
  --datasets imdb_binary collab twitter \
  --limit 3
```

Compute graph features:

```bash
python -m gco_hpif.cli.compute_graph_features \
  --graphs-index artifacts/slice_a_smoke/manifests/graphs_index.csv \
  --interim-dir artifacts/slice_a_smoke/interim \
  --output-dir artifacts/features_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 3 \
  --timeout-seconds 60
```
