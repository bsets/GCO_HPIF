# GCO-HPIF

**GCO-HPIF** stands for **Graph-Based Combinatorial Optimization — Hardness Prediction and Interpretation Framework**.

This repository provides a modular research pipeline for studying graph-instance hardness for the **Maximum Clique Problem (MCP)**. It combines graph-feature extraction, solver runs, runtime-consensus hardness labels, machine-learning hardness classification, association-rule interpretation, and runtime prediction.

---

## Pipeline at a glance

| Part | Stage | Detailed docs |
|---|---|---|
| A | Raw graph ingestion and graph manifests | [docs/parts/part_a_raw_graphs.md](docs/parts/part_a_raw_graphs.md) |
| B | Fixed TWITTER train/validation/test split | [docs/parts/part_b_twitter_split.md](docs/parts/part_b_twitter_split.md) |
| C | NetworkX graph-feature computation | [docs/parts/part_c_features.md](docs/parts/part_c_features.md) |
| D | Solver wrappers for MCP algorithms | [docs/parts/part_d_solvers.md](docs/parts/part_d_solvers.md) |
| E | Runtime-consensus hardness labels | [docs/parts/part_e_hardness_labels.md](docs/parts/part_e_hardness_labels.md) |
| F | ML hardness classification | [docs/parts/part_f_hardness_classification.md](docs/parts/part_f_hardness_classification.md) |
| G | FP-Growth association-rule interpretation | [docs/parts/part_g_association_rules.md](docs/parts/part_g_association_rules.md) |
| H | Runtime prediction from graph features | [docs/parts/part_h_runtime_prediction.md](docs/parts/part_h_runtime_prediction.md) |

For the full documentation map, see [docs/index.md](docs/index.md).

---

## Datasets

The current experiments use:

- **TWITTER** ego-network graphs from SNAP.
- **COLLAB** graph instances from TU-format graph datasets.
- **IMDB-BINARY** graph instances from TU-format graph datasets.

Generated data, solver logs, models, plots, and reports are written under `artifacts/` and are not committed to GitHub.

---

## Installation

Create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -U pip
pip install -r requirements_part_f.txt
pip install -r requirements_part_g.txt
```

After packaging cleanup is complete, the preferred editable install will be:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

---

## Quick start

Run a small raw-graph smoke test:

```bash
python -m gco_hpif.cli.prepare_raw_graphs \
  --output-root artifacts/slice_a_smoke \
  --datasets imdb_binary collab twitter \
  --limit 3
```

Compute graph features for the smoke graphs:

```bash
python -m gco_hpif.cli.compute_graph_features \
  --graphs-index artifacts/slice_a_smoke/manifests/graphs_index.csv \
  --interim-dir artifacts/slice_a_smoke/interim \
  --output-dir artifacts/features_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 3 \
  --timeout-seconds 60
```

See [docs/quickstart.md](docs/quickstart.md) and [docs/cli_reference.md](docs/cli_reference.md) for stage-specific commands.

---

## Main output locations

Typical generated outputs include:

```text
artifacts/slice_a_full/
artifacts/features_full/
artifacts/solver_runs/
artifacts/hardness_labels_consensus_runtime/
artifacts/ml_hardness_part_f_feature_prefix_fixed/
artifacts/association_rules_part_g/
artifacts/runtime_prediction_part_h/
```

See [docs/outputs.md](docs/outputs.md) for details.

---

## External dependencies

Some stages require external tools or licenses:

- **Gurobi** requires `gurobipy` and a valid Gurobi license.
- **CliSAT** and **MoMC** require solver binaries or source builds.
- **EGN** and **HGS** are optional external integrations.
- Raw datasets and large generated artifacts are not bundled in this repository.

---

## Development

See [docs/development.md](docs/development.md).

---

## Citation

If you use this repository in academic work, please cite the associated pre-print:

> Sharman, Bharat and Hassini, Elkafi, *A General Framework for Predicting and Explaining the Hardness of Graph-Based Combinatorial Optimization Problems using Machine Learning and Association Rule Mining*. Available at SSRN: https://ssrn.com/abstract=5975002 or http://dx.doi.org/10.2139/ssrn.5975002

A `CITATION.cff` file is planned as part of the packaging cleanup.

---

## License

This repository is released under the MIT License.
