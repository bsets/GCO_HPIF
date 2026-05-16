# GCO-HPIF

**GCO-HPIF** stands for **Graph-Based Combinatorial Optimization — Hardness Prediction and Interpretation Framework**.

This repository provides a modular research pipeline for studying graph-instance hardness for the **Maximum Clique Problem (MCP)**. It combines graph-feature extraction, exact/heuristic/learned solver runs, runtime-consensus hardness labels, machine-learning hardness classification, association-rule interpretation, and runtime prediction.

The project is designed so that each stage can be run, tested, and inspected independently, while still supporting an eventual end-to-end reproducible pipeline.

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
