# GCO-HPIF

**GCO-HPIF** stands for **Graph-Based Combinatorial Optimization — Hardness Prediction and Interpretation Framework**.

This repository contains code for reproducing the pipeline used to study graph-instance hardness for the Maximum Clique Problem (MCP). The pipeline includes:

1. raw graph dataset ingestion,
2. fixed TWITTER train/validation/test split generation,
3. NetworkX-based graph feature computation,
4. solver wrappers and per-instance solver result logging,
5. machine-learning-based hardness prediction,
6. feature selection, and
7. association-rule-mining-based explanation of hardness patterns.

The repository is being built in modular parts so that each stage can be tested independently before being integrated into a full end-to-end pipeline.

---

## Pipeline status

- [x] **Part A:** Raw graph ingestion and graph manifests
- [x] **Part B:** TWITTER train/validation/test split manifest
- [x] **Part C:** 23 NetworkX graph features with 60-second timeout logging
- [x] **Part D.1:** Gurobi maximum-clique solver wrapper
- [x] **Part D.2:** CliSAT solver wrapper
- [x] **Part D.3:** MOMC solver wrapper
- [ ] **Part D.4:** EGN training/inference wrapper
- [ ] **Part D.5:** HGS training/inference wrapper
- [ ] **Part E:** Hardness label construction
- [ ] **Part F:** Hardness prediction models
- [ ] **Part G:** Feature selection and association-rule mining
- [ ] **Part H:** Computation time prediction models

---

## Datasets

The current pipeline uses the following graph datasets:

- **IMDB-BINARY**
- **COLLAB**
- **TWITTER ego networks**

Dataset handling is automated as much as possible.

- **IMDB-BINARY** and **COLLAB** are downloaded from the TU-format source used by PyTorch Geometric.
- **TWITTER** is downloaded from SNAP as `twitter.tar.gz` and parsed as one ego-network per `.edges` file.
- TWITTER graphs are converted to simple undirected NetworkX graphs.
- The TWITTER ego node is added back and connected to every node that appears in the corresponding `.edges` file.

Generated data files are written under `artifacts/` and are not committed to GitHub.

---

## Installation

Create and activate a Python virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -U pip
pip install -e .[dev]
```

Run tests:

```bash
pytest -q
```

---

## Part A: Raw graph ingestion

Part A downloads and normalizes the raw graph datasets.

It does **not** compute graph features and does **not** exclude any graphs.

Run a smoke test:

```bash
python -m gco_hpif.cli.prepare_raw_graphs \
  --output-root ./artifacts/slice_a_smoke \
  --datasets imdb_binary collab twitter \
  --limit 3
```

Run the full Part A pipeline:

```bash
python -m gco_hpif.cli.prepare_raw_graphs \
  --output-root ./artifacts/slice_a_full \
  --datasets imdb_binary collab twitter
```

Expected full graph counts:

| Dataset | Graph count |
|---|---:|
| IMDB-BINARY | 1000 |
| COLLAB | 5000 |
| TWITTER | 973 |

Part A creates:

```text
artifacts/slice_a_full/
├── raw/
├── interim/
│   ├── imdb_binary_graphs.pkl.gz
│   ├── collab_graphs.pkl.gz
│   └── twitter_graphs.pkl.gz
└── manifests/
    ├── graphs_index.csv
    └── dataset_summary.csv
```

---

## Part B: TWITTER split manifest

Part B creates a fixed TWITTER train/validation/test split manifest.

The split follows the original experimental setup used for the trainable solvers:

- first 60% of TWITTER graphs for training,
- next 20% for validation,
- remaining graphs for testing.

For the 973 TWITTER graphs loaded by Part A, this gives:

| Split | Graph count |
|---|---:|
| train | 584 |
| validation | 195 |
| test | 194 |

Create the split manifest:

```bash
python -m gco_hpif.cli.make_twitter_split \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --output data/manifests/twitter_split_60_20_20.csv
```

The resulting manifest is committed to the repository because it defines the fixed TWITTER split used later by EGN and HGS.

---

## Part C: NetworkX graph feature computation

Part C computes 23 graph-level features using NetworkX/NumPy.

Only graphs whose features are successfully computed within the per-graph time limit are retained for later solver and machine-learning stages.

The per-graph feature-computation time limit is:

```text
60 seconds
```

Run a smoke test:

```bash
python -m gco_hpif.cli.compute_graph_features \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/features_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 3 \
  --timeout-seconds 60
```

Run the full feature-computation step:

```bash
python -m gco_hpif.cli.compute_graph_features \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/features_full \
  --datasets twitter collab imdb_binary \
  --timeout-seconds 60
```

Part C creates:

```text
artifacts/features_full/
├── graph_features.csv
├── feature_failures.csv
├── feature_timing_summary.csv
└── feature_column_manifest.csv
```

`graph_features.csv` contains successful feature rows.

`feature_failures.csv` records graphs that failed or exceeded the 60-second time limit. This makes timeout-based exclusions explicit and reproducible.

The adjacency eigenvalue features are defined as:

```text
feature_Smallest_Eigenvalue_Adjacency
    Algebraically smallest eigenvalue of the adjacency matrix.

feature_Second_Smallest_Eigenvalue_Adjacency
    Algebraically second-smallest eigenvalue of the adjacency matrix.
```

---

## Part D.1: Gurobi maximum-clique solver wrapper

Part D.1 adds a Gurobi-based exact maximum-clique solver wrapper.

The wrapper uses the complement-graph mixed-integer programming formulation:

- create one binary variable per node,
- add one constraint for each edge in the complement graph,
- maximize the number of selected nodes.

The Gurobi wrapper runs only on graph instances whose features were successfully computed in Part C.

Gurobi is an optional solver backend. To run this part, the local machine must have:

- `gurobipy`,
- a valid Gurobi license,
- the `GRB_LICENSE_FILE` environment variable configured if the license is stored in a non-default location.

Example:

```bash
export GRB_LICENSE_FILE=/home/bharat/opt/gurobi1301/gurobi.lic
```

Run a Gurobi smoke test:

```bash
python -m gco_hpif.cli.run_gurobi_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/gurobi_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 2 \
  --time-limit-seconds 30 \
  --threads 1
```

Run the full Gurobi solver step:

```bash
python -m gco_hpif.cli.run_gurobi_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/gurobi_full \
  --datasets twitter collab imdb_binary \
  --time-limit-seconds 1800 \
  --threads 1
```

Part D.1 creates:

```text
artifacts/solver_runs/gurobi_full/
├── solver_runs.csv
├── solver_errors.csv
└── run_summary.csv
```

The Gurobi smoke test and full run have been validated locally after license renewal. Generated solver outputs are not committed to GitHub.

---

## Generated artifacts

The following folders contain generated outputs and should not be committed:

```text
artifacts/
data/raw/
data/interim/
```

The repository commits source code, tests, documentation, and small reproducibility manifests only.

---

## Development workflow

Before starting a new pipeline part:

```bash
git checkout main
git pull origin main
git checkout -b part-name
```

After making changes:

```bash
pytest -q
git status
git add <relevant files>
git commit -m "Descriptive commit message"
git push -u origin part-name
```

Then open a pull request on GitHub and merge into `main`.

---

## License

This repository is released under the MIT License.