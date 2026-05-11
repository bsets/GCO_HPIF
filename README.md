# GCO-HPIF

**GCO-HPIF** stands for **Graph-Based Combinatorial Optimization — Hardness Prediction and Interpretation Framework**.

This repository contains a modular, reproducible pipeline for studying graph-instance hardness for the Maximum Clique Problem (MCP). The completed stages currently cover raw graph ingestion, fixed TWITTER train/validation/test splitting, graph-feature computation, solver wrappers for exact, heuristic, and learned maximum-clique solvers, runtime-consensus hardness label construction, and downstream ML hardness classification from graph features.## Completed pipeline status

| Part | Status | Description |
|---|---:|---|
| Part A | Complete | Raw graph ingestion and graph manifests |
| Part B | Complete | Fixed TWITTER 60/20/20 train/validation/test split manifest |
| Part C | Complete | 23 NetworkX graph features with 60-second timeout logging |
| Part D.1 | Complete | Gurobi maximum-clique solver wrapper |
| Part D.2 | Complete | CliSAT maximum-clique solver wrapper |
| Part D.3 | Complete | MoMC maximum-clique solver wrapper |
| Part D.4 | Complete | Optional EGN training/inference wrapper |
| Part D.5 | Complete | Optional HGS training/inference wrapper |
| Part E | Complete | Runtime-consensus hardness label construction from five solver outputs |
| Part F | Complete | ML hardness classification from graph features using feature selection and tuned classifiers |

Future stages planned for this repository include association-rule mining, percentile-bin interpretation of selected graph features, and computation-time prediction.

---

## Completed pipeline status

| Part     | Status   | Description                                                 |
| -------- | -------- | ----------------------------------------------------------- |
| Part A   | Complete | Raw graph ingestion and graph manifests                     |
| Part B   | Complete | Fixed TWITTER 60/20/20 train/validation/test split manifest |
| Part C   | Complete | 23 NetworkX graph features with 60-second timeout logging   |
| Part D.1 | Complete | Gurobi maximum-clique solver wrapper                        |
| Part D.2 | Complete | CliSAT maximum-clique solver wrapper                        |
| Part D.3 | Complete | MoMC maximum-clique solver wrapper                          |
| Part D.4 | Complete | Optional EGN training/inference wrapper                     |
| Part D.5 | Complete | Optional HGS training/inference wrapper                     |

Future stages planned for this repository include hardness label construction, hardness prediction models, feature selection, association-rule mining, and computation-time prediction.

---

## Datasets

The current pipeline uses:

* `TWITTER` ego-network graphs from SNAP.
* `COLLAB` from TU-format graph datasets.
* `IMDB-BINARY` from TU-format graph datasets.

Part A writes graph stores under:

```text
artifacts/slice_a_full/interim/
├── twitter_graphs.pkl.gz
├── collab_graphs.pkl.gz
└── imdb_binary_graphs.pkl.gz
```

Generated data and solver outputs are written under `artifacts/` and are not committed to GitHub.

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

## Part A — Raw graph ingestion

Smoke test:

```bash
python -m gco_hpif.cli.prepare_raw_graphs \
  --output-root artifacts/slice_a_smoke \
  --datasets imdb_binary collab twitter \
  --limit 3
```

Full run:

```bash
python -m gco_hpif.cli.prepare_raw_graphs \
  --output-root artifacts/slice_a_full \
  --datasets imdb_binary collab twitter
```

Expected full graph counts:

| Dataset     | Count |
| ----------- | ----- |
| TWITTER     | 973   |
| COLLAB      | 5000  |
| IMDB-BINARY | 1000  |

Main outputs:

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

---

## Part B — TWITTER split manifest

Create the fixed 60/20/20 TWITTER split:

```bash
python -m gco_hpif.cli.make_twitter_split \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --output data/manifests/twitter_split_60_20_20.csv
```

Expected split counts:

| Split      | Count |
| ---------- | ----- |
| train      | 584   |
| validation | 195   |
| test       | 194   |

The split manifest is committed because it defines the reproducible TWITTER split used by EGN and HGS.

---

## Part C — NetworkX graph features

Part C computes 23 graph-level features using NetworkX/NumPy. Graphs whose features fail or exceed the timeout are logged explicitly.

Smoke test:

```bash
python -m gco_hpif.cli.compute_graph_features \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/features_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 3 \
  --timeout-seconds 60
```

Full run:

```bash
python -m gco_hpif.cli.compute_graph_features \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/features_full \
  --datasets twitter collab imdb_binary \
  --timeout-seconds 60
```

Main outputs:

```text
artifacts/features_full/
├── graph_features.csv
├── feature_failures.csv
├── feature_timing_summary.csv
└── feature_column_manifest.csv
```

---

## Common solver-output contract

All solver wrappers write standardized outputs:

```text
solver_runs.csv
solver_errors.csv
run_summary.csv
```

Common columns include:

```text
dataset
graph_id
source_index
solver_name
run_type
time_limit_seconds
status
optimality_status
runtime_seconds
best_clique_size
best_clique_nodes
clique_valid
num_nodes
num_edges
seed
threads
mip_gap
objective_bound
error_message
```

---

## Part D.1 — Gurobi solver wrapper

Gurobi is an optional exact solver backend. You need:

* `gurobipy`
* a valid Gurobi license
* `GRB_LICENSE_FILE` configured if the license is not in the default location

Example:

```bash
export GRB_LICENSE_FILE=/home/bharat/opt/gurobi1301/gurobi.lic
```

Smoke test:

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

Full run:

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

---

## Part D.2 — CliSAT solver wrapper

CliSAT runs on graphs that passed Part C feature computation. The wrapper writes DIMACS input files and standardized solver logs.

Smoke test:

```bash
python -m gco_hpif.cli.run_clisat_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/clisat_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 2 \
  --time-limit-seconds 30 \
  --threads 1
```

Full run:

```bash
python -m gco_hpif.cli.run_clisat_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/clisat_full \
  --datasets twitter collab imdb_binary \
  --time-limit-seconds 1800 \
  --threads 1
```

Default executable path:

```text
external/CliSAT/bin/CliSAT
```

---

## Part D.3 — MoMC solver wrapper

MoMC runs on graphs that passed Part C feature computation. If the executable is missing, the CLI can compile it from the bundled C source unless `--no-compile` is used.

Smoke test:

```bash
python -m gco_hpif.cli.run_momc_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/momc_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 2 \
  --time-limit-seconds 30 \
  --threads 1
```

Full run:

```bash
python -m gco_hpif.cli.run_momc_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/momc_full \
  --datasets twitter collab imdb_binary \
  --time-limit-seconds 1800 \
  --threads 1
```

Default source/executable locations:

```text
external/MOMC/src/
external/MOMC/bin/MoMC
```

---

## Part D.4 — EGN training and all-test inference

EGN is treated as an optional external integration. The upstream EGN source is not vendored into this repository; clone it locally under:

```text
external/EGN/erdos_neu
```

Smoke test:

```bash
python -m gco_hpif.cli.run_egn_solver \
  --mode smoke \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --egn-root external/EGN/erdos_neu \
  --output-dir artifacts/solver_runs/egn_smoke \
  --dataset twitter \
  --limit-per-split 2 \
  --epochs 2 \
  --inference-samples 2
```

Train on the TWITTER training split and infer on the TWITTER test split:

```bash
python -m gco_hpif.cli.run_egn_solver \
  --mode train-and-infer \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --egn-root external/EGN/erdos_neu \
  --output-dir artifacts/solver_runs/egn_full \
  --dataset twitter \
  --epochs 100 \
  --train-batch-size 4 \
  --infer-batch-size 1 \
  --num-layers 5 \
  --hidden-1 64 \
  --hidden-2 1 \
  --learning-rate 0.001 \
  --penalty-coeff 4.0 \
  --seed 66 \
  --inference-samples 8
```

Run all-test inference using the trained EGN checkpoint:

```bash
EGN_CKPT="artifacts/solver_runs/egn_full/trained_egn_model.pt"

python -m gco_hpif.cli.run_egn_solver \
  --mode infer \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --egn-root external/EGN/erdos_neu \
  --checkpoint "$EGN_CKPT" \
  --output-dir artifacts/solver_runs/egn_full_all_test_graphs \
  --dataset twitter \
  --infer-batch-size 1 \
  --inference-samples 8 \
  --extra-infer-graph-store collab=artifacts/slice_a_full/interim/collab_graphs.pkl.gz \
  --extra-infer-graph-store imdb_binary=artifacts/slice_a_full/interim/imdb_binary_graphs.pkl.gz
```

Expected all-test inference count:

| Dataset      | Inference graphs |
| ------------ | ---------------- |
| TWITTER test | 194              |
| COLLAB       | 5000             |
| IMDB-BINARY  | 1000             |

---

## Part D.5 — HGS training and all-test inference

HGS is treated as an optional external integration. The upstream HGS source is not vendored into this repository; clone it locally under:

```text
external/HGS/GeometricScatteringMaximalClique
```

Example local setup:

```bash
mkdir -p external/HGS

git clone https://github.com/yimengmin/GeometricScatteringMaximalClique.git \
  external/HGS/GeometricScatteringMaximalClique
```

Smoke test:

```bash
python -m gco_hpif.cli.run_hgs_solver \
  --mode smoke \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --raw-graph-store artifacts/slice_a_full/interim/twitter_graphs.pkl.gz \
  --hgs-root external/HGS/GeometricScatteringMaximalClique \
  --output-dir artifacts/solver_runs/hgs_smoke \
  --dataset twitter \
  --limit-per-split 2 \
  --epochs 2 \
  --infer-splits test \
  --num-walkers 4
```

Train on the TWITTER training split and run all-test inference:

```bash
python -m gco_hpif.cli.run_hgs_solver \
  --mode train-and-infer \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --raw-graph-store artifacts/slice_a_full/interim/twitter_graphs.pkl.gz \
  --hgs-root external/HGS/GeometricScatteringMaximalClique \
  --output-dir artifacts/solver_runs/hgs_full_all_test_graphs \
  --dataset twitter \
  --epochs 20 \
  --train-batch-size 1 \
  --infer-splits test \
  --hidden 8 \
  --num-layers 4 \
  --learning-rate 0.001 \
  --penalty-coeff 2.0 \
  --seed 42 \
  --num-walkers 20 \
  --sample-length 90 \
  --extra-infer-graph-store collab=artifacts/slice_a_full/interim/collab_graphs.pkl.gz \
  --extra-infer-graph-store imdb_binary=artifacts/slice_a_full/interim/imdb_binary_graphs.pkl.gz
```

Run all-test inference using an existing HGS checkpoint:

```bash
python -m gco_hpif.cli.run_hgs_solver \
  --mode infer \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --raw-graph-store artifacts/slice_a_full/interim/twitter_graphs.pkl.gz \
  --hgs-root external/HGS/GeometricScatteringMaximalClique \
  --checkpoint-path artifacts/solver_runs/hgs_full/hgs_checkpoint.pt \
  --output-dir artifacts/solver_runs/hgs_full_all_test_graphs \
  --dataset twitter \
  --infer-splits test \
  --extra-infer-graph-store collab=artifacts/slice_a_full/interim/collab_graphs.pkl.gz \
  --extra-infer-graph-store imdb_binary=artifacts/slice_a_full/interim/imdb_binary_graphs.pkl.gz \
  --num-walkers 20 \
  --sample-length 90
```

Expected all-test inference count:

| Dataset      | Inference graphs |
| ------------ | ---------------- |
| TWITTER test | 194              |
| COLLAB       | 5000             |
| IMDB-BINARY  | 1000             |

---

## Checking solver outputs

After any solver run, inspect the output folder:

```bash
ls -lh artifacts/solver_runs/<run_name>
cat artifacts/solver_runs/<run_name>/run_summary.csv
head artifacts/solver_runs/<run_name>/solver_runs.csv
cat artifacts/solver_runs/<run_name>/solver_errors.csv
```

For all-test EGN/HGS runs, check dataset-level counts:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("artifacts/solver_runs/<run_name>/solver_runs.csv")
df = pd.read_csv(path)

print("Total rows:", len(df))
print(df.groupby("dataset").size())
print()
print("Invalid clique counts:")
print(df.groupby("dataset")["clique_valid"].apply(lambda s: (~s.astype(bool)).sum()))
PY
```

---

## Generated artifacts

The following are generated locally and should not be committed:

```text
artifacts/
data/raw/
data/interim/
external/EGN/erdos_neu/
external/HGS/GeometricScatteringMaximalClique/
*.pt
*.pth
*.ckpt
__pycache__/
.pytest_cache/
```

The repository commits source code, tests, documentation, small fixed manifests, and external-dependency placeholders only.

---

## Development workflow

Start a new branch:

```bash
git checkout main
git pull origin main
git checkout -b <branch-name>
```

Run checks:

```bash
pytest -q
python -m py_compile src/gco_hpif/cli/*.py
```

Commit and push:

```bash
git status
git add <relevant source/docs/tests only>
git commit -m "Descriptive commit message"
git push -u origin <branch-name>
```

Open a pull request into `main`, merge online, then sync locally:

```bash
git checkout main
git pull origin main
git branch -d <branch-name>
git status
```

---

## License

This repository is released under the MIT License.

