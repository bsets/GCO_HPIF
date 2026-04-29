# Part D.4: EGN training/inference wrapper

Part D.4 adds an optional external integration for EGN / Erdos Goes Neural.

The wrapper is aligned with the Part D solver-output contract used by Gurobi, CliSAT, and MoMC. It writes:

```text
solver_runs.csv
solver_errors.csv
run_summary.csv
```

Training logs and checkpoints are written under the selected output directory:

```text
artifacts/solver_runs/egn_full/
├── trained_egn_model.pt
├── training_log.csv
├── solver_runs.csv
├── solver_errors.csv
└── run_summary.csv
```

## Why EGN is external and not vendored

The upstream EGN repository is:

```text
https://github.com/Stalence/erdos_neu
```

At the time this wrapper was prepared, the upstream repository did not include a detectable root `LICENSE` file. Therefore, GCO-HPIF does not copy or redistribute upstream EGN source files. Instead, this repository provides only original adapter/wrapper code. Users who want to reproduce EGN results should separately obtain the upstream code and provide its local path via `--egn-root` or `GCO_HPIF_EGN_ROOT`.

Expected local upstream files:

```text
external/EGN/erdos_neu/
├── models.py
├── cut_utils.py
├── modules_and_utils.py
└── myfuncs.py
```

## Input manifest

Part D.4 uses the committed Part B split manifest:

```text
data/manifests/twitter_split_60_20_20.csv
```

Required columns:

```text
dataset
graph_id
source_index
split
n_nodes
n_edges
graph_hash_sha256
source_loader
source_name
source_ego_id
```

The wrapper uses:

- `split == train` for training;
- `split == test` for inference;
- `graph_id` and `source_index` to load the corresponding graph from the Part A graph store.

## Setup external EGN code

From the GCO-HPIF repo root:

```bash
mkdir -p external/EGN
git clone https://github.com/Stalence/erdos_neu.git external/EGN/erdos_neu
```

For stronger reproducibility, record and pin the commit you used:

```bash
cd external/EGN/erdos_neu
git rev-parse HEAD
git checkout <commit-sha-used-for-your-run>
cd -
```

Do not commit `external/EGN/erdos_neu` to this repository.

## Dependencies

EGN requires a compatible PyTorch / PyTorch Geometric stack, including:

```text
torch
torch_geometric
torch_scatter
```

Install these according to your machine's CUDA/PyTorch version. The exact command depends on your CUDA environment.

## Smoke test

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
  --inference-samples 2 \
  --device cpu
```

The CPU smoke test is mainly for plumbing. Full EGN runs should normally use CUDA.

## Full train-and-infer run

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

## Train only

```bash
python -m gco_hpif.cli.run_egn_solver \
  --mode train \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --egn-root external/EGN/erdos_neu \
  --output-dir artifacts/solver_runs/egn_train \
  --dataset twitter \
  --epochs 100 \
  --seed 66
```

## Inference from an existing checkpoint

```bash
python -m gco_hpif.cli.run_egn_solver \
  --mode infer \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --egn-root external/EGN/erdos_neu \
  --checkpoint artifacts/solver_runs/egn_full/trained_egn_model.pt \
  --output-dir artifacts/solver_runs/egn_infer \
  --dataset twitter \
  --seed 66 \
  --inference-samples 8
```

## Output columns

`solver_runs.csv` uses the common Part D columns:

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

EGN is a neural/heuristic method, so:

```text
optimality_status = heuristic
```

It should not be labelled `optimal` unless separately compared against an exact solver result.
