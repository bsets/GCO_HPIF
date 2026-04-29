# Part D.5: HGS training/inference wrapper

## Purpose

This part adds an optional integration for the HGS / geometric scattering maximal clique implementation from:

https://github.com/yimengmin/GeometricScatteringMaximalClique

The wrapper follows the same design as the EGN integration:

1. Do not vendor upstream third-party code into GCO-HPIF.
2. Clone the upstream code locally under `external/HGS/GeometricScatteringMaximalClique`.
3. Use the Part B split manifest as the reproducible source of train/validation/test splits.
4. Produce standardized solver outputs:
   - `solver_runs.csv`
   - `solver_errors.csv`
   - `run_summary.csv`

## Licensing approach

The upstream HGS repository does not appear to include a clear open-source license. Because a public repository without an explicit license is not automatically reusable or redistributable, this integration treats HGS as an optional external dependency.

GCO-HPIF commits only:
- a wrapper,
- a CLI,
- tests,
- documentation,
- an ignored external-dependency placeholder.

GCO-HPIF does **not** commit:
- the upstream HGS source code,
- trained HGS model checkpoints,
- generated solver artifacts.

## Local setup

From the GCO-HPIF repository root:

```bash
mkdir -p external/HGS
git clone https://github.com/yimengmin/GeometricScatteringMaximalClique.git \
  external/HGS/GeometricScatteringMaximalClique
```

Install or confirm the core dependencies already used by the EGN part:

```bash
source .venv/bin/activate
python -m pip install pandas numpy scipy networkx tqdm torch torch_geometric
```

Depending on your PyTorch/PyG install, you may also need the matching wheels for:
- `torch_scatter`
- `torch_sparse`
- `torch_cluster`

## Smoke run

```bash
python -m gco_hpif.cli.run_hgs_solver \
  --mode smoke \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --hgs-root external/HGS/GeometricScatteringMaximalClique \
  --output-dir artifacts/solver_runs/hgs_smoke \
  --dataset twitter \
  --limit-per-split 2 \
  --epochs 2 \
  --infer-splits test \
  --num-walkers 4
```

## Full run

```bash
python -m gco_hpif.cli.run_hgs_solver \
  --mode train-and-infer \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --hgs-root external/HGS/GeometricScatteringMaximalClique \
  --output-dir artifacts/solver_runs/hgs_full \
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
  --sample-length 90
```

## Implementation notes

The wrapper dynamically imports the upstream `models.py` from the locally cloned HGS repository. It does not copy upstream source into GCO-HPIF.

The wrapper uses simple reproducible node features with input dimension 3:

1. normalized degree,
2. normalized core number,
3. clustering coefficient.

The HGS decoder is implemented in GCO-HPIF as a small greedy multi-start clique decoder over the model's node scores. This keeps the output contract stable and avoids copying the upstream sampler source.

## Outputs

The wrapper writes:

- `solver_runs.csv`: one row per inferred graph.
- `solver_errors.csv`: any graph-level failures.
- `run_summary.csv`: aggregate run metadata and basic statistics.
- `hgs_train_metrics.csv`: per-epoch training loss.
- `hgs_checkpoint.pt`: local generated checkpoint; do not commit it.

## Expected Git exclusions

The following should stay out of Git:

```text
external/HGS/GeometricScatteringMaximalClique/
artifacts/solver_runs/hgs_smoke/
artifacts/solver_runs/hgs_full/
*.pt
*.pth
*.ckpt
__pycache__/
.pytest_cache/
```
