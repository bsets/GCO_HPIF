## Part D.4: EGN training/inference wrapper

Part D.4 adds an optional external integration for EGN / Erdos Goes Neural.

The wrapper uses the committed Part B split manifest:

```text
data/manifests/twitter_split_60_20_20.csv
```

and writes standardized Part D outputs:

```text
artifacts/solver_runs/egn_full/
├── trained_egn_model.pt
├── training_log.csv
├── solver_runs.csv
├── solver_errors.csv
└── run_summary.csv
```

EGN is treated as an optional external dependency. The upstream code is not vendored in this repository because the upstream repository does not include a detectable open-source license. To run EGN:

```bash
mkdir -p external/EGN
git clone https://github.com/Stalence/erdos_neu.git external/EGN/erdos_neu
```

Run a smoke test:

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

Run the full train-and-infer workflow:

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

Generated EGN outputs under `artifacts/` and the external upstream checkout under `external/EGN/erdos_neu/` should not be committed.
