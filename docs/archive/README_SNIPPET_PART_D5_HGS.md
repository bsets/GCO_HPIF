### Part D.5: HGS training/inference wrapper

HGS is integrated as an optional external solver because the upstream repository does not appear to include a clear open-source license. The GCO-HPIF repository therefore does **not** vendor the upstream HGS source code. To run HGS locally, clone the upstream repository into `external/HGS/GeometricScatteringMaximalClique`, which is ignored by Git.

Smoke run:

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

Full run:

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

Outputs:

- `solver_runs.csv`
- `solver_errors.csv`
- `run_summary.csv`
- `hgs_train_metrics.csv`
- `hgs_checkpoint.pt`
