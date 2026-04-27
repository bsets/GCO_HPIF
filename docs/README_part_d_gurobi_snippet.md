## Part D.1: Gurobi maximum-clique solver

Part D.1 runs the Gurobi exact maximum-clique wrapper on graph instances that passed Part C feature computation.

The candidate set is defined by:

```text
Part A graph manifest ∩ Part C successful feature rows
```

This means graphs that timed out or failed during feature computation are not solved in this stage.

### Smoke test

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

### Full run

```bash
python -m gco_hpif.cli.run_gurobi_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/gurobi_full \
  --datasets twitter collab imdb_binary \
  --time-limit-seconds 1800
```

### Outputs

```text
artifacts/solver_runs/gurobi_full/
├── solver_runs.csv
├── solver_errors.csv
└── run_summary.csv
```

`solver_runs.csv` stores one row per graph instance with graph identifiers, time limit, Gurobi status, runtime, best clique size, clique nodes, clique-validity check, optimality status, MIP gap, and objective bound where available.

Generated solver outputs are not committed to Git.
