## Part E — Runtime-consensus hardness labels

Part E constructs empirical hardness labels from the five solver output files. The primary target is `consensus5_runtime_hardness_label_binary`.

Definition:

1. Keep only graph instances common to Gurobi, CliSAT, MOMC, EGN, and HGS.
2. For each dataset and each algorithm, compute the 75th percentile of `runtime_seconds`.
3. Mark an instance as top-25 runtime for an algorithm if its runtime is greater than or equal to the corresponding dataset-algorithm 75th percentile.
4. Label an instance as Consensus-5 Hard if it is in the top-25 runtime group for all five algorithms.

Sensitivity labels are also written:

- `consensus4_runtime_hardness_label_binary`: Hard if top-25 runtime for at least 4 of 5 algorithms.
- `majority3_runtime_hardness_label_binary`: Hard if top-25 runtime for at least 3 of 5 algorithms.

Run:

```bash
python -m gco_hpif.cli.build_hardness_labels \
  --solver-runs \
    artifacts/solver_runs/gurobi_full/solver_runs.csv \
    artifacts/solver_runs/clisat_full/solver_runs.csv \
    artifacts/solver_runs/momc_full/solver_runs.csv \
    artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv \
    artifacts/solver_runs/hgs_full_all_test_graphs/solver_runs.csv \
  --features artifacts/features_full/graph_features.csv \
  --output-dir artifacts/hardness_labels_consensus_runtime
```
