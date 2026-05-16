# Part E — Runtime-consensus hardness labels

Part E constructs empirical hardness labels using solver runtimes.

## Required solver outputs

```text
artifacts/solver_runs/gurobi_full/solver_runs.csv
artifacts/solver_runs/clisat_full/solver_runs.csv
artifacts/solver_runs/momc_full/solver_runs.csv
artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv
artifacts/solver_runs/hgs_full_all_test_graphs/solver_runs.csv
```

Use the fixed EGN file:

```text
artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv
```

## Common five-solver instance set

```text
6194 instances total
5000 COLLAB
1000 IMDB-BINARY
194 TWITTER
```

## Labels

Primary label:

```text
consensus5_runtime_hardness_label_binary
```

Sensitivity labels:

```text
consensus4_runtime_hardness_label_binary
majority3_runtime_hardness_label_binary
```

## Definition

An instance is labelled `Hard` under the consensus-5 label if it is in the top runtime quartile for all five solvers, with the top runtime quartile computed separately by dataset and solver.

## Main output

```text
artifacts/hardness_labels_consensus_runtime/hardness_labels.csv
```
