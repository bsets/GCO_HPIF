# Part D — Solver wrappers

Part D contains wrappers for Maximum Clique Problem solver runs.

## Solver integrations

| Part | Solver | Notes |
|---|---|---|
| D.1 | Gurobi | Requires `gurobipy` and a valid Gurobi license |
| D.2 | CliSAT | Requires CliSAT executable or build |
| D.3 | MoMC | Requires MoMC source or executable |
| D.4 | EGN | Optional learned solver integration |
| D.5 | HGS | Optional heuristic/learned integration |

## Purpose

Each solver wrapper should run a solver on graph instances and produce standardized output that can be joined in Part E.

## Common output contract

Each solver output directory should contain a standardized file:

```text
solver_runs.csv
```

Typical output directories include:

```text
artifacts/solver_runs/gurobi_full/
artifacts/solver_runs/clisat_full/
artifacts/solver_runs/momc_full/
artifacts/solver_runs/egn_full_all_test_graphs_fixed/
artifacts/solver_runs/hgs_full_all_test_graphs/
```

## Notes

- Gurobi requires a license.
- EGN and HGS may require external repositories and environment-specific setup.
- Part E expects all solver outputs to use a consistent `solver_runs.csv` structure.
