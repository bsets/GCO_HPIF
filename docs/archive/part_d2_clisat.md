# Part D.2: CliSAT maximum-clique solver wrapper

Part D.2 adds a CliSAT-based exact maximum-clique solver wrapper to GCO-HPIF.

It is designed to align with Part D.1 Gurobi by writing the same canonical output files:

```text
solver_runs.csv
solver_errors.csv
run_summary.csv
```

The wrapper also preserves CliSAT-specific raw outputs so the execution remains easy to debug and comparable with the original Jupyter notebook workflow.

---

## What this part does

For each graph that passed Part C feature computation, the CliSAT wrapper:

1. loads the graph from the Part A interim graph pickle files,
2. converts the NetworkX graph to a DIMACS `.clq` file,
3. runs the bundled CliSAT binary,
4. parses the CliSAT output fields `omega`, clique node ids, `ts(s)`, `tp(s)`, and `tr(s)`,
5. validates that the parsed node list is a clique in the original graph, and
6. writes aligned solver-output CSV files.

The binary call follows the same pattern as the original notebook:

```bash
external/CliSAT/bin/CliSAT <graph_path.clq> <time_limit_seconds> <threads>
```

For the full run, that becomes:

```bash
external/CliSAT/bin/CliSAT <graph_path.clq> 1800 1
```

---

## Files added by this patch

```text
.gitattributes
README_PART_D2_CLISAT_PATCH.md
README_SNIPPET_PART_D2_CLISAT.md
docs/part_d2_clisat.md
external/CliSAT/LICENSE
external/CliSAT/NOTICE.md
external/CliSAT/README.md
external/CliSAT/bin/CliSAT
src/gco_hpif/cli/run_clisat_solver.py
src/gco_hpif/solvers/clisat_solver.py
tests/test_clisat_solver.py
```

---

## Third-party binary and license

The bundled binary is stored at:

```text
external/CliSAT/bin/CliSAT
```

The source project is:

```text
https://github.com/psanse/CliSAT
```

The license text included with the user-provided CliSAT material says the project is released under the Unlicense / public-domain dedication. Therefore this patch includes:

```text
external/CliSAT/LICENSE
external/CliSAT/NOTICE.md
```

The binary is an ELF 64-bit Linux x86-64 executable. It is expected to run on compatible Linux systems. For another platform, replace the binary or pass another executable with `--clisat-executable`.

---

## Node-index convention

GCO-HPIF stores graph nodes internally with 0-based integer labels.

DIMACS `.clq` edge endpoints are written as 1-based ids, so internal edge `(0, 2)` becomes:

```text
e 1 3
```

The uploaded CliSAT binary reports clique node ids using 0-based ids. Therefore the CLI default is:

```bash
--clisat-output-node-base 0
```

Use this option only if you replace the binary with a different build:

```bash
--clisat-output-node-base 1
```

---

## Required upstream steps

Before running CliSAT, Parts A and C should already have been run.

Part A creates the graph index and interim graph pickle files:

```text
artifacts/slice_a_full/manifests/graphs_index.csv
artifacts/slice_a_full/interim/twitter_graphs.pkl.gz
artifacts/slice_a_full/interim/collab_graphs.pkl.gz
artifacts/slice_a_full/interim/imdb_binary_graphs.pkl.gz
```

Part C creates the graph feature file. CliSAT runs only on graphs present in this file:

```text
artifacts/features_full/graph_features.csv
```

---

## Smoke test command

Run only a few graphs from each dataset:

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

This uses the default bundled executable:

```text
external/CliSAT/bin/CliSAT
```

---

## Full run command

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

---

## Canonical outputs

The canonical outputs are aligned with the Gurobi wrapper:

```text
artifacts/solver_runs/clisat_full/
├── solver_runs.csv
├── solver_errors.csv
└── run_summary.csv
```

`solver_runs.csv` uses these columns from the shared solver contract:

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

`best_clique_nodes` is written as a compact JSON list, using the repository's internal 0-based node convention.

---

## Additional CliSAT-specific outputs

The wrapper also writes reproducibility/debug files:

```text
artifacts/solver_runs/clisat_full/
├── dimacs/
│   ├── twitter/
│   ├── collab/
│   └── imdb_binary/
├── raw_clisat_outputs/
│   ├── twitter/
│   ├── collab/
│   └── imdb_binary/
├── twitter_dataset_CliSAT_results.txt
├── collab_dataset_CliSAT_results.txt
└── imdb_binary_dataset_CliSAT_results.txt
```

The `*_dataset_CliSAT_results.txt` files preserve the raw block style of the original notebook:

```text
Graph: <graph_id>
<raw CliSAT stdout>
**************************************************
```

---

## Tests

Run the CliSAT-specific test file:

```bash
pytest -q tests/test_clisat_solver.py
```

Run the full test suite:

```bash
pytest -q
```

The tests cover:

1. DIMACS conversion,
2. CliSAT stdout parsing,
3. fake-executable solver calls,
4. node-index conversion,
5. the bundled binary smoke test on Linux, and
6. common output CSV creation.

---

## Troubleshooting

### Permission denied

Run:

```bash
chmod +x external/CliSAT/bin/CliSAT
```

### `CliSAT executable not found`

Confirm that the binary exists:

```bash
ls -l external/CliSAT/bin/CliSAT
```

Or pass an explicit path:

```bash
python -m gco_hpif.cli.run_clisat_solver \
  ... \
  --clisat-executable /absolute/path/to/CliSAT
```

### Parsed clique is invalid

The most likely cause is a node-index convention mismatch. The bundled binary should use:

```bash
--clisat-output-node-base 0
```

If you replaced the binary with a build that reports 1-based node ids, run with:

```bash
--clisat-output-node-base 1
```
