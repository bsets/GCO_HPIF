# Part D.3: MoMC solver wrapper

Part D.3 adds a MoMC-based exact maximum-clique solver wrapper to the GCO-HPIF pipeline.

The wrapper is aligned with the Part D.1 Gurobi and Part D.2 CliSAT solver-output contract. It writes:

```text
solver_runs.csv
solver_errors.csv
run_summary.csv
```

It also writes DIMACS inputs and raw MoMC logs for reproducibility:

```text
artifacts/solver_runs/momc_full/
├── dimacs/
├── raw_momc_outputs/
├── twitter_dataset_MOMC_results.txt
├── collab_dataset_MOMC_results.txt
├── imdb_binary_dataset_MOMC_results.txt
├── solver_runs.csv
├── solver_errors.csv
└── run_summary.csv
```

## Third-party source and license

The MoMC source is included under:

```text
external/MOMC/src/MOMC2016_1800sec_timeout_aware.c
```

Original source URL:

```text
https://home.mis.u-picardie.fr/~cli/MoMC2016.c
```

The source file includes the copyright and permissive license notice from Chu-Min Li and Hua Jiang. The license is preserved in:

```text
external/MOMC/LICENSE
```

The attribution notice is preserved in:

```text
external/MOMC/NOTICE.md
```

## Build behavior

The wrapper auto-compiles the MoMC executable when it is missing:

```bash
gcc -O3 -DMOMC external/MOMC/src/MOMC2016_1800sec_timeout_aware.c -o external/MOMC/bin/MoMC
```

This mirrors the original notebook compilation command.

You can also build manually:

```bash
make -C external/MOMC
```

The generated binary is written to:

```text
external/MOMC/bin/MoMC
```

## Run a smoke test

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

Note: the uploaded MoMC source has an internal 1800-second timeout. The `--time-limit-seconds` argument is recorded in the standardized output and is used for the Python subprocess timeout. For the final full run, use `1800` to match the source-level timeout.

## Runtime recording

MoMC can print `Time 0.00000000` for very fast instances. When that happens, the wrapper records a high-resolution Python wall-clock measurement in `runtime_seconds` instead of writing `0`. When MoMC reports a positive runtime, the wrapper keeps the solver-reported value.

## Run the full MoMC solver step

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

## Optional manual compile

If you want to compile before running:

```bash
make -C external/MOMC
```

or:

```bash
gcc -O3 -DMOMC external/MOMC/src/MOMC2016_1800sec_timeout_aware.c -o external/MOMC/bin/MoMC
chmod +x external/MOMC/bin/MoMC
```

Then run the CLI with `--no-compile`:

```bash
python -m gco_hpif.cli.run_momc_solver \
  --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
  --features artifacts/features_full/graph_features.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --output-dir artifacts/solver_runs/momc_smoke \
  --datasets twitter collab imdb_binary \
  --limit-per-dataset 2 \
  --time-limit-seconds 30 \
  --threads 1 \
  --no-compile
```

## Output columns

`solver_runs.csv` uses the shared solver columns:

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

For MoMC:

- `solver_name` is `momc`.
- `best_clique_nodes` are stored as 0-based internal node ids.
- The raw MoMC output is kept under `raw_momc_outputs/` and in dataset-level text files.
- `mip_gap` and `objective_bound` are not applicable and are left blank.
