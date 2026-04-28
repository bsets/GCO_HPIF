# Part D.3 MoMC solver patch

This patch adds the Part D.3 MoMC solver wrapper to the GCO-HPIF GitHub repository.

## Files added

```text
docs/part_d3_momc.md
external/MOMC/LICENSE
external/MOMC/Makefile
external/MOMC/NOTICE.md
external/MOMC/README.md
external/MOMC/src/MOMC2016_1800sec_timeout_aware.c
external/MOMC/bin/.gitignore
src/gco_hpif/cli/run_momc_solver.py
src/gco_hpif/solvers/momc_solver.py
tests/test_momc_solver.py
README_SNIPPET_PART_D3_MOMC.md
```

## Output alignment

MoMC uses the same standardized solver outputs as the Gurobi and CliSAT parts:

```text
solver_runs.csv
solver_errors.csv
run_summary.csv
```

The wrapper also keeps raw per-graph and dataset-level MoMC outputs for debugging and reproducibility.

## Step-by-step application instructions

Start from your local repo:

```bash
cd /home/bharat/Desktop/PhD_Research/Clean_Code_for_GitHub/GCO-HPIF
source .venv/bin/activate
```

Synchronize local `main`:

```bash
git checkout main
git pull origin main
```

Create a new branch:

```bash
git checkout -b part-d3-momc
```

Unzip the patch into the repo root:

```bash
unzip /path/to/gco_hpif_part_d3_momc_patch.zip -d .
```

Run the MoMC unit tests:

```bash
pytest -q tests/test_momc_solver.py
```

Optionally run the full test suite:

```bash
pytest -q
```

Run a smoke test on 2 graphs per dataset:

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

Run the full MoMC solver step:

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

Review generated outputs:

```bash
ls -R artifacts/solver_runs/momc_smoke
head -5 artifacts/solver_runs/momc_smoke/solver_runs.csv
cat artifacts/solver_runs/momc_smoke/run_summary.csv
```

Generated solver outputs under `artifacts/` should stay local and should not be committed.

Stage only source, tests, documentation, and third-party source/license files:

```bash
git add \
  README_SNIPPET_PART_D3_MOMC.md \
  docs/part_d3_momc.md \
  external/MOMC \
  src/gco_hpif/cli/run_momc_solver.py \
  src/gco_hpif/solvers/momc_solver.py \
  tests/test_momc_solver.py
```

Commit:

```bash
git commit -m "Add Part D.3 MoMC solver wrapper"
```

Push:

```bash
git push -u origin part-d3-momc
```

Open a pull request on GitHub, review the changed files, and merge into `main`.

## Commit / PR description

```text
Add Part D.3 MoMC solver wrapper

This update adds the MoMC integration for Part D.3 of the GCO-HPIF pipeline. It provides a reusable solver wrapper that compiles the bundled MoMC C source, runs MoMC on generated DIMACS graph instances, and writes outputs in the same standardized format used by the Gurobi and CliSAT solver integrations.

Key additions:
- Added MoMC solver wrapper under src/gco_hpif/solvers/momc_solver.py.
- Added CLI entry point: python -m gco_hpif.cli.run_momc_solver.
- Added third-party MoMC C source under external/MOMC/src.
- Added MoMC license, attribution notice, Makefile, and documentation under external/MOMC and docs.
- Added parser logic for MoMC output including clique size, clique nodes, runtime, branching count, and timeout detection.
- Added aligned output files:
  - solver_runs.csv
  - solver_errors.csv
  - run_summary.csv
- Added raw per-dataset MoMC output text files for reproducibility/debugging.
- Added tests for parser behavior, executable invocation, compilation helper, and output-format compatibility.

The wrapper supports smoke tests and full runs across twitter, collab, and imdb_binary datasets using the same graph index, features file, interim directory, and artifact layout as the rest of the GCO-HPIF execution pipeline.
```


Runtime note: MoMC can print `Time 0.00000000` for very fast instances. The wrapper falls back to a high-resolution wall-clock measurement in those cases so `runtime_seconds` does not become `0`.
