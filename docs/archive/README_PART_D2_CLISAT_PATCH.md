# GCO-HPIF Part D.2: CliSAT patch

This patch adds the Part D.2 CliSAT maximum-clique solver wrapper to the GCO-HPIF repository.

It includes the uploaded CliSAT binary and runs graph instances using the same command pattern as the original notebook:

```bash
external/CliSAT/bin/CliSAT <graph_path.clq> <time_limit_seconds> <threads>
```

For the full run, this becomes:

```bash
external/CliSAT/bin/CliSAT <graph_path.clq> 1800 1
```

The Python CLI wraps this call, converts graphs to DIMACS, parses CliSAT output, validates cliques, and writes solver outputs aligned with Part D.1 Gurobi.

---

## 1. Files included

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

## 2. Third-party CliSAT attribution

The CliSAT binary is included under:

```text
external/CliSAT/bin/CliSAT
```

The original CliSAT repository is:

```text
https://github.com/psanse/CliSAT
```

The license text provided for CliSAT says it is released under the Unlicense / public-domain dedication. This patch therefore includes:

```text
external/CliSAT/LICENSE
external/CliSAT/NOTICE.md
```

The main GCO-HPIF repository can remain MIT-licensed; the `external/CliSAT/` directory separately documents the third-party CliSAT license and attribution.

---

## 3. Step-by-step application instructions

### Step 1: Go to your local repo

```bash
cd GCO_HPIF
```

### Step 2: Make sure you are on the latest `main`

```bash
git checkout main
git pull origin main
```

### Step 3: Create a new branch

```bash
git checkout -b part-d2-clisat
```

### Step 4: Unzip this patch into the repo root

If the zip file is in your Downloads folder, for example:

```bash
unzip ~/Downloads/gco_hpif_part_d2_clisat_final_patch.zip -d .
```

Confirm that the files landed in the expected locations:

```bash
ls -l external/CliSAT/bin/CliSAT
ls -l src/gco_hpif/solvers/clisat_solver.py
ls -l src/gco_hpif/cli/run_clisat_solver.py
ls -l tests/test_clisat_solver.py
```

### Step 5: Ensure the binary is executable

```bash
chmod +x external/CliSAT/bin/CliSAT
```

Check the binary:

```bash
file external/CliSAT/bin/CliSAT
```

Expected style of output:

```text
ELF 64-bit LSB executable, x86-64, ... GNU/Linux ...
```

### Step 6: Install the repo in editable mode

Use your existing project environment if you already have one. Otherwise:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

### Step 7: Run the CliSAT tests

```bash
pytest -q tests/test_clisat_solver.py
```

### Step 8: Run the full repository tests

```bash
pytest -q
```

### Step 9: Add the README section

Open:

```text
README_SNIPPET_PART_D2_CLISAT.md
```

Copy the suggested section into your main repository `README.md`, preferably after the existing Part D.1 Gurobi section.

### Step 10: Commit the patch

```bash
git status
git add .gitattributes \
  README_PART_D2_CLISAT_PATCH.md \
  README_SNIPPET_PART_D2_CLISAT.md \
  docs/part_d2_clisat.md \
  external/CliSAT \
  src/gco_hpif/solvers/clisat_solver.py \
  src/gco_hpif/cli/run_clisat_solver.py \
  tests/test_clisat_solver.py \
  README.md

git commit -m "Add Part D.2 CliSAT solver wrapper"
```

If you have not edited `README.md` yet, remove `README.md` from the `git add` command.

### Step 11: Push the branch

```bash
git push -u origin part-d2-clisat
```

Then open a pull request on GitHub.

---

## 4. Required upstream artifacts before running CliSAT

Before running CliSAT on the real datasets, Parts A and C should already have been completed.

Required Part A files:

```text
artifacts/slice_a_full/manifests/graphs_index.csv
artifacts/slice_a_full/interim/twitter_graphs.pkl.gz
artifacts/slice_a_full/interim/collab_graphs.pkl.gz
artifacts/slice_a_full/interim/imdb_binary_graphs.pkl.gz
```

Required Part C file:

```text
artifacts/features_full/graph_features.csv
```

The CliSAT wrapper solves only the graphs present in `graph_features.csv`. This keeps it aligned with Gurobi and with the Part C feature-timeout filtering.

---

## 5. Smoke test execution

Run a small test before the full 1800-second run:

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

This writes:

```text
artifacts/solver_runs/clisat_smoke/solver_runs.csv
artifacts/solver_runs/clisat_smoke/solver_errors.csv
artifacts/solver_runs/clisat_smoke/run_summary.csv
```

Also check the raw CliSAT outputs:

```text
artifacts/solver_runs/clisat_smoke/raw_clisat_outputs/
artifacts/solver_runs/clisat_smoke/twitter_dataset_CliSAT_results.txt
artifacts/solver_runs/clisat_smoke/collab_dataset_CliSAT_results.txt
artifacts/solver_runs/clisat_smoke/imdb_binary_dataset_CliSAT_results.txt
```

---

## 6. Full execution

Run CliSAT with the same timeout and final integer argument used in your notebook code:

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

Internally, for each graph, the wrapper runs:

```bash
external/CliSAT/bin/CliSAT <generated_dimacs_graph.clq> 1800 1
```

---

## 7. Expected full-run output structure

```text
artifacts/solver_runs/clisat_full/
├── solver_runs.csv
├── solver_errors.csv
├── run_summary.csv
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

The canonical file for downstream integration is:

```text
solver_runs.csv
```

The raw text files are reproducibility/debug artifacts.

---

## 8. Output-column alignment with Gurobi

`solver_runs.csv` uses the same column contract as the Gurobi wrapper:

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

CliSAT-specific fields that do not apply, such as `mip_gap` and `objective_bound`, are written as empty values so that all solver outputs can be concatenated later.

---

## 9. Important node-index note

The generated DIMACS files use 1-based edge endpoints, as expected by `.clq` format.

The uploaded CliSAT binary reports clique node ids using 0-based ids. Therefore the default is:

```bash
--clisat-output-node-base 0
```

Do not change this for the bundled binary.

If you later replace the binary with a different build that reports 1-based clique node ids, use:

```bash
--clisat-output-node-base 1
```

---

## 10. Useful verification commands

Check for errors:

```bash
python - <<'PY'
import pandas as pd
errors = pd.read_csv('artifacts/solver_runs/clisat_smoke/solver_errors.csv')
print(errors.head())
print('error rows:', len(errors))
PY
```

Check result columns:

```bash
python - <<'PY'
import pandas as pd
runs = pd.read_csv('artifacts/solver_runs/clisat_smoke/solver_runs.csv')
print(runs.columns.tolist())
print(runs[['dataset', 'graph_id', 'status', 'best_clique_size', 'clique_valid']].head())
PY
```

Check generated DIMACS files:

```bash
find artifacts/solver_runs/clisat_smoke/dimacs -name '*.clq' | head
```

Check notebook-style raw text:

```bash
head -40 artifacts/solver_runs/clisat_smoke/twitter_dataset_CliSAT_results.txt
```

---

## 11. Troubleshooting

### Permission denied

```bash
chmod +x external/CliSAT/bin/CliSAT
```

### Binary does not run on your OS

The bundled file is a Linux x86-64 executable. For macOS, Windows, or a different architecture, rebuild CliSAT for that platform and either replace:

```text
external/CliSAT/bin/CliSAT
```

or pass:

```bash
--clisat-executable /path/to/your/CliSAT
```

### Parsed clique is invalid

For the bundled binary, keep:

```bash
--clisat-output-node-base 0
```

If using another build, try:

```bash
--clisat-output-node-base 1
```

### Missing Part A or Part C files

Re-run Parts A and C first. CliSAT depends on:

```text
artifacts/slice_a_full/manifests/graphs_index.csv
artifacts/slice_a_full/interim/
artifacts/features_full/graph_features.csv
```
