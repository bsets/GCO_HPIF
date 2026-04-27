# Local steps for Part D.1 Gurobi

```bash
cd /home/bharat/Desktop/PhD_Research/Clean_Code_for_GitHub/GCO-HPIF
git checkout main
git pull origin main
git checkout -b part-d-gurobi

unzip -o ~/Downloads/gco_hpif_part_d_gurobi_files.zip -d .

source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

Run the smoke test:

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

Validate:

```bash
python - <<'PY'
import pandas as pd

runs = pd.read_csv("artifacts/solver_runs/gurobi_smoke/solver_runs.csv")
errors = pd.read_csv("artifacts/solver_runs/gurobi_smoke/solver_errors.csv")
summary = pd.read_csv("artifacts/solver_runs/gurobi_smoke/run_summary.csv")

print(runs[["dataset", "graph_id", "status", "optimality_status", "best_clique_size", "clique_valid", "runtime_seconds"]])
print("\\nErrors:")
print(errors)
print("\\nSummary:")
print(summary)
PY
```

Commit source only:

```bash
git status
git add README.md docs/README_part_d_gurobi_snippet.md docs/README_part_d_gurobi_steps.md \
  src/gco_hpif/solvers/__init__.py \
  src/gco_hpif/solvers/common.py \
  src/gco_hpif/solvers/gurobi_solver.py \
  src/gco_hpif/cli/run_gurobi_solver.py \
  tests/test_solver_common.py \
  tests/test_gurobi_solver_optional.py

git commit -m "Add Part D Gurobi solver wrapper"
git push -u origin part-d-gurobi
```
