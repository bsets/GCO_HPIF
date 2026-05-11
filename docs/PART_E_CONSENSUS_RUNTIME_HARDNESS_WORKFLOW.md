# Part E consensus-runtime hardness workflow

## Test

```bash
cd /home/bharat/Desktop/PhD_Research/Clean_Code_for_GitHub/GCO-HPIF
source .venv/bin/activate
pip install -e .[dev]
pytest -q tests/test_runtime_hardness_labels.py
python -m py_compile src/gco_hpif/labels/hardness.py src/gco_hpif/cli/build_hardness_labels.py
```

## Run Part E

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

## Inspect

```bash
cat artifacts/hardness_labels_consensus_runtime/hardness_label_summary.csv
cat artifacts/hardness_labels_consensus_runtime/hardness_percentage_summary.csv
head artifacts/hardness_labels_consensus_runtime/hardness_labels.csv
```
