# EGN naming fix: exact local workflow

This workflow patches `src/gco_hpif/solvers/egn_solver.py` so EGN all-test inference writes canonical graph IDs such as `collab_graph000001` and `imdb_binary_graph000001`.

## 1. Unzip and apply the patch

```bash
cd ~/Downloads
unzip gco_hpif_egn_patch_exact_sequence.zip
cd gco_hpif_egn_patch_exact_sequence

bash scripts/apply_egn_naming_fix_to_repo.sh \
  /home/bharat/Desktop/PhD_Research/Clean_Code_for_GitHub/GCO-HPIF
```

## 2. Activate environment and run checks

```bash
cd /home/bharat/Desktop/PhD_Research/Clean_Code_for_GitHub/GCO-HPIF

source .venv/bin/activate

python -m py_compile src/gco_hpif/solvers/egn_solver.py
pytest -q
```

## 3. Rerun EGN all-test inference using existing checkpoint

```bash
EGN_CKPT="artifacts/solver_runs/egn_full/trained_egn_model.pt"

python -m gco_hpif.cli.run_egn_solver \
  --mode infer \
  --split-manifest data/manifests/twitter_split_60_20_20.csv \
  --interim-dir artifacts/slice_a_full/interim \
  --egn-root external/EGN/erdos_neu \
  --checkpoint "$EGN_CKPT" \
  --output-dir artifacts/solver_runs/egn_full_all_test_graphs_fixed \
  --dataset twitter \
  --infer-batch-size 1 \
  --inference-samples 8 \
  --extra-infer-graph-store collab=artifacts/slice_a_full/interim/collab_graphs.pkl.gz \
  --extra-infer-graph-store imdb_binary=artifacts/slice_a_full/interim/imdb_binary_graphs.pkl.gz
```

## 4. Verify EGN graph IDs

```bash
python - <<'PY'
import pandas as pd

path = "artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv"
df = pd.read_csv(path)

print("\nCounts by dataset:")
print(df.groupby("dataset").size())

print("\nCOLLAB first rows:")
print(df[df["dataset"]=="collab"][["graph_id","source_index"]].head(10).to_string(index=False))

print("\nIMDB-BINARY first rows:")
print(df[df["dataset"]=="imdb_binary"][["graph_id","source_index"]].head(10).to_string(index=False))

print("\nTWITTER first rows:")
print(df[df["dataset"]=="twitter"][["graph_id","source_index"]].head(10).to_string(index=False))
PY
```

Expected COLLAB/IMDB-BINARY style:

```text
collab_graph000001       1
collab_graph000002       2
imdb_binary_graph000001  1
imdb_binary_graph000002  2
```

## 5. Rerun Part E using the fixed EGN file

```bash
python -m gco_hpif.cli.build_hardness_labels \
  --solver-runs \
    artifacts/solver_runs/gurobi_full/solver_runs.csv \
    artifacts/solver_runs/clisat_full/solver_runs.csv \
    artifacts/solver_runs/momc_full/solver_runs.csv \
    artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv \
    artifacts/solver_runs/hgs_full_all_test_graphs/solver_runs.csv \
  --features artifacts/features_full/graph_features.csv \
  --output-dir artifacts/hardness_labels_full_strict_after_egn_fix
```

## 6. Inspect Part E output

```bash
cat artifacts/hardness_labels_full_strict_after_egn_fix/hardness_label_summary.csv

python - <<'PY'
import pandas as pd

path = "artifacts/hardness_labels_full_strict_after_egn_fix/hardness_labels.csv"
df = pd.read_csv(path)

print("\nRows:", len(df))
print("\nLabel counts:")
print(df.groupby(["dataset","hardness_label"]).size().reset_index(name="count").to_string(index=False))

cols = [
    "dataset", "graph_id", "source_index", "hardness_label",
    "clique_gurobi", "clique_clisat", "clique_momc", "clique_egn", "clique_hgs",
]
available = [c for c in cols if c in df.columns]
print("\nSample rows:")
print(df[available].head(20).to_string(index=False))
PY
```

## 7. Commit

```bash
git status

git checkout -b fix-egn-canonical-graph-ids

git add \
  src/gco_hpif/solvers/egn_solver.py \
  scripts/repair_egn_solver_runs_ids.py \
  docs/EGN_NAMING_FIX_EXACT_WORKFLOW.md \
  docs/NEW_SESSION_SUMMARY_EGN_NAMING_FIX.txt

git commit -m "Fix EGN extra inference graph IDs"
git push -u origin fix-egn-canonical-graph-ids
```

## Fallback: repair existing EGN file without rerunning inference

```bash
python scripts/repair_egn_solver_runs_ids.py \
  --input artifacts/solver_runs/egn_full_all_test_graphs/solver_runs.csv \
  --output artifacts/solver_runs/egn_full_all_test_graphs_fixed/solver_runs.csv
```
