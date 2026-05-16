from pathlib import Path
import shutil
import subprocess

ROOT = Path(".")
DOCS = ROOT / "docs"
ARCHIVE = DOCS / "archive"
PARTS = DOCS / "parts"

ARCHIVE.mkdir(parents=True, exist_ok=True)
PARTS.mkdir(parents=True, exist_ok=True)

# Preserve original README only once.
readme_archive = ARCHIVE / "README_before_phase1_docs_cleanup.md"
if Path("README.md").exists() and not readme_archive.exists():
    shutil.copy2("README.md", readme_archive)

root_files_to_archive = [
    "README_PART_D2_CLISAT_PATCH.md",
    "README_PART_D3_MOMC_PATCH.md",
    "README_SNIPPET_PART_D2_CLISAT.md",
    "README_SNIPPET_PART_D3_MOMC.md",
    "README_SNIPPET_PART_D4_EGN.md",
    "README_SNIPPET_PART_D5_HGS.md",
    "README_SNIPPET_PART_E_CONSENSUS_RUNTIME.md",
    "README_SNIPPET_PART_F_ML_HARDNESS_CLASSIFICATION.md",
    "README_SNIPPET_PART_G_ASSOCIATION_RULE_MINING.md",
]

docs_files_to_archive = [
    "docs/PART_E_CONSENSUS_RUNTIME_HARDNESS_WORKFLOW.md",
    "docs/PART_F_ML_HARDNESS_CLASSIFICATION_WORKFLOW.md",
    "docs/PART_G_ASSOCIATION_RULE_MINING_WORKFLOW.md",
    "docs/PART_H_RUNTIME_PREDICTION_WORKFLOW.md",
    "docs/README_part_b_snippet.md",
    "docs/README_part_c_snippet.md",
    "docs/README_part_d_gurobi_snippet.md",
    "docs/README_part_d_gurobi_steps.md",
    "docs/part_d2_clisat.md",
    "docs/part_d3_momc.md",
    "docs/part_d4_egn.md",
    "docs/part_d5_hgs.md",
]


def git_mv_if_exists(src: str, dst: Path) -> None:
    src_path = Path(src)
    if not src_path.exists():
        return
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", str(src_path), str(dst)], check=True)


for f in root_files_to_archive:
    git_mv_if_exists(f, ARCHIVE / f)

for f in docs_files_to_archive:
    git_mv_if_exists(f, ARCHIVE / Path(f).name)


readme = """# GCO-HPIF

**GCO-HPIF** stands for **Graph-Based Combinatorial Optimization — Hardness Prediction and Interpretation Framework**.

This repository provides a modular research pipeline for studying graph-instance hardness for the **Maximum Clique Problem (MCP)**. It combines graph-feature extraction, solver runs, runtime-consensus hardness labels, machine-learning hardness classification, association-rule interpretation, and runtime prediction.

---

## Pipeline at a glance

| Part | Stage | Detailed docs |
|---|---|---|
| A | Raw graph ingestion and graph manifests | [docs/parts/part_a_raw_graphs.md](docs/parts/part_a_raw_graphs.md) |
| B | Fixed TWITTER train/validation/test split | [docs/parts/part_b_twitter_split.md](docs/parts/part_b_twitter_split.md) |
| C | NetworkX graph-feature computation | [docs/parts/part_c_features.md](docs/parts/part_c_features.md) |
| D | Solver wrappers for MCP algorithms | [docs/parts/part_d_solvers.md](docs/parts/part_d_solvers.md) |
| E | Runtime-consensus hardness labels | [docs/parts/part_e_hardness_labels.md](docs/parts/part_e_hardness_labels.md) |
| F | ML hardness classification | [docs/parts/part_f_hardness_classification.md](docs/parts/part_f_hardness_classification.md) |
| G | FP-Growth association-rule interpretation | [docs/parts/part_g_association_rules.md](docs/parts/part_g_association_rules.md) |
| H | Runtime prediction from graph features | [docs/parts/part_h_runtime_prediction.md](docs/parts/part_h_runtime_prediction.md) |

For the full documentation map, see [docs/index.md](docs/index.md).

---

## Datasets

The current experiments use:

- **TWITTER** ego-network graphs from SNAP.
- **COLLAB** graph instances from TU-format graph datasets.
- **IMDB-BINARY** graph instances from TU-format graph datasets.

Generated data, solver logs, models, plots, and reports are written under `artifacts/` and are not committed to GitHub.

---

## Installation

Create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
