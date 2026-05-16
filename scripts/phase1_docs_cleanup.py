from pathlib import Path
import shutil
import subprocess

DOCS = Path("docs")
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


def git_mv_if_exists(src, dst):
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists():
        return
    if dst_path.exists():
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", str(src_path), str(dst_path)], check=True)


for file_name in root_files_to_archive:
    git_mv_if_exists(file_name, ARCHIVE / file_name)

for file_name in docs_files_to_archive:
    git_mv_if_exists(file_name, ARCHIVE / Path(file_name).name)


def write_file(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


readme_lines = [
    "# GCO-HPIF",
    "",
    "**GCO-HPIF** stands for **Graph-Based Combinatorial Optimization — Hardness Prediction and Interpretation Framework**.",
    "",
    "This repository provides a modular research pipeline for studying graph-instance hardness for the **Maximum Clique Problem (MCP)**. It combines graph-feature extraction, solver runs, runtime-consensus hardness labels, machine-learning hardness classification, association-rule interpretation, and runtime prediction.",
    "",
    "---",
    "",
    "## Pipeline at a glance",
    "",
    "| Part | Stage | Detailed docs |",
    "|---|---|---|",
    "| A | Raw graph ingestion and graph manifests | [docs/parts/part_a_raw_graphs.md](docs/parts/part_a_raw_graphs.md) |",
    "| B | Fixed TWITTER train/validation/test split | [docs/parts/part_b_twitter_split.md](docs/parts/part_b_twitter_split.md) |",
    "| C | NetworkX graph-feature computation | [docs/parts/part_c_features.md](docs/parts/part_c_features.md) |",
    "| D | Solver wrappers for MCP algorithms | [docs/parts/part_d_solvers.md](docs/parts/part_d_solvers.md) |",
    "| E | Runtime-consensus hardness labels | [docs/parts/part_e_hardness_labels.md](docs/parts/part_e_hardness_labels.md) |",
    "| F | ML hardness classification | [docs/parts/part_f_hardness_classification.md](docs/parts/part_f_hardness_classification.md) |",
    "| G | FP-Growth association-rule interpretation | [docs/parts/part_g_association_rules.md](docs/parts/part_g_association_rules.md) |",
    "| H | Runtime prediction from graph features | [docs/parts/part_h_runtime_prediction.md](docs/parts/part_h_runtime_prediction.md) |",
    "",
    "For the full documentation map, see [docs/index.md](docs/index.md).",
    "",
    "---",
    "",
    "## Datasets",
    "",
    "The current experiments use:",
    "",
    "- **TWITTER** ego-network graphs from SNAP.",
    "- **COLLAB** graph instances from TU-format graph datasets.",
    "- **IMDB-BINARY** graph instances from TU-format graph datasets.",
    "",
    "Generated data, solver logs, models, plots, and reports are written under `artifacts/` and are not committed to GitHub.",
    "",
    "---",
    "",
    "## Installation",
    "",
    "Create and activate a virtual environment:",
    "",
    "```bash",
    "python3.10 -m venv .venv",
    "source .venv/bin/activate",
    "```",
    "",
    "Install dependencies:",
    "",
    "```bash",
    "pip install -U pip",
    "pip install -r requirements_part_f.txt",
    "pip install -r requirements_part_g.txt",
    "```",
    "",
    "After packaging cleanup is complete, the preferred editable install will be:",
    "",
    "```bash",
    "pip install -e \".[dev]\"",
    "```",
    "",
    "Run tests:",
    "",
    "```bash",
    "pytest -q",
    "```",
    "",
    "---",
    "",
    "## Quick start",
    "",
    "Run a small raw-graph smoke test:",
    "",
    "```bash",
    "python -m gco_hpif.cli.prepare_raw_graphs \\",
    "  --output-root artifacts/slice_a_smoke \\",
    "  --datasets imdb_binary collab twitter \\",
    "  --limit 3",
    "```",
    "",
    "Compute graph features for the smoke graphs:",
    "",
    "```bash",
    "python -m gco_hpif.cli.compute_graph_features \\",
    "  --graphs-index artifacts/slice_a_smoke/manifests/graphs_index.csv \\",
    "  --interim-dir artifacts/slice_a_smoke/interim \\",
    "  --output-dir artifacts/features_smoke \\",
    "  --datasets twitter collab imdb_binary \\",
    "  --limit-per-dataset 3 \\",
    "  --timeout-seconds 60",
    "```",
    "",
    "See [docs/quickstart.md](docs/quickstart.md) and [docs/cli_reference.md](docs/cli_reference.md) for stage-specific commands.",
    "",
    "---",
    "",
    "## Main output locations",
    "",
    "Typical generated outputs include:",
    "",
    "```text",
    "artifacts/slice_a_full/",
    "artifacts/features_full/",
    "artifacts/solver_runs/",
    "artifacts/hardness_labels_consensus_runtime/",
    "artifacts/ml_hardness_part_f_feature_prefix_fixed/",
    "artifacts/association_rules_part_g/",
    "artifacts/runtime_prediction_part_h/",
    "```",
    "",
    "See [docs/outputs.md](docs/outputs.md) for details.",
    "",
    "---",
    "",
    "## External dependencies",
    "",
    "Some stages require external tools or licenses:",
    "",
    "- **Gurobi** requires `gurobipy` and a valid Gurobi license.",
    "- **CliSAT** and **MoMC** require solver binaries or source builds.",
    "- **EGN** and **HGS** are optional external integrations.",
    "- Raw datasets and large generated artifacts are not bundled in this repository.",
    "",
    "---",
    "",
    "## Development",
    "",
    "See [docs/development.md](docs/development.md).",
    "",
    "---",
    "",
    "## Citation",
    "",
    "If you use this repository in academic work, please cite the associated pre-print:",
    "",
    "> Sharman, Bharat and Hassini, Elkafi, *A General Framework for Predicting and Explaining the Hardness of Graph-Based Combinatorial Optimization Problems using Machine Learning and Association Rule Mining*. Available at SSRN: https://ssrn.com/abstract=5975002 or http://dx.doi.org/10.2139/ssrn.5975002",
    "",
    "A `CITATION.cff` file is planned as part of the packaging cleanup.",
    "",
    "---",
    "",
    "## License",
    "",
    "This repository is released under the MIT License.",
]

write_file("README.md", readme_lines)

docs = {
    "docs/index.md": [
        "# GCO-HPIF documentation",
        "",
        "This documentation expands the short front-page `README.md`.",
        "",
        "## Start here",
        "",
        "- [Quickstart](quickstart.md)",
        "- [Pipeline overview](pipeline_overview.md)",
        "- [CLI reference](cli_reference.md)",
        "- [Outputs](outputs.md)",
        "- [Development](development.md)",
        "",
        "## Pipeline parts",
        "",
        "- [Part A — Raw graph ingestion](parts/part_a_raw_graphs.md)",
        "- [Part B — TWITTER split](parts/part_b_twitter_split.md)",
        "- [Part C — Graph features](parts/part_c_features.md)",
        "- [Part D — Solver wrappers](parts/part_d_solvers.md)",
        "- [Part E — Hardness labels](parts/part_e_hardness_labels.md)",
        "- [Part F — Hardness classification](parts/part_f_hardness_classification.md)",
        "- [Part G — Association rules](parts/part_g_association_rules.md)",
        "- [Part H — Runtime prediction](parts/part_h_runtime_prediction.md)",
    ],
    "docs/quickstart.md": [
        "# Quickstart",
        "",
        "Install dependencies:",
        "",
        "```bash",
        "python3.10 -m venv .venv",
        "source .venv/bin/activate",
        "pip install -U pip",
        "pip install -r requirements_part_f.txt",
        "pip install -r requirements_part_g.txt",
        "```",
        "",
        "Run a small raw-graph smoke test:",
        "",
        "```bash",
        "python -m gco_hpif.cli.prepare_raw_graphs \\",
        "  --output-root artifacts/slice_a_smoke \\",
        "  --datasets imdb_binary collab twitter \\",
        "  --limit 3",
        "```",
        "",
        "Compute graph features:",
        "",
        "```bash",
        "python -m gco_hpif.cli.compute_graph_features \\",
        "  --graphs-index artifacts/slice_a_smoke/manifests/graphs_index.csv \\",
        "  --interim-dir artifacts/slice_a_smoke/interim \\",
        "  --output-dir artifacts/features_smoke \\",
        "  --datasets twitter collab imdb_binary \\",
        "  --limit-per-dataset 3 \\",
        "  --timeout-seconds 60",
        "```",
    ],
    "docs/pipeline_overview.md": [
        "# Pipeline overview",
        "",
        "| Part | Stage | Main output |",
        "|---|---|---|",
        "| A | Raw graph ingestion | `artifacts/slice_a_full/` |",
        "| B | TWITTER split | `data/manifests/twitter_split_60_20_20.csv` |",
        "| C | Graph features | `artifacts/features_full/graph_features.csv` |",
        "| D | Solver wrappers | `artifacts/solver_runs/*/solver_runs.csv` |",
        "| E | Hardness labels | `artifacts/hardness_labels_consensus_runtime/hardness_labels.csv` |",
        "| F | Hardness classification | `artifacts/ml_hardness_part_f_feature_prefix_fixed/` |",
        "| G | Association rules | `artifacts/association_rules_part_g/` |",
        "| H | Runtime prediction | `artifacts/runtime_prediction_part_h/` |",
    ],
    "docs/outputs.md": [
        "# Outputs",
        "",
        "Generated outputs are written mainly under `artifacts/`.",
        "",
        "```text",
        "artifacts/slice_a_full/",
        "artifacts/features_full/",
        "artifacts/solver_runs/",
        "artifacts/hardness_labels_consensus_runtime/",
        "artifacts/ml_hardness_part_f_feature_prefix_fixed/",
        "artifacts/association_rules_part_g/",
        "artifacts/runtime_prediction_part_h/",
        "```",
        "",
        "Do not commit large generated artifacts.",
    ],
    "docs/development.md": [
        "# Development workflow",
        "",
        "```bash",
        "git checkout main",
        "git pull origin main",
        "git checkout -b descriptive-branch-name",
        "pytest -q",
        "git status",
        "```",
        "",
        "The front `README.md` should remain short. Put detailed commands and explanations in `docs/`, especially under `docs/parts/`.",
    ],
    "docs/cli_reference.md": [
        "# CLI reference",
        "",
        "This page lists the main stage-specific CLIs. A unified end-to-end `run_pipeline` CLI is planned for the next cleanup phase.",
        "",
        "See the part-specific docs under `docs/parts/` for commands.",
    ],
    "docs/parts/part_a_raw_graphs.md": [
        "# Part A — Raw graph ingestion",
        "",
        "Raw graph ingestion and graph manifests.",
    ],
    "docs/parts/part_b_twitter_split.md": [
        "# Part B — TWITTER split manifest",
        "",
        "Fixed 60/20/20 TWITTER split.",
    ],
    "docs/parts/part_c_features.md": [
        "# Part C — NetworkX graph features",
        "",
        "Graph-feature computation from normalized graph inputs.",
    ],
    "docs/parts/part_d_solvers.md": [
        "# Part D — Solver wrappers",
        "",
        "Solver wrappers for Gurobi, CliSAT, MoMC, EGN, and HGS.",
    ],
    "docs/parts/part_e_hardness_labels.md": [
        "# Part E — Runtime-consensus hardness labels",
        "",
        "Runtime-consensus hardness label construction.",
    ],
    "docs/parts/part_f_hardness_classification.md": [
        "# Part F — ML hardness classification",
        "",
        "Machine-learning hardness classification from graph features.",
    ],
    "docs/parts/part_g_association_rules.md": [
        "# Part G — Association-rule mining",
        "",
        "FP-Growth association-rule mining for hardness interpretation.",
    ],
    "docs/parts/part_h_runtime_prediction.md": [
        "# Part H — Runtime prediction",
        "",
        "Runtime prediction from graph features and solver runtime outputs.",
    ],
}

for path, lines in docs.items():
    write_file(path, lines)

print("Phase 1 docs cleanup files written.")
print("Next: run git status, inspect README.md and docs/, then commit.")
