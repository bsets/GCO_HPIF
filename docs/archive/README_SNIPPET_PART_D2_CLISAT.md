## Suggested README.md insertion for Part D.2

Add the following after the existing Part D.1 Gurobi section in the main repository `README.md`.

```markdown
## Part D.2: CliSAT maximum-clique solver wrapper

Part D.2 adds a CliSAT-based exact maximum-clique solver wrapper.

The wrapper runs only on graph instances whose features were successfully computed in Part C. It converts each NetworkX graph to DIMACS `.clq` format and invokes the bundled CliSAT executable as:

    external/CliSAT/bin/CliSAT <graph_path.clq> <time_limit_seconds> <threads>

The bundled executable is stored at:

    external/CliSAT/bin/CliSAT

CliSAT is third-party software originally published at https://github.com/psanse/CliSAT and released under the Unlicense / public-domain dedication. The license and attribution notice are included under `external/CliSAT/`.

Run a CliSAT smoke test:

    python -m gco_hpif.cli.run_clisat_solver \
      --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
      --features artifacts/features_full/graph_features.csv \
      --interim-dir artifacts/slice_a_full/interim \
      --output-dir artifacts/solver_runs/clisat_smoke \
      --datasets twitter collab imdb_binary \
      --limit-per-dataset 2 \
      --time-limit-seconds 30 \
      --threads 1

Run the full CliSAT solver step:

    python -m gco_hpif.cli.run_clisat_solver \
      --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
      --features artifacts/features_full/graph_features.csv \
      --interim-dir artifacts/slice_a_full/interim \
      --output-dir artifacts/solver_runs/clisat_full \
      --datasets twitter collab imdb_binary \
      --time-limit-seconds 1800 \
      --threads 1

Part D.2 creates:

    artifacts/solver_runs/clisat_full/
    ├── solver_runs.csv
    ├── solver_errors.csv
    ├── run_summary.csv
    ├── dimacs/
    ├── raw_clisat_outputs/
    ├── twitter_dataset_CliSAT_results.txt
    ├── collab_dataset_CliSAT_results.txt
    └── imdb_binary_dataset_CliSAT_results.txt

Generated solver outputs are not committed to GitHub.
```
