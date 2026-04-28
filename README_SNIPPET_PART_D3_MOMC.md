## README snippet for Part D.3

After applying and validating this patch, update the repository README as follows.

### Pipeline status

Change:

```markdown
* Part D.3: MOMC solver wrapper
```

or:

```markdown
- [ ] Part D.3: MOMC solver wrapper
```

to:

```markdown
- [x] Part D.3: MOMC solver wrapper
```

### Add this section after Part D.2 CliSAT

```markdown
## Part D.3: MoMC maximum-clique solver wrapper

Part D.3 adds a MoMC-based exact maximum-clique solver wrapper.

The wrapper compiles the bundled MoMC C source when needed and then runs each generated DIMACS graph instance using the same pattern as the original notebook workflow:

    gcc -O3 -DMOMC external/MOMC/src/MOMC2016_1800sec_timeout_aware.c -o external/MOMC/bin/MoMC
    external/MOMC/bin/MoMC <graph_path.clq>

The MoMC source is included under `external/MOMC/src/`. The source file was obtained from:

    https://home.mis.u-picardie.fr/~cli/MoMC2016.c

The source file includes a permissive license notice from Chu-Min Li and Hua Jiang. The license text is preserved in `external/MOMC/LICENSE`, and the attribution notice is preserved in `external/MOMC/NOTICE.md`.

Run a MoMC smoke test:

    python -m gco_hpif.cli.run_momc_solver \
      --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
      --features artifacts/features_full/graph_features.csv \
      --interim-dir artifacts/slice_a_full/interim \
      --output-dir artifacts/solver_runs/momc_smoke \
      --datasets twitter collab imdb_binary \
      --limit-per-dataset 2 \
      --time-limit-seconds 30 \
      --threads 1

Run the full MoMC solver step:

    python -m gco_hpif.cli.run_momc_solver \
      --graphs-index artifacts/slice_a_full/manifests/graphs_index.csv \
      --features artifacts/features_full/graph_features.csv \
      --interim-dir artifacts/slice_a_full/interim \
      --output-dir artifacts/solver_runs/momc_full \
      --datasets twitter collab imdb_binary \
      --time-limit-seconds 1800 \
      --threads 1

Part D.3 creates:

    artifacts/solver_runs/momc_full/
    ├── solver_runs.csv
    ├── solver_errors.csv
    ├── run_summary.csv
    ├── dimacs/
    ├── raw_momc_outputs/
    ├── twitter_dataset_MOMC_results.txt
    ├── collab_dataset_MOMC_results.txt
    └── imdb_binary_dataset_MOMC_results.txt

Generated solver outputs are not committed to GitHub.
```
