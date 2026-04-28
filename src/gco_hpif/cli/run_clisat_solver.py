"""CLI for running the CliSAT maximum-clique wrapper on Part C-valid graphs."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from tqdm.auto import tqdm

from gco_hpif.solvers.clisat_solver import (
    DEFAULT_CLISAT_EXECUTABLE,
    append_notebook_style_clisat_output,
    solve_clisat_max_clique,
    write_raw_clisat_output,
)
from gco_hpif.solvers.common import (
    iter_candidate_graphs,
    json_dumps_compact,
    normalize_dataset_names,
    write_solver_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CliSAT maximum-clique solver on graphs that passed Part C."
    )
    parser.add_argument("--graphs-index", required=True, help="Path to Part A graphs_index.csv.")
    parser.add_argument(
        "--features",
        required=True,
        help="Path to Part C graph_features.csv. Only graphs in this file are solved.",
    )
    parser.add_argument(
        "--interim-dir",
        required=True,
        help="Path to Part A interim directory containing *_graphs.pkl.gz files.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for CliSAT solver logs.")
    parser.add_argument(
        "--clisat-executable",
        default=str(DEFAULT_CLISAT_EXECUTABLE),
        help=(
            "Path to the compiled CliSAT executable. Defaults to the bundled "
            "Linux binary at external/CliSAT/bin/CliSAT."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["twitter", "collab", "imdb_binary"],
        help="Datasets to solve: twitter collab imdb_binary.",
    )
    parser.add_argument("--limit-per-dataset", type=int, default=None, help="Optional smoke-test limit per dataset.")
    parser.add_argument("--time-limit-seconds", type=float, default=1800, help="Per-instance CliSAT time limit.")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help=(
            "Final integer argument passed to CliSAT. The original notebook used 1. "
            "It is stored in the common solver-output 'threads' column for alignment."
        ),
    )
    parser.add_argument(
        "--clisat-output-node-base",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Node-index base used by CliSAT stdout. The bundled binary reports 0-based "
            "clique nodes, even though DIMACS edges are written 1-based. Use 1 only "
            "for a different CliSAT build that reports 1-based nodes."
        ),
    )
    parser.add_argument(
        "--python-timeout-buffer-seconds",
        type=float,
        default=60.0,
        help="Extra Python subprocess timeout buffer added to --time-limit-seconds.",
    )
    parser.add_argument(
        "--no-raw-output-files",
        action="store_true",
        help="Do not write per-graph raw CliSAT stdout/stderr files or notebook-style raw text files.",
    )
    return parser.parse_args(argv)


def make_error_row(instance, args: argparse.Namespace, elapsed_seconds: float, error: Exception) -> dict:
    """Create a standardized error row for one graph instance."""
    return {
        "dataset": instance.dataset,
        "graph_id": instance.graph_id,
        "source_index": instance.source_index,
        "solver_name": "clisat",
        "run_type": "solve",
        "time_limit_seconds": args.time_limit_seconds,
        "status": "error",
        "optimality_status": "error",
        "runtime_seconds": elapsed_seconds,
        "best_clique_size": None,
        "best_clique_nodes": json_dumps_compact([]),
        "clique_valid": False,
        "num_nodes": instance.graph.number_of_nodes(),
        "num_edges": instance.graph.number_of_edges(),
        "seed": None,
        "threads": args.threads,
        "mip_gap": None,
        "objective_bound": None,
        "error_message": f"{type(error).__name__}: {error}",
    }


def _initialise_notebook_style_files(output_dir: Path, datasets: list[str]) -> dict[str, Path]:
    paths = {dataset: output_dir / f"{dataset}_dataset_CliSAT_results.txt" for dataset in datasets}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    dimacs_dir = output_dir / "dimacs"
    raw_output_dir = output_dir / "raw_clisat_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_datasets = normalize_dataset_names(args.datasets)
    notebook_style_files = {}
    if not args.no_raw_output_files:
        notebook_style_files = _initialise_notebook_style_files(output_dir, selected_datasets)

    candidate_instances = list(
        iter_candidate_graphs(
            graphs_index_csv=args.graphs_index,
            features_csv=args.features,
            interim_dir=args.interim_dir,
            datasets=selected_datasets,
            limit_per_dataset=args.limit_per_dataset,
        )
    )

    rows: list[dict] = []
    python_timeout_seconds = args.time_limit_seconds + args.python_timeout_buffer_seconds

    for instance in tqdm(candidate_instances, desc="CliSAT graphs", unit="graph"):
        start = time.perf_counter()
        dimacs_path = dimacs_dir / instance.dataset / f"{instance.graph_id}.clq"
        try:
            result = solve_clisat_max_clique(
                instance.graph,
                clisat_executable=args.clisat_executable,
                dimacs_path=dimacs_path,
                time_limit_seconds=args.time_limit_seconds,
                threads=args.threads,
                python_timeout_seconds=python_timeout_seconds,
                clisat_output_node_base=args.clisat_output_node_base,
            )

            if not args.no_raw_output_files:
                write_raw_clisat_output(
                    output_dir=raw_output_dir / instance.dataset,
                    graph_id=instance.graph_id,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
                append_notebook_style_clisat_output(
                    output_file=notebook_style_files[instance.dataset],
                    graph_name=instance.graph_id,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )

            row = {
                "dataset": instance.dataset,
                "graph_id": instance.graph_id,
                "source_index": instance.source_index,
                "solver_name": "clisat",
                "run_type": "solve",
                "time_limit_seconds": args.time_limit_seconds,
                "status": result.status,
                "optimality_status": result.optimality_status,
                "runtime_seconds": result.runtime_seconds,
                "best_clique_size": result.best_clique_size,
                "best_clique_nodes": json_dumps_compact(result.best_clique_nodes),
                "clique_valid": result.clique_valid,
                "num_nodes": instance.graph.number_of_nodes(),
                "num_edges": instance.graph.number_of_edges(),
                "seed": None,
                "threads": args.threads,
                "mip_gap": None,
                "objective_bound": None,
                "error_message": result.error_message,
            }
        except Exception as error:
            elapsed = time.perf_counter() - start
            row = make_error_row(instance, args, elapsed, error)

        rows.append(row)

    runs_file, errors_file, summary_file = write_solver_outputs(rows, output_dir, solver_name="clisat")

    print("\nPart D.2 CliSAT run complete.")
    print(f"CliSAT executable: {args.clisat_executable}")
    print(f"Candidate graphs: {len(candidate_instances)}")
    print(f"Solver runs: {runs_file}")
    print(f"Solver errors: {errors_file}")
    print(f"Run summary: {summary_file}")
    print(f"DIMACS files: {dimacs_dir}")
    if not args.no_raw_output_files:
        print(f"Raw CliSAT stdout/stderr: {raw_output_dir}")
        print("Notebook-style raw text files:")
        for path in notebook_style_files.values():
            print(f"  {path}")


if __name__ == "__main__":
    main()
