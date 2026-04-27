"""CLI for running the Gurobi maximum-clique wrapper on Part C-valid graphs."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from tqdm.auto import tqdm

from gco_hpif.solvers.common import (
    iter_candidate_graphs,
    json_dumps_compact,
    write_solver_outputs,
)
from gco_hpif.solvers.gurobi_solver import solve_gurobi_max_clique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gurobi maximum-clique solver on graphs that passed Part C."
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
    parser.add_argument("--output-dir", required=True, help="Output directory for Gurobi solver logs.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["twitter", "collab", "imdb_binary"],
        help="Datasets to solve: twitter collab imdb_binary.",
    )
    parser.add_argument("--limit-per-dataset", type=int, default=None, help="Optional smoke-test limit per dataset.")
    parser.add_argument("--time-limit-seconds", type=float, default=1800, help="Per-instance Gurobi time limit.")
    parser.add_argument("--seed", type=int, default=2026, help="Gurobi seed.")
    parser.add_argument("--threads", type=int, default=None, help="Optional Gurobi thread count.")
    parser.add_argument("--output-flag", action="store_true", help="Print Gurobi optimizer output.")
    return parser.parse_args()


def make_error_row(instance, args, elapsed_seconds: float, error: Exception) -> dict:
    """Create a standardized error row for one graph instance."""
    return {
        "dataset": instance.dataset,
        "graph_id": instance.graph_id,
        "source_index": instance.source_index,
        "solver_name": "gurobi",
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
        "seed": args.seed,
        "threads": args.threads,
        "mip_gap": None,
        "objective_bound": None,
        "error_message": f"{type(error).__name__}: {error}",
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_instances = list(
        iter_candidate_graphs(
            graphs_index_csv=args.graphs_index,
            features_csv=args.features,
            interim_dir=args.interim_dir,
            datasets=args.datasets,
            limit_per_dataset=args.limit_per_dataset,
        )
    )

    rows: list[dict] = []

    for instance in tqdm(candidate_instances, desc="Gurobi graphs", unit="graph"):
        start = time.perf_counter()
        try:
            result = solve_gurobi_max_clique(
                instance.graph,
                time_limit_seconds=args.time_limit_seconds,
                seed=args.seed,
                threads=args.threads,
                output_flag=args.output_flag,
            )
            row = {
                "dataset": instance.dataset,
                "graph_id": instance.graph_id,
                "source_index": instance.source_index,
                "solver_name": "gurobi",
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
                "seed": args.seed,
                "threads": args.threads,
                "mip_gap": result.mip_gap,
                "objective_bound": result.objective_bound,
                "error_message": result.error_message,
            }
        except Exception as error:
            elapsed = time.perf_counter() - start
            row = make_error_row(instance, args, elapsed, error)

        rows.append(row)

    runs_file, errors_file, summary_file = write_solver_outputs(rows, output_dir, solver_name="gurobi")

    print("\nPart D Gurobi run complete.")
    print(f"Candidate graphs: {len(candidate_instances)}")
    print(f"Solver runs:      {runs_file}")
    print(f"Solver errors:    {errors_file}")
    print(f"Run summary:      {summary_file}")


if __name__ == "__main__":
    main()
