from __future__ import annotations

import gzip
import math
import pickle
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from gco_hpif.utils.graph_utils import DATASET_ORDER


DEFAULT_FEATURE_TIMEOUT_SECONDS = 60


FEATURE_COLUMNS: list[str] = [
    "feature_Number_of_Nodes",
    "feature_Number_of_Edges",
    "feature_Density",
    "feature_Radius",
    "feature_Diameter",
    "feature_Median_Degree_Centrality",
    "feature_Median_Betweenness_Centrality",
    "feature_Median_Closeness_Centrality",
    "feature_Global_Clustering_Coefficient",
    "feature_Median_Node_Eccentricity",
    "feature_Algebraic_Connectivity",
    "feature_Median_Neighbour_Degree",
    "feature_Spectral_Radius",
    "feature_Laplacian_Spectral_Radius",
    "feature_Median_Geodesic_Distance",
    "feature_Smallest_NonZero_Eigenvalue_Laplacian",
    "feature_Second_Smallest_NonZero_Eigenvalue_Laplacian",
    "feature_Second_Largest_Eigenvalue_Laplacian",
    "feature_Smallest_Eigenvalue_Adjacency",
    "feature_Second_Smallest_Eigenvalue_Adjacency",
    "feature_Second_Largest_Eigenvalue_Adjacency",
    "feature_Gap_Largest_Second_Largest_Adjacency",
    "feature_Gap_Largest_Smallest_Laplacian",
]




FAILURE_COLUMNS: list[str] = [
    "dataset",
    "graph_id",
    "source_index",
    "failure_reason",
    "exception_type",
    "error_message",
    "timeout_seconds",
    "elapsed_seconds",
]


FEATURE_DEFINITIONS: dict[str, str] = {
    "feature_Number_of_Nodes": "Number of nodes in the simple undirected graph.",
    "feature_Number_of_Edges": "Number of edges in the simple undirected graph.",
    "feature_Density": "NetworkX graph density.",
    "feature_Radius": "Graph radius for connected graphs; NaN for disconnected graphs.",
    "feature_Diameter": "Graph diameter for connected graphs; NaN for disconnected graphs.",
    "feature_Median_Degree_Centrality": "Median node degree centrality.",
    "feature_Median_Betweenness_Centrality": "Median node betweenness centrality.",
    "feature_Median_Closeness_Centrality": "Median node closeness centrality.",
    "feature_Global_Clustering_Coefficient": "Global clustering coefficient / transitivity.",
    "feature_Median_Node_Eccentricity": "Median node eccentricity for connected graphs; NaN for disconnected graphs.",
    "feature_Algebraic_Connectivity": "Second-smallest Laplacian eigenvalue computed from a dense NumPy Laplacian matrix; zero for disconnected graphs up to numerical tolerance.",
    "feature_Median_Neighbour_Degree": "Median across nodes of the median degree of each node's neighbours.",
    "feature_Spectral_Radius": "Maximum absolute eigenvalue of the adjacency matrix.",
    "feature_Laplacian_Spectral_Radius": "Largest eigenvalue of the Laplacian matrix.",
    "feature_Median_Geodesic_Distance": "Median shortest-path distance over unordered node pairs for connected graphs; NaN for disconnected graphs.",
    "feature_Smallest_NonZero_Eigenvalue_Laplacian": "Smallest strictly positive Laplacian eigenvalue using tolerance 1e-10.",
    "feature_Second_Smallest_NonZero_Eigenvalue_Laplacian": "Second smallest strictly positive Laplacian eigenvalue using tolerance 1e-10.",
    "feature_Second_Largest_Eigenvalue_Laplacian": "Second largest Laplacian eigenvalue after sorting eigenvalues in ascending order.",
    "feature_Smallest_Eigenvalue_Adjacency": "Algebraically smallest eigenvalue of the adjacency matrix, i.e., the first value after ascending sort.",
    "feature_Second_Smallest_Eigenvalue_Adjacency": "Algebraically second smallest eigenvalue of the adjacency matrix, i.e., the second value after ascending sort.",
    "feature_Second_Largest_Eigenvalue_Adjacency": "Second largest adjacency eigenvalue after sorting eigenvalues in ascending order.",
    "feature_Gap_Largest_Second_Largest_Adjacency": "Difference between the largest and second largest adjacency eigenvalues.",
    "feature_Gap_Largest_Smallest_Laplacian": "Difference between the largest and smallest Laplacian eigenvalues.",
}

@dataclass(frozen=True)
class FeatureComputationResult:
    features_path: Path
    failures_path: Path
    summary_path: Path
    feature_manifest_path: Path


class FeatureComputationTimeout(TimeoutError):
    """Raised when one graph exceeds the configured feature-computation time limit."""


def safe_median(values: Iterable[Any]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def is_connected_nontrivial(G: nx.Graph) -> bool:
    return G.number_of_nodes() > 0 and nx.is_connected(G)


def adjacency_eigenvalues(G: nx.Graph) -> np.ndarray:
    A = nx.to_numpy_array(G, dtype=float)
    if A.size == 0:
        return np.array([], dtype=float)
    return np.linalg.eigvalsh(A)


def laplacian_eigenvalues(G: nx.Graph) -> np.ndarray:
    """Return Laplacian eigenvalues without requiring SciPy.

    NetworkX's ``laplacian_matrix`` imports SciPy. To keep this repository
    lightweight and reproducible in a minimal environment, we build the dense
    Laplacian matrix directly from the NumPy adjacency matrix:

        L = D - A

    The project datasets are treated as simple undirected, unweighted graphs
    after Part A normalization, so this dense construction matches the intended
    feature definitions.
    """
    if G.number_of_nodes() == 0:
        return np.array([], dtype=float)

    A = nx.to_numpy_array(G, dtype=float, nodelist=list(G.nodes()))
    if A.size == 0:
        return np.array([], dtype=float)

    degrees = A.sum(axis=1)
    L = np.diag(degrees) - A
    return np.linalg.eigvalsh(L)


def unordered_geodesic_distances(G: nx.Graph) -> list[int]:
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return []

    dists: list[int] = []
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            if v in lengths[u]:
                dists.append(lengths[u][v])
    return dists


def median_of_median_neighbor_degrees(G: nx.Graph) -> float:
    deg = dict(G.degree())
    per_node_neighbor_degree_medians: list[float] = []

    for node in G.nodes():
        neighbor_degrees = [deg[nbr] for nbr in G.neighbors(node)]
        if len(neighbor_degrees) == 0:
            per_node_neighbor_degree_medians.append(float("nan"))
        else:
            per_node_neighbor_degree_medians.append(float(np.median(neighbor_degrees)))

    return safe_median(per_node_neighbor_degree_medians)


# ---------------------------------------------------------------------------
# Individual feature functions.
# ---------------------------------------------------------------------------

def feature_number_of_nodes(G: nx.Graph) -> float:
    return float(G.number_of_nodes())


def feature_number_of_edges(G: nx.Graph) -> float:
    return float(G.number_of_edges())


def feature_density(G: nx.Graph) -> float:
    return float(nx.density(G))


def feature_radius(G: nx.Graph) -> float:
    if is_connected_nontrivial(G):
        return float(nx.radius(G))
    return float("nan")


def feature_diameter(G: nx.Graph) -> float:
    if is_connected_nontrivial(G):
        return float(nx.diameter(G))
    return float("nan")


def feature_median_degree_centrality(G: nx.Graph) -> float:
    return safe_median(nx.degree_centrality(G).values())


def feature_median_betweenness_centrality(G: nx.Graph) -> float:
    return safe_median(nx.betweenness_centrality(G).values())


def feature_median_closeness_centrality(G: nx.Graph) -> float:
    return safe_median(nx.closeness_centrality(G).values())


def feature_global_clustering_coefficient(G: nx.Graph) -> float:
    return float(nx.transitivity(G))


def feature_median_node_eccentricity(G: nx.Graph) -> float:
    if is_connected_nontrivial(G):
        return safe_median(nx.eccentricity(G).values())
    return float("nan")


def feature_algebraic_connectivity(G: nx.Graph) -> float:
    """Return algebraic connectivity without requiring SciPy.

    Algebraic connectivity is the second-smallest Laplacian eigenvalue.
    For disconnected graphs, this value is zero up to numerical tolerance.
    """
    eigs = np.sort(laplacian_eigenvalues(G))
    if eigs.size < 2:
        return float("nan")
    value = float(eigs[1])
    if abs(value) < 1e-10:
        return 0.0
    return value


def feature_median_neighbour_degree(G: nx.Graph) -> float:
    return median_of_median_neighbor_degrees(G)


def feature_spectral_radius(G: nx.Graph) -> float:
    eigs = adjacency_eigenvalues(G)
    if eigs.size == 0:
        return float("nan")
    return float(np.max(np.abs(eigs)))


def feature_laplacian_spectral_radius(G: nx.Graph) -> float:
    eigs = laplacian_eigenvalues(G)
    if eigs.size == 0:
        return float("nan")
    return float(np.max(eigs))


def feature_median_geodesic_distance(G: nx.Graph) -> float:
    if is_connected_nontrivial(G):
        return safe_median(unordered_geodesic_distances(G))
    return float("nan")


def feature_smallest_nonzero_eigenvalue_laplacian(G: nx.Graph) -> float:
    eigs = np.sort(laplacian_eigenvalues(G))
    nz = eigs[eigs > 1e-10]
    if nz.size == 0:
        return float("nan")
    return float(nz[0])


def feature_second_smallest_nonzero_eigenvalue_laplacian(G: nx.Graph) -> float:
    eigs = np.sort(laplacian_eigenvalues(G))
    nz = eigs[eigs > 1e-10]
    if nz.size < 2:
        return float("nan")
    return float(nz[1])


def feature_second_largest_eigenvalue_laplacian(G: nx.Graph) -> float:
    eigs = np.sort(laplacian_eigenvalues(G))
    if eigs.size < 2:
        return float("nan")
    return float(eigs[-2])


def feature_smallest_eigenvalue_adjacency(G: nx.Graph) -> float:
    eigs = np.sort(adjacency_eigenvalues(G))
    if eigs.size == 0:
        return float("nan")
    return float(eigs[0])


def feature_second_smallest_eigenvalue_adjacency(G: nx.Graph) -> float:
    eigs = np.sort(adjacency_eigenvalues(G))
    if eigs.size < 2:
        return float("nan")
    return float(eigs[1])


def feature_second_largest_eigenvalue_adjacency(G: nx.Graph) -> float:
    eigs = np.sort(adjacency_eigenvalues(G))
    if eigs.size < 2:
        return float("nan")
    return float(eigs[-2])


def feature_gap_largest_second_largest_adjacency(G: nx.Graph) -> float:
    eigs = np.sort(adjacency_eigenvalues(G))
    if eigs.size < 2:
        return float("nan")
    return float(eigs[-1] - eigs[-2])


def feature_gap_largest_smallest_laplacian(G: nx.Graph) -> float:
    eigs = np.sort(laplacian_eigenvalues(G))
    if eigs.size == 0:
        return float("nan")
    return float(eigs[-1] - eigs[0])


FEATURE_FUNCTIONS: list[tuple[str, Callable[[nx.Graph], float]]] = [
    ("feature_Number_of_Nodes", feature_number_of_nodes),
    ("feature_Number_of_Edges", feature_number_of_edges),
    ("feature_Density", feature_density),
    ("feature_Radius", feature_radius),
    ("feature_Diameter", feature_diameter),
    ("feature_Median_Degree_Centrality", feature_median_degree_centrality),
    ("feature_Median_Betweenness_Centrality", feature_median_betweenness_centrality),
    ("feature_Median_Closeness_Centrality", feature_median_closeness_centrality),
    ("feature_Global_Clustering_Coefficient", feature_global_clustering_coefficient),
    ("feature_Median_Node_Eccentricity", feature_median_node_eccentricity),
    ("feature_Algebraic_Connectivity", feature_algebraic_connectivity),
    ("feature_Median_Neighbour_Degree", feature_median_neighbour_degree),
    ("feature_Spectral_Radius", feature_spectral_radius),
    ("feature_Laplacian_Spectral_Radius", feature_laplacian_spectral_radius),
    ("feature_Median_Geodesic_Distance", feature_median_geodesic_distance),
    ("feature_Smallest_NonZero_Eigenvalue_Laplacian", feature_smallest_nonzero_eigenvalue_laplacian),
    ("feature_Second_Smallest_NonZero_Eigenvalue_Laplacian", feature_second_smallest_nonzero_eigenvalue_laplacian),
    ("feature_Second_Largest_Eigenvalue_Laplacian", feature_second_largest_eigenvalue_laplacian),
    ("feature_Smallest_Eigenvalue_Adjacency", feature_smallest_eigenvalue_adjacency),
    ("feature_Second_Smallest_Eigenvalue_Adjacency", feature_second_smallest_eigenvalue_adjacency),
    ("feature_Second_Largest_Eigenvalue_Adjacency", feature_second_largest_eigenvalue_adjacency),
    ("feature_Gap_Largest_Second_Largest_Adjacency", feature_gap_largest_second_largest_adjacency),
    ("feature_Gap_Largest_Smallest_Laplacian", feature_gap_largest_smallest_laplacian),
]


@contextmanager
def graph_time_limit(timeout_seconds: int | float | None):
    if timeout_seconds is None or timeout_seconds <= 0:
        yield
        return

    old_handler = signal.getsignal(signal.SIGALRM)

    def _handler(signum, frame):  # pragma: no cover - depends on OS signal timing
        raise FeatureComputationTimeout(f"Graph feature computation exceeded {timeout_seconds} seconds")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def compute_feature_row(G: nx.Graph, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute the 23 NetworkX features for one graph.

    This function does not apply a timeout by itself. Use
    ``compute_feature_row_with_timeout`` or the CLI for timeout-controlled runs.
    """
    row: dict[str, Any] = dict(metadata or {})
    graph_start = time.perf_counter()

    for feature_name, feature_func in FEATURE_FUNCTIONS:
        t0 = time.perf_counter()
        value = feature_func(G)
        t1 = time.perf_counter()
        row[feature_name] = value
        row[f"Time_{feature_name}"] = t1 - t0

    row["total_feature_time_seconds"] = time.perf_counter() - graph_start
    return row


def compute_feature_row_with_timeout(
    G: nx.Graph,
    metadata: dict[str, Any] | None = None,
    timeout_seconds: int | float = DEFAULT_FEATURE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    with graph_time_limit(timeout_seconds):
        return compute_feature_row(G, metadata=metadata)


def load_graph_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rb") as f:
        records = pickle.load(f)
    if not isinstance(records, list):
        raise TypeError(f"Expected a list of graph records in {path}; got {type(records)!r}")
    return records


def _default_interim_path(interim_dir: Path, dataset: str) -> Path:
    return interim_dir / f"{dataset}_graphs.pkl.gz"


def _metadata_from_graph_index(graphs_index_csv: Path) -> dict[str, dict[str, Any]]:
    df = pd.read_csv(graphs_index_csv)
    metadata: dict[str, dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        metadata[str(row["graph_id"])] = row
    return metadata


def _sort_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]):
        return (DATASET_ORDER.get(str(row.get("dataset")), 999), int(row.get("source_index", 10**12)))

    return sorted(rows, key=key)


def compute_features_from_part_a_artifacts(
    graphs_index_csv: Path,
    interim_dir: Path,
    output_dir: Path,
    datasets: list[str] | None = None,
    timeout_seconds: int = DEFAULT_FEATURE_TIMEOUT_SECONDS,
    limit_per_dataset: int | None = None,
    show_progress: bool = True,
) -> FeatureComputationResult:
    """Compute all graph features from the Part A interim graph pickle files.

    Outputs:
      - graph_features.csv: successful graph feature rows
      - feature_failures.csv: timeout/error rows
      - feature_timing_summary.csv: per-dataset timing summary
      - feature_column_manifest.csv: ordered list of feature columns
    """
    graphs_index_csv = Path(graphs_index_csv)
    interim_dir = Path(interim_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None or len(datasets) == 0:
        datasets = ["twitter", "collab", "imdb_binary"]

    graph_index_metadata = _metadata_from_graph_index(graphs_index_csv)

    feature_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    dataset_iter = tqdm(datasets, desc="Datasets", unit="dataset", disable=not show_progress)

    for dataset in dataset_iter:
        pickle_path = _default_interim_path(interim_dir, dataset)
        if not pickle_path.exists():
            raise FileNotFoundError(f"Missing Part A interim graph pickle for {dataset}: {pickle_path}")

        records = load_graph_records(pickle_path)
        if limit_per_dataset is not None:
            records = records[:limit_per_dataset]

        dataset_success_times: list[float] = []
        dataset_timeout_count = 0
        dataset_error_count = 0

        record_iter = tqdm(
            records,
            desc=f"{dataset}: features",
            unit="graph",
            leave=False,
            disable=not show_progress,
        )

        for record in record_iter:
            graph_id = str(record["graph_id"])
            G = record["graph"]
            metadata = dict(graph_index_metadata.get(graph_id, {}))
            if not metadata:
                metadata = {
                    "dataset": record.get("dataset", dataset),
                    "graph_id": graph_id,
                    "source_index": record.get("source_index"),
                }

            graph_start = time.perf_counter()
            try:
                row = compute_feature_row_with_timeout(G, metadata=metadata, timeout_seconds=timeout_seconds)
                feature_rows.append(row)
                dataset_success_times.append(float(row["total_feature_time_seconds"]))

            except FeatureComputationTimeout as exc:
                elapsed = time.perf_counter() - graph_start
                dataset_timeout_count += 1
                failure_rows.append({
                    "dataset": metadata.get("dataset", dataset),
                    "graph_id": graph_id,
                    "source_index": metadata.get("source_index"),
                    "failure_reason": "timeout",
                    "exception_type": type(exc).__name__,
                    "error_message": str(exc),
                    "timeout_seconds": timeout_seconds,
                    "elapsed_seconds": elapsed,
                })
                tqdm.write(f"Timeout > {timeout_seconds}s: {graph_id}")

            except Exception as exc:  # noqa: BLE001 - capture failures for reproducible manifest
                elapsed = time.perf_counter() - graph_start
                dataset_error_count += 1
                failure_rows.append({
                    "dataset": metadata.get("dataset", dataset),
                    "graph_id": graph_id,
                    "source_index": metadata.get("source_index"),
                    "failure_reason": "error",
                    "exception_type": type(exc).__name__,
                    "error_message": str(exc),
                    "timeout_seconds": timeout_seconds,
                    "elapsed_seconds": elapsed,
                })
                tqdm.write(f"Failed: {graph_id} | {type(exc).__name__}: {exc}")

        total_graphs = len(records)
        successful_graphs = len(dataset_success_times)
        summary_rows.append({
            "dataset": dataset,
            "total_graphs_attempted": total_graphs,
            "successful_graphs": successful_graphs,
            "timeout_failures": dataset_timeout_count,
            "other_failures": dataset_error_count,
            "timeout_seconds": timeout_seconds,
            "total_feature_time_seconds": float(np.sum(dataset_success_times)) if dataset_success_times else 0.0,
            "mean_feature_time_seconds": float(np.mean(dataset_success_times)) if dataset_success_times else math.nan,
            "median_feature_time_seconds": float(np.median(dataset_success_times)) if dataset_success_times else math.nan,
            "max_feature_time_seconds": float(np.max(dataset_success_times)) if dataset_success_times else math.nan,
            "min_feature_time_seconds": float(np.min(dataset_success_times)) if dataset_success_times else math.nan,
        })

    feature_rows = _sort_feature_rows(feature_rows)
    failure_rows = _sort_feature_rows(failure_rows)

    features_path = output_dir / "graph_features.csv"
    failures_path = output_dir / "feature_failures.csv"
    summary_path = output_dir / "feature_timing_summary.csv"
    feature_manifest_path = output_dir / "feature_column_manifest.csv"

    pd.DataFrame(feature_rows).to_csv(features_path, index=False)
    pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS).to_csv(failures_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["dataset_order"] = summary_df["dataset"].map(DATASET_ORDER)
        summary_df = summary_df.sort_values("dataset_order").drop(columns=["dataset_order"])
    summary_df.to_csv(summary_path, index=False)

    pd.DataFrame({
        "feature_order": list(range(1, len(FEATURE_COLUMNS) + 1)),
        "feature_column": FEATURE_COLUMNS,
        "feature_definition": [FEATURE_DEFINITIONS[col] for col in FEATURE_COLUMNS],
    }).to_csv(feature_manifest_path, index=False)

    return FeatureComputationResult(
        features_path=features_path,
        failures_path=failures_path,
        summary_path=summary_path,
        feature_manifest_path=feature_manifest_path,
    )
