"""Common utilities for solver wrappers.

These utilities keep all solver wrappers aligned on the same graph-loading,
candidate-selection, clique-validation, and result-output conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import json
import pickle
from typing import Iterator, Sequence

import networkx as nx
import pandas as pd


DATASET_ORDER = {"twitter": 0, "collab": 1, "imdb_binary": 2}

DATASET_PICKLE_FILES = {
    "twitter": "twitter_graphs.pkl.gz",
    "collab": "collab_graphs.pkl.gz",
    "imdb_binary": "imdb_binary_graphs.pkl.gz",
}

SOLVER_RUN_COLUMNS = [
    "dataset",
    "graph_id",
    "source_index",
    "solver_name",
    "run_type",
    "time_limit_seconds",
    "status",
    "optimality_status",
    "runtime_seconds",
    "best_clique_size",
    "best_clique_nodes",
    "clique_valid",
    "num_nodes",
    "num_edges",
    "seed",
    "threads",
    "mip_gap",
    "objective_bound",
    "error_message",
]


@dataclass(frozen=True)
class GraphInstance:
    """One graph instance selected for a solver run."""

    dataset: str
    graph_id: str
    source_index: int
    graph: nx.Graph


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset names used across CLI arguments and CSV manifests."""
    cleaned = name.strip().lower().replace("-", "_")
    aliases = {
        "imdb_binary": "imdb_binary",
        "imdb": "imdb_binary",
        "collab": "collab",
        "twitter": "twitter",
    }
    if cleaned not in aliases:
        raise ValueError(f"Unsupported dataset name: {name!r}")
    return aliases[cleaned]


def normalize_dataset_names(names: Sequence[str] | None) -> list[str]:
    """Normalize a sequence of dataset names, preserving the canonical order."""
    if names is None or len(names) == 0:
        names = ["twitter", "collab", "imdb_binary"]
    normalized = [normalize_dataset_name(name) for name in names]
    return sorted(dict.fromkeys(normalized), key=lambda d: DATASET_ORDER.get(d, 999))


def load_pickle_graphs(pickle_path: str | Path) -> list:
    """Load one Part A compressed graph pickle file.

    Part A stores each graph as a small record dictionary such as::

        {
            "graph_id": "twitter_graph000001",
            "dataset": "twitter",
            "source_index": 1,
            "graph": <networkx.Graph>,
        }

    Older/local scripts may store either raw NetworkX graphs or dictionaries.
    This function deliberately returns the loaded records unchanged;
    :func:`extract_networkx_graph` unwraps them later so solver wrappers can
    support both formats.
    """
    path = Path(pickle_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph pickle file not found: {path}")

    with gzip.open(path, "rb") as f:
        records = pickle.load(f)

    if isinstance(records, dict):
        records = list(records.values())

    return list(records)


def extract_networkx_graph(graph_record, graph_id: str | None = None) -> nx.Graph:
    """Extract a NetworkX graph from a Part A graph record.

    Supported inputs:
    - a raw ``networkx.Graph``;
    - a dictionary containing a ``"graph"`` key whose value is a ``networkx.Graph``.

    The returned graph is copied and normalized to a simple undirected graph
    with self-loops removed. This prevents solver wrappers from mutating the
    cached Part A graph objects.
    """
    if isinstance(graph_record, nx.Graph):
        graph = graph_record
    elif isinstance(graph_record, dict) and isinstance(graph_record.get("graph"), nx.Graph):
        graph = graph_record["graph"]
    else:
        label = f" for {graph_id}" if graph_id else ""
        raise TypeError(
            f"Expected a NetworkX graph or Part A graph record{label}, "
            f"got {type(graph_record).__name__}"
        )

    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return nx.convert_node_labels_to_integers(graph, first_label=0, ordering="default")


def graph_pickle_path(interim_dir: str | Path, dataset: str) -> Path:
    """Return the Part A interim pickle path for one dataset."""
    dataset = normalize_dataset_name(dataset)
    return Path(interim_dir) / DATASET_PICKLE_FILES[dataset]


def load_graph_index(graphs_index_csv: str | Path) -> pd.DataFrame:
    """Load the Part A graph manifest."""
    path = Path(graphs_index_csv)
    if not path.exists():
        raise FileNotFoundError(f"Graph index file not found: {path}")

    df = pd.read_csv(path)
    required = {"dataset", "graph_id", "source_index", "n_nodes", "n_edges"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Graph index is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["dataset"] = df["dataset"].map(normalize_dataset_name)
    df["source_index"] = df["source_index"].astype(int)
    return df


def load_successful_feature_rows(features_csv: str | Path) -> pd.DataFrame:
    """Load Part C successful graph-feature rows."""
    path = Path(features_csv)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_csv(path)
    required = {"dataset", "graph_id", "source_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Feature file is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["dataset"] = df["dataset"].map(normalize_dataset_name)
    df["source_index"] = df["source_index"].astype(int)
    return df


def candidate_graph_metadata(
    graphs_index_csv: str | Path,
    features_csv: str | Path,
    datasets: Sequence[str] | None = None,
    limit_per_dataset: int | None = None,
) -> pd.DataFrame:
    """Return metadata for graphs that passed Part C feature computation.

    The solver candidate set is the intersection of:
    1. graphs present in the Part A graph index, and
    2. graphs present in Part C's successful `graph_features.csv`.
    """
    selected_datasets = normalize_dataset_names(datasets)
    graph_index = load_graph_index(graphs_index_csv)
    features = load_successful_feature_rows(features_csv)

    graph_index = graph_index[graph_index["dataset"].isin(selected_datasets)].copy()
    features = features[features["dataset"].isin(selected_datasets)].copy()

    successful_ids = features[["dataset", "graph_id", "source_index"]].drop_duplicates()

    merged = graph_index.merge(
        successful_ids,
        on=["dataset", "graph_id", "source_index"],
        how="inner",
        validate="one_to_one",
    )

    merged = merged.sort_values(
        by=["dataset", "source_index"],
        key=lambda s: s.map(DATASET_ORDER) if s.name == "dataset" else s,
    ).reset_index(drop=True)

    if limit_per_dataset is not None:
        if limit_per_dataset <= 0:
            raise ValueError("limit_per_dataset must be positive when provided.")
        merged = (
            merged.groupby("dataset", group_keys=False, sort=False)
            .head(limit_per_dataset)
            .reset_index(drop=True)
        )

    return merged


def iter_candidate_graphs(
    graphs_index_csv: str | Path,
    features_csv: str | Path,
    interim_dir: str | Path,
    datasets: Sequence[str] | None = None,
    limit_per_dataset: int | None = None,
) -> Iterator[GraphInstance]:
    """Yield graph instances that are eligible for solver runs."""
    metadata = candidate_graph_metadata(
        graphs_index_csv=graphs_index_csv,
        features_csv=features_csv,
        datasets=datasets,
        limit_per_dataset=limit_per_dataset,
    )

    graph_cache: dict[str, list[nx.Graph]] = {}

    for row in metadata.itertuples(index=False):
        dataset = row.dataset
        if dataset not in graph_cache:
            graph_cache[dataset] = load_pickle_graphs(graph_pickle_path(interim_dir, dataset))

        zero_based_index = int(row.source_index) - 1
        graphs = graph_cache[dataset]
        if zero_based_index < 0 or zero_based_index >= len(graphs):
            raise IndexError(
                f"source_index={row.source_index} is out of range for dataset={dataset}"
            )

        graph_record = graphs[zero_based_index]
        graph = extract_networkx_graph(graph_record, graph_id=row.graph_id)

        yield GraphInstance(
            dataset=dataset,
            graph_id=row.graph_id,
            source_index=int(row.source_index),
            graph=graph,
        )


def is_valid_clique(graph: nx.Graph, nodes: Sequence[int]) -> bool:
    """Return True if the selected nodes form a clique in `graph`."""
    if nodes is None:
        return False

    node_list = list(nodes)
    if len(set(node_list)) != len(node_list):
        return False

    if not set(node_list).issubset(set(graph.nodes())):
        return False

    for i, u in enumerate(node_list):
        for v in node_list[i + 1:]:
            if not graph.has_edge(u, v):
                return False

    return True


def json_dumps_compact(value) -> str:
    """Serialize list-like solver outputs compactly for CSV storage."""
    return json.dumps(value, separators=(",", ":"))


def write_solver_outputs(rows: Sequence[dict], output_dir: str | Path, solver_name: str) -> tuple[Path, Path, Path]:
    """Write solver run rows, error rows, and a compact run summary."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if rows:
        runs_df = pd.DataFrame(rows)
        for col in SOLVER_RUN_COLUMNS:
            if col not in runs_df.columns:
                runs_df[col] = None
        runs_df = runs_df[SOLVER_RUN_COLUMNS]
    else:
        runs_df = pd.DataFrame(columns=SOLVER_RUN_COLUMNS)

    runs_file = output_path / "solver_runs.csv"
    errors_file = output_path / "solver_errors.csv"
    summary_file = output_path / "run_summary.csv"

    runs_df.to_csv(runs_file, index=False)

    errors_df = runs_df[runs_df["status"].isin(["error", "skipped"])].copy()
    errors_df.to_csv(errors_file, index=False)

    if len(runs_df) > 0:
        summary_df = (
            runs_df.groupby(["dataset", "status"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["dataset", "status"])
        )
        summary_df.insert(0, "solver_name", solver_name)
    else:
        summary_df = pd.DataFrame(columns=["solver_name", "dataset", "status", "count"])

    summary_df.to_csv(summary_file, index=False)
    return runs_file, errors_file, summary_file
