"""Part D.4 EGN optional external integration.

This module intentionally does NOT vendor/copy the upstream EGN implementation.
It imports the upstream Erdos Goes Neural files from a user-provided local path
at runtime, for example:

    external/EGN/erdos_neu/

The wrapper writes the same canonical output files as the other Part D solver
wrappers:

    solver_runs.csv
    solver_errors.csv
    run_summary.csv
"""

from __future__ import annotations

import re
import csv
import gzip
import importlib
import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Literal

import networkx as nx
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional but recommended
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


SPLIT_MANIFEST_COLUMNS = [
    "dataset",
    "graph_id",
    "source_index",
    "split",
    "n_nodes",
    "n_edges",
    "graph_hash_sha256",
    "source_loader",
    "source_name",
    "source_ego_id",
]

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

ERROR_COLUMNS = [
    "dataset",
    "graph_id",
    "source_index",
    "stage",
    "error_type",
    "error_message",
]


@dataclass(frozen=True)
class EGNConfig:
    split_manifest: Path
    interim_dir: Path
    output_dir: Path
    egn_root: Path
    extra_infer_graph_stores: list[str] | None = None
    dataset: str = "twitter"
    mode: Literal["train", "infer", "train-and-infer", "smoke"] = "smoke"
    checkpoint: Path | None = None
    epochs: int = 100
    train_batch_size: int = 4
    infer_batch_size: int = 1
    num_layers: int = 5
    hidden_1: int = 64
    hidden_2: int = 1
    learning_rate: float = 0.001
    penalty_coeff: float = 4.0
    seed: int = 66
    inference_samples: int = 8
    receptive_field: int | None = None
    effective_volume_range: float = 0.15
    lr_decay_step_size: int = 5
    lr_decay_factor: float = 0.95
    edge_drop_p: float = 0.0
    edge_dropout_decay: float = 0.90
    warmup_epochs: int = 2
    limit_per_split: int | None = None
    device: str | None = None


def require_columns(df: pd.DataFrame, required: Iterable[str], source_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}. "
            f"Expected at least: {list(required)}"
        )


def read_split_manifest(
    split_manifest: str | Path,
    dataset: str = "twitter",
    splits: Iterable[str] = ("train", "validation", "test"),
    limit_per_split: int | None = None,
) -> pd.DataFrame:
    """Read and validate the fixed Part B split manifest.

    The manifest is expected to contain the columns:
    dataset, graph_id, source_index, split, n_nodes, n_edges,
    graph_hash_sha256, source_loader, source_name, source_ego_id.
    """

    path = Path(split_manifest)
    df = pd.read_csv(path)
    require_columns(df, SPLIT_MANIFEST_COLUMNS, str(path))

    allowed = set(splits)
    out = df[(df["dataset"] == dataset) & (df["split"].isin(allowed))].copy()

    if out.empty:
        raise ValueError(
            f"No rows found in {path} for dataset={dataset!r} and splits={sorted(allowed)}."
        )

    # Preserve manifest order within each split. This mirrors the original notebook
    # slicing while making the selected graph IDs explicit and reproducible.
    out["_manifest_order"] = np.arange(len(out))

    if limit_per_split is not None:
        out = (
            out.sort_values(["split", "_manifest_order"])
            .groupby("split", group_keys=False)
            .head(limit_per_split)
            .copy()
        )

    split_order = {"train": 0, "validation": 1, "test": 2}
    out["_split_order"] = out["split"].map(split_order).fillna(99).astype(int)
    out = out.sort_values(["_split_order", "_manifest_order"]).drop(
        columns=["_split_order", "_manifest_order"]
    )
    return out.reset_index(drop=True)


def candidate_graph_store_paths(interim_dir: str | Path, dataset: str) -> list[Path]:
    root = Path(interim_dir)
    dataset = dataset.lower()
    return [
        root / f"{dataset}_graphs.pkl.gz",
        root / f"{dataset}_graphs.pkl",
        root / f"{dataset}.pkl.gz",
        root / f"{dataset}.pkl",
        root / f"{dataset}_data.pkl.gz",
        root / f"{dataset}_data.pkl",
    ]


def load_graph_store(interim_dir: str | Path, dataset: str) -> Any:
    """Load the Part A graph store for a dataset from the interim directory."""

    candidates = candidate_graph_store_paths(interim_dir, dataset)
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(
            "Could not find a graph store for dataset "
            f"{dataset!r} in {interim_dir}. Tried: {[str(p) for p in candidates]}"
        )

    path = existing[0]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def get_graph_from_store(store: Any, row: pd.Series) -> Any:
    """Return a graph object from a Part A graph store.

    The Part B manifest column ``source_index`` is an original source identifier;
    it is not guaranteed to be the Python list position in every serialized graph
    store. This function therefore tries several safe lookup strategies:

    1. unwrap common container keys such as ``graphs``;
    2. dictionary lookup by graph_id / source_index;
    3. list-position lookup only when source_index is within bounds;
    4. linear scan of list/dict records by graph_id, source_index, source_name,
       and source_ego_id metadata.
    """

    graph_id = str(row["graph_id"])
    source_index = int(row["source_index"])
    source_name = "" if pd.isna(row.get("source_name", "")) else str(row.get("source_name", ""))
    source_ego_id = "" if pd.isna(row.get("source_ego_id", "")) else str(row.get("source_ego_id", ""))

    # Some stores are dictionaries with the actual graph collection nested inside.
    if isinstance(store, dict):
        for key in ("graphs", "graph_store", "items", "records", "data"):
            value = store.get(key)
            if isinstance(value, (list, tuple, dict)):
                try:
                    return get_graph_from_store(value, row)
                except (KeyError, IndexError, TypeError):
                    pass

        # Direct dictionary lookups.
        for key in (graph_id, source_index, str(source_index), source_name, source_ego_id):
            if key != "" and key in store:
                return store[key]

        # Some stores keep metadata arrays plus graph arrays.
        for graph_key in ("nx_graphs", "networkx_graphs", "graph_objects"):
            if graph_key in store and isinstance(store[graph_key], (list, tuple)):
                graphs = store[graph_key]

                for ids_key in ("graph_ids", "ids"):
                    ids = store.get(ids_key)
                    if isinstance(ids, (list, tuple)):
                        for idx, value in enumerate(ids):
                            if str(value) == graph_id and idx < len(graphs):
                                return graphs[idx]

                for idx_key in ("source_indices", "source_index"):
                    indices = store.get(idx_key)
                    if isinstance(indices, (list, tuple)):
                        for idx, value in enumerate(indices):
                            try:
                                if int(value) == source_index and idx < len(graphs):
                                    return graphs[idx]
                            except Exception:
                                pass

        # Dict-of-records scan.
        for value in store.values():
            if _record_matches_manifest_row(value, graph_id, source_index, source_name, source_ego_id):
                return value

    if isinstance(store, (list, tuple)):
        # Use source_index as list position only if valid.
        if 0 <= source_index < len(store):
            candidate = store[source_index]

            # If the candidate carries metadata, require a match.
            # If it does not carry metadata, this is the conventional raw-list format.
            if _record_has_metadata(candidate):
                if _record_matches_manifest_row(candidate, graph_id, source_index, source_name, source_ego_id):
                    return candidate
            else:
                return candidate

        # Otherwise scan all records. This fixes stores where source_index is an
        # original source ID rather than the list offset.
        for candidate in store:
            if _record_matches_manifest_row(candidate, graph_id, source_index, source_name, source_ego_id):
                return candidate

    raise KeyError(
        f"Could not locate graph_id={graph_id!r}, source_index={source_index!r}, "
        f"source_name={source_name!r}, source_ego_id={source_ego_id!r} "
        f"in graph store of type {type(store).__name__}."
    )


def _record_has_metadata(record: Any) -> bool:
    """Return True if a graph-store record appears to include manifest metadata."""

    if isinstance(record, dict):
        keys = set(record.keys())
        return bool(keys & {"graph_id", "source_index", "source_name", "source_ego_id", "id"})

    return any(
        hasattr(record, attr)
        for attr in ("graph_id", "source_index", "source_name", "source_ego_id", "id")
    )


def _record_value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _record_matches_manifest_row(
    record: Any,
    graph_id: str,
    source_index: int,
    source_name: str,
    source_ego_id: str,
) -> bool:
    """Check whether a graph-store record matches a manifest row."""

    if record is None:
        return False

    for key in ("graph_id", "id"):
        value = _record_value(record, key)
        if value is not None and str(value) == graph_id:
            return True

    value = _record_value(record, "source_index")
    if value is not None:
        try:
            if int(value) == source_index:
                return True
        except Exception:
            if str(value) == str(source_index):
                return True

    value = _record_value(record, "source_name")
    if source_name and value is not None and str(value) == source_name:
        return True

    value = _record_value(record, "source_ego_id")
    if source_ego_id and value is not None and str(value) == source_ego_id:
        return True

    return False


def normalise_networkx_graph(graph: Any) -> nx.Graph:
    """Convert a graph-like object to a simple undirected NetworkX graph.

    Handles common Part A graph-store formats:
    - raw NetworkX graph;
    - wrapper object with graph / nx_graph / G attribute;
    - dict with graph / nx_graph / G / networkx_graph key;
    - dict with edge_index;
    - dict with edges / nodes;
    - dict with adjacency / adj / adj_matrix.

    Node labels are relabelled to 0..n-1 so that EGN output nodes match
    PyTorch Geometric local graph indices.
    """

    def _to_numpy(value: Any):
        """Best-effort conversion for torch / numpy / list-like objects."""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def _from_edge_index(edge_index: Any, num_nodes: int | None = None) -> nx.Graph:
        arr = _to_numpy(edge_index)

        if arr.ndim != 2:
            raise TypeError(f"edge_index must be 2-dimensional; got shape {arr.shape}")

        if arr.shape[0] == 2:
            edges = list(zip(arr[0].tolist(), arr[1].tolist()))
        elif arr.shape[1] == 2:
            edges = list(zip(arr[:, 0].tolist(), arr[:, 1].tolist()))
        else:
            raise TypeError(f"edge_index must have shape [2, E] or [E, 2]; got {arr.shape}")

        g = nx.Graph()
        if num_nodes is not None:
            g.add_nodes_from(range(int(num_nodes)))
        g.add_edges_from((int(u), int(v)) for u, v in edges)
        return g

    def _from_adjacency(adj: Any) -> nx.Graph:
        try:
            import scipy.sparse as sp_sparse

            if sp_sparse.issparse(adj):
                if hasattr(nx, "from_scipy_sparse_array"):
                    return nx.from_scipy_sparse_array(adj)
                return nx.from_scipy_sparse_matrix(adj)
        except Exception:
            pass

        arr = _to_numpy(adj)
        return nx.from_numpy_array(arr)

    def _coerce(obj: Any) -> nx.Graph | None:
        if isinstance(obj, nx.Graph):
            return obj.copy()

        if isinstance(obj, dict):
            for key in (
                "graph",
                "nx_graph",
                "networkx_graph",
                "G",
                "nx",
                "data",
                "object",
            ):
                if key in obj and obj[key] is not obj:
                    candidate = _coerce(obj[key])
                    if candidate is not None:
                        return candidate

            if "edge_index" in obj:
                num_nodes = (
                    obj.get("num_nodes")
                    or obj.get("n_nodes")
                    or obj.get("number_of_nodes")
                )
                return _from_edge_index(obj["edge_index"], num_nodes=num_nodes)

            if "edges" in obj:
                g = nx.Graph()
                if "nodes" in obj:
                    g.add_nodes_from(list(obj["nodes"]))
                elif "num_nodes" in obj:
                    g.add_nodes_from(range(int(obj["num_nodes"])))
                elif "n_nodes" in obj:
                    g.add_nodes_from(range(int(obj["n_nodes"])))
                g.add_edges_from(obj["edges"])
                return g

            for key in ("adjacency", "adj", "adj_matrix", "A"):
                if key in obj:
                    return _from_adjacency(obj[key])

            raise TypeError(
                "Graph record is a dict, but it does not contain a recognized graph key. "
                f"Available keys: {sorted(map(str, obj.keys()))}"
            )

        for attr in ("graph", "nx_graph", "networkx_graph", "G"):
            if hasattr(obj, attr):
                candidate = _coerce(getattr(obj, attr))
                if candidate is not None:
                    return candidate

        return None

    g = _coerce(graph)
    if g is None:
        raise TypeError(
            f"Expected a NetworkX graph, graph wrapper, or dict graph record; got {type(graph).__name__}."
        )

    g = nx.Graph(g)
    g.remove_edges_from(nx.selfloop_edges(g))

    try:
        ordered_nodes = sorted(g.nodes())
    except TypeError:
        ordered_nodes = list(g.nodes())

    mapping = {node: i for i, node in enumerate(ordered_nodes)}
    return nx.relabel_nodes(g, mapping, copy=True)


def networkx_to_pyg_data(graph: nx.Graph) -> Any:
    """Convert a NetworkX graph to a PyTorch Geometric Data object."""

    torch, _, _, _, _, from_networkx, _ = require_torch_geometric()
    data = from_networkx(graph)
    data.x = torch.ones((graph.number_of_nodes(),), dtype=torch.float)
    data.num_nodes = graph.number_of_nodes()
    return data


def build_pyg_split_datasets(
    split_manifest: str | Path,
    interim_dir: str | Path,
    dataset: str = "twitter",
    limit_per_split: int | None = None,
) -> tuple[list[Any], list[Any], list[Any], pd.DataFrame, dict[str, nx.Graph]]:
    """Load train/validation/test PyG datasets using the fixed split manifest."""

    split_df = read_split_manifest(
        split_manifest,
        dataset=dataset,
        splits=("train", "validation", "test"),
        limit_per_split=limit_per_split,
    )
    store = load_graph_store(interim_dir, dataset)

    pyg_by_split: dict[str, list[Any]] = {"train": [], "validation": [], "test": []}
    nx_by_graph_id: dict[str, nx.Graph] = {}

    for _, row in split_df.iterrows():
        graph = normalise_networkx_graph(get_graph_from_store(store, row))
        graph_id = str(row["graph_id"])
        nx_by_graph_id[graph_id] = graph

        data = networkx_to_pyg_data(graph)
        data.graph_id = graph_id
        data.source_index = int(row["source_index"])
        pyg_by_split[str(row["split"])].append(data)

    return (
        pyg_by_split["train"],
        pyg_by_split["validation"],
        pyg_by_split["test"],
        split_df,
        nx_by_graph_id,
    )



def _parse_extra_graph_store_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(
            "Invalid --extra-infer-graph-store value. Expected format: dataset=path"
        )

    dataset_name, raw_path = spec.split("=", 1)
    dataset_name = dataset_name.strip()
    raw_path = raw_path.strip()

    if not dataset_name:
        raise ValueError(f"Invalid dataset name in graph-store spec: {spec!r}")
    if not raw_path:
        raise ValueError(f"Invalid path in graph-store spec: {spec!r}")

    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Extra inference graph store not found: {path}")

    return dataset_name, path


def _load_direct_graph_store(path: Path) -> Any:
    name = path.name.lower()

    if name.endswith((".pkl.gz", ".pickle.gz", ".gpickle.gz")):
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)

    if name.endswith((".pkl", ".pickle", ".gpickle")):
        with path.open("rb") as handle:
            return pickle.load(handle)

    raise ValueError(
        f"Unsupported extra graph-store extension for {path}. "
        "Expected .pkl, .pickle, .gpickle, .pkl.gz, .pickle.gz, or .gpickle.gz."
    )


def _iter_extra_graph_items(store: Any) -> list[tuple[Any, Any]]:
    """Return (graph_id_hint, graph_like_object) pairs from common graph stores."""
    if isinstance(store, dict):
        if "graphs" in store:
            return _iter_extra_graph_items(store["graphs"])
        if "records" in store:
            return _iter_extra_graph_items(store["records"])
        if "data" in store:
            return _iter_extra_graph_items(store["data"])

        return list(store.items())

    if isinstance(store, (list, tuple)):
        return list(enumerate(store))

    raise TypeError(f"Unsupported extra graph-store type: {type(store).__name__}")



def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip() == "":
            return None
        return int(value)
    except Exception:
        return None


def canonical_graph_id(dataset_name: str, source_index: int) -> str:
    """Return the repository-standard graph ID for a dataset/source index."""
    return f"{dataset_name}_graph{int(source_index):06d}"


def _extract_record_graph_id(record: Any) -> str | None:
    for key in ("graph_id", "id"):
        value = _record_value(record, key, None)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _extract_record_source_index(record: Any) -> int | None:
    for key in ("source_index", "index", "idx"):
        value = _record_value(record, key, None)
        out = _int_or_none(value)
        if out is not None:
            return out
    return None


def _canonicalize_extra_graph_identity(
    dataset_name: str,
    graph_id_hint: Any,
    graph_like: Any,
    position: int,
) -> tuple[str, int]:
    """Return a canonical `(graph_id, source_index)` pair for extra all-test stores."""
    dataset_name = str(dataset_name).strip()

    record_graph_id = _extract_record_graph_id(graph_like)
    record_source_index = _extract_record_source_index(graph_like)

    hint_str = None if graph_id_hint is None else str(graph_id_hint).strip()
    hint_int = _int_or_none(graph_id_hint)

    chosen_graph_id = None
    if record_graph_id:
        chosen_graph_id = record_graph_id
    elif hint_str:
        chosen_graph_id = hint_str

    if chosen_graph_id:
        m = re.fullmatch(rf"{re.escape(dataset_name)}_graph(\d{{6}})", chosen_graph_id)
        if m:
            source_index = int(m.group(1))
            return chosen_graph_id, source_index

        m = re.fullmatch(rf"{re.escape(dataset_name)}_(\d{{6}})", chosen_graph_id)
        if m:
            zero_based = int(m.group(1))
            source_index = zero_based + 1
            return canonical_graph_id(dataset_name, source_index), source_index

    if record_source_index is not None:
        source_index = int(record_source_index)
        if source_index == position:
            source_index = source_index + 1
        return canonical_graph_id(dataset_name, source_index), source_index

    if hint_int is not None:
        source_index = int(hint_int) + 1
        return canonical_graph_id(dataset_name, source_index), source_index

    source_index = int(position) + 1
    return canonical_graph_id(dataset_name, source_index), source_index


def build_extra_pyg_infer_datasets(
    specs: list[str],
) -> tuple[list[Any], pd.DataFrame, dict[str, nx.Graph]]:
    """Build PyG test data from full graph stores outside the Twitter split manifest.

    Each spec is formatted as dataset=path/to/graphs.pkl.gz. All graphs in these stores
    are treated as test/inference graphs.

    This function uses repository-standard graph IDs for extra datasets, e.g.:
        collab_graph000001, collab_graph000002, ...
        imdb_binary_graph000001, imdb_binary_graph000002, ...
    """
    extra_test_data: list[Any] = []
    extra_rows: list[dict[str, Any]] = []
    extra_nx_by_graph_id: dict[str, nx.Graph] = {}

    for spec in specs:
        dataset_name, graph_store_path = _parse_extra_graph_store_spec(spec)
        store = _load_direct_graph_store(graph_store_path)
        items = _iter_extra_graph_items(store)

        for position, (graph_id_hint, graph_like) in tqdm(
            list(enumerate(items)),
            total=len(items),
            desc=f"EGN materialise all-test {dataset_name}",
        ):
            graph = normalise_networkx_graph(graph_like)
            graph_id, source_index = _canonicalize_extra_graph_identity(
                dataset_name=dataset_name,
                graph_id_hint=graph_id_hint,
                graph_like=graph_like,
                position=position,
            )

            data = networkx_to_pyg_data(graph)
            data.graph_id = str(graph_id)
            data.source_index = int(source_index)

            extra_test_data.append(data)
            extra_nx_by_graph_id[str(graph_id)] = graph
            extra_rows.append(
                {
                    "dataset": dataset_name,
                    "graph_id": str(graph_id),
                    "source_index": int(source_index),
                    "split": "test",
                    "n_nodes": int(graph.number_of_nodes()),
                    "n_edges": int(graph.number_of_edges()),
                    "graph_hash_sha256": "",
                    "source_loader": "extra_infer_graph_store",
                    "source_name": str(graph_store_path),
                    "source_ego_id": "",
                }
            )

    extra_df = pd.DataFrame(extra_rows, columns=SPLIT_MANIFEST_COLUMNS)
    return extra_test_data, extra_df, extra_nx_by_graph_id


def require_torch_geometric() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Lazy-import torch/PyG so non-EGN tests can run without GPU dependencies."""

    try:
        import torch
        from torch.optim import Adam
        from torch_geometric.loader import DataLoader
        from torch_geometric.utils import from_networkx
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "EGN requires torch, torch_geometric, and related PyG packages. "
            "Install a CUDA/CPU-compatible PyTorch + PyG stack before running Part D.4."
        ) from exc

    try:
        # Importing torch_scatter explicitly gives a clearer error in environments
        # where PyG is installed but compiled extensions are missing.
        import torch_scatter  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "EGN requires torch_scatter. Install the torch_scatter wheel that "
            "matches your PyTorch/CUDA version."
        ) from exc

    return torch, Adam, DataLoader, None, None, from_networkx, None


def load_external_egn(egn_root: str | Path) -> SimpleNamespace:
    """Import upstream EGN modules from a local, user-provided checkout.

    Expected files at egn_root:
    - models.py
    - cut_utils.py
    - modules_and_utils.py
    """

    root = Path(egn_root).resolve()
    expected = ["models.py", "cut_utils.py", "modules_and_utils.py"]
    missing = [name for name in expected if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"EGN root {root} is missing {missing}. "
            "Clone https://github.com/Stalence/erdos_neu into external/EGN/erdos_neu "
            "or pass --egn-root /path/to/erdos_neu."
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        models = importlib.import_module("models")
        cut_utils = importlib.import_module("cut_utils")
        modules_and_utils = importlib.import_module("modules_and_utils")

        # Upstream EGN notebooks import GraphSizeNorm in the notebook scope.
        # When models.py is imported as a standalone external module, the
        # symbol may be missing from models.py's global namespace.
        try:
            from torch_geometric.nn.norm.graph_size_norm import GraphSizeNorm
        except Exception:
            from torch_geometric.nn.norm import GraphSizeNorm

        if not hasattr(models, "GraphSizeNorm"):
            models.GraphSizeNorm = GraphSizeNorm
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import upstream EGN modules from {root}. "
            "Check that the EGN dependencies are installed and that --egn-root points "
            "to the directory containing models.py, cut_utils.py, and modules_and_utils.py."
        ) from exc

    return SimpleNamespace(
        clique_MPNN=models.clique_MPNN,
        get_diracs=cut_utils.get_diracs,
        decode_clique_final_speed=modules_and_utils.decode_clique_final_speed,
    )


def set_reproducibility_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def validate_clique_nodes(graph: nx.Graph, nodes: Iterable[int]) -> bool:
    nodes = [int(n) for n in nodes]
    if len(nodes) <= 1:
        return all(n in graph for n in nodes)

    if any(n not in graph for n in nodes):
        return False

    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            if not graph.has_edge(u, v):
                return False
    return True


def train_egn_model(
    train_data: list[Any],
    egn: SimpleNamespace,
    config: EGNConfig,
) -> tuple[Any, pd.DataFrame, Path]:
    """Train EGN using the notebook hyperparameters exposed through config."""

    torch, Adam, DataLoader, *_ = require_torch_geometric()

    set_reproducibility_seed(config.seed)

    device_name = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    receptive_field = config.receptive_field or (config.num_layers + 1)

    train_loader = DataLoader(train_data, config.train_batch_size, shuffle=True)

    net = egn.clique_MPNN(
        train_data,
        config.num_layers,
        config.hidden_1,
        config.hidden_2,
        1,
    )
    net.to(device).reset_parameters()
    optimizer = Adam(net.parameters(), lr=config.learning_rate, weight_decay=0.0)

    logs: list[dict[str, Any]] = []
    edge_drop_p = config.edge_drop_p
    penalty_coeff = config.penalty_coeff

    train_start = time.perf_counter()

    net.train()
    for epoch in tqdm(range(config.epochs), desc="EGN training epochs", unit="epoch"):
        epoch_start = time.perf_counter()
        if epoch % 5 == 0:
            edge_drop_p = edge_drop_p * config.edge_dropout_decay

        if epoch % config.lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr_decay_factor * param_group["lr"]

        loss_values: list[float] = []
        batches = 0

        for data in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            unit="batch",
            leave=False,
        ):
            batches += 1
            optimizer.zero_grad()
            data = data.to(device)

            data_prime = egn.get_diracs(
                data,
                1,
                sparse=True,
                effective_volume_range=config.effective_volume_range,
                receptive_field=receptive_field,
            )

            data_prime = data_prime.to(device)
            retdict = net(data_prime, None, penalty_coeff)

            if "loss" in retdict:
                loss_tensor = retdict["loss"][0]
                try:
                    loss_values.append(float(loss_tensor.detach().mean().cpu()))
                except Exception:
                    pass

            if epoch > config.warmup_epochs:
                retdict["loss"][0].backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                optimizer.step()

        logs.append(
            {
                "epoch": epoch,
                "batches": batches,
                "mean_loss": float(np.mean(loss_values)) if loss_values else np.nan,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "penalty_coeff": penalty_coeff,
                "edge_drop_p": edge_drop_p,
                "runtime_seconds": time.perf_counter() - epoch_start,
            }
        )

    total_runtime = time.perf_counter() - train_start

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.checkpoint or (config.output_dir / "trained_egn_model.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # Saving the full model mirrors the user's notebook:
    # torch.save(net, 'trained_egn_model_on_twitter_dataset.pth')
    torch.save(net, checkpoint)

    logs_df = pd.DataFrame(logs)
    logs_df["total_training_runtime_seconds"] = total_runtime
    logs_df.to_csv(config.output_dir / "training_log.csv", index=False)

    return net, logs_df, checkpoint



def _torch_load_trusted_checkpoint(checkpoint: Path, map_location: Any) -> Any:
    """Load a locally generated trusted PyTorch checkpoint.

    PyTorch 2.6 changed torch.load's default to weights_only=True. The EGN
    wrapper saves the full upstream model object, so inference needs
    weights_only=False. Only use this for checkpoints you generated locally or
    otherwise trust.
    """
    import torch as _torch

    try:
        return _torch.load(checkpoint, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return _torch.load(checkpoint, map_location=map_location)



def infer_egn_model(
    test_data: list[Any],
    test_rows: pd.DataFrame,
    nx_by_graph_id: dict[str, nx.Graph],
    egn: SimpleNamespace,
    config: EGNConfig,
    net: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run EGN inference/decoding and return solver_runs + solver_errors."""

    torch, _, DataLoader, *_ = require_torch_geometric()

    set_reproducibility_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    device_name = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    receptive_field = config.receptive_field or (config.num_layers + 1)

    if net is None:
        checkpoint = config.checkpoint or (config.output_dir / "trained_egn_model.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}. Run --mode train-and-infer "
                "or pass --checkpoint /path/to/trained_egn_model.pt."
            )
        net = _torch_load_trusted_checkpoint(checkpoint, map_location=device)

    net.to(device)
    net.eval()

    loader = DataLoader(test_data, config.infer_batch_size, shuffle=False)

    runs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    test_lookup = {
        str(row["graph_id"]): row for _, row in test_rows[test_rows["split"] == "test"].iterrows()
    }

    with torch.no_grad():
        for data in tqdm(loader, desc="EGN inference batches", unit="batch"):
            batch_graph_ids = [str(x) for x in list(getattr(data, "graph_id", []))]
            if not batch_graph_ids:
                # PyG may collate custom attributes differently; batch_size=1 is the
                # recommended mode here. Fall back to a positional lookup.
                start_idx = len(runs) + len(errors)
                batch_graph_ids = [str(test_rows[test_rows["split"] == "test"].iloc[start_idx]["graph_id"])]

            data = data.to(device)
            num_graphs = int(data.batch.max().item()) + 1
            batch_start = time.perf_counter()

            # Pre-sample seed nodes once per graph, matching the notebook logic.
            total_samples: list[np.ndarray] = []
            for graph_idx in range(num_graphs):
                curr_inds = data.batch == graph_idx
                g_size = int(curr_inds.sum().item())
                replace = config.inference_samples > g_size
                total_samples.append(
                    rng.choice(g_size, config.inference_samples, replace=replace)
                )

            best_sets: dict[int, Any] = {}
            best_edges = np.zeros((num_graphs,))
            max_sets = np.zeros((num_graphs,))

            try:
                for sample_idx in tqdm(
                    range(config.inference_samples),
                    desc="Dirac samples",
                    unit="sample",
                    leave=False,
                ):
                    sample_start = time.perf_counter()

                    data_prime = egn.get_diracs(
                        data.to(device),
                        1,
                        sparse=True,
                        effective_volume_range=config.effective_volume_range,
                        receptive_field=receptive_field,
                    )

                    data_prime.x = torch.zeros_like(data_prime.x)
                    g_offset = 0

                    for graph_idx in range(num_graphs):
                        curr_inds = data_prime.batch == graph_idx
                        g_size = int(curr_inds.sum().item())
                        seed_node = int(total_samples[graph_idx][sample_idx]) + g_offset
                        data_prime.x[seed_node] = 1.0
                        g_offset += g_size

                    retdict = net(data_prime)
                    sets, set_edges, set_cardinality = egn.decode_clique_final_speed(
                        data_prime,
                        retdict["output"][0],
                        weight_factor=0.0,
                        draw=False,
                        beam=1,
                    )

                    for graph_idx in range(num_graphs):
                        indices = data.batch == graph_idx
                        candidate_size = float(set_cardinality[graph_idx].item())
                        if candidate_size > max_sets[graph_idx]:
                            max_sets[graph_idx] = candidate_size
                            best_sets[graph_idx] = sets[indices].detach().cpu()
                            best_edges[graph_idx] = float(set_edges[graph_idx].item())

                    _ = time.perf_counter() - sample_start

                batch_runtime = time.perf_counter() - batch_start

                for graph_idx, graph_id in enumerate(batch_graph_ids):
                    row = test_lookup[graph_id]
                    graph = nx_by_graph_id[graph_id]

                    tensor = best_sets.get(graph_idx)
                    if tensor is None:
                        nodes: list[int] = []
                    else:
                        values = tensor.reshape(-1).tolist()
                        nodes = [idx for idx, val in enumerate(values) if float(val) == 1.0]

                    clique_valid = validate_clique_nodes(graph, nodes)

                    runs.append(
                        {
                            "dataset": row["dataset"],
                            "graph_id": graph_id,
                            "source_index": int(row["source_index"]),
                            "solver_name": "EGN",
                            "run_type": "infer",
                            "time_limit_seconds": "",
                            "status": "success",
                            "optimality_status": "heuristic",
                            "runtime_seconds": batch_runtime / max(num_graphs, 1),
                            "best_clique_size": len(nodes),
                            "best_clique_nodes": json.dumps(nodes),
                            "clique_valid": bool(clique_valid),
                            "num_nodes": int(row["n_nodes"]),
                            "num_edges": int(row["n_edges"]),
                            "seed": int(config.seed),
                            "threads": "",
                            "mip_gap": "",
                            "objective_bound": "",
                            "error_message": "",
                        }
                    )

            except Exception as exc:
                for graph_id in batch_graph_ids:
                    row = test_lookup.get(graph_id)
                    errors.append(
                        {
                            "dataset": config.dataset if row is None else row["dataset"],
                            "graph_id": graph_id,
                            "source_index": "" if row is None else int(row["source_index"]),
                            "stage": "infer",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )

    return pd.DataFrame(runs, columns=SOLVER_RUN_COLUMNS), pd.DataFrame(errors, columns=ERROR_COLUMNS)


def write_empty_outputs(output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=SOLVER_RUN_COLUMNS).to_csv(output / "solver_runs.csv", index=False)
    pd.DataFrame(columns=ERROR_COLUMNS).to_csv(output / "solver_errors.csv", index=False)
    pd.DataFrame(
        [
            {
                "solver_name": "EGN",
                "run_type": "none",
                "num_success": 0,
                "num_errors": 0,
                "mean_runtime_seconds": "",
                "mean_best_clique_size": "",
            }
        ]
    ).to_csv(output / "run_summary.csv", index=False)


def write_solver_outputs(
    output_dir: str | Path,
    runs_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    extra_summary: dict[str, Any] | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if runs_df.empty:
        runs_df = pd.DataFrame(columns=SOLVER_RUN_COLUMNS)
    else:
        runs_df = runs_df.reindex(columns=SOLVER_RUN_COLUMNS)

    if errors_df.empty:
        errors_df = pd.DataFrame(columns=ERROR_COLUMNS)
    else:
        errors_df = errors_df.reindex(columns=ERROR_COLUMNS)

    runs_df.to_csv(output / "solver_runs.csv", index=False)
    errors_df.to_csv(output / "solver_errors.csv", index=False)

    summary = {
        "solver_name": "EGN",
        "run_type": "infer",
        "num_success": int((runs_df["status"] == "success").sum()) if not runs_df.empty else 0,
        "num_errors": int(len(errors_df)),
        "mean_runtime_seconds": (
            float(pd.to_numeric(runs_df["runtime_seconds"], errors="coerce").mean())
            if not runs_df.empty
            else ""
        ),
        "mean_best_clique_size": (
            float(pd.to_numeric(runs_df["best_clique_size"], errors="coerce").mean())
            if not runs_df.empty
            else ""
        ),
    }
    if extra_summary:
        summary.update(extra_summary)

    pd.DataFrame([summary]).to_csv(output / "run_summary.csv", index=False)


def run_egn_pipeline(config: EGNConfig) -> None:
    """Run Part D.4 EGN train/infer/smoke workflow."""

    if config.mode == "smoke":
        config = EGNConfig(**{**config.__dict__, "limit_per_split": config.limit_per_split or 2, "epochs": min(config.epochs, 2), "inference_samples": min(config.inference_samples, 2)})

    egn = load_external_egn(config.egn_root)
    train_data, _validation_data, test_data, split_df, nx_by_graph_id = build_pyg_split_datasets(
        split_manifest=config.split_manifest,
        interim_dir=config.interim_dir,
        dataset=config.dataset,
        limit_per_split=config.limit_per_split,
    )

    if config.extra_infer_graph_stores:
        extra_test_data, extra_split_df, extra_nx_by_graph_id = build_extra_pyg_infer_datasets(
            config.extra_infer_graph_stores
        )
        test_data.extend(extra_test_data)
        if not extra_split_df.empty:
            split_df = pd.concat([split_df, extra_split_df], ignore_index=True)
        nx_by_graph_id.update(extra_nx_by_graph_id)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    net = None
    train_logs = None
    checkpoint = config.checkpoint

    if config.mode in {"train", "train-and-infer", "smoke"}:
        net, train_logs, checkpoint = train_egn_model(train_data, egn, config)

    if config.mode in {"infer", "train-and-infer", "smoke"}:
        infer_config = config
        if checkpoint is not None:
            infer_config = EGNConfig(**{**config.__dict__, "checkpoint": Path(checkpoint)})

        runs_df, errors_df = infer_egn_model(
            test_data=test_data,
            test_rows=split_df,
            nx_by_graph_id=nx_by_graph_id,
            egn=egn,
            config=infer_config,
            net=net,
        )

        extra_summary: dict[str, Any] = {
            "dataset": config.dataset,
            "mode": config.mode,
            "checkpoint": str(checkpoint or (config.output_dir / "trained_egn_model.pt")),
            "epochs": config.epochs if config.mode != "infer" else "",
            "seed": config.seed,
        }
        if train_logs is not None and not train_logs.empty:
            extra_summary["final_train_mean_loss"] = train_logs["mean_loss"].iloc[-1]

        write_solver_outputs(config.output_dir, runs_df, errors_df, extra_summary)
    else:
        write_empty_outputs(config.output_dir)
        if train_logs is not None:
            # run_summary.csv exists even for train-only mode, but the useful training
            # details are stored in training_log.csv.
            summary_path = config.output_dir / "run_summary.csv"
            pd.DataFrame(
                [
                    {
                        "solver_name": "EGN",
                        "run_type": "train",
                        "num_success": "",
                        "num_errors": "",
                        "mean_runtime_seconds": "",
                        "mean_best_clique_size": "",
                        "dataset": config.dataset,
                        "mode": config.mode,
                        "checkpoint": str(checkpoint or (config.output_dir / "trained_egn_model.pt")),
                        "epochs": config.epochs,
                        "seed": config.seed,
                        "final_train_mean_loss": train_logs["mean_loss"].iloc[-1],
                    }
                ]
            ).to_csv(summary_path, index=False)
