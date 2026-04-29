"""Optional HGS wrapper for GCO-HPIF.

This module deliberately does not vendor upstream HGS code.  It dynamically
imports the upstream model implementation from a local clone supplied via
``--hgs-root`` and keeps generated checkpoints/artifacts outside Git.

The public outputs follow the common solver contract used by the other GCO-HPIF
solver wrappers:
    - solver_runs.csv
    - solver_errors.csv
    - run_summary.csv
"""

from __future__ import annotations

import contextlib
import gzip
import importlib
import json
import math
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm.auto import tqdm

try:  # torch is intentionally optional at import time for lightweight tests.
    import torch
except Exception:  # pragma: no cover - exercised only on systems without torch.
    torch = None  # type: ignore[assignment]


SOLVER_NAME = "hgs"

SOLVER_COLUMNS = [
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
class HGSConfig:
    mode: str
    split_manifest: Path
    interim_dir: Path
    hgs_root: Path
    output_dir: Path
    raw_graph_store: Path | None = None
    extra_infer_graph_stores: list[str] | None = None
    dataset: str = "twitter"
    limit_per_split: int | None = None
    train_splits: list[str] | None = None
    infer_splits: list[str] | None = None
    epochs: int = 20
    train_batch_size: int = 1
    hidden: int = 8
    num_layers: int = 4
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    penalty_coeff: float = 2.0
    moment: int = 1
    smooth: float = 0.1
    use_smooth_residual: bool = False
    seed: int = 42
    device: str = "auto"
    checkpoint_path: Path | None = None
    num_walkers: int = 20
    sample_length: int = 90
    time_limit_seconds: float | None = None


@dataclass
class GraphExample:
    row: dict[str, Any]
    graph: nx.Graph
    contiguous_graph: nx.Graph
    original_nodes: list[Any]
    features: np.ndarray


def run_hgs_experiment(config: HGSConfig) -> None:
    """Run the requested HGS mode and write standardized output files."""
    _require_torch()
    _set_seed(config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoint_path or (config.output_dir / "hgs_checkpoint.pt")

    manifest = _read_manifest(config)
    graph_records = _load_graph_records(config.raw_graph_store or config.interim_dir)

    train_splits = config.train_splits or ["train"]
    infer_splits = config.infer_splits or ["test"]

    train_metrics: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    device = _resolve_device(config.device)

    if config.mode in {"train", "train-and-infer"}:
        train_df = manifest[manifest["split"].astype(str).str.lower().isin([s.lower() for s in train_splits])]
        train_examples, train_errors = _materialise_examples(train_df, graph_records, "train")
        errors.extend(train_errors)

        if not train_examples:
            raise RuntimeError("No HGS training examples were materialised. Check --split-manifest and --interim-dir.")

        hgs_modules = _import_hgs_modules(config.hgs_root)
        model = _build_model(hgs_modules["models"], config).to(device)
        optimizer = torch.optim.RMSprop(  # type: ignore[union-attr]
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        train_metrics = _train_model(
            model=model,
            optimizer=optimizer,
            examples=train_examples,
            config=config,
            device=device,
        )

        _save_checkpoint(checkpoint_path, model, config)

    if config.mode in {"infer", "train-and-infer"}:
        infer_df = manifest[manifest["split"].astype(str).str.lower().isin([s.lower() for s in infer_splits])]
        infer_examples, infer_errors = _materialise_examples(infer_df, graph_records, "infer")
        errors.extend(infer_errors)

        extra_examples: list[GraphExample] = []
        extra_errors: list[dict[str, Any]] = []
        if config.extra_infer_graph_stores:
            extra_examples, extra_errors = _materialise_extra_infer_graph_stores(
                config.extra_infer_graph_stores
            )
            infer_examples.extend(extra_examples)
            errors.extend(extra_errors)

        if not infer_examples:
            raise RuntimeError(
                "No HGS inference examples were materialised. Check --split-manifest, "
                "--interim-dir, --raw-graph-store, and --extra-infer-graph-store."
            )

        hgs_modules = _import_hgs_modules(config.hgs_root)
        model = _load_model_for_inference(hgs_modules["models"], checkpoint_path, config, device)

        infer_runs, infer_errors = _infer_model(
            model=model,
            examples=infer_examples,
            config=config,
            device=device,
        )
        runs.extend(infer_runs)
        errors.extend(infer_errors)

    if config.mode == "train":
        # Keep the common output contract even for train-only runs.
        runs = []

    _write_outputs(config, runs, errors, train_metrics, checkpoint_path)


def greedy_decode_clique(
    graph: nx.Graph,
    scores: Iterable[float],
    *,
    num_walkers: int = 20,
    sample_length: int = 90,
) -> list[Any]:
    """Decode node scores into a clique using greedy multi-start search.

    The node ordering is the order returned by ``list(graph.nodes())``.
    """
    nodes = list(graph.nodes())
    score_array = np.asarray(list(scores), dtype=float).reshape(-1)

    if len(nodes) != len(score_array):
        raise ValueError(f"Expected {len(nodes)} scores, received {len(score_array)}.")

    if not nodes:
        return []

    order = np.argsort(-score_array)
    max_starts = min(num_walkers, len(order))
    max_candidates = min(sample_length, len(order))

    best: list[Any] = []
    for start_rank in range(max_starts):
        clique = [nodes[int(order[start_rank])]]

        for rank in range(max_candidates):
            candidate = nodes[int(order[rank])]
            if candidate in clique:
                continue
            if all(graph.has_edge(candidate, chosen) for chosen in clique):
                clique.append(candidate)

        if len(clique) > len(best):
            best = clique

    return best


def validate_clique(graph: nx.Graph, nodes: Iterable[Any]) -> bool:
    """Return True when all supplied nodes form a clique in ``graph``."""
    clique_nodes = list(nodes)
    if len(set(clique_nodes)) != len(clique_nodes):
        return False
    if any(node not in graph for node in clique_nodes):
        return False
    for i, left in enumerate(clique_nodes):
        for right in clique_nodes[i + 1 :]:
            if not graph.has_edge(left, right):
                return False
    return True


def normalise_networkx_graph(obj: Any) -> nx.Graph:
    """Convert common stored graph representations into a NetworkX graph."""
    if isinstance(obj, nx.Graph):
        graph = obj.copy()
        graph.remove_edges_from(nx.selfloop_edges(graph))
        return graph

    if isinstance(obj, dict):
        for key in ("graph", "nx_graph", "networkx_graph", "G"):
            if key in obj:
                return normalise_networkx_graph(obj[key])

        for key in ("adjacency", "adj", "adj_matrix"):
            if key in obj:
                return _graph_from_adjacency(obj[key])

        if "edge_index" in obj:
            return _graph_from_edge_index(
                obj["edge_index"],
                num_nodes=obj.get("num_nodes") or obj.get("n_nodes"),
                nodes=obj.get("nodes"),
            )

        if "edges" in obj:
            graph = nx.Graph()
            if "nodes" in obj:
                graph.add_nodes_from(obj["nodes"])
            graph.add_edges_from(obj["edges"])
            graph.remove_edges_from(nx.selfloop_edges(graph))
            return graph

    for attr in ("graph", "nx_graph", "networkx_graph", "G"):
        if hasattr(obj, attr):
            return normalise_networkx_graph(getattr(obj, attr))

    if hasattr(obj, "edge_index"):
        num_nodes = getattr(obj, "num_nodes", None)
        return _graph_from_edge_index(getattr(obj, "edge_index"), num_nodes=num_nodes)

    raise TypeError(f"Unsupported graph record type: {type(obj)!r}")


def _graph_from_adjacency(adj: Any) -> nx.Graph:
    if sps.issparse(adj):
        graph = nx.from_scipy_sparse_array(adj)
    else:
        array = np.asarray(adj)
        graph = nx.from_numpy_array(array)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def _graph_from_edge_index(edge_index: Any, num_nodes: int | None = None, nodes: Iterable[Any] | None = None) -> nx.Graph:
    if torch is not None and hasattr(edge_index, "detach"):
        edge_index = edge_index.detach().cpu().numpy()

    array = np.asarray(edge_index)
    if array.size == 0:
        edge_pairs = np.empty((0, 2), dtype=int)
    elif array.ndim == 2 and array.shape[0] == 2:
        edge_pairs = array.T
    elif array.ndim == 2 and array.shape[1] == 2:
        edge_pairs = array
    else:
        raise ValueError(f"Unsupported edge_index shape: {array.shape}")

    graph = nx.Graph()
    if nodes is not None:
        graph.add_nodes_from(list(nodes))
    elif num_nodes is not None:
        graph.add_nodes_from(range(int(num_nodes)))
    elif edge_pairs.size:
        graph.add_nodes_from(range(int(edge_pairs.max()) + 1))

    graph.add_edges_from((int(u), int(v)) for u, v in edge_pairs if int(u) != int(v))
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("The HGS wrapper requires torch. Activate .venv and install torch/PyG dependencies first.")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[union-attr]
    if device == "cuda" and not torch.cuda.is_available():  # type: ignore[union-attr]
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(device)  # type: ignore[union-attr]


def _read_manifest(config: HGSConfig) -> pd.DataFrame:
    if not config.split_manifest.exists():
        raise FileNotFoundError(f"Split manifest not found: {config.split_manifest}")

    df = pd.read_csv(config.split_manifest)
    required = {"dataset", "graph_id", "source_index", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Split manifest is missing required columns: {sorted(missing)}")

    dataset_key = config.dataset.lower()
    df = df[df["dataset"].astype(str).str.lower() == dataset_key].copy()

    if config.limit_per_split is not None:
        df = (
            df.sort_values(["split", "graph_id", "source_index"])
            .groupby("split", group_keys=False)
            .head(config.limit_per_split)
            .copy()
        )

    if df.empty:
        raise RuntimeError(f"No manifest rows found for dataset={config.dataset!r}.")

    return df.reset_index(drop=True)


@contextlib.contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    path = path.resolve()
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(path))


def _import_hgs_modules(hgs_root: Path) -> dict[str, Any]:
    if not hgs_root.exists():
        raise FileNotFoundError(
            f"HGS root not found: {hgs_root}. Clone the upstream repository into external/HGS first."
        )

    expected = ["models.py", "diff_module.py", "layers.py"]
    missing = [name for name in expected if not (hgs_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"HGS root is missing expected files: {missing}")

    # Avoid collisions with other optional integrations that also expose modules
    # named models/utils/layers.
    for module_name in ["models", "diff_module", "layers", "utils", "Sampler"]:
        sys.modules.pop(module_name, None)

    with _prepend_sys_path(hgs_root):
        models = importlib.import_module("models")

    return {"models": models}


def _build_model(hgs_models: Any, config: HGSConfig):
    return hgs_models.GNN(
        input_dim=3,
        hidden_dim=config.hidden,
        output_dim=1,
        n_layers=config.num_layers,
        dropout=config.dropout,
        Withgres=config.use_smooth_residual,
        smooth=config.smooth,
    )


def _load_model_for_inference(hgs_models: Any, checkpoint_path: Path, config: HGSConfig, device: Any):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"HGS checkpoint not found: {checkpoint_path}")

    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)  # type: ignore[union-attr]
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)  # type: ignore[union-attr]

    model_config = payload.get("model_config", {})
    effective_config = HGSConfig(**{**asdict(config), **model_config})
    model = _build_model(hgs_models, effective_config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _save_checkpoint(checkpoint_path: Path, model: Any, config: HGSConfig) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(  # type: ignore[union-attr]
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "mode": config.mode,
                "split_manifest": config.split_manifest,
                "interim_dir": config.interim_dir,
                "hgs_root": config.hgs_root,
                "output_dir": config.output_dir,
                "dataset": config.dataset,
                "hidden": config.hidden,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "smooth": config.smooth,
                "use_smooth_residual": config.use_smooth_residual,
                "moment": config.moment,
                "seed": config.seed,
            },
        },
        checkpoint_path,
    )


def _load_graph_records(graph_source: Path) -> list[dict[str, Any]]:
    """Load graph records from a directory or a direct graph-store file.

    This supports the current GCO-HPIF workflow and the HGS notebook workflow.
    ``graph_source`` may be:
      - a directory such as artifacts/slice_a_full/interim
      - a direct pickle/torch/json graph-store file
    """
    if not graph_source.exists():
        raise FileNotFoundError(f"Graph source not found: {graph_source}")

    candidates = _iter_candidate_graph_files(graph_source)
    records: list[dict[str, Any]] = []
    load_errors: list[str] = []

    for path in candidates:
        try:
            payload = _load_serialized_payload(path)
        except Exception as exc:
            load_errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue

        for idx, item in enumerate(_flatten_graph_payload(payload)):
            try:
                graph = normalise_networkx_graph(item)
            except Exception:
                continue

            meta = _extract_record_meta(item, idx)
            meta["source_file"] = str(path)
            records.append({"meta": meta, "graph": graph, "raw": item})

    if not records:
        preview = "\n".join(f"  - {p}" for p in candidates[:80]) or "  - no candidate files found"
        error_preview = "\n".join(f"  - {e}" for e in load_errors[:40]) or "  - no loader errors captured"
        raise RuntimeError(
            "No usable graph records were found under "
            f"{graph_source}.\n\n"
            "Candidate files checked:\n"
            f"{preview}\n\n"
            "Loader errors, if any:\n"
            f"{error_preview}\n\n"
            "Use --raw-graph-store to point directly to the pickle file that "
            "contains the NetworkX graphs, for example TWITTER_dataset.pkl."
        )

    return records


def _iter_candidate_graph_files(graph_source: Path) -> list[Path]:
    supported_endings = (
        ".pkl",
        ".pickle",
        ".gpickle",
        ".pkl.gz",
        ".pickle.gz",
        ".gpickle.gz",
        ".pt",
        ".pth",
        ".json",
        ".jsonl",
    )

    if graph_source.is_file():
        return [graph_source]

    skip_parts = {"__pycache__", ".pytest_cache", ".git"}
    candidates: list[Path] = []

    for path in graph_source.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_parts for part in path.parts):
            continue

        name = path.name.lower()
        if name.endswith(supported_endings):
            candidates.append(path)

    def sort_key(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        graphish = any(token in name for token in ("graph", "graphs", "dataset", "nx", "edge", "data"))
        return (0 if graphish else 1, str(path))

    return sorted(candidates, key=sort_key)


def _load_serialized_payload(path: Path) -> Any:
    name = path.name.lower()
    suffix = path.suffix.lower()

    if name.endswith((".pkl.gz", ".pickle.gz", ".gpickle.gz")):
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)

    if suffix in {".pkl", ".pickle", ".gpickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)

    if suffix in {".pt", ".pth"}:
        if torch is None:
            raise RuntimeError("torch is required to load .pt/.pth graph artifacts.")
        try:
            return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[union-attr]
        except TypeError:
            return torch.load(path, map_location="cpu")  # type: ignore[union-attr]

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    raise ValueError(f"Unsupported graph artifact extension: {path.name}")


def _flatten_graph_payload(payload: Any) -> Iterator[Any]:
    """Flatten common graph-store containers into graph-like records."""
    if isinstance(payload, dict):
        if any(
            key in payload
            for key in (
                "graph",
                "nx_graph",
                "networkx_graph",
                "G",
                "edge_index",
                "edges",
                "adjacency",
                "adj",
                "adj_matrix",
            )
        ):
            yield payload
            return

        for collection_key in (
            "records",
            "graphs",
            "graph_records",
            "graph_store",
            "data",
            "items",
            "examples",
            "objects",
        ):
            if collection_key in payload and isinstance(payload[collection_key], (list, tuple, dict)):
                yield from _flatten_graph_payload(payload[collection_key])
                return

        for key, value in payload.items():
            if isinstance(value, nx.Graph) or hasattr(value, "edge_index"):
                yield {"graph": value, "graph_id": key, "_dict_key": key}
            elif isinstance(value, dict):
                nested = dict(value)
                nested.setdefault("graph_id", key)
                nested.setdefault("_dict_key", key)
                yield from _flatten_graph_payload(nested)
        return

    if isinstance(payload, (list, tuple)):
        for item in payload:
            yield item
        return

    yield payload


def _extract_record_meta(item: Any, idx: int) -> dict[str, Any]:
    meta: dict[str, Any] = {"record_position": idx}

    if isinstance(item, dict):
        for key in ("graph_id", "source_index", "source_name", "source_ego_id", "dataset", "_dict_key"):
            if key in item:
                meta[key] = item[key]
        nested = item.get("metadata") or item.get("meta")
        if isinstance(nested, dict):
            for key in ("graph_id", "source_index", "source_name", "source_ego_id", "dataset"):
                if key in nested:
                    meta[key] = nested[key]
    else:
        for key in ("graph_id", "source_index", "source_name", "source_ego_id", "dataset"):
            if hasattr(item, key):
                meta[key] = getattr(item, key)

    return meta


def _materialise_examples(
    manifest_df: pd.DataFrame,
    graph_records: list[dict[str, Any]],
    stage: str,
) -> tuple[list[GraphExample], list[dict[str, Any]]]:
    examples: list[GraphExample] = []
    errors: list[dict[str, Any]] = []

    for _, row_series in tqdm(
        manifest_df.iterrows(),
        total=len(manifest_df),
        desc=f"HGS materialise {stage}",
    ):
        row = row_series.to_dict()
        try:
            graph = _lookup_graph(row, graph_records)
            contiguous, original_nodes = _to_contiguous_graph(graph)
            features = _node_features(contiguous)
            examples.append(
                GraphExample(
                    row=row,
                    graph=graph,
                    contiguous_graph=contiguous,
                    original_nodes=original_nodes,
                    features=features,
                )
            )
        except Exception as exc:
            errors.append(_error_row(row, run_type=stage, exc=exc))

    return examples, errors



def _materialise_extra_infer_graph_stores(
    specs: list[str],
) -> tuple[list[GraphExample], list[dict[str, Any]]]:
    """Materialise all graph records from extra dataset graph stores.

    Each spec must be formatted as:

        dataset=path/to/graphs.pkl.gz

    This is used for all-test inference on datasets that do not have a
    train/validation/test split manifest, such as full COLLAB and IMDB-BINARY.
    """
    examples: list[GraphExample] = []
    errors: list[dict[str, Any]] = []

    for spec in specs:
        dataset_name, graph_store_path = _parse_extra_graph_store_spec(spec)
        records = _load_graph_records(graph_store_path)

        for position, record in tqdm(
            list(enumerate(records)),
            total=len(records),
            desc=f"HGS materialise all-test {dataset_name}",
        ):
            meta = record.get("meta", {})
            row = {
                "dataset": dataset_name,
                "graph_id": meta.get("graph_id") or meta.get("_dict_key") or f"{dataset_name}_{position:06d}",
                "source_index": meta.get("source_index", position),
                "source_name": meta.get("source_name"),
                "source_ego_id": meta.get("source_ego_id"),
                "split": "test",
            }

            try:
                graph = record["graph"].copy()
                contiguous, original_nodes = _to_contiguous_graph(graph)
                features = _node_features(contiguous)

                examples.append(
                    GraphExample(
                        row=row,
                        graph=graph,
                        contiguous_graph=contiguous,
                        original_nodes=original_nodes,
                        features=features,
                    )
                )
            except Exception as exc:
                errors.append(_error_row(row, run_type="infer", exc=exc))

    return examples, errors


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



def _lookup_graph(row: dict[str, Any], records: list[dict[str, Any]]) -> nx.Graph:
    scored: list[tuple[int, int, nx.Graph]] = []
    for position, record in enumerate(records):
        score = _record_match_score(row, record["meta"], position)
        if score > 0:
            scored.append((score, -position, record["graph"]))

    if not scored:
        raise KeyError(
            "Could not match graph for "
            f"graph_id={row.get('graph_id')!r}, source_index={row.get('source_index')!r}, "
            f"source_name={row.get('source_name')!r}, source_ego_id={row.get('source_ego_id')!r}."
        )

    scored.sort(reverse=True)
    return scored[0][2].copy()


def _record_match_score(row: dict[str, Any], meta: dict[str, Any], position: int) -> int:
    score = 0

    if _same(row.get("graph_id"), meta.get("graph_id")) or _same(row.get("graph_id"), meta.get("_dict_key")):
        score += 8
    if _same(row.get("source_index"), meta.get("source_index")):
        score += 4
    if _same(row.get("source_index"), meta.get("record_position")):
        score += 2
    if _same(row.get("source_index"), position):
        score += 1
    if _same(row.get("source_name"), meta.get("source_name")):
        score += 2
    if _same(row.get("source_ego_id"), meta.get("source_ego_id")):
        score += 2
    if _same(row.get("dataset"), meta.get("dataset")):
        score += 1

    return score


def _same(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    if isinstance(left, float) and math.isnan(left):
        return False
    if isinstance(right, float) and math.isnan(right):
        return False
    return str(left) == str(right)


def _to_contiguous_graph(graph: nx.Graph) -> tuple[nx.Graph, list[Any]]:
    original_nodes = list(graph.nodes())
    mapping = {node: idx for idx, node in enumerate(original_nodes)}
    contiguous = nx.relabel_nodes(graph, mapping, copy=True)
    contiguous.remove_edges_from(nx.selfloop_edges(contiguous))
    return contiguous, original_nodes


def _node_features(graph: nx.Graph) -> np.ndarray:
    """Compute HGS notebook-style 3D node features.

    Features:
      1. eccentricity within the graph/connected component
      2. log degree
      3. clustering coefficient

    This mirrors the HGS preprocessing notebook but preserves row order by
    assigning features by contiguous node id.
    """
    n = graph.number_of_nodes()
    features = np.zeros((n, 3), dtype=np.float32)

    if n == 0:
        return features

    # The wrapper relabels graphs to contiguous integer nodes before this
    # function is called.
    if n == 1:
        node = list(graph.nodes())[0]
        features[int(node)] = [0.0, 0.0, 0.0]
        return features

    if nx.is_connected(graph):
        components = [graph]
    else:
        components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

    clustering = nx.clustering(graph)

    for subgraph in components:
        try:
            eccentricity = nx.eccentricity(subgraph)
        except Exception:
            eccentricity = {node: 0.0 for node in subgraph.nodes()}

        for node in subgraph.nodes():
            degree = graph.degree(node)
            log_degree = float(np.log(degree)) if degree > 0 else 0.0
            features[int(node)] = [
                float(eccentricity.get(node, 0.0)),
                log_degree,
                float(clustering.get(node, 0.0)),
            ]

    return features.astype(np.float32)


def _to_torch_sparse_adjacency(graph: nx.Graph, device: Any):
    n = graph.number_of_nodes()
    if n == 0:
        indices = torch.zeros((2, 0), dtype=torch.long, device=device)  # type: ignore[union-attr]
        values = torch.zeros((0,), dtype=torch.float32, device=device)  # type: ignore[union-attr]
        return torch.sparse_coo_tensor(indices, values, (0, 0), device=device)  # type: ignore[union-attr]

    rows: list[int] = []
    cols: list[int] = []
    for u, v in graph.edges():
        rows.extend([int(u), int(v)])
        cols.extend([int(v), int(u)])

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)  # type: ignore[union-attr]
    values = torch.ones(len(rows), dtype=torch.float32, device=device)  # type: ignore[union-attr]
    return torch.sparse_coo_tensor(indices, values, (n, n), device=device).coalesce()  # type: ignore[union-attr]


def _hgs_unsupervised_loss(output: Any, adj_sparse: Any, penalty_coeff: float, device: Any):
    n = output.shape[0]
    adj = adj_sparse.to_dense()
    eye = torch.eye(n, dtype=adj.dtype, device=device)  # type: ignore[union-attr]
    non_edge = torch.ones((n, n), dtype=adj.dtype, device=device) - eye - adj  # type: ignore[union-attr]

    p = output.reshape(n, 1)
    edge_reward = (p.T @ adj @ p).sum()
    non_edge_penalty = (p * (non_edge @ p)).sum()
    return -edge_reward + penalty_coeff * non_edge_penalty


def _train_model(
    model: Any,
    optimizer: Any,
    examples: list[GraphExample],
    config: HGSConfig,
    device: Any,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    model.train()

    for epoch in tqdm(range(config.epochs), desc="HGS training epochs"):
        random.shuffle(examples)
        losses: list[float] = []

        batches = [
            examples[i : i + config.train_batch_size]
            for i in range(0, len(examples), config.train_batch_size)
        ]

        for batch in tqdm(batches, desc=f"HGS epoch {epoch + 1}", leave=False):
            optimizer.zero_grad()
            batch_loss = None

            for example in batch:
                features = torch.tensor(example.features, dtype=torch.float32, device=device)  # type: ignore[union-attr]
                adj_sparse = _to_torch_sparse_adjacency(example.contiguous_graph, device)
                output = model(features, adj_sparse, moment=config.moment, device=str(device))
                loss = _hgs_unsupervised_loss(output, adj_sparse, config.penalty_coeff, device)
                batch_loss = loss if batch_loss is None else batch_loss + loss

            batch_loss = batch_loss / max(len(batch), 1)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore[union-attr]
            optimizer.step()
            losses.append(float(batch_loss.detach().cpu().item()))

        metrics.append(
            {
                "epoch": epoch + 1,
                "mean_loss": float(np.mean(losses)) if losses else np.nan,
                "num_batches": len(batches),
                "num_examples": len(examples),
            }
        )

    return metrics


def _infer_model(
    model: Any,
    examples: list[GraphExample],
    config: HGSConfig,
    device: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():  # type: ignore[union-attr]
        for example in tqdm(examples, desc="HGS inference"):
            row = example.row
            started = time.perf_counter()
            try:
                features = torch.tensor(example.features, dtype=torch.float32, device=device)  # type: ignore[union-attr]
                adj_sparse = _to_torch_sparse_adjacency(example.contiguous_graph, device)
                output = model(features, adj_sparse, moment=config.moment, device=str(device))
                scores = output.detach().cpu().numpy().reshape(-1)

                contiguous_clique = greedy_decode_clique(
                    example.contiguous_graph,
                    scores,
                    num_walkers=config.num_walkers,
                    sample_length=config.sample_length,
                )
                clique_nodes = [example.original_nodes[int(node)] for node in contiguous_clique]
                runtime = time.perf_counter() - started
                valid = validate_clique(example.graph, clique_nodes)

                runs.append(
                    {
                        "dataset": row.get("dataset", config.dataset),
                        "graph_id": row.get("graph_id"),
                        "source_index": row.get("source_index"),
                        "solver_name": SOLVER_NAME,
                        "run_type": "infer",
                        "time_limit_seconds": config.time_limit_seconds,
                        "status": "ok",
                        "optimality_status": "heuristic",
                        "runtime_seconds": runtime,
                        "best_clique_size": len(clique_nodes),
                        "best_clique_nodes": json.dumps(clique_nodes),
                        "clique_valid": bool(valid),
                        "num_nodes": example.graph.number_of_nodes(),
                        "num_edges": example.graph.number_of_edges(),
                        "seed": config.seed,
                        "threads": None,
                        "mip_gap": None,
                        "objective_bound": None,
                        "error_message": "",
                    }
                )
            except Exception as exc:
                errors.append(_error_row(row, run_type="infer", exc=exc))

    return runs, errors


def _error_row(row: dict[str, Any], run_type: str, exc: Exception) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset"),
        "graph_id": row.get("graph_id"),
        "source_index": row.get("source_index"),
        "solver_name": SOLVER_NAME,
        "run_type": run_type,
        "time_limit_seconds": None,
        "status": "error",
        "optimality_status": "heuristic",
        "runtime_seconds": None,
        "best_clique_size": None,
        "best_clique_nodes": "[]",
        "clique_valid": False,
        "num_nodes": row.get("n_nodes"),
        "num_edges": row.get("n_edges"),
        "seed": None,
        "threads": None,
        "mip_gap": None,
        "objective_bound": None,
        "error_message": repr(exc),
    }


def _write_outputs(
    config: HGSConfig,
    runs: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    train_metrics: list[dict[str, Any]],
    checkpoint_path: Path,
) -> None:
    runs_df = pd.DataFrame(runs, columns=SOLVER_COLUMNS)
    errors_df = pd.DataFrame(errors, columns=SOLVER_COLUMNS)
    metrics_df = pd.DataFrame(train_metrics)

    runs_df.to_csv(config.output_dir / "solver_runs.csv", index=False)
    errors_df.to_csv(config.output_dir / "solver_errors.csv", index=False)
    metrics_df.to_csv(config.output_dir / "hgs_train_metrics.csv", index=False)

    summary = {
        "solver_name": SOLVER_NAME,
        "mode": config.mode,
        "dataset": config.dataset,
        "num_runs": int(len(runs_df)),
        "num_errors": int(len(errors_df)),
        "num_train_epochs": int(config.epochs if config.mode in {"train", "train-and-infer"} else 0),
        "checkpoint_path": str(checkpoint_path),
        "mean_runtime_seconds": float(runs_df["runtime_seconds"].mean()) if not runs_df.empty else None,
        "mean_best_clique_size": float(runs_df["best_clique_size"].mean()) if not runs_df.empty else None,
        "num_invalid_cliques": int((runs_df["clique_valid"] == False).sum()) if not runs_df.empty else 0,
    }
    pd.DataFrame([summary]).to_csv(config.output_dir / "run_summary.csv", index=False)
