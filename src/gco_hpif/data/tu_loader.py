from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import networkx as nx
from tqdm.auto import tqdm

from gco_hpif.data.common import download_file, ensure_dir, parse_int_pair, read_text_lines
from gco_hpif.utils.graph_utils import canonicalize_simple_undirected, graph_sha256, make_graph_id

TU_BASE_URL = "https://www.chrsmrrs.com/graphkerneldatasets"
TU_DATASET_MAP = {
    "imdb_binary": "IMDB-BINARY",
    "collab": "COLLAB",
}


def _extract_if_needed(zip_path: Path, extract_root: Path) -> Path:
    dataset_root = extract_root / zip_path.stem
    if dataset_root.exists():
        return dataset_root

    ensure_dir(extract_root)
    tqdm.write(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    return dataset_root


def _find_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return matches[0]


def load_tu_dataset(dataset_key: str, raw_root: Path, force_download: bool = False, limit: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_name = TU_DATASET_MAP[dataset_key]
    zip_path = raw_root / "tu" / f"{dataset_name}.zip"
    download_file(
        f"{TU_BASE_URL}/{dataset_name}.zip",
        zip_path,
        force=force_download,
        desc=f"Downloading {dataset_name}",
    )
    dataset_root = _extract_if_needed(zip_path, raw_root / "tu")

    a_path = _find_file(dataset_root, f"{dataset_name}_A.txt")
    indicator_path = _find_file(dataset_root, f"{dataset_name}_graph_indicator.txt")
    graph_label_matches = list(dataset_root.rglob(f"{dataset_name}_graph_labels.txt"))
    graph_labels = read_text_lines(graph_label_matches[0]) if graph_label_matches else None

    graph_indicator = [int(x) for x in read_text_lines(indicator_path)]
    num_graphs = max(graph_indicator)

    node_to_graph = {idx: gid for idx, gid in enumerate(graph_indicator, start=1)}
    graphs = [nx.Graph() for _ in range(num_graphs)]

    for node_id, graph_id in tqdm(node_to_graph.items(), desc=f"{dataset_key}: nodes", unit="node", leave=False):
        graphs[graph_id - 1].add_node(node_id)

    edge_lines = read_text_lines(a_path)
    for line in tqdm(edge_lines, desc=f"{dataset_key}: edges", unit="edge", leave=False):
        u, v = parse_int_pair(line)
        g_u = node_to_graph[u]
        g_v = node_to_graph[v]
        if g_u != g_v:
            raise ValueError(f"Edge spans multiple graphs in {dataset_name}: ({u}, {v})")
        graphs[g_u - 1].add_edge(u, v)

    records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    iterable = enumerate(graphs, start=1)
    if limit is not None:
        total = min(len(graphs), limit)
    else:
        total = len(graphs)

    for idx, raw_graph in tqdm(iterable, total=total, desc=f"{dataset_key}: graphs", unit="graph"):
        if limit is not None and idx > limit:
            break
        G = canonicalize_simple_undirected(raw_graph)
        graph_id = make_graph_id(dataset_key, idx)
        sha = graph_sha256(G)
        label = int(graph_labels[idx - 1]) if graph_labels is not None else None

        records.append({
            "graph_id": graph_id,
            "dataset": dataset_key,
            "source_index": idx,
            "graph": G,
        })
        manifest_rows.append({
            "dataset": dataset_key,
            "graph_id": graph_id,
            "source_index": idx,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "graph_hash_sha256": sha,
            "source_loader": "tu_manual_parser",
            "source_name": dataset_name,
            "graph_label": label,
        })

    return records, manifest_rows
