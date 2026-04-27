from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any

import networkx as nx
from tqdm.auto import tqdm

from gco_hpif.data.common import download_file, parse_int_pair
from gco_hpif.utils.graph_utils import canonicalize_simple_undirected, graph_sha256, make_graph_id

TWITTER_URL = "https://snap.stanford.edu/data/twitter.tar.gz"


def load_twitter_ego_graphs(raw_root: Path, force_download: bool = False, limit: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tar_path = raw_root / "twitter" / "twitter.tar.gz"
    download_file(TWITTER_URL, tar_path, force=force_download, desc="Downloading Twitter ego graphs")

    records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    with tarfile.open(tar_path, "r:gz") as tf:
        edge_members = [m for m in tf.getmembers() if m.name.endswith(".edges")]
        edge_members.sort(key=lambda m: int(Path(m.name).stem))

        if limit is not None:
            edge_members = edge_members[:limit]

        for idx, member in enumerate(tqdm(edge_members, desc="twitter: ego graphs", unit="graph"), start=1):
            ego_id = Path(member.name).stem
            extracted = tf.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not extract member: {member.name}")

            G = nx.Graph()
            neighbor_nodes: set[int] = set()

            for raw_line in io.TextIOWrapper(extracted, encoding="utf-8"):
                line = raw_line.strip()
                if not line:
                    continue
                u, v = parse_int_pair(line)
                G.add_edge(u, v)
                neighbor_nodes.add(u)
                neighbor_nodes.add(v)

            # SNAP ego files omit the ego node itself; add it back and connect it
            # to every node appearing in the edge file.
            ego_node = int(ego_id)
            G.add_node(ego_node)
            for node in neighbor_nodes:
                if node != ego_node:
                    G.add_edge(ego_node, node)

            G = canonicalize_simple_undirected(G)
            graph_id = make_graph_id("twitter", idx)
            sha = graph_sha256(G)

            records.append({
                "graph_id": graph_id,
                "dataset": "twitter",
                "source_index": idx,
                "source_ego_id": ego_id,
                "graph": G,
            })
            manifest_rows.append({
                "dataset": "twitter",
                "graph_id": graph_id,
                "source_index": idx,
                "source_ego_id": ego_id,
                "n_nodes": G.number_of_nodes(),
                "n_edges": G.number_of_edges(),
                "graph_hash_sha256": sha,
                "source_loader": "snap_twitter_tar_parser",
                "source_name": "ego-Twitter",
                "graph_label": None,
            })

    return records, manifest_rows
