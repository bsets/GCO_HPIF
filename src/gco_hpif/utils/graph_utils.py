from __future__ import annotations

import hashlib
from typing import Iterable

import networkx as nx


DATASET_ORDER = {
    "twitter": 0,
    "collab": 1,
    "imdb_binary": 2,
}


def canonicalize_simple_undirected(graph: nx.Graph | nx.DiGraph) -> nx.Graph:
    """Return a simple undirected graph with integer node labels starting at 0."""
    G = nx.Graph(graph)
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def graph_sha256(G: nx.Graph) -> str:
    """Stable hash based on sorted undirected edge list and node count."""
    edges = sorted((min(u, v), max(u, v)) for u, v in G.edges())
    payload = f"n={G.number_of_nodes()}|m={G.number_of_edges()}|edges={edges}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_graph_id(dataset: str, source_index: int) -> str:
    return f"{dataset}_graph{source_index:06d}"
