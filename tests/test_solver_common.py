import gzip
import pickle

import networkx as nx
import pandas as pd

from gco_hpif.solvers.common import (
    candidate_graph_metadata,
    extract_networkx_graph,
    is_valid_clique,
    iter_candidate_graphs,
    write_solver_outputs,
)


def test_is_valid_clique():
    G = nx.path_graph(4)
    assert is_valid_clique(G, [0, 1])
    assert is_valid_clique(G, [1, 2])
    assert not is_valid_clique(G, [0, 2])
    assert not is_valid_clique(G, [0, 0])
    assert not is_valid_clique(G, [99])


def test_candidate_graph_metadata_uses_feature_success_intersection(tmp_path):
    graph_index = pd.DataFrame(
        [
            {"dataset": "twitter", "graph_id": "twitter_graph000001", "source_index": 1, "n_nodes": 3, "n_edges": 2},
            {"dataset": "twitter", "graph_id": "twitter_graph000002", "source_index": 2, "n_nodes": 4, "n_edges": 3},
            {"dataset": "collab", "graph_id": "collab_graph000001", "source_index": 1, "n_nodes": 5, "n_edges": 4},
        ]
    )
    features = pd.DataFrame(
        [
            {"dataset": "twitter", "graph_id": "twitter_graph000002", "source_index": 2, "feature_Number_of_Nodes": 4},
            {"dataset": "collab", "graph_id": "collab_graph000001", "source_index": 1, "feature_Number_of_Nodes": 5},
        ]
    )

    graph_index_path = tmp_path / "graphs_index.csv"
    features_path = tmp_path / "graph_features.csv"
    graph_index.to_csv(graph_index_path, index=False)
    features.to_csv(features_path, index=False)

    selected = candidate_graph_metadata(graph_index_path, features_path)
    assert selected["graph_id"].tolist() == ["twitter_graph000002", "collab_graph000001"]


def test_iter_candidate_graphs_loads_from_part_a_pickles(tmp_path):
    interim = tmp_path / "interim"
    interim.mkdir()

    twitter_graphs = [
        {
            "dataset": "twitter",
            "graph_id": "twitter_graph000001",
            "source_index": 1,
            "graph": nx.path_graph(3),
        },
        {
            "dataset": "twitter",
            "graph_id": "twitter_graph000002",
            "source_index": 2,
            "graph": nx.complete_graph(4),
        },
    ]
    with gzip.open(interim / "twitter_graphs.pkl.gz", "wb") as f:
        pickle.dump(twitter_graphs, f)

    graph_index = pd.DataFrame(
        [
            {"dataset": "twitter", "graph_id": "twitter_graph000001", "source_index": 1, "n_nodes": 3, "n_edges": 2},
            {"dataset": "twitter", "graph_id": "twitter_graph000002", "source_index": 2, "n_nodes": 4, "n_edges": 6},
        ]
    )
    features = pd.DataFrame(
        [
            {"dataset": "twitter", "graph_id": "twitter_graph000002", "source_index": 2, "feature_Number_of_Nodes": 4},
        ]
    )

    graph_index_path = tmp_path / "graphs_index.csv"
    features_path = tmp_path / "graph_features.csv"
    graph_index.to_csv(graph_index_path, index=False)
    features.to_csv(features_path, index=False)

    instances = list(
        iter_candidate_graphs(
            graphs_index_csv=graph_index_path,
            features_csv=features_path,
            interim_dir=interim,
            datasets=["twitter"],
        )
    )

    assert len(instances) == 1
    assert instances[0].graph_id == "twitter_graph000002"
    assert instances[0].graph.number_of_edges() == 6


def test_extract_networkx_graph_supports_raw_graph_and_part_a_record():
    raw = nx.path_graph(3)
    raw.add_edge(0, 0)

    extracted_raw = extract_networkx_graph(raw, graph_id="raw")
    assert isinstance(extracted_raw, nx.Graph)
    assert extracted_raw.number_of_nodes() == 3
    assert not list(nx.selfloop_edges(extracted_raw))

    record = {
        "dataset": "twitter",
        "graph_id": "twitter_graph000001",
        "source_index": 1,
        "graph": nx.complete_graph(4),
    }
    extracted_record = extract_networkx_graph(record, graph_id=record["graph_id"])
    assert isinstance(extracted_record, nx.Graph)
    assert extracted_record.number_of_nodes() == 4
    assert extracted_record.number_of_edges() == 6


def test_write_solver_outputs_writes_headers_for_empty_results(tmp_path):
    runs_file, errors_file, summary_file = write_solver_outputs([], tmp_path, "gurobi")

    runs = pd.read_csv(runs_file)
    errors = pd.read_csv(errors_file)
    summary = pd.read_csv(summary_file)

    assert list(runs.columns)[0:3] == ["dataset", "graph_id", "source_index"]
    assert list(errors.columns)[0:3] == ["dataset", "graph_id", "source_index"]
    assert list(summary.columns) == ["solver_name", "dataset", "status", "count"]
