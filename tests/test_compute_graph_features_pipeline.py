import gzip
import pickle

import networkx as nx
import pandas as pd

from gco_hpif.features.networkx_features import FAILURE_COLUMNS, compute_features_from_part_a_artifacts


def test_compute_features_from_part_a_artifacts_smoke(tmp_path):
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "features"
    manifest_dir = tmp_path / "manifests"
    interim_dir.mkdir()
    manifest_dir.mkdir()

    records = [
        {"dataset": "twitter", "graph_id": "twitter_graph000001", "source_index": 1, "graph": nx.path_graph(3)},
        {"dataset": "twitter", "graph_id": "twitter_graph000002", "source_index": 2, "graph": nx.cycle_graph(4)},
    ]
    with gzip.open(interim_dir / "twitter_graphs.pkl.gz", "wb") as f:
        pickle.dump(records, f)

    pd.DataFrame([
        {
            "dataset": "twitter",
            "graph_id": "twitter_graph000001",
            "source_index": 1,
            "n_nodes": 3,
            "n_edges": 2,
            "graph_hash_sha256": "hash1",
            "source_loader": "unit",
            "source_name": "unit",
            "graph_label": None,
            "source_ego_id": 123,
        },
        {
            "dataset": "twitter",
            "graph_id": "twitter_graph000002",
            "source_index": 2,
            "n_nodes": 4,
            "n_edges": 4,
            "graph_hash_sha256": "hash2",
            "source_loader": "unit",
            "source_name": "unit",
            "graph_label": None,
            "source_ego_id": 456,
        },
    ]).to_csv(manifest_dir / "graphs_index.csv", index=False)

    result = compute_features_from_part_a_artifacts(
        graphs_index_csv=manifest_dir / "graphs_index.csv",
        interim_dir=interim_dir,
        output_dir=output_dir,
        datasets=["twitter"],
        timeout_seconds=10,
        show_progress=False,
    )

    features = pd.read_csv(result.features_path)
    failures = pd.read_csv(result.failures_path)
    summary = pd.read_csv(result.summary_path)
    feature_manifest = pd.read_csv(result.feature_manifest_path)

    assert len(features) == 2
    assert len(failures) == 0
    assert list(failures.columns) == FAILURE_COLUMNS
    assert summary.loc[0, "successful_graphs"] == 2
    assert summary.loc[0, "timeout_failures"] == 0
    assert len(feature_manifest) == 23
    assert "feature_definition" in feature_manifest.columns
    assert "feature_Number_of_Nodes" in features.columns
    adjacency_defs = feature_manifest.set_index("feature_column")["feature_definition"].to_dict()
    assert "Algebraically smallest eigenvalue" in adjacency_defs["feature_Smallest_Eigenvalue_Adjacency"]
    assert "Algebraically second smallest eigenvalue" in adjacency_defs["feature_Second_Smallest_Eigenvalue_Adjacency"]
