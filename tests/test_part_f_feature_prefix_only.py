from pathlib import Path

import pandas as pd

from gco_hpif.ml.hardness_classification import build_ml_dataset


def test_build_ml_dataset_uses_only_feature_prefix_columns(tmp_path: Path) -> None:
    features_csv = tmp_path / "graph_features.csv"
    labels_csv = tmp_path / "hardness_labels.csv"

    features = pd.DataFrame(
        {
            "dataset": ["twitter", "twitter", "collab", "collab"],
            "graph_id": ["twitter_graph000001", "twitter_graph000002", "collab_graph000001", "collab_graph000002"],
            "source_index": [1, 2, 1, 2],
            "feature_Number_of_Nodes": [10, 20, 30, 40],
            "feature_Number_of_Edges": [15, 30, 45, 60],
            "n_nodes": [10, 20, 30, 40],
            "n_edges": [15, 30, 45, 60],
            "Time_feature_Number_of_Nodes": [0.01, 0.02, 0.03, 0.04],
        }
    )

    labels = pd.DataFrame(
        {
            "dataset": ["twitter", "twitter", "collab", "collab"],
            "graph_id": ["twitter_graph000001", "twitter_graph000002", "collab_graph000001", "collab_graph000002"],
            "source_index": [1, 2, 1, 2],
            "consensus5_runtime_hardness_label_binary": [0, 1, 0, 1],
        }
    )

    features.to_csv(features_csv, index=False)
    labels.to_csv(labels_csv, index=False)

    _, feature_cols = build_ml_dataset(
        features_csv=features_csv,
        labels_csv=labels_csv,
        target_cols=["consensus5_runtime_hardness_label_binary"],
    )

    assert feature_cols == ["feature_Number_of_Nodes", "feature_Number_of_Edges"]
    assert "n_nodes" not in feature_cols
    assert "n_edges" not in feature_cols
    assert "Time_feature_Number_of_Nodes" not in feature_cols
