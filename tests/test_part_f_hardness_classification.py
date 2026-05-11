from pathlib import Path

import pandas as pd

from gco_hpif.ml.hardness_classification import (
    build_ml_dataset,
    choose_first_peak_k,
)


def test_choose_first_peak_k_detects_first_plateau():
    scores = [0.10, 0.18, 0.24, 0.241, 0.240, 0.27]
    assert choose_first_peak_k(scores, min_delta=0.002, patience=2, min_k=2) == 3


def test_choose_first_peak_k_falls_back_to_global_best():
    scores = [0.10, 0.18, 0.24, 0.25, 0.27]
    assert choose_first_peak_k(scores, min_delta=0.002, patience=2, min_k=2) == 5


def test_build_ml_dataset_uses_graph_features_only(tmp_path: Path):
    """Part F must use only canonical feature_* graph-feature columns.

    Metadata columns such as n_nodes/n_edges/source_index, timing columns such as
    Time_feature_*, and non-prefixed numeric helper columns must not be used as
    ML predictors. Constant feature_* columns should also be dropped.
    """

    features = pd.DataFrame(
        {
            "dataset": ["twitter", "twitter", "collab", "collab"],
            "graph_id": [
                "twitter_graph000001",
                "twitter_graph000002",
                "collab_graph000001",
                "collab_graph000002",
            ],
            "source_index": [1, 2, 1, 2],
            # Non-feature metadata/helper columns: must be excluded.
            "num_nodes": [10, 20, 30, 40],
            "n_nodes": [10, 20, 30, 40],
            "n_edges": [11, 21, 31, 41],
            "density": [0.1, 0.2, 0.3, 0.4],
            "Time_feature_Number_of_Nodes": [0.01, 0.01, 0.02, 0.02],
            # True graph features: only names beginning with feature_ count.
            "feature_Number_of_Nodes": [10, 20, 30, 40],
            "feature_Density": [0.1, 0.2, 0.3, 0.4],
            # This begins with feature_, but should be dropped because constant.
            "feature_Constant": [1, 1, 1, 1],
        }
    )

    labels = pd.DataFrame(
        {
            "dataset": ["twitter", "twitter", "collab", "collab"],
            "graph_id": [
                "twitter_graph000001",
                "twitter_graph000002",
                "collab_graph000001",
                "collab_graph000002",
            ],
            "source_index": [1, 2, 1, 2],
            # Runtime columns from Part E must not be predictors.
            "runtime_gurobi": [0.1, 0.2, 0.3, 0.4],
            "consensus5_runtime_hardness_label_binary": [0, 1, 0, 1],
        }
    )

    features_csv = tmp_path / "graph_features.csv"
    labels_csv = tmp_path / "hardness_labels.csv"
    features.to_csv(features_csv, index=False)
    labels.to_csv(labels_csv, index=False)

    data, feature_cols = build_ml_dataset(
        features_csv,
        labels_csv,
        ["consensus5_runtime_hardness_label_binary"],
    )

    assert len(data) == 4
    assert feature_cols == ["feature_Number_of_Nodes", "feature_Density"]

    assert "feature_Constant" not in feature_cols
    assert "source_index" not in feature_cols
    assert "num_nodes" not in feature_cols
    assert "n_nodes" not in feature_cols
    assert "n_edges" not in feature_cols
    assert "density" not in feature_cols
    assert "Time_feature_Number_of_Nodes" not in feature_cols
    assert "runtime_gurobi" not in feature_cols

