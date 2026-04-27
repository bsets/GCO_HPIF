import math

import networkx as nx
import numpy as np

from gco_hpif.features.networkx_features import (
    FEATURE_COLUMNS,
    compute_feature_row,
    feature_median_neighbour_degree,
    feature_second_smallest_eigenvalue_adjacency,
    feature_smallest_eigenvalue_adjacency,
)


def test_feature_column_count_and_names():
    assert len(FEATURE_COLUMNS) == 23
    assert FEATURE_COLUMNS[0] == "feature_Number_of_Nodes"
    assert FEATURE_COLUMNS[-1] == "feature_Gap_Largest_Smallest_Laplacian"


def test_compute_feature_row_path_graph():
    G = nx.path_graph(3)
    row = compute_feature_row(G, metadata={"dataset": "unit", "graph_id": "unit_graph000001", "source_index": 1})

    assert row["dataset"] == "unit"
    assert row["graph_id"] == "unit_graph000001"
    assert row["feature_Number_of_Nodes"] == 3.0
    assert row["feature_Number_of_Edges"] == 2.0
    assert math.isclose(row["feature_Density"], 2 / 3)
    assert row["feature_Radius"] == 1.0
    assert row["feature_Diameter"] == 2.0
    assert row["feature_Median_Node_Eccentricity"] == 2.0
    assert row["total_feature_time_seconds"] >= 0.0

    # Every feature and every per-feature timing column should be present.
    for col in FEATURE_COLUMNS:
        assert col in row
        assert f"Time_{col}" in row


def test_disconnected_graph_features_use_nan_for_connected_only_features():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (2, 3)])
    row = compute_feature_row(G)

    assert math.isnan(row["feature_Radius"])
    assert math.isnan(row["feature_Diameter"])
    assert math.isnan(row["feature_Median_Node_Eccentricity"])
    assert math.isnan(row["feature_Median_Geodesic_Distance"])


def test_median_neighbour_degree_handles_isolates():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1)

    value = feature_median_neighbour_degree(G)
    assert math.isclose(value, 1.0)


def test_adjacency_smallest_eigenvalue_definitions_are_algebraic():
    G = nx.path_graph(3)

    # Path graph P3 has adjacency eigenvalues [-sqrt(2), 0, sqrt(2)].
    # The repository feature definitions use algebraic ascending order,
    # not the nonzero eigenvalue closest to zero.
    assert np.isclose(feature_smallest_eigenvalue_adjacency(G), -np.sqrt(2))
    assert np.isclose(feature_second_smallest_eigenvalue_adjacency(G), 0.0)

    row = compute_feature_row(G)
    assert np.isclose(row["feature_Smallest_Eigenvalue_Adjacency"], -np.sqrt(2))
    assert np.isclose(row["feature_Second_Smallest_Eigenvalue_Adjacency"], 0.0)
