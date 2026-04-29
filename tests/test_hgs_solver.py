import networkx as nx

from gco_hpif.solvers.hgs_solver import greedy_decode_clique, normalise_networkx_graph, validate_clique


def test_validate_clique_accepts_complete_subgraph():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])

    assert validate_clique(graph, [0, 1, 2])
    assert not validate_clique(graph, [0, 1, 2, 3])


def test_greedy_decode_clique_returns_valid_high_scoring_clique():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
    scores = [0.95, 0.90, 0.85, 0.80]

    clique = greedy_decode_clique(graph, scores, num_walkers=2, sample_length=4)

    assert set(clique) == {0, 1, 2}
    assert validate_clique(graph, clique)


def test_normalise_networkx_graph_from_edge_index_dict():
    record = {"edge_index": [[0, 1, 2], [1, 2, 0]], "num_nodes": 4}
    graph = normalise_networkx_graph(record)

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3
    assert validate_clique(graph, [0, 1, 2])
