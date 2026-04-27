import networkx as nx

from gco_hpif.utils.graph_utils import canonicalize_simple_undirected, make_graph_id


def test_canonicalize_removes_self_loops_and_relabels():
    G = nx.Graph()
    G.add_nodes_from([10, 20, 30])
    G.add_edge(10, 10)
    G.add_edge(10, 20)
    H = canonicalize_simple_undirected(G)
    assert sorted(H.nodes()) == [0, 1, 2]
    assert list(nx.selfloop_edges(H)) == []
    assert H.number_of_edges() == 1


def test_make_graph_id():
    assert make_graph_id("twitter", 7) == "twitter_graph000007"
