import networkx as nx
import pytest


def test_gurobi_solver_on_small_graph_when_gurobi_available():
    pytest.importorskip("gurobipy")

    from gco_hpif.solvers.gurobi_solver import solve_gurobi_max_clique

    G = nx.path_graph(4)
    result = solve_gurobi_max_clique(G, time_limit_seconds=30, seed=2026, threads=1)

    assert result.status == "success"
    assert result.optimality_status == "optimal"
    assert result.best_clique_size == 2
    assert result.clique_valid is True
