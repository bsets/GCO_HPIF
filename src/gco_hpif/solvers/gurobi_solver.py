"""Gurobi wrapper for the maximum clique problem.

The formulation solves maximum clique in G by solving maximum independent set
in the complement graph. For every edge (u, v) in the complement graph, the
constraint x_u + x_v <= 1 prevents selecting two non-adjacent vertices from
the original graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import networkx as nx

from gco_hpif.solvers.common import is_valid_clique


@dataclass(frozen=True)
class GurobiCliqueResult:
    """Result returned by the Gurobi maximum-clique wrapper."""

    status: str
    optimality_status: str
    runtime_seconds: float
    best_clique_size: int | None
    best_clique_nodes: list[int]
    clique_valid: bool
    mip_gap: float | None
    objective_bound: float | None
    error_message: str = ""


def _import_gurobi():
    """Import Gurobi lazily so the package can be installed without Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(
            "gurobipy is not installed or Gurobi is not available in this environment. "
            "Install/configure Gurobi before running the Gurobi solver wrapper."
        ) from exc
    return gp, GRB


def _status_name(status_code: int, GRB: Any) -> str:
    """Convert a Gurobi status code to a readable lowercase status name."""
    status_map = {
        getattr(GRB, "OPTIMAL", None): "optimal",
        getattr(GRB, "TIME_LIMIT", None): "time_limit",
        getattr(GRB, "INFEASIBLE", None): "infeasible",
        getattr(GRB, "INF_OR_UNBD", None): "inf_or_unbd",
        getattr(GRB, "UNBOUNDED", None): "unbounded",
        getattr(GRB, "INTERRUPTED", None): "interrupted",
        getattr(GRB, "NUMERIC", None): "numeric",
        getattr(GRB, "SUBOPTIMAL", None): "suboptimal",
    }
    return status_map.get(status_code, f"status_{status_code}")


def _pipeline_status(optimality_status: str) -> str:
    """Map Gurobi's status to the common pipeline status."""
    if optimality_status == "optimal":
        return "success"
    if optimality_status == "time_limit":
        return "timeout"
    return "error"


def solve_gurobi_max_clique(
    graph: nx.Graph,
    time_limit_seconds: float = 1800,
    seed: int = 2026,
    threads: int | None = None,
    output_flag: bool = False,
) -> GurobiCliqueResult:
    """Solve maximum clique with Gurobi."""
    gp, GRB = _import_gurobi()

    G = nx.Graph(graph)
    G.remove_edges_from(nx.selfloop_edges(G))
    nodes = list(G.nodes())

    if len(nodes) == 0:
        return GurobiCliqueResult(
            status="success",
            optimality_status="optimal",
            runtime_seconds=0.0,
            best_clique_size=0,
            best_clique_nodes=[],
            clique_valid=True,
            mip_gap=0.0,
            objective_bound=0.0,
        )

    complement = nx.complement(G)

    model = gp.Model("gco_hpif_max_clique")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.TimeLimit = float(time_limit_seconds)
    model.Params.Seed = int(seed)
    if threads is not None:
        model.Params.Threads = int(threads)

    x = {node: model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i, node in enumerate(nodes)}

    for constraint_index, (u, v) in enumerate(complement.edges()):
        model.addConstr(x[u] + x[v] <= 1, name=f"non_edge_{constraint_index}")

    model.setObjective(gp.quicksum(x[node] for node in nodes), GRB.MAXIMIZE)
    model.optimize()

    optimality_status = _status_name(model.Status, GRB)
    status = _pipeline_status(optimality_status)

    has_solution = getattr(model, "SolCount", 0) > 0
    if has_solution:
        selected_nodes = [node for node in nodes if x[node].X > 0.5]
        best_clique_size = len(selected_nodes)
    else:
        selected_nodes = []
        best_clique_size = None

    clique_valid = is_valid_clique(G, selected_nodes) if has_solution else False

    try:
        mip_gap = float(model.MIPGap) if has_solution else None
        if math.isinf(mip_gap) or math.isnan(mip_gap):
            mip_gap = None
    except Exception:
        mip_gap = None

    try:
        objective_bound = float(model.ObjBound)
        if math.isinf(objective_bound) or math.isnan(objective_bound):
            objective_bound = None
    except Exception:
        objective_bound = None

    return GurobiCliqueResult(
        status=status,
        optimality_status=optimality_status,
        runtime_seconds=float(getattr(model, "Runtime", 0.0)),
        best_clique_size=best_clique_size,
        best_clique_nodes=list(selected_nodes),
        clique_valid=bool(clique_valid),
        mip_gap=mip_gap,
        objective_bound=objective_bound,
    )
