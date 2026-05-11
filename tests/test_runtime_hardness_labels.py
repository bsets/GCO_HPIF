import pandas as pd

from gco_hpif.labels.hardness import build_runtime_hardness_labels


def _row(dataset, graph_id, source_index, solver, runtime, clique=10):
    return {
        "dataset": dataset,
        "graph_id": graph_id,
        "source_index": source_index,
        "solver_name": solver,
        "runtime_seconds": runtime,
        "best_clique_size": clique,
    }


def test_consensus5_detects_instance_top25_for_all_solvers():
    rows = []
    solvers = ["gurobi", "clisat", "momc", "egn", "hgs"]
    for i, graph_id in enumerate(["g1", "g2", "g3", "g4"], start=1):
        for solver in solvers:
            rows.append(_row("collab", graph_id, i, solver, runtime=float(i)))
    labels, incomplete, thresholds, coverage = build_runtime_hardness_labels(pd.DataFrame(rows))
    assert len(labels) == 4
    assert len(incomplete) == 0
    g4 = labels[labels["graph_id"] == "g4"].iloc[0]
    assert g4["consensus5_runtime_hardness_label"] == "Hard"
    assert g4["consensus5_runtime_hardness_label_binary"] == 1


def test_incomplete_graph_excluded_when_missing_solver():
    rows = []
    for solver in ["gurobi", "clisat", "momc", "egn", "hgs"]:
        rows.append(_row("collab", "g1", 1, solver, runtime=1.0))
    for solver in ["gurobi", "clisat", "momc", "egn"]:
        rows.append(_row("collab", "g2", 2, solver, runtime=2.0))
    labels, incomplete, thresholds, coverage = build_runtime_hardness_labels(pd.DataFrame(rows))
    assert set(labels["graph_id"]) == {"g1"}
    assert set(incomplete["graph_id"]) == {"g2"}


def test_consensus4_and_majority3_labels():
    solvers = ["gurobi", "clisat", "momc", "egn", "hgs"]
    rows = []
    for i, graph_id in enumerate(["g1", "g2", "g3", "g4"], start=1):
        for solver in solvers:
            runtime = float(i)
            if graph_id == "g4" and solver == "hgs":
                runtime = 0.1
            rows.append(_row("twitter", graph_id, i, solver, runtime=runtime))
    labels, incomplete, thresholds, coverage = build_runtime_hardness_labels(pd.DataFrame(rows))
    g4 = labels[labels["graph_id"] == "g4"].iloc[0]
    assert g4["num_algorithms_runtime_top25"] == 4
    assert g4["consensus5_runtime_hardness_label"] == "Not Hard"
    assert g4["consensus4_runtime_hardness_label"] == "Hard"
    assert g4["majority3_runtime_hardness_label"] == "Hard"
