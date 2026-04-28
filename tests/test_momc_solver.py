import gzip
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

from gco_hpif.cli.run_momc_solver import main as run_momc_main
from gco_hpif.solvers.common import SOLVER_RUN_COLUMNS
from gco_hpif.solvers.momc_solver import (
    build_momc_executable,
    graph_to_dimacs_text,
    parse_momc_stdout,
    solve_momc_max_clique,
)


def _write_fake_momc_executable(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "echo 'c Hello! I am MoMC (published in May 2019) build at test.'",
                "echo \"c reading $1 ...\"",
                "echo 'c Initial clique size: 2'",
                "echo 'M 2 1'",
                "echo \"s Instance $1 Max_CLQ 2 Branching 4 Time 0.01000000 ProveBranching 0 ProveTime 0.00000000\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path


def test_graph_to_dimacs_text_uses_one_based_node_ids():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 2)

    dimacs = graph_to_dimacs_text(G, comment="unit test")

    assert "p edge 3 1" in dimacs
    assert "e 1 3" in dimacs


def test_parse_momc_stdout_extracts_solver_output():
    stdout = """
    c Hello! I am MoMC (published in May 2019) build at test.
    c reading path3.clq ...
    c Initial clique size: 2
    M 2 1
    s Instance path3.clq Max_CLQ 2 Branching 4 Time 0.01000000 ProveBranching 0 ProveTime 0.00000000
    """

    parsed = parse_momc_stdout(stdout)

    assert parsed.instance_path == "path3.clq"
    assert parsed.max_clique_size == 2
    assert parsed.clique_nodes_reported == [2, 1]
    assert parsed.branching_count == 4
    assert parsed.solver_runtime_seconds == 0.01
    assert parsed.prove_runtime_seconds == 0.0
    assert parsed.timed_out is False


def test_parse_momc_stdout_detects_timeout_best_found():
    stdout = """
    c Timeout reached. Saving best solution found so far.
    M 5 4 3
    s Instance hard.clq Max_CLQ 3 Branching 12 Time 1800.00000000 ProveBranching 0 ProveTime 0.00000000
    """

    parsed = parse_momc_stdout(stdout)

    assert parsed.max_clique_size == 3
    assert parsed.clique_nodes_reported == [5, 4, 3]
    assert parsed.timed_out is True


def test_solve_momc_max_clique_with_fake_executable(tmp_path):
    fake_momc = _write_fake_momc_executable(tmp_path / "MoMC")
    dimacs_path = tmp_path / "graph.clq"

    result = solve_momc_max_clique(
        nx.path_graph(3),
        momc_executable=fake_momc,
        dimacs_path=dimacs_path,
        python_timeout_seconds=30,
        momc_output_node_base=1,
    )

    assert result.status == "success"
    assert result.optimality_status == "reported_by_momc"
    assert result.best_clique_size == 2
    assert result.best_clique_nodes == [1, 0]
    assert result.clique_valid is True
    assert dimacs_path.exists()



def test_solve_momc_uses_wall_clock_when_reported_time_is_zero(tmp_path):
    fake_momc = tmp_path / "MoMC"
    fake_momc.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "sleep 0.02",
                "echo 'M 2 1'",
                "echo \"s Instance $1 Max_CLQ 2 Branching 4 Time 0.00000000 ProveBranching 0 ProveTime 0.00000000\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_momc.chmod(fake_momc.stat().st_mode | 0o111)

    result = solve_momc_max_clique(
        nx.path_graph(3),
        momc_executable=fake_momc,
        dimacs_path=tmp_path / "graph.clq",
        python_timeout_seconds=30,
        momc_output_node_base=1,
    )

    assert result.status == "success"
    assert result.runtime_seconds > 0.0
    assert result.runtime_seconds != 0.0


def test_solve_momc_can_accept_zero_based_stdout_when_requested(tmp_path):
    fake_momc = tmp_path / "MoMC"
    fake_momc.write_text(
        "#!/bin/sh\necho 'M 1 0'\necho \"s Instance $1 Max_CLQ 2 Branching 4 Time 0.01000000 ProveBranching 0 ProveTime 0.00000000\"\n",
        encoding="utf-8",
    )
    fake_momc.chmod(fake_momc.stat().st_mode | 0o111)

    result = solve_momc_max_clique(
        nx.path_graph(3),
        momc_executable=fake_momc,
        dimacs_path=tmp_path / "graph.clq",
        python_timeout_seconds=30,
        momc_output_node_base=0,
    )

    assert result.best_clique_nodes == [1, 0]
    assert result.clique_valid is True


def test_build_momc_executable_uses_compiler_flags_on_tiny_fixture(tmp_path):
    source = tmp_path / "tiny_momc.c"
    source.write_text(
        '#include <stdio.h>\nint main(int argc, char **argv) { printf("M 2 1\\n"); return 0; }\n',
        encoding="utf-8",
    )
    executable = build_momc_executable(
        source_path=source,
        output_executable=tmp_path / "MoMC",
        cflags="-O0 -DMOMC",
        compile_timeout_seconds=30,
    )

    assert executable.exists()
    assert executable.stat().st_mode & 0o111


def test_run_momc_cli_writes_common_solver_outputs(tmp_path):
    fake_momc = _write_fake_momc_executable(tmp_path / "MoMC")

    interim = tmp_path / "interim"
    interim.mkdir()
    twitter_graphs = [
        {
            "dataset": "twitter",
            "graph_id": "twitter_graph000001",
            "source_index": 1,
            "graph": nx.path_graph(3),
        }
    ]
    with gzip.open(interim / "twitter_graphs.pkl.gz", "wb") as f:
        pickle.dump(twitter_graphs, f)

    graph_index = pd.DataFrame(
        [
            {
                "dataset": "twitter",
                "graph_id": "twitter_graph000001",
                "source_index": 1,
                "n_nodes": 3,
                "n_edges": 2,
            }
        ]
    )
    features = pd.DataFrame(
        [
            {
                "dataset": "twitter",
                "graph_id": "twitter_graph000001",
                "source_index": 1,
                "feature_Number_of_Nodes": 3,
            }
        ]
    )
    graphs_index_path = tmp_path / "graphs_index.csv"
    features_path = tmp_path / "graph_features.csv"
    graph_index.to_csv(graphs_index_path, index=False)
    features.to_csv(features_path, index=False)

    output_dir = tmp_path / "momc_run"
    run_momc_main(
        [
            "--graphs-index",
            str(graphs_index_path),
            "--features",
            str(features_path),
            "--interim-dir",
            str(interim),
            "--output-dir",
            str(output_dir),
            "--momc-executable",
            str(fake_momc),
            "--no-compile",
            "--datasets",
            "twitter",
            "--time-limit-seconds",
            "30",
            "--threads",
            "1",
        ]
    )

    runs = pd.read_csv(output_dir / "solver_runs.csv")
    errors = pd.read_csv(output_dir / "solver_errors.csv")
    summary = pd.read_csv(output_dir / "run_summary.csv")

    assert list(runs.columns) == SOLVER_RUN_COLUMNS
    assert len(runs) == 1
    assert runs.loc[0, "solver_name"] == "momc"
    assert runs.loc[0, "status"] == "success"
    assert runs.loc[0, "best_clique_size"] == 2
    assert runs.loc[0, "best_clique_nodes"] == "[1,0]"
    assert errors.empty
    assert summary.loc[0, "solver_name"] == "momc"
    assert summary.loc[0, "status"] == "success"
    assert (output_dir / "dimacs" / "twitter" / "twitter_graph000001.clq").exists()
    assert (output_dir / "raw_momc_outputs" / "twitter" / "twitter_graph000001.stdout.txt").exists()
    assert "Graph: twitter_graph000001" in (output_dir / "twitter_dataset_MOMC_results.txt").read_text()
