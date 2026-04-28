import gzip
import pickle
import platform
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from gco_hpif.cli.run_clisat_solver import main as run_clisat_main
from gco_hpif.solvers.clisat_solver import (
    DEFAULT_CLISAT_EXECUTABLE,
    graph_to_dimacs_text,
    parse_clisat_stdout,
    solve_clisat_max_clique,
)
from gco_hpif.solvers.common import SOLVER_RUN_COLUMNS


def _write_fake_clisat_executable(path):
    path.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "echo 'CliSAT fake solver output'",
                "echo '1 2  [2]'",
                "echo 'omega:2'",
                "echo 'ts(s):0.1'",
                "echo 'tp(s):0.2'",
                "echo 'tr(s):0.3'",
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


def test_parse_clisat_stdout_extracts_actual_binary_style_output():
    stdout = """
    \x1b[1;32m*****************************
    1 2  [2]
    omega:2\tts(s):1.2\ttp(s):0.3\ttr(s):0.05\tsteps:0
    """

    parsed = parse_clisat_stdout(stdout)

    assert parsed.omega == 2
    assert parsed.clique_nodes_reported == [1, 2]
    assert parsed.solver_runtime_seconds == 1.55


def test_parse_clisat_stdout_accepts_older_star_style_output():
    stdout = """
    solver log line
    * 1 5 8 9 [4]
    omega:4
    ts(s):1.2
    tp(s):0.3
    tr(s):0.05
    """

    parsed = parse_clisat_stdout(stdout)

    assert parsed.omega == 4
    assert parsed.clique_nodes_reported == [1, 5, 8, 9]
    assert parsed.solver_runtime_seconds == 1.55


def test_solve_clisat_max_clique_with_fake_executable(tmp_path):
    fake_clisat = _write_fake_clisat_executable(tmp_path / "CliSAT")
    dimacs_path = tmp_path / "graph.clq"

    result = solve_clisat_max_clique(
        nx.path_graph(3),
        clisat_executable=fake_clisat,
        dimacs_path=dimacs_path,
        time_limit_seconds=30,
        threads=1,
        clisat_output_node_base=0,
    )

    assert result.status == "success"
    assert result.optimality_status == "reported_by_clisat"
    assert result.best_clique_size == 2
    assert result.best_clique_nodes == [1, 2]
    assert result.clique_valid is True
    assert dimacs_path.exists()


def test_solve_clisat_can_convert_one_based_stdout_when_requested(tmp_path):
    fake_clisat = tmp_path / "CliSAT"
    fake_clisat.write_text(
        "#!/bin/sh\necho '1 2 [2]'\necho 'omega:2'\necho 'ts(s):0'\necho 'tp(s):0'\necho 'tr(s):0'\n",
        encoding="utf-8",
    )
    fake_clisat.chmod(fake_clisat.stat().st_mode | 0o111)

    result = solve_clisat_max_clique(
        nx.path_graph(3),
        clisat_executable=fake_clisat,
        dimacs_path=tmp_path / "graph.clq",
        time_limit_seconds=30,
        threads=1,
        clisat_output_node_base=1,
    )

    assert result.best_clique_nodes == [0, 1]
    assert result.clique_valid is True


def test_bundled_clisat_binary_smoke_if_available(tmp_path):
    binary_path = Path(DEFAULT_CLISAT_EXECUTABLE)
    if not binary_path.exists():
        # When the test file is run from an installed package, look relative to repo root.
        binary_path = Path(__file__).resolve().parents[1] / DEFAULT_CLISAT_EXECUTABLE
    if platform.system() != "Linux" or not binary_path.exists():
        pytest.skip("Bundled CliSAT binary smoke test requires Linux and external/CliSAT/bin/CliSAT.")

    result = solve_clisat_max_clique(
        nx.path_graph(3),
        clisat_executable=binary_path,
        dimacs_path=tmp_path / "path_graph.clq",
        time_limit_seconds=30,
        threads=1,
        clisat_output_node_base=0,
    )

    assert result.status == "success"
    assert result.best_clique_size == 2
    assert result.best_clique_nodes in ([0, 1], [1, 2])
    assert result.clique_valid is True


def test_run_clisat_cli_writes_common_solver_outputs(tmp_path):
    fake_clisat = _write_fake_clisat_executable(tmp_path / "CliSAT")

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

    output_dir = tmp_path / "clisat_run"
    run_clisat_main(
        [
            "--graphs-index",
            str(graphs_index_path),
            "--features",
            str(features_path),
            "--interim-dir",
            str(interim),
            "--output-dir",
            str(output_dir),
            "--clisat-executable",
            str(fake_clisat),
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
    assert runs.loc[0, "solver_name"] == "clisat"
    assert runs.loc[0, "status"] == "success"
    assert runs.loc[0, "best_clique_size"] == 2
    assert runs.loc[0, "best_clique_nodes"] == "[1,2]"
    assert errors.empty
    assert summary.loc[0, "solver_name"] == "clisat"
    assert summary.loc[0, "status"] == "success"
    assert (output_dir / "dimacs" / "twitter" / "twitter_graph000001.clq").exists()
    assert (output_dir / "raw_clisat_outputs" / "twitter" / "twitter_graph000001.stdout.txt").exists()
    assert "Graph: twitter_graph000001" in (output_dir / "twitter_dataset_CliSAT_results.txt").read_text()
