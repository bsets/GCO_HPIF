from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from gco_hpif.solvers.egn_solver import (
    ERROR_COLUMNS,
    SOLVER_RUN_COLUMNS,
    SPLIT_MANIFEST_COLUMNS,
    load_external_egn,
    read_split_manifest,
    validate_clique_nodes,
    write_solver_outputs,
)


def _write_split_manifest(path: Path) -> None:
    rows = []
    for i, split in enumerate(["train", "train", "validation", "test"]):
        rows.append(
            {
                "dataset": "twitter",
                "graph_id": f"twitter_{i}",
                "source_index": i,
                "split": split,
                "n_nodes": 4,
                "n_edges": 3,
                "graph_hash_sha256": f"hash-{i}",
                "source_loader": "snap_twitter",
                "source_name": f"{i}.edges",
                "source_ego_id": str(i),
            }
        )
    pd.DataFrame(rows, columns=SPLIT_MANIFEST_COLUMNS).to_csv(path, index=False)


def test_read_split_manifest_uses_fixed_part_b_schema(tmp_path: Path) -> None:
    split_path = tmp_path / "twitter_split_60_20_20.csv"
    _write_split_manifest(split_path)

    df = read_split_manifest(split_path, dataset="twitter", splits=("train", "test"))

    assert list(df["split"]) == ["train", "train", "test"]
    assert list(df["graph_id"]) == ["twitter_0", "twitter_1", "twitter_3"]


def test_read_split_manifest_limit_per_split(tmp_path: Path) -> None:
    split_path = tmp_path / "twitter_split_60_20_20.csv"
    _write_split_manifest(split_path)

    df = read_split_manifest(
        split_path,
        dataset="twitter",
        splits=("train", "validation", "test"),
        limit_per_split=1,
    )

    assert df.groupby("split").size().to_dict() == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


def test_validate_clique_nodes() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3)])

    assert validate_clique_nodes(graph, [0, 1, 2]) is True
    assert validate_clique_nodes(graph, [0, 1, 3]) is False
    assert validate_clique_nodes(graph, []) is True
    assert validate_clique_nodes(graph, [10]) is False


def test_write_solver_outputs_has_standard_columns(tmp_path: Path) -> None:
    runs = pd.DataFrame(
        [
            {
                "dataset": "twitter",
                "graph_id": "twitter_0",
                "source_index": 0,
                "solver_name": "EGN",
                "run_type": "infer",
                "time_limit_seconds": "",
                "status": "success",
                "optimality_status": "heuristic",
                "runtime_seconds": 0.1,
                "best_clique_size": 3,
                "best_clique_nodes": json.dumps([0, 1, 2]),
                "clique_valid": True,
                "num_nodes": 4,
                "num_edges": 3,
                "seed": 66,
                "threads": "",
                "mip_gap": "",
                "objective_bound": "",
                "error_message": "",
            }
        ]
    )

    errors = pd.DataFrame(columns=ERROR_COLUMNS)

    write_solver_outputs(tmp_path, runs, errors)

    written_runs = pd.read_csv(tmp_path / "solver_runs.csv")
    written_errors = pd.read_csv(tmp_path / "solver_errors.csv")
    summary = pd.read_csv(tmp_path / "run_summary.csv")

    assert list(written_runs.columns) == SOLVER_RUN_COLUMNS
    assert list(written_errors.columns) == ERROR_COLUMNS
    assert int(summary.loc[0, "num_success"]) == 1


def test_external_egn_missing_path_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        load_external_egn(tmp_path / "does_not_exist")
