from pathlib import Path

import pandas as pd

from gco_hpif.data.splits import (
    infer_split_counts,
    load_dataset_rows_from_graph_index,
    make_ordered_split_manifest,
    build_twitter_split_manifest,
)


def _fake_graph_index() -> pd.DataFrame:
    twitter = pd.DataFrame(
        {
            "dataset": ["twitter"] * 10,
            "graph_id": [f"twitter_graph{i:06d}" for i in range(1, 11)],
            "source_index": list(range(1, 11)),
            "n_nodes": [100 + i for i in range(10)],
            "n_edges": [200 + i for i in range(10)],
            "graph_hash_sha256": [f"hash{i}" for i in range(10)],
            "source_loader": ["snap_twitter"] * 10,
            "source_name": ["ego-Twitter"] * 10,
            "source_ego_id": list(range(1000, 1010)),
        }
    )

    collab = pd.DataFrame(
        {
            "dataset": ["collab"] * 2,
            "graph_id": ["collab_graph000001", "collab_graph000002"],
            "source_index": [1, 2],
            "n_nodes": [10, 12],
            "n_edges": [20, 24],
            "graph_hash_sha256": ["c1", "c2"],
            "source_loader": ["tu"] * 2,
            "source_name": ["COLLAB"] * 2,
            "source_ego_id": [None, None],
        }
    )

    return pd.concat([collab, twitter], ignore_index=True)


def test_infer_split_counts_default_rounding_for_973_graphs():
    assert infer_split_counts(973) == (584, 195, 194)


def test_load_dataset_rows_from_graph_index_filters_and_orders(tmp_path: Path):
    csv_path = tmp_path / "graphs_index.csv"

    _fake_graph_index().iloc[::-1].to_csv(csv_path, index=False)

    rows = load_dataset_rows_from_graph_index(csv_path, dataset="twitter")

    assert len(rows) == 10
    assert rows["dataset"].str.lower().eq("twitter").all()
    assert rows["source_index"].tolist() == list(range(1, 11))


def test_make_ordered_split_manifest_with_explicit_counts():
    rows = _fake_graph_index()
    rows = rows[rows["dataset"] == "twitter"].sort_values("source_index")

    manifest, summary = make_ordered_split_manifest(
        rows,
        train_count=6,
        validation_count=2,
    )

    assert summary.total_graphs == 10
    assert summary.train_count == 6
    assert summary.validation_count == 2
    assert summary.test_count == 2
    assert manifest["split"].tolist() == ["train"] * 6 + ["validation"] * 2 + ["test"] * 2


def test_build_twitter_split_manifest_writes_csv(tmp_path: Path):
    graph_index_path = tmp_path / "graphs_index.csv"
    output_path = tmp_path / "twitter_split.csv"

    _fake_graph_index().to_csv(graph_index_path, index=False)

    written_path, summary = build_twitter_split_manifest(
        graph_index_path,
        output_path,
        train_count=6,
        validation_count=2,
    )

    assert written_path == output_path
    assert output_path.exists()
    assert summary.as_dict() == {
        "dataset": "twitter",
        "total_graphs": 10,
        "train": 6,
        "validation": 2,
        "test": 2,
    }

    df = pd.read_csv(output_path)
    assert len(df) == 10
    assert df.groupby("split").size().to_dict() == {
        "test": 2,
        "train": 6,
        "validation": 2,
    }
