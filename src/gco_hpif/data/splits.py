"""Utilities for creating reproducible graph dataset split manifests.

Part B of the GCO-HPIF pipeline creates a committed train/validation/test
manifest for the TWITTER graphs produced by Part A. The manifest is small,
human-readable, and stable across runs, so downstream EGN/HGS training and
evaluation can use the exact same graph split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class SplitSummary:
    """Summary of a generated split manifest."""

    dataset: str
    total_graphs: int
    train_count: int
    validation_count: int
    test_count: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "dataset": self.dataset,
            "total_graphs": self.total_graphs,
            "train": self.train_count,
            "validation": self.validation_count,
            "test": self.test_count,
        }


def load_dataset_rows_from_graph_index(
    graphs_index_csv: str | Path,
    dataset: str = "twitter",
) -> pd.DataFrame:
    """Load ordered rows for one dataset from the Part A ``graphs_index.csv``."""

    graphs_index_csv = Path(graphs_index_csv)
    if not graphs_index_csv.exists():
        raise FileNotFoundError(f"Graph index file not found: {graphs_index_csv}")

    df = pd.read_csv(graphs_index_csv)

    required_cols = {"dataset", "graph_id", "source_index"}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(
            f"Graph index is missing required columns: {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )

    dataset_key = dataset.lower()
    out = df[df["dataset"].astype(str).str.lower() == dataset_key].copy()

    if out.empty:
        raise ValueError(f"No rows found for dataset={dataset!r} in {graphs_index_csv}")

    out["source_index"] = pd.to_numeric(out["source_index"], errors="raise").astype(int)
    out = out.sort_values("source_index", kind="mergesort").reset_index(drop=True)

    if out["graph_id"].duplicated().any():
        dupes = out.loc[out["graph_id"].duplicated(), "graph_id"].tolist()
        raise ValueError(f"Duplicate graph_id values found for {dataset}: {dupes[:5]}")

    return out


def infer_split_counts(
    total_graphs: int,
    train_count: Optional[int] = None,
    validation_count: Optional[int] = None,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
) -> tuple[int, int, int]:
    """Infer train/validation/test counts.

    If explicit ``train_count`` and/or ``validation_count`` are provided, they
    are used directly. Otherwise counts are computed by rounding the requested
    ratios to the nearest integer, and the test count receives the remainder.

    For the current Slice A TWITTER manifest with 973 graphs, the default
    60/20/remainder rule gives 584 train, 195 validation, and 194 test graphs.
    """

    if total_graphs <= 0:
        raise ValueError("total_graphs must be positive")

    if train_count is None:
        train_count = int(round(train_ratio * total_graphs))
    if validation_count is None:
        validation_count = int(round(validation_ratio * total_graphs))

    for name, value in {
        "train_count": train_count,
        "validation_count": validation_count,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative; got {value}")

    test_count = total_graphs - train_count - validation_count
    if test_count < 0:
        raise ValueError(
            "Requested split counts exceed total graphs: "
            f"train={train_count}, validation={validation_count}, "
            f"total={total_graphs}"
        )

    return train_count, validation_count, test_count


def make_ordered_split_manifest(
    dataset_rows: pd.DataFrame,
    dataset: str = "twitter",
    train_count: Optional[int] = None,
    validation_count: Optional[int] = None,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
) -> tuple[pd.DataFrame, SplitSummary]:
    """Create an ordered split manifest for one dataset.

    The first ``train_count`` rows become training graphs, the next
    ``validation_count`` rows become validation graphs, and the remainder become
    test graphs. The input rows should already be sorted by ``source_index``.
    """

    total = len(dataset_rows)
    n_train, n_val, n_test = infer_split_counts(
        total_graphs=total,
        train_count=train_count,
        validation_count=validation_count,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    split_labels = ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test

    preferred_cols = [
        "dataset",
        "graph_id",
        "source_index",
        "n_nodes",
        "n_edges",
        "graph_hash_sha256",
        "source_loader",
        "source_name",
        "source_ego_id",
    ]
    cols = [c for c in preferred_cols if c in dataset_rows.columns]

    manifest = dataset_rows.loc[:, cols].copy()
    insert_at = 3 if "source_index" in manifest.columns else len(manifest.columns)
    manifest.insert(insert_at, "split", split_labels)

    summary = SplitSummary(
        dataset=dataset,
        total_graphs=total,
        train_count=n_train,
        validation_count=n_val,
        test_count=n_test,
    )
    return manifest, summary


def write_split_manifest(manifest: pd.DataFrame, output_csv: str | Path) -> Path:
    """Write a split manifest CSV, creating parent folders if needed."""

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)
    return output_csv


def build_twitter_split_manifest(
    graphs_index_csv: str | Path,
    output_csv: str | Path,
    train_count: Optional[int] = None,
    validation_count: Optional[int] = None,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
) -> tuple[Path, SplitSummary]:
    """High-level helper used by the Part B CLI."""

    rows = load_dataset_rows_from_graph_index(graphs_index_csv, dataset="twitter")
    manifest, summary = make_ordered_split_manifest(
        rows,
        dataset="twitter",
        train_count=train_count,
        validation_count=validation_count,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    output_path = write_split_manifest(manifest, output_csv)
    return output_path, summary
