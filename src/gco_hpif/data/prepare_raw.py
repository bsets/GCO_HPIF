from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from gco_hpif.data.common import ensure_dir, write_pickle_gz
from gco_hpif.data.tu_loader import load_tu_dataset
from gco_hpif.data.twitter_loader import load_twitter_ego_graphs
from gco_hpif.utils.graph_utils import DATASET_ORDER


@dataclass
class PrepareResult:
    manifest_path: Path
    summary_path: Path
    interim_paths: list[Path]


DATASET_LOADERS = {
    "imdb_binary": load_tu_dataset,
    "collab": load_tu_dataset,
    "twitter": load_twitter_ego_graphs,
}


def prepare_raw_graphs(output_root: Path, datasets: list[str], force_download: bool = False, limit: int | None = None) -> PrepareResult:
    raw_root = output_root / "raw"
    interim_root = output_root / "interim"
    manifest_root = output_root / "manifests"

    ensure_dir(raw_root)
    ensure_dir(interim_root)
    ensure_dir(manifest_root)

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    interim_paths: list[Path] = []

    for dataset in tqdm(datasets, desc="Datasets", unit="dataset"):
        if dataset not in DATASET_LOADERS:
            raise ValueError(f"Unknown dataset: {dataset}")

        loader = DATASET_LOADERS[dataset]
        if dataset in {"imdb_binary", "collab"}:
            records, manifest_rows = loader(dataset, raw_root=raw_root, force_download=force_download, limit=limit)
        else:
            records, manifest_rows = loader(raw_root=raw_root, force_download=force_download, limit=limit)

        out_path = interim_root / f"{dataset}_graphs.pkl.gz"
        tqdm.write(f"Writing {len(records)} graphs to {out_path}")
        write_pickle_gz(records, out_path)
        interim_paths.append(out_path)

        all_rows.extend(manifest_rows)
        summary_rows.append({
            "dataset": dataset,
            "graph_count": len(records),
            "pickle_file": str(out_path.relative_to(output_root)),
        })

    index_df = pd.DataFrame(all_rows)
    if not index_df.empty:
        index_df["dataset_order"] = index_df["dataset"].map(DATASET_ORDER)
        index_df = index_df.sort_values(["dataset_order", "source_index", "graph_id"]).drop(columns=["dataset_order"])

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["dataset_order"] = summary_df["dataset"].map(DATASET_ORDER)
        summary_df = summary_df.sort_values(["dataset_order"]).drop(columns=["dataset_order"])

    manifest_path = manifest_root / "graphs_index.csv"
    summary_path = manifest_root / "dataset_summary.csv"
    index_df.to_csv(manifest_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return PrepareResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        interim_paths=interim_paths,
    )
