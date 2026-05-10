#!/usr/bin/env python3
"""Repair an already-generated EGN solver_runs.csv with old non-canonical graph IDs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def canonical_graph_id(dataset: str, source_index: int) -> str:
    return f"{dataset}_graph{int(source_index):06d}"


def repair_row(row: pd.Series) -> pd.Series:
    dataset = str(row["dataset"]).strip()
    graph_id = str(row["graph_id"]).strip()

    m = re.fullmatch(rf"{re.escape(dataset)}_(\d{{6}})", graph_id)
    if m:
        zero_based = int(m.group(1))
        source_index = zero_based + 1
        row["graph_id"] = canonical_graph_id(dataset, source_index)
        row["source_index"] = source_index
        return row

    m = re.fullmatch(rf"{re.escape(dataset)}_graph(\d{{6}})", graph_id)
    if m:
        row["source_index"] = int(m.group(1))
        return row

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    repaired = df.apply(repair_row, axis=1)
    repaired.to_csv(output_path, index=False)

    print(f"Wrote repaired EGN solver runs to: {output_path}")
    print(repaired[["dataset", "graph_id", "source_index"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
