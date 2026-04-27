from __future__ import annotations

import csv
import gzip
import os
import pickle
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class _TqdmReportHook:
    def __init__(self, desc: str):
        self._bar = tqdm(total=None, unit="B", unit_scale=True, unit_divisor=1024, desc=desc, leave=False)

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            self._bar.total = total_size
        downloaded = block_num * block_size
        self._bar.update(downloaded - self._bar.n)

    def close(self) -> None:
        self._bar.close()


def download_file(url: str, dest: Path, force: bool = False, desc: str | None = None) -> Path:
    ensure_dir(dest.parent)
    if dest.exists() and not force:
        return dest

    hook = _TqdmReportHook(desc or f"Downloading {dest.name}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=hook)
    finally:
        hook.close()
    return dest


def write_pickle_gz(obj, dest: Path) -> None:
    ensure_dir(dest.parent)
    with gzip.open(dest, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_text_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_int_pair(line: str) -> tuple[int, int]:
    parts = [p.strip() for p in line.replace("	", " ").replace(",", " ").split() if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Cannot parse edge line: {line!r}")
    return int(parts[0]), int(parts[1])
