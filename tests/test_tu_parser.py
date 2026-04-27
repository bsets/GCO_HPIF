from pathlib import Path
import zipfile

from gco_hpif.data.tu_loader import load_tu_dataset


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_tu_loader_roundtrip(tmp_path: Path):
    raw_root = tmp_path / "raw"
    zip_path = raw_root / "tu" / "IMDB-BINARY.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    build_root = tmp_path / "build" / "IMDB-BINARY"
    build_root.mkdir(parents=True, exist_ok=True)

    _write_lines(build_root / "IMDB-BINARY_graph_indicator.txt", ["1", "1", "2", "2"])
    _write_lines(build_root / "IMDB-BINARY_A.txt", ["1, 2", "3, 4"])
    _write_lines(build_root / "IMDB-BINARY_graph_labels.txt", ["1", "-1"])

    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in build_root.iterdir():
            zf.write(file, arcname=f"IMDB-BINARY/{file.name}")

    records, manifest = load_tu_dataset("imdb_binary", raw_root=raw_root, force_download=False)
    assert len(records) == 2
    assert len(manifest) == 2
    assert manifest[0]["n_nodes"] == 2
    assert manifest[0]["n_edges"] == 1
