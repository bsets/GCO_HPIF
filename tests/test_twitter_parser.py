from pathlib import Path
import tarfile
import io

from gco_hpif.data.twitter_loader import load_twitter_ego_graphs


def test_twitter_loader_roundtrip(tmp_path: Path):
    raw_root = tmp_path / "raw"
    tar_path = raw_root / "twitter" / "twitter.tar.gz"
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "w:gz") as tf:
        content = b"10 11\n11 12\n"
        info = tarfile.TarInfo(name="123.edges")
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))

    records, manifest = load_twitter_ego_graphs(raw_root=raw_root, force_download=False)
    assert len(records) == 1
    G = records[0]["graph"]
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 5
    assert manifest[0]["source_ego_id"] == "123"
