"""Tests for remapping crop NPZs into sorted shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from symsorter.clip_encode import load_existing_embeddings, save_folder_embeddings
from symsorter.commands.remap_sorted_npz import remap_sorted_npz_command


def _write_source_npz(crop_dir: Path, names: list[str], model_type: str = "dinov3") -> Path:
    crop_dir.mkdir(parents=True, exist_ok=True)
    emb = {n: np.arange(8, dtype=np.float32) + i for i, n in enumerate(names)}
    dims = {n: (224, 224) for n in names}
    npz = crop_dir / "encoding_dinov3_224.npz"
    save_folder_embeddings(npz, emb, dimensions_dict=dims, crop_size=224, model_type=model_type)
    return npz


def test_remap_writes_shard_npz(tmp_path: Path) -> None:
    source = tmp_path / "turtle_v01_run02" / "GRAV_x" / "turtle_3" / "crops_224"
    names = [
        "050_DJIP4_RGB_AU_GRAV_20221023T061720_0040_det_7_turtle_3_224px.jpg",
        "051_DJIP4_RGB_AU_GRAV_20221023T064910_0723_det_10_turtle_3_224px.jpg",
    ]
    _write_source_npz(source, names)

    shard = tmp_path / "sorted" / "turtle-surface" / "shard_00001"
    shard.mkdir(parents=True)
    for n in names:
        Image.new("RGB", (224, 224), color=(1, 2, 3)).save(shard / n)
    # AppleDouble junk must not break matching
    (shard / f"._{names[0]}").write_bytes(b"x")

    stats = remap_sorted_npz_command(
        thumbs=tmp_path / "sorted",
        source=tmp_path / "turtle_v01_run02",
        npz_name="encoding_dinov3_224.npz",
        classes="turtle-surface",
        overwrite=True,
        dry_run=False,
    )
    assert stats["written"] == 1
    assert stats["matched"] == 2
    assert stats["missing"] == 0

    out = shard / "encoding_dinov3_224.npz"
    assert out.exists()
    emb, _dims, meta = load_existing_embeddings(out)
    assert set(emb.keys()) == set(names)
    assert meta.get("model_type") == "dinov3"
