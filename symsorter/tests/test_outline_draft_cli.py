"""Tests for outline-draft enumeration and LabelMe output."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from symsorter.commands.outline_draft import iter_thumb_paths, outline_draft_command
from symsorter.outline.backends import box_polygon, draft_polygon, sam_polygon


def _make_thumb_tree(root: Path) -> Path:
    """Create a mini sorted tree with a nested duplicate to skip."""
    shard = root / "turtle-surface" / "shard_00001"
    shard.mkdir(parents=True)
    nested = shard / "turtle-surface"
    nested.mkdir()

    name = "051_DJIP4_RGB_AU_GRAV_20221023T064910_0723_det_10_turtle_3_224px.jpg"
    Image.new("RGB", (224, 224), color=(40, 90, 60)).save(shard / name)
    # Nested duplicate copy — must be skipped
    Image.new("RGB", (224, 224), color=(40, 90, 60)).save(nested / name)

    deep_shard = root / "turtle-deep" / "shard_00001"
    deep_shard.mkdir(parents=True)
    deep_name = "040_DJIP4_RGB_AU_NEDS_20190526T125451_0242_det_0_turtle_3_224px.jpg"
    Image.new("RGB", (224, 224), color=(20, 40, 80)).save(deep_shard / deep_name)
    return root


def test_iter_skips_nested_duplicate_trees(tmp_path: Path) -> None:
    root = _make_thumb_tree(tmp_path)
    found = list(iter_thumb_paths(root, ("turtle-surface", "turtle-deep")))
    rels = [str(rel) for _path, rel in found]
    assert len(found) == 2
    assert all("shard_00001/turtle-surface/" not in r for r in rels)
    assert any("turtle-surface/shard_00001/" in r for r in rels)
    assert any("turtle-deep/shard_00001/" in r for r in rels)


def test_iter_skips_appledouble_dot_underscore_files(tmp_path: Path) -> None:
    shard = tmp_path / "turtle-surface" / "shard_00001"
    shard.mkdir(parents=True)
    good = "051_DJIP4_RGB_AU_GRAV_20221023T064910_0723_det_10_turtle_3_224px.jpg"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(shard / good)
    # macOS AppleDouble sidecar that looks like a jpg
    (shard / f"._{good}").write_bytes(b"fake-appledouble")
    (shard / ".DS_Store").write_bytes(b"store")

    found = list(iter_thumb_paths(tmp_path, ("turtle-surface",)))
    names = [p.name for p, _rel in found]
    assert names == [good]
    assert not any(n.startswith("._") for n in names)


def test_box_polygon_inset() -> None:
    draft = box_polygon(size=224, inset=8)
    assert draft.backend == "box"
    assert draft.points == [[8.0, 8.0], [216.0, 8.0], [216.0, 216.0], [8.0, 216.0]]


def test_outline_draft_writes_json_beside_thumbs(tmp_path: Path) -> None:
    thumbs = _make_thumb_tree(tmp_path / "sorted")
    out = tmp_path / "outline_out"

    stats = outline_draft_command(
        thumbs=thumbs,
        out=out,
        classes="turtle-surface,turtle-deep",
        backend="box",
        limit=None,
        overwrite=False,
        dry_run=False,
    )
    assert stats["written"] == 2
    assert stats["broken_symlink"] == 0

    manifest = (out / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest) == 2
    row = json.loads(manifest[0])
    assert row["turtle_group_id"] is None
    assert row["status"] == "drafted"
    assert row["polygon_thumb"] is not None
    assert row["conf"] is not None
    assert row["det_index"] is not None

    # Labels live next to thumbs, not under out/labels/
    assert not (out / "labels").exists()
    surface_jpg = next((thumbs / "turtle-surface").rglob("*_224px.jpg"))
    # Prefer a non-nested path
    for candidate in (thumbs / "turtle-surface").rglob("*_224px.jpg"):
        if "shard_" in candidate.parts and candidate.parent.name.startswith("shard_"):
            surface_jpg = candidate
            break
    label_path = surface_jpg.with_suffix(".json")
    assert label_path.exists()
    assert label_path.parent == surface_jpg.parent

    data = json.loads(label_path.read_text(encoding="utf-8"))
    assert data["imageHeight"] == 224
    assert data["imageWidth"] == 224
    shape = data["shapes"][0]
    assert shape["label"] == "turtle"
    assert shape["shape_type"] == "polygon"
    assert shape["flags"]["draft"] is True
    assert shape["flags"]["backend"] == "box"
    assert shape["flags"]["turtle_group_id"] is None


def test_outline_draft_dry_run_writes_nothing(tmp_path: Path) -> None:
    thumbs = _make_thumb_tree(tmp_path / "sorted")
    stats = outline_draft_command(
        thumbs=thumbs,
        out=None,
        backend="box",
        dry_run=True,
    )
    assert stats["enumerated"] == 2
    assert list(thumbs.rglob("*.json")) == []


def test_outline_draft_skips_broken_symlink(tmp_path: Path) -> None:
    shard = tmp_path / "turtle-surface" / "shard_00001"
    shard.mkdir(parents=True)
    good = shard / "051_DJIP4_RGB_AU_GRAV_20221023T064910_0723_det_10_turtle_3_224px.jpg"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(good)
    broken = shard / "052_DJIP4_RGB_AU_GRAV_20221023T064910_0724_det_1_turtle_3_224px.jpg"
    broken.symlink_to(tmp_path / "does_not_exist.jpg")

    out = tmp_path / "out"
    stats = outline_draft_command(
        thumbs=tmp_path,
        out=out,
        classes="turtle-surface",
        backend="box",
    )
    assert stats["written"] == 1
    assert stats["broken_symlink"] == 1
    assert good.with_suffix(".json").exists()
    lines = (out / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    statuses = {json.loads(line)["status"] for line in lines}
    assert "broken_symlink" in statuses
    assert "drafted" in statuses


def test_threshold_backend_returns_polygon(tmp_path: Path) -> None:
    path = tmp_path / "thumb.jpg"
    # Dark blob on bright sand-like background
    img = Image.new("RGB", (224, 224), color=(220, 210, 180))
    for x in range(60, 160):
        for y in range(70, 150):
            img.putpixel((x, y), (40, 70, 50))
    img.save(path)
    draft = draft_polygon("threshold", image_path=path)
    assert draft.backend == "threshold"
    assert len(draft.points) >= 3
    assert draft.flags.get("draft") is True


def test_overwrite_false_skips_existing(tmp_path: Path) -> None:
    thumbs = _make_thumb_tree(tmp_path / "sorted")
    outline_draft_command(thumbs=thumbs, out=None, backend="box")
    stats = outline_draft_command(thumbs=thumbs, out=None, backend="box", overwrite=False)
    assert stats["written"] == 0
    assert stats["skipped_existing"] == 2


def test_sam_polygon_uses_ultralytics_masks_xy(tmp_path: Path) -> None:
    """SAM backend reads masks.xy like turtledrone segment.py (mocked model)."""
    path = tmp_path / "thumb.jpg"
    Image.new("RGB", (224, 224), color=(30, 60, 40)).save(path)

    poly = np.array([[20.0, 30.0], [180.0, 25.0], [190.0, 200.0], [15.0, 195.0]])
    fake_result = SimpleNamespace(masks=SimpleNamespace(xy=[poly]))

    class FakeSAM:
        def __call__(self, _source, **_kwargs):
            return [fake_result]

    draft = sam_polygon(path, model=FakeSAM(), sam_model="sam_b.pt")
    assert draft.backend == "sam"
    assert draft.points == poly.tolist()
    assert draft.flags["draft"] is True
    assert draft.flags["prompt_bbox"] == [8.0, 8.0, 216.0, 216.0]
