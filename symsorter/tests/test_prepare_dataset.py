"""Tests for ROI prepare and JSON update commands."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from symsorter.commands.prepare_dataset import (
    prepare_dataset_command,
    update_json_from_symsorter_command,
)


def _write_labelme_pair(root: Path) -> Path:
    """Create one image and matching Labelme-style JSON file."""
    img_path = root / "frame_001.jpg"
    Image.new("RGB", (400, 300), color=(100, 120, 140)).save(img_path)

    ann_path = root / "frame_001.json"
    ann = {
        "imagePath": "frame_001.jpg",
        "shapes": [
            {
                "label": "old_label",
                "shape_type": "rectangle",
                "points": [[100, 80], [220, 200]],
            }
        ],
    }
    ann_path.write_text(json.dumps(ann), encoding="utf-8")
    return ann_path


def test_prepare_creates_thumbnails_and_map(tmp_path: Path) -> None:
    """Prepare command should generate ROI thumbnails and mapping JSON."""
    _write_labelme_pair(tmp_path)
    out_dir = tmp_path / "symsorter_prepared"

    prepare_dataset_command(root=tmp_path, out_dir=out_dir, thumb_size=224, context_scale=1.6)

    map_path = out_dir / "symsorter_prepare_map.json"
    assert map_path.exists()

    data = json.loads(map_path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    assert len(entries) == 1

    thumb_path = out_dir / entries[0]["thumb_file"]
    assert thumb_path.exists()
    with Image.open(thumb_path) as thumb:
        assert thumb.size == (224, 224)


def test_update_json_applies_symsorter_class(tmp_path: Path) -> None:
    """update-json should write class info back to source shape metadata."""
    ann_path = _write_labelme_pair(tmp_path)
    out_dir = tmp_path / "symsorter_prepared"
    prepare_dataset_command(root=tmp_path, out_dir=out_dir, thumb_size=224, context_scale=1.6)

    map_data = json.loads((out_dir / "symsorter_prepare_map.json").read_text(encoding="utf-8"))
    thumb_name = map_data["entries"][0]["thumb_file"]

    classifications = {
        "images": [{"id": 1, "file_name": thumb_name}],
        "categories": [{"id": 7, "name": "turtle"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 7}],
    }
    (out_dir / "classifications.json").write_text(
        json.dumps(classifications),
        encoding="utf-8",
    )

    update_json_from_symsorter_command(
        prepared_dir=out_dir,
        source_root=tmp_path,
        replace_label=True,
    )

    updated = json.loads(ann_path.read_text(encoding="utf-8"))
    shape = updated["shapes"][0]
    assert shape["label"] == "turtle"
    assert shape["symsorter"]["class_name"] == "turtle"
    assert shape["symsorter"]["class_id"] == 7
