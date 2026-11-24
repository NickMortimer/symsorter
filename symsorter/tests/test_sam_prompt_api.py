"""Tests for interactive SAM prompt kwargs and LabelMe upsert merge."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from symsorter.outline.backends import sam_polygon
from symsorter.outline.export_labelme import extract_polygons, load_labelme_json, upsert_turtle_polygon


class _FakeSAM:
    def __init__(self):
        self.last_kwargs = None

    def __call__(self, _source, **kwargs):
        self.last_kwargs = kwargs
        poly = np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]])
        return [SimpleNamespace(masks=SimpleNamespace(xy=[poly]))]


def test_sam_point_prompt_passes_points(tmp_path: Path) -> None:
    thumb = tmp_path / "t.jpg"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(thumb)
    model = _FakeSAM()
    draft = sam_polygon(thumb, model=model, points=[[100.0, 110.0]], point_labels=[1])
    assert draft.flags["prompt_kind"] == "points"
    assert model.last_kwargs["points"] == [[100.0, 110.0]]
    assert model.last_kwargs["labels"] == [1]
    assert "bboxes" not in model.last_kwargs


def test_sam_box_prompt_passes_bboxes(tmp_path: Path) -> None:
    thumb = tmp_path / "t.jpg"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(thumb)
    model = _FakeSAM()
    draft = sam_polygon(thumb, model=model, bboxes=[20.0, 30.0, 180.0, 190.0])
    assert draft.flags["prompt_kind"] == "box"
    assert model.last_kwargs["bboxes"] == [20.0, 30.0, 180.0, 190.0]


def test_upsert_keeps_measurement_lines(tmp_path: Path) -> None:
    thumb = tmp_path / "t.jpg"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(thumb)
    json_path = thumb.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "turtle_length",
                        "shape_type": "line",
                        "points": [[10, 10], [200, 10]],
                    }
                ],
                "imagePath": thumb.name,
            }
        ),
        encoding="utf-8",
    )
    upsert_turtle_polygon(
        json_path,
        image_path=thumb.name,
        image_height=224,
        image_width=224,
        points=[[8, 8], [216, 8], [216, 216], [8, 216]],
        flags={"backend": "sam", "prompt_kind": "box", "draft": True},
    )
    data = load_labelme_json(json_path)
    assert data is not None
    labels = [s.get("label") for s in data["shapes"]]
    assert "turtle_length" in labels
    assert "turtle" in labels
    polys = extract_polygons(data)
    assert len(polys) == 1
    assert polys[0]["flags"]["prompt_kind"] == "box"
