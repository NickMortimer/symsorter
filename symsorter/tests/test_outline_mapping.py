"""Tests for YOLO/slice → thumb box projection (turtle-repo geometry)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from symsorter.outline.export_labelme import extract_rectangles, upsert_turtle_boxes
from symsorter.outline.mapping import (
    CropWindow,
    FrameBox,
    fixed_size_crop_window,
    load_boxes_from_slices,
    project_box_to_crop,
    rectangles_to_labelme_shapes,
)


def test_fixed_size_crop_window_centered() -> None:
    # Near the origin, turtledrone expand clamps the center so the 224 window stays in-bounds
    crop = fixed_size_crop_window(100, 100, 120, 140, img_w=1000, img_h=1000, fixed_size=224)
    assert crop.width == 224
    assert crop.height == 224
    assert crop.left == 0
    assert crop.top == 8  # cy clamped to 112 → top 0 would be for cy=112; box cy=120 → 8
    # Well inside the frame: exact center
    mid = fixed_size_crop_window(400, 400, 440, 460, img_w=1000, img_h=1000, fixed_size=224)
    assert mid.left == 420 - 112
    assert mid.top == 430 - 112


def test_project_box_to_crop_clamps() -> None:
    crop = CropWindow(100, 200, 324, 424)
    box = FrameBox(80, 210, 150, 300)  # hangs off left of crop
    local = project_box_to_crop(box, crop)
    assert local is not None
    x1, y1, x2, y2 = local.as_xyxy()
    assert x1 == 0.0  # clamped
    assert y1 == 10.0
    assert x2 == 50.0
    assert y2 == 100.0


def test_project_no_overlap() -> None:
    crop = CropWindow(0, 0, 224, 224)
    box = FrameBox(500, 500, 600, 600)
    assert project_box_to_crop(box, crop) is None


def test_load_boxes_from_slices(tmp_path: Path) -> None:
    slices = tmp_path / "slices"
    slices.mkdir()
    stem = "DJIP4_RGB_AU_GRAV_20221023T064910_0723"
    # tile origin 2560,1536
    path = slices / f"{stem}_2560_1536_3200_2176.json"
    path.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "turtle_3_yolo",
                        "shape_type": "rectangle",
                        "points": [[10.0, 20.0], [40.0, 50.0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    boxes = load_boxes_from_slices(slices, stem)
    assert len(boxes) == 1
    assert boxes[0].as_xyxy() == (2570.0, 1556.0, 2600.0, 1586.0)
    assert boxes[0].source == "slice"


def test_upsert_turtle_boxes_keeps_measurements(tmp_path: Path) -> None:
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
    shapes = rectangles_to_labelme_shapes(
        [FrameBox(20, 30, 80, 90, source="image")]
    )
    upsert_turtle_boxes(
        json_path,
        shapes,
        image_path=thumb.name,
        image_height=224,
        image_width=224,
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))
    labels = [s["label"] for s in data["shapes"]]
    assert "turtle_length" in labels
    assert "turtle_box" in labels
    rects = extract_rectangles(data)
    assert len(rects) == 1
    assert rects[0]["points"][0] == [20.0, 30.0]
