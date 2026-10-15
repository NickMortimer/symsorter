"""Tests for multi-turtle polygons, delete, lasso simplify, crop shift."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from symsorter.outline.export_labelme import (
    add_turtle_polygon,
    delete_all_turtle_polygons,
    delete_turtle_polygon_at,
    extract_polygons,
    find_turtle_polygon_index_at,
    load_labelme_json,
    replace_turtle_polygon_at,
    set_turtle_polygons,
    simplify_polyline,
    upsert_turtle_polygon,
)
from symsorter.outline.overlay import apply_outline_overlay_if_present, shift_polygons


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _square(n: float = 0.0) -> list[list[float]]:
    return [
        [10.0 + n, 10.0 + n],
        [50.0 + n, 10.0 + n],
        [50.0 + n, 50.0 + n],
        [10.0 + n, 50.0 + n],
    ]


def test_add_keeps_multiple_turtle_polygons(tmp_path: Path) -> None:
    thumb = tmp_path / "t.jpg"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(thumb)
    json_path = thumb.with_suffix(".json")
    # Seed with a measurement line
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
    add_turtle_polygon(
        json_path,
        image_path=thumb.name,
        image_height=224,
        image_width=224,
        points=_square(0),
        flags={"backend": "sam", "draft": True},
    )
    add_turtle_polygon(
        json_path,
        image_path=thumb.name,
        image_height=224,
        image_width=224,
        points=_square(40),
        flags={"backend": "sam", "draft": True},
    )
    data = load_labelme_json(json_path)
    assert data is not None
    labels = [s.get("label") for s in data["shapes"]]
    assert labels.count("turtle") == 2
    assert "turtle_length" in labels
    assert len(extract_polygons(data)) == 2


def test_upsert_default_adds_not_replaces(tmp_path: Path) -> None:
    thumb = tmp_path / "t.jpg"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(thumb)
    json_path = thumb.with_suffix(".json")
    upsert_turtle_polygon(
        json_path,
        image_path=thumb.name,
        image_height=224,
        image_width=224,
        points=_square(0),
    )
    upsert_turtle_polygon(
        json_path,
        image_path=thumb.name,
        image_height=224,
        image_width=224,
        points=_square(30),
    )
    assert len(extract_polygons(load_labelme_json(json_path))) == 2

    upsert_turtle_polygon(
        json_path,
        image_path=thumb.name,
        image_height=224,
        image_width=224,
        points=_square(60),
        replace_all=True,
    )
    assert len(extract_polygons(load_labelme_json(json_path))) == 1


def test_delete_and_find_polygon(tmp_path: Path) -> None:
    thumb = tmp_path / "t.jpg"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(thumb)
    json_path = thumb.with_suffix(".json")
    add_turtle_polygon(
        json_path,
        image_path=thumb.name,
        image_height=224,
        image_width=224,
        points=_square(0),
    )
    add_turtle_polygon(
        json_path,
        image_path=thumb.name,
        image_height=224,
        image_width=224,
        points=_square(80),
    )
    # Point inside second square (~90,90)
    idx = find_turtle_polygon_index_at(json_path, 90.0, 90.0)
    assert idx == 1
    assert delete_turtle_polygon_at(json_path, 1) is True
    assert len(extract_polygons(load_labelme_json(json_path))) == 1
    assert delete_all_turtle_polygons(json_path) == 1
    assert extract_polygons(load_labelme_json(json_path)) == []


def test_replace_and_set_polygons(tmp_path: Path) -> None:
    thumb = tmp_path / "t.jpg"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(thumb)
    json_path = thumb.with_suffix(".json")
    add_turtle_polygon(
        json_path,
        image_path=thumb.name,
        image_height=224,
        image_width=224,
        points=_square(0),
        flags={"backend": "sam"},
    )
    replace_turtle_polygon_at(
        json_path,
        0,
        image_path=thumb.name,
        image_height=224,
        image_width=224,
        points=_square(5),
        flags={"backend": "lasso", "prompt_kind": "lasso"},
    )
    polys = extract_polygons(load_labelme_json(json_path))
    assert len(polys) == 1
    assert polys[0]["flags"]["backend"] == "lasso"
    assert polys[0]["points"][0][0] == 15.0

    set_turtle_polygons(
        json_path,
        [
            {"points": _square(0), "flags": {"draft": True}},
            {"points": _square(40), "flags": {"draft": True}},
        ],
        image_path=thumb.name,
        image_height=224,
        image_width=224,
    )
    assert len(extract_polygons(load_labelme_json(json_path))) == 2


def test_simplify_polyline() -> None:
    pts = [[float(i), float(i)] for i in range(100)]
    out = simplify_polyline(pts, max_points=10)
    assert len(out) == 10
    assert out[0] == [0.0, 0.0]
    assert out[-1] == [99.0, 99.0]
    assert simplify_polyline(pts[:5], max_points=10) == pts[:5]


def test_shift_polygons_for_crop() -> None:
    polys = [{"label": "turtle", "points": _square(50), "flags": {}}]
    shifted = shift_polygons(polys, dx=40.0, dy=40.0)
    assert shifted[0]["points"][0] == [20.0, 20.0]


def test_overlay_with_crop_xyxy(tmp_path: Path) -> None:
    _ensure_qapp()
    thumb = tmp_path / "thumb.jpg"
    Image.new("RGB", (224, 224), color=(40, 50, 60)).save(thumb)
    ann = {
        "shapes": [
            {
                "label": "turtle",
                "shape_type": "polygon",
                "points": [[60, 60], [160, 60], [160, 160], [60, 160]],
                "flags": {},
            }
        ],
        "imageHeight": 224,
        "imageWidth": 224,
    }
    thumb.with_suffix(".json").write_text(json.dumps(ann), encoding="utf-8")
    # Simulate center crop of 112×112 starting at (56,56)
    crop = QPixmap(str(thumb)).copy(56, 56, 112, 112)
    overlaid = apply_outline_overlay_if_present(
        crop,
        thumb,
        outline_only=True,
        crop_xyxy=(56, 56, 168, 168),
    )
    assert not overlaid.isNull()
    assert overlaid.width() == 112
