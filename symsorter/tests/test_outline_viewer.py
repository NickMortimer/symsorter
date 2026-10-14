"""Tests for outline LabelMe load + pixmap overlay helpers."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QApplication

from symsorter.outline.export_labelme import (
    extract_polygons,
    labelme_path_for_thumb,
    load_labelme_json,
)
from symsorter.ui.outline_dialog import render_thumb_with_polygons


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_labelme_path_and_extract(tmp_path: Path) -> None:
    thumb = tmp_path / "051_DJIP4_RGB_AU_GRAV_20221023T064910_0723_det_10_turtle_3_224px.jpg"
    Image.new("RGB", (224, 224), color=(10, 20, 30)).save(thumb)
    ann = {
        "shapes": [
            {
                "label": "turtle",
                "shape_type": "polygon",
                "points": [[8, 8], [216, 8], [216, 216], [8, 216]],
                "flags": {"draft": True, "backend": "sam"},
            }
        ],
        "imagePath": thumb.name,
        "imageHeight": 224,
        "imageWidth": 224,
    }
    json_path = labelme_path_for_thumb(thumb)
    json_path.write_text(json.dumps(ann), encoding="utf-8")

    loaded = load_labelme_json(json_path)
    polys = extract_polygons(loaded)
    assert len(polys) == 1
    assert polys[0]["label"] == "turtle"
    assert len(polys[0]["points"]) == 4
    assert polys[0]["flags"]["backend"] == "sam"


def test_extract_ignores_non_polygon() -> None:
    data = {
        "shapes": [
            {"label": "box", "shape_type": "rectangle", "points": [[0, 0], [10, 10]]},
            {
                "label": "turtle",
                "shape_type": "polygon",
                "points": [[1, 1], [2, 1], [2, 2]],
            },
        ]
    }
    polys = extract_polygons(data)
    assert len(polys) == 1
    assert polys[0]["label"] == "turtle"


def test_render_thumb_with_polygons(tmp_path: Path) -> None:
    _ensure_qapp()
    thumb = tmp_path / "thumb.jpg"
    Image.new("RGB", (224, 224), color=(80, 90, 100)).save(thumb)
    polys = [
        {
            "label": "turtle",
            "points": [[20.0, 20.0], [200.0, 20.0], [200.0, 200.0], [20.0, 200.0]],
            "flags": {},
        }
    ]
    pix = render_thumb_with_polygons(thumb, polys, display_size=224)
    assert not pix.isNull()
    assert pix.width() == 224
    assert pix.height() == 224
