"""Tests for grid/viewer polygon overlay helpers."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from symsorter.outline.overlay import (
    apply_outline_overlay_if_present,
    load_polygons_for_thumb,
)


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_load_and_draw_outline_on_scaled_thumb(tmp_path: Path) -> None:
    _ensure_qapp()
    thumb = tmp_path / "thumb.jpg"
    Image.new("RGB", (224, 224), color=(40, 50, 60)).save(thumb)
    ann = {
        "shapes": [
            {
                "label": "turtle",
                "shape_type": "polygon",
                "points": [[10, 10], [200, 10], [200, 200], [10, 200]],
                "flags": {"backend": "sam", "draft": True},
            }
        ],
        "imageHeight": 224,
        "imageWidth": 224,
    }
    thumb.with_suffix(".json").write_text(json.dumps(ann), encoding="utf-8")

    polys = load_polygons_for_thumb(thumb)
    assert len(polys) == 1

    raw = QPixmap(str(thumb))
    scaled = raw.scaled(128, 128)
    overlaid = apply_outline_overlay_if_present(
        scaled,
        thumb,
        source_width=224,
        source_height=224,
        outline_only=True,
    )
    assert not overlaid.isNull()
    assert overlaid.width() == 128
    # Drawing should produce a distinct pixmap copy
    assert overlaid.cacheKey() != scaled.cacheKey() or overlaid is not scaled


def test_measurement_lines_do_not_count_as_polygons(tmp_path: Path) -> None:
    thumb = tmp_path / "meas.jpg"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(thumb)
    ann = {
        "shapes": [
            {
                "label": "turtle_length",
                "shape_type": "line",
                "points": [[10, 10], [200, 10]],
            }
        ]
    }
    thumb.with_suffix(".json").write_text(json.dumps(ann), encoding="utf-8")
    assert load_polygons_for_thumb(thumb) == []
