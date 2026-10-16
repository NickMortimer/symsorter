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


def test_lock_auto_saves_labelme(tmp_path: Path) -> None:
    """Locking a polygon (K / Lock button) persists flags immediately."""
    _ensure_qapp()
    from symsorter.outline.export_labelme import is_polygon_locked, load_labelme_json
    from symsorter.ui.outline_dialog import OutlineViewerDialog

    thumb = tmp_path / "051_DJIP4_RGB_AU_GRAV_20221023T064910_0723_det_10_turtle_3_224px.jpg"
    Image.new("RGB", (224, 224), color=(10, 20, 30)).save(thumb)
    ann = {
        "shapes": [
            {
                "label": "turtle",
                "shape_type": "polygon",
                "points": [[8, 8], [216, 8], [216, 216], [8, 216]],
                "flags": {"draft": True},
            }
        ],
        "imagePath": thumb.name,
        "imageHeight": 224,
        "imageWidth": 224,
    }
    thumb.with_suffix(".json").write_text(json.dumps(ann), encoding="utf-8")

    dlg = OutlineViewerDialog(
        None, entries=[(thumb, thumb.name)], start_index=0, display_size=224
    )
    assert len(dlg._polygons) == 1
    assert is_polygon_locked(dlg._polygons[0]) is False
    dlg.toggle_lock_selected()
    assert dlg._dirty is False
    loaded = load_labelme_json(thumb.with_suffix(".json"))
    shapes = [s for s in loaded["shapes"] if s.get("shape_type") == "polygon"]
    assert len(shapes) == 1
    assert shapes[0]["flags"]["locked"] is True
    assert shapes[0]["flags"]["quality"] == "good"
    dlg.close()


def test_canvas_double_click_removes_vertex(tmp_path: Path) -> None:
    _ensure_qapp()
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent

    from symsorter.ui.outline_dialog import OutlineCanvas

    thumb = tmp_path / "t.jpg"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(thumb)
    canvas = OutlineCanvas()
    canvas.resize(224, 224)
    canvas.set_mode("edit")
    from PySide6.QtGui import QPixmap

    pix = QPixmap(str(thumb))
    polys = [
        {
            "label": "turtle",
            "points": [[20.0, 20.0], [200.0, 20.0], [200.0, 200.0], [20.0, 200.0], [110.0, 110.0]],
            "flags": {},
        }
    ]
    canvas.set_scene(
        pix,
        polys,
        selected_index=0,
        image_width=224,
        image_height=224,
        refit=True,
    )
    assert len(canvas._polygons[0]["points"]) == 5

    # Map image (20,20) → widget coords using canvas transform
    wx = int(20.0 * canvas._zoom + canvas._ox)
    wy = int(20.0 * canvas._zoom + canvas._oy)
    from PySide6.QtCore import QPointF

    event = QMouseEvent(
        QEvent.Type.MouseButtonDblClick,
        QPointF(wx, wy),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    canvas.mouseDoubleClickEvent(event)
    assert len(canvas._polygons[0]["points"]) == 4
