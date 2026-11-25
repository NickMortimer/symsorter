"""Tests for OutlineCanvas frame overlays, crop rect, and status message."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QMouseEvent, QPixmap

from symsorter.ui.outline_dialog import OutlineCanvas


def test_overlays_are_not_hit_testable(qapp, tmp_path) -> None:
    canvas = OutlineCanvas()
    canvas.resize(400, 400)
    canvas.set_mode("delete")
    pix = QPixmap(640, 480)
    pix.fill(Qt.black)
    own = [{"label": "turtle", "points": [[10, 10], [60, 10], [60, 60]], "flags": {}}]
    canvas.set_scene(
        pix,
        own,
        selected_index=0,
        image_width=640,
        image_height=480,
        refit=True,
    )
    canvas.set_overlays([[[300, 300], [400, 300], [400, 400], [300, 400]]])
    canvas.set_crop_rect((188, 88, 412, 312))
    w = canvas._img_to_widget(350, 350)
    canvas.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            w,
            w,
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    assert len(canvas._polygons) == 1
    canvas.grab()  # paint path runs without error
    # A click inside the editable polygon still deletes it.
    inside = canvas._img_to_widget(40, 20)
    canvas.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            inside,
            inside,
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    assert len(canvas._polygons) == 0


def test_message_replaces_no_image_when_pixmap_missing(qapp) -> None:
    canvas = OutlineCanvas()
    canvas.resize(180, 80)
    blank = canvas.grab()
    canvas.set_message("frame_missing")
    labelled = canvas.grab()
    assert canvas._message == "frame_missing"
    assert blank.toImage() != labelled.toImage()
    canvas.set_message(None)
    assert canvas._message is None
