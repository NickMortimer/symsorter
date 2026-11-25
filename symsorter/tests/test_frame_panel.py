"""Tests for OutlineCanvas frame overlays and FramePanel."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import NamedTuple

from PIL import Image
from PySide6.QtCore import QEvent, QSettings, Qt, QThreadPool
from PySide6.QtGui import QKeySequence, QMouseEvent, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
)

from symsorter.outline.export_labelme import extract_polygons, load_labelme_json
from symsorter.ui import frame_panel as frame_panel_mod
from symsorter.ui.frame_panel import FramePanel
from symsorter.ui.frame_settings import (
    FrameSettings,
    edit_frame_settings_dialog,
    load_frame_settings,
    roi_csv_for,
    save_frame_settings,
)
from symsorter.ui.outline_dialog import OutlineCanvas, OutlineViewerDialog

THUMB = "050_DJIP4_RGB_AU_MANGB_20230503T113703_2116_det_0_turtle_3_224px.jpg"
DEEP_THUMB = "050_DJIP4_RGB_AU_MANGB_20230503T113703_2116_det_1_turtle_3_224px.jpg"
STEM = "DJIP4_RGB_AU_MANGB_20230503T113703_2116"
FLIGHT = "MANGB_20230503T1137"
ROI_HEADER = (
    "view,crop_basename,flight,x1,y1,x2,y2,confidence,class_name,original_image,status"
)
# 640x480 frame; box centred at (300, 200) → crop window (188, 88, 412, 312)


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


class _PanelTree(NamedTuple):
    finds: Path
    csv: Path
    surface_thumb: Path
    surface_json: Path
    frame: Path


def _roi_line(view: str, name: str, stem: str, box: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = box
    return (
        f"{view},{name},{FLIGHT},{x1},{y1},{x2},{y2},0.5,turtle_3,"
        f"/media/x/Timor2/surveys/AU/MANGB/{FLIGHT}/{stem}.JPG,ok"
    )


def _write_labelme(thumb: Path, points: list[list[int]], *, locked: bool) -> Path:
    thumb.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224)).save(thumb, "JPEG")
    payload = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
            {
                "label": "turtle",
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {"locked": locked, "draft": not locked},
            }
        ],
        "imagePath": thumb.name,
        "imageData": None,
        "imageHeight": 224,
        "imageWidth": 224,
    }
    json_path = thumb.with_suffix(".json")
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return json_path


def _panel_tree(tmp_path: Path) -> _PanelTree:
    """Surface plus one locked deep neighbour, sharing a 640×480 frame."""
    training = tmp_path / "training"
    finds = tmp_path / "finds"
    surface_json = _write_labelme(
        training / "surface" / THUMB,
        [[10, 10], [50, 10], [50, 50]],
        locked=False,
    )
    _write_labelme(
        training / "deep" / DEEP_THUMB,
        [[0, 0], [10, 0], [10, 10]],
        locked=True,
    )
    frame = finds / FLIGHT / "images" / f"{STEM}.JPG"
    frame.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color=(30, 30, 30)).save(frame, "JPEG")
    csv_path = training / "thumb_roi.csv"
    csv_path.write_text(
        ROI_HEADER
        + "\n"
        + _roi_line("surface", THUMB, STEM, (290, 190, 310, 210))
        + "\n"
        + _roi_line("deep", DEEP_THUMB, STEM, (490, 390, 510, 410))
        + "\n",
        encoding="utf-8",
    )
    return _PanelTree(
        finds, csv_path, surface_json.with_suffix(".jpg"), surface_json, frame
    )


def _assert_reason(panel: FramePanel, reason: str) -> None:
    assert panel.reason == reason
    message = panel.canvas._message
    assert isinstance(message, str)
    assert reason in message
    assert message.strip() != reason


def test_show_thumb_maps_polys_crop_and_neighbours(qapp, tmp_path: Path) -> None:
    t = _panel_tree(tmp_path)
    panel = FramePanel(FrameSettings([t.finds], t.csv), load_async=False)
    assert panel._load_async is False
    polys = extract_polygons(load_labelme_json(t.surface_json))
    panel.show_thumb(t.surface_thumb, polys, 0)
    assert panel.reason is None
    assert panel.canvas._message is None
    assert panel.canvas._polygons[0]["points"] == [[198, 98], [238, 98], [238, 138]]
    assert panel.canvas._crop_rect == (188, 88, 412, 312)
    assert len(panel.canvas._overlays) == 1  # deep neighbour, locked
    assert panel.canvas._overlays[0] == [[388, 256], [398, 256], [398, 266]]
    assert panel.thumb_polygons()[0]["points"] == polys[0]["points"]
    assert panel.canvas._polygons[0]["flags"] is not polys[0]["flags"]

    polys[0]["flags"]["marker"] = "caller"
    assert "marker" not in panel.canvas._polygons[0]["flags"]
    edited = extract_polygons(load_labelme_json(t.surface_json))
    edited[0]["points"] = [[1, 2], [8, 2], [8, 9]]
    t.frame.unlink()
    panel.sync_from_thumb(edited, None)
    assert panel.reason is None
    assert panel.canvas._polygons[0]["points"] == [[189, 90], [196, 90], [196, 97]]
    assert panel.canvas._selected is None
    assert panel.canvas._src is not None and not panel.canvas._src.isNull()
    assert (panel.canvas._src.width(), panel.canvas._src.height()) == (640, 480)
    assert panel.thumb_polygons()[0]["points"] == [[1, 2], [8, 2], [8, 9]]
    edited[0]["flags"]["marker"] = "after-sync"
    assert panel.canvas._polygons[0]["flags"].get("marker") != "after-sync"


def test_reasons(qapp, tmp_path: Path) -> None:
    t = _panel_tree(tmp_path)
    panel = FramePanel(FrameSettings([t.finds], t.csv), load_async=False)
    missing = t.surface_thumb.with_name("not_in_the_csv_224px.jpg")
    panel.show_thumb(missing, [], None)
    _assert_reason(panel, "no_roi_row")

    t.frame.unlink()
    polys = extract_polygons(load_labelme_json(t.surface_json))
    panel.show_thumb(t.surface_thumb, polys, 0)
    _assert_reason(panel, "frame_missing")

    bare = FramePanel(FrameSettings(), load_async=False)
    bare.show_thumb(t.surface_thumb, polys, 0)
    _assert_reason(bare, "no_survey_roots")


def test_sam_modes_disabled(qapp) -> None:
    panel = FramePanel(FrameSettings(), load_async=False)
    assert panel._load_async is False
    panel.set_tool_mode("box")
    assert panel.canvas.mode == "view"
    panel.set_tool_mode("point")
    assert panel.canvas.mode == "view"
    panel.set_tool_mode("lasso")
    assert panel.canvas.mode == "lasso"
    seen: list[object] = []
    panel.polygonsEdited.connect(lambda: seen.append("edited"))
    panel.selectionChanged.connect(seen.append)
    panel.canvas.polygonsChanged.emit()
    panel.canvas.selectionChanged.emit(2)
    assert seen == ["edited", 2]


def test_stale_load_is_ignored(qapp, tmp_path: Path, monkeypatch) -> None:
    training = tmp_path / "training"
    finds = tmp_path / "finds"
    surface = training / "surface"
    surface.mkdir(parents=True)
    images = finds / FLIGHT / "images"
    images.mkdir(parents=True)
    thumb_a = surface / "a_det_0_224px.jpg"
    thumb_b = surface / "b_det_0_224px.jpg"
    Image.new("RGB", (32, 32)).save(thumb_a, "JPEG")
    Image.new("RGB", (32, 32)).save(thumb_b, "JPEG")
    stem_a = "FRAME_A_800x600"
    stem_b = "FRAME_B_640x480"
    Image.new("RGB", (800, 600), color=(200, 0, 0)).save(
        images / f"{stem_a}.JPG", "JPEG"
    )
    Image.new("RGB", (640, 480), color=(0, 0, 200)).save(
        images / f"{stem_b}.JPG", "JPEG"
    )
    csv_path = training / "thumb_roi.csv"
    box = (100, 100, 140, 140)
    csv_path.write_text(
        ROI_HEADER
        + "\n"
        + _roi_line("surface", thumb_a.name, stem_a, box)
        + "\n"
        + _roi_line("surface", thumb_b.name, stem_b, box)
        + "\n",
        encoding="utf-8",
    )
    original_run = frame_panel_mod._FrameLoadRunnable.run

    def _delay_previous(self: frame_panel_mod._FrameLoadRunnable) -> None:
        if "FRAME_A_800x600" in self._path:
            time.sleep(0.25)
        original_run(self)

    monkeypatch.setattr(frame_panel_mod._FrameLoadRunnable, "run", _delay_previous)
    panel = FramePanel(FrameSettings([finds], csv_path), load_async=True)
    assert panel._load_async is True
    panel.show_thumb(thumb_a, [], None)
    panel.show_thumb(thumb_b, [], None)
    QThreadPool.globalInstance().waitForDone()
    qapp.processEvents()
    pix = panel.canvas._src
    assert pix is not None and not pix.isNull()
    assert (pix.width(), pix.height()) == (640, 480)


def test_settings_round_trip(qapp, tmp_path: Path, monkeypatch) -> None:
    """Round-trip settings without writing the user's real QSettings."""
    ini = tmp_path / "qt-settings.ini"
    calls: list[tuple[object, ...]] = []

    def _factory(*args: object, **_kwargs: object) -> QSettings:
        calls.append(args)
        return QSettings(str(ini), QSettings.Format.IniFormat)

    monkeypatch.setattr("symsorter.ui.frame_settings.QSettings", _factory)
    save_frame_settings(FrameSettings([tmp_path], tmp_path / "r.csv"))
    loaded = load_frame_settings()
    assert loaded.survey_roots == [tmp_path] and loaded.roi_csv == tmp_path / "r.csv"
    assert calls and all(call[:2] == ("symsorter", "symsorter") for call in calls)

    stored = QSettings(str(ini), QSettings.Format.IniFormat)
    assert stored.value("outline/roi_csv") == str(tmp_path / "r.csv")
    roots = stored.value("outline/survey_roots")
    root_values = roots if isinstance(roots, list) else [roots]
    assert str(tmp_path) in [str(item) for item in root_values]

    folder = tmp_path / "training" / "surface"
    assert roi_csv_for(folder, loaded) == tmp_path / "r.csv"
    assert roi_csv_for(folder, FrameSettings()) is None
    sibling = tmp_path / "training" / "thumb_roi.csv"
    sibling.parent.mkdir(parents=True, exist_ok=True)
    sibling.write_text("view\n", encoding="utf-8")
    assert roi_csv_for(folder, FrameSettings()) == sibling


def test_edit_frame_settings_dialog_ok_saves_cancel_returns_none(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    ini = tmp_path / "dialog-settings.ini"

    def _factory(*_args: object, **_kwargs: object) -> QSettings:
        return QSettings(str(ini), QSettings.Format.IniFormat)

    monkeypatch.setattr("symsorter.ui.frame_settings.QSettings", _factory)
    monkeypatch.setattr(QDialog, "exec", lambda _dialog: QDialog.DialogCode.Rejected)
    assert edit_frame_settings_dialog(None) is None

    root = tmp_path / "survey"
    csv_path = tmp_path / "roi.csv"

    def _ok(dialog: QDialog) -> QDialog.DialogCode:
        roots = dialog.findChild(QPlainTextEdit)
        csv_edit = dialog.findChild(QLineEdit)
        assert roots is not None and csv_edit is not None
        labels = [button.text() for button in dialog.findChildren(QPushButton)]
        assert "Add folder…" in labels
        assert "Browse…" in labels
        roots.setPlainText(f"\n{root}\n\n")
        csv_edit.setText(str(csv_path))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", _ok)
    saved = edit_frame_settings_dialog(None)
    assert saved is not None
    assert saved.survey_roots == [root]
    assert saved.roi_csv == csv_path
    loaded = load_frame_settings()
    assert loaded.survey_roots == [root] and loaded.roi_csv == csv_path


def _outline_dialog(tmp_path: Path) -> tuple[_PanelTree, OutlineViewerDialog]:
    tree = _panel_tree(tmp_path)
    dialog = OutlineViewerDialog(
        None,
        entries=[(tree.surface_thumb, THUMB)],
        display_size=224,
        frame_settings=FrameSettings([tree.finds], tree.csv),
    )
    dialog.frame_panel._load_async = False
    return tree, dialog


def _shortcut(dialog: OutlineViewerDialog, key: str) -> QShortcut:
    target = QKeySequence(key)
    for shortcut in dialog.findChildren(QShortcut):
        if shortcut.key() == target:
            return shortcut
    raise AssertionError(f"missing shortcut {key}")


def test_frame_edit_marks_dirty_and_saves_thumb_coords(qapp, tmp_path: Path) -> None:
    tree, dialog = _outline_dialog(tmp_path)
    dialog.toggle_frame_panel()
    assert dialog.frame_panel.isVisibleTo(dialog)
    frame_canvas = dialog.frame_panel.canvas
    frame_canvas._polygons[0]["points"][0] = [200.0, 100.0]  # frame px
    frame_canvas.polygonsChanged.emit()
    assert dialog._dirty is True
    assert dialog._polygons[0]["points"][0] == [12.0, 12.0]
    assert dialog.canvas._polygons is dialog._polygons
    dialog.save_current()
    saved = extract_polygons(load_labelme_json(tree.surface_json))
    assert saved[0]["points"][0] == [12.0, 12.0]
    assert not (tree.finds / FLIGHT / "images" / f"{STEM}.json").exists()
    dialog.close()


def test_thumb_edit_updates_frame_panel(qapp, tmp_path: Path) -> None:
    _tree, dialog = _outline_dialog(tmp_path)
    dialog.toggle_frame_panel()
    dialog._polygons[0]["points"][0] = [20.0, 20.0]
    dialog.canvas.polygonsChanged.emit()
    assert dialog.frame_panel.canvas._polygons[0]["points"][0] == [208.0, 108.0]
    dialog.close()


def test_locked_polygon_blocked_in_frame_panel(qapp, tmp_path: Path) -> None:
    _tree, dialog = _outline_dialog(tmp_path)
    dialog.toggle_frame_panel()
    dialog.toggle_lock_selected()
    frame_canvas = dialog.frame_panel.canvas
    frame_canvas.set_mode("edit")
    assert frame_canvas.mode == "edit"
    assert frame_canvas._selected == 0
    before = [list(point) for point in frame_canvas._polygons[0]["points"]]
    messages: list[str] = []
    frame_canvas.statusMessage.connect(messages.append)
    frame_canvas._set_selected_points([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]])
    assert frame_canvas._polygons[0]["points"] == before
    assert messages == ["Polygon is locked — press K to unlock before editing."]
    assert dialog.status_label.text() == messages[0]
    dialog.close()


def test_shift_f_hidden_by_default(qapp, tmp_path: Path) -> None:
    _tree, dialog = _outline_dialog(tmp_path)
    assert not dialog.frame_panel.isVisibleTo(dialog)
    assert dialog.frame_btn.text() == "Frame"
    assert dialog.frame_btn.isCheckable()
    assert not dialog.frame_btn.isChecked()
    splitter = dialog.canvas.parentWidget()
    assert isinstance(splitter, QSplitter)
    assert splitter.orientation() == Qt.Orientation.Horizontal
    assert splitter.indexOf(dialog.canvas) >= 0
    assert splitter.indexOf(dialog.frame_panel) >= 0
    dialog.toggle_frame_panel()
    assert dialog.frame_panel.isVisibleTo(dialog)
    assert dialog.frame_btn.isChecked()
    dialog.toggle_frame_panel()
    assert not dialog.frame_panel.isVisibleTo(dialog)
    assert not dialog.frame_btn.isChecked()
    _shortcut(dialog, "Shift+F").activated.emit()
    assert dialog.frame_panel.isVisibleTo(dialog)
    assert dialog.frame_btn.isChecked()
    dialog.frame_btn.clicked.emit()
    assert not dialog.frame_panel.isVisibleTo(dialog)
    dialog.close()
