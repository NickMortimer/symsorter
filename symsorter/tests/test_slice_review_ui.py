import time
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

import symsorter.ui.slice_review as sr
from symsorter.slices.session import load_local
from symsorter.tests.test_slice_session import make, open_session
from symsorter.ui.slice_review import SliceReviewDialog, open_slice_project_dialog

CROSSING = [[1350.0, 2000.0], [1420.0, 2000.0], [1420.0, 2040.0]]


@pytest.fixture
def session(tmp_path):
    proj, local = make(tmp_path)
    return open_session(proj, local)


def dialog(session, **kw):
    dlg = SliceReviewDialog(session, load_async=False, **kw)
    dlg.show_current()
    return dlg


def test_dialog_shows_slice_and_completes_with_key(qapp, session):
    dlg = dialog(session)
    qapp.processEvents()
    assert "example_r00" in dlg.windowTitle() and "1/2" in dlg.windowTitle()
    assert dlg.canvas._crop_rect == (64.0, 64.0, 688.0, 688.0)
    assert len(dlg.canvas._polygons) == 2
    assert dlg.canvas._polygons[0]["points"][0] == [1000.0 - 694, 2000.0 - 1669]
    QTest.keyClick(dlg, Qt.Key_C)
    assert session.status(0) == "complete" and session.index == 1


def test_crossing_polygon_shows_refusal_and_recentre_fixes_it(qapp, session):
    session.set_polygons(session.polygons() + [CROSSING])
    dlg = dialog(session)
    QTest.keyClick(dlg, Qt.Key_C)
    assert "Re-centre" in dlg.status_label.text() and session.status(0) is None
    QTest.keyClick(dlg, Qt.Key_R)
    assert session.position() == (796, 1733)
    QTest.keyClick(dlg, Qt.Key_C)
    assert session.status(0) == "complete"


def test_missing_frame_shows_message_not_crash(qapp, tmp_path):
    proj, local = make(tmp_path, frames=())
    dlg = dialog(open_session(proj, local))
    assert "frame missing" in dlg.canvas._message
    QTest.keyClick(dlg, Qt.Key_C)
    assert "Cannot save" in dlg.status_label.text()
    QTest.keyClick(dlg, Qt.Key_N)
    QTest.keyClick(dlg, Qt.Key_P)


def test_reject_asks_for_reason_and_skip_and_next(qapp, session, monkeypatch):
    dlg = dialog(session)
    monkeypatch.setattr(dlg, "_ask_reason", lambda: "blurry")
    QTest.keyClick(dlg, Qt.Key_X)
    assert session.status(0) == "reject" and session.index == 1
    QTest.keyClick(dlg, Qt.Key_Space)
    assert session.status(1) == "skip"


def test_reject_cancelled_saves_nothing(qapp, session, monkeypatch):
    dlg = dialog(session)
    monkeypatch.setattr(dlg, "_ask_reason", lambda: None)
    QTest.keyClick(dlg, Qt.Key_X)
    assert session.status(0) is None


def test_canvas_edit_pushes_frame_pixels_to_session(qapp, session):
    dlg = dialog(session)
    dlg.canvas._polygons[0]["points"][0] = [10.0, 20.0]
    dlg.canvas.polygonsChanged.emit()
    assert session.polygons()[0][0] == [10.0 + 694, 20.0 + 1669] and session.dirty


def test_adopt_neighbours_key(qapp, session):
    dlg = dialog(session)
    QTest.keyClick(dlg, Qt.Key_A)
    assert "dopted 0" in dlg.status_label.text()


@pytest.mark.parametrize("choice,moved", [("discard", True), ("cancel", False)])
def test_moving_with_unsaved_edits_asks(qapp, tmp_path, monkeypatch, choice, moved):
    proj, local = make(tmp_path)
    s = open_session(proj, local)
    s.index = 1
    dlg = dialog(s)
    dlg.canvas._polygons.append({"label": "turtle", "flags": {}, "shape_type": "polygon",
                                 "points": [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0]]})
    dlg.canvas.polygonsChanged.emit()
    monkeypatch.setattr(dlg, "_ask_leave", lambda: choice)
    QTest.keyClick(dlg, Qt.Key_P)
    assert (s.index == 0) is moved


def test_moving_with_unsaved_edits_save_as_complete(qapp, tmp_path, monkeypatch):
    proj, local = make(tmp_path)
    s = open_session(proj, local)
    s.index = 1
    dlg = dialog(s)
    dlg.canvas._polygons.append({"label": "turtle", "flags": {}, "shape_type": "polygon",
                                 "points": [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0]]})
    dlg.canvas.polygonsChanged.emit()
    monkeypatch.setattr(dlg, "_ask_leave", lambda: "save")
    QTest.keyClick(dlg, Qt.Key_P)
    assert s.status(1) == "complete" and s.index == 0


def test_async_load_delivers_image(qapp, session):
    dlg = SliceReviewDialog(session)
    dlg.show_current()
    end = time.time() + 20
    while dlg.canvas._src is None or dlg.canvas._src.isNull():
        qapp.processEvents()
        assert time.time() < end
        time.sleep(0.01)
    assert (dlg.canvas._src.width(), dlg.canvas._src.height()) == (752, 752)


def test_sam_click_shifts_points_to_frame_pixels(qapp, session, monkeypatch):
    seen = {}

    class Draft:
        points = [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0]]
        flags = {}

    def stub(path, model=None, **kw):
        seen.update(kw, exists=Path(path).exists())
        return Draft()

    monkeypatch.setattr(sr, "sam_polygon", stub)
    dlg = dialog(session)
    monkeypatch.setattr(dlg, "_ensure_sam", lambda: object())
    # view origin is (694, 1669); click at view (306, 331) -> frame (1000, 2000)
    dlg._on_point_seed(306.0, 331.0)
    left, top = 1000 - 112, 2000 - 112
    assert seen["exists"] and seen["size"] == 224
    assert seen["points"] == [[112.0, 112.0]]
    new = session.polygons()[-1]
    assert new[0] == [10.0 + left, 10.0 + top] and len(session.polygons()) == 3


def test_open_dialog_writes_local_settings(qapp, tmp_path, monkeypatch):
    proj, _ = make(tmp_path)
    monkeypatch.setattr(sr, "_ask_settings",
                        lambda parent, roots: ([tmp_path / "root"], "nick"))
    monkeypatch.setattr(SliceReviewDialog, "show", lambda self: None)
    dlg = open_slice_project_dialog(None, proj)
    assert dlg is not None
    local = load_local(proj)
    assert local.reviewer == "nick" and local.survey_roots == [tmp_path / "root"]


def test_open_dialog_cancelled_settings_returns_none(qapp, tmp_path, monkeypatch):
    proj, _ = make(tmp_path)
    monkeypatch.setattr(sr, "_ask_settings", lambda parent, roots: None)
    assert open_slice_project_dialog(None, proj) is None
