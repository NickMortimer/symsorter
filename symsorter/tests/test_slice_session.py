import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from symsorter.slices.contract import load_reviews
from symsorter.slices.session import (
    LocalSettings, ReviewSession, SaveRefused, load_local, save_local,
)

FIX = Path(__file__).parent / "fixtures" / "slices" / "project.json"
FLIGHT = "AU/TESTA/TESTA_20240101T0900"
S1 = "AU--TESTA--TESTA_20240101T0900__DJI_0001__758_1733"
S2 = "AU--TESTA--TESTA_20240101T0900__DJI_0002__3900_400"
CLOCK = lambda: "2026-11-28T10:00:00+10:00"  # noqa: E731


def make(tmp_path, frames=("DJI_0001", "DJI_0002"), project=None):
    proj = tmp_path / "proj"
    proj.mkdir()
    if project is None:
        shutil.copy(FIX, proj / "project.json")
    else:
        (proj / "project.json").write_text(json.dumps(project))
    d = tmp_path / "root" / "AU" / "TESTA" / "TESTA_20240101T0900"
    d.mkdir(parents=True)
    for f in frames:
        Image.new("RGB", (5472, 3648)).save(d / f"{f}.JPG")
    return proj, LocalSettings([tmp_path / "root"], "nick")


def open_session(proj, local):
    return ReviewSession(proj, local, clock=CLOCK)


@pytest.fixture
def env(tmp_path):
    proj, local = make(tmp_path)
    return proj, local, open_session(proj, local)


@pytest.fixture
def session(env):
    return env[2]


def test_starts_at_first_unreviewed_and_saves_complete(env):
    proj, _, session = env
    assert session.current.frame_stem == "DJI_0001"
    row = session.save("complete")
    assert row.status == "complete" and len(row.polygons) == 2 and row.x0 is None
    assert row.time == "2026-11-28T10:00:00+10:00" and row.reviewer == "nick"
    assert load_reviews(proj / "review.jsonl")[row.slice_id] == row


def test_complete_refused_when_polygon_crosses_edge_then_recentre_allows_it(session):
    session.set_polygons(session.polygons() + [[[1350.0, 2000.0], [1420.0, 2000.0], [1420.0, 2040.0]]])
    with pytest.raises(SaveRefused, match="Re-centre"):
        session.save("complete")
    assert session.recentre_current() and session.position() == (796, 1733)
    row = session.save("complete")
    assert (row.x0, row.y0) == (796, 1733)


def test_polygon_wholly_in_margin_is_dropped_on_save(session):
    session.set_polygons(session.polygons() + [[[700.0, 1700.0], [740.0, 1700.0], [720.0, 1720.0]]])
    assert len(session.save("complete").polygons) == 2


def test_degenerate_polygon_dropped_on_save(session):
    session.set_polygons(session.polygons() + [[], [[1000.0, 2000.0]]])
    assert len(session.save("complete").polygons) == 2


def test_reject_keeps_reason_and_ignores_fit(env):
    proj, _, session = env
    session.set_polygons(session.polygons() + [[[1350.0, 2000.0], [1420.0, 2000.0], [1420.0, 2040.0]]])
    row = session.save("reject", "blurry")
    assert row.status == "reject" and row.reason == "blurry" and len(row.polygons) == 3
    assert load_reviews(proj / "review.jsonl")[S1].reason == "blurry"
    assert session.status() == "reject"


def test_resume_after_reopen_starts_at_next_unreviewed(tmp_path):
    proj, local = make(tmp_path)
    s = open_session(proj, local)
    s.save("complete")
    s2 = open_session(proj, local)
    assert s2.current.slice_id == S2 and s2.index == 1
    assert s2.status(0) == "complete"


def test_resume_uses_saved_position_and_polygons(tmp_path):
    proj, local = make(tmp_path)
    s = open_session(proj, local)
    s.set_polygons(s.polygons() + [[[1350.0, 2000.0], [1420.0, 2000.0], [1420.0, 2040.0]]])
    s.recentre_current()
    s.save("complete")
    s2 = open_session(proj, local)
    assert s2.index == 1 and s2.prev() is True and s2.index == 0
    assert s2.position() == (796, 1733) and len(s2.polygons()) == 3


def test_missing_frame_is_skipped_and_counted(tmp_path):
    proj, local = make(tmp_path, frames=("DJI_0001",))
    s = open_session(proj, local)
    assert s.progress()["missing"] == 1
    assert s.lookup(1).problem == "frame missing"
    s.save("complete")
    assert s.next_unreviewed() is False and s.index == 0
    p = s.progress()
    assert p == {"total": 2, "complete": 1, "reject": 0, "skip": 0, "unreviewed": 0, "missing": 1}


def test_unreadable_frame_behaves_as_missing(tmp_path):
    proj, local = make(tmp_path)
    (tmp_path / "root/AU/TESTA/TESTA_20240101T0900/DJI_0002.JPG").write_bytes(b"junk")
    s = open_session(proj, local)
    assert s.lookup(1).problem == "frame unreadable"
    assert s.progress()["missing"] == 1 and s.next_unreviewed() is False


def test_dirty_session_does_not_move(session):
    session.set_polygons([])
    assert session.next_unreviewed() is False and session.index == 0
    session.save("skip")
    assert session.next_unreviewed() is True and session.index == 1
    session.set_polygons([[[3950.0, 450.0], [3990.0, 450.0], [3990.0, 480.0]]])
    assert session.prev() is False and session.index == 1


def test_prev_moves_back_when_clean(session):
    session.save("skip")
    session.next_unreviewed()
    assert session.prev() is True and session.index == 0
    assert session.prev() is False


def test_reviews_for_slices_not_in_project_are_ignored(tmp_path):
    proj, local = make(tmp_path)
    s = open_session(proj, local)
    row = s.save("complete")
    extra = json.dumps({"schema": "symsorter.slice_review/1", "project_id": "x",
                        "slice_id": "ghost", "status": "skip", "polygons": [],
                        "reviewer": "r", "time": "t"})
    with open(proj / "review.jsonl", "a") as f:
        f.write(extra + "\n")
    s2 = open_session(proj, local)
    assert s2.progress()["total"] == 2
    assert s2.progress()["skip"] == 0 and s2.progress()["complete"] == 1
    s2.save("skip")
    reviews = load_reviews(proj / "review.jsonl")
    assert "ghost" in reviews and row.slice_id in reviews


def _two_slice_project():
    sq = lambda x, y: {"label": "turtle", "points": [[x, y], [x + 40, y], [x + 40, y + 40]]}  # noqa: E731
    def spec(sid, x0, polys):
        return {"slice_id": sid, "survey_id": FLIGHT, "frame_id": "DJI_0001.JPG",
                "frame_wh": [5472, 3648], "x0": x0, "y0": 1000, "size": 624,
                "kind": "turtle", "polygons": polys}
    return {"schema": "marine_count.slice_project/1", "project_id": "p", "slice_px": 624,
            "margin_px": 64, "slices": [
                spec("a", 1000, [sq(1100, 1100)]),
                spec("b", 1400, [sq(1100, 1100), sq(1500, 1100)])]}


def test_adopt_neighbour_adds_once(tmp_path):
    proj, local = make(tmp_path, project=_two_slice_project())
    s = open_session(proj, local)
    assert s.current.slice_id == "a" and len(s.polygons()) == 1
    assert len(s.neighbours()) == 2          # b's two polygons touch a's view rect
    assert s.adopt_neighbours() == 1         # duplicate of a's own skipped
    assert len(s.polygons()) == 2 and s.dirty
    assert s.adopt_neighbours() == 0
    assert len(s.polygons()) == 2


def test_neighbours_use_latest_review(tmp_path):
    proj, local = make(tmp_path, project=_two_slice_project())
    s = open_session(proj, local)
    s.set_polygons([[[1500.0, 1100.0], [1540.0, 1100.0], [1540.0, 1140.0]]])
    s.save("complete")
    s.next_unreviewed()
    assert s.current.slice_id == "b"
    s.set_polygons([])
    # a's latest review (edited polygon) is the source; its project polygon at 1100 is not
    assert s.neighbours() == [[[1500.0, 1100.0], [1540.0, 1100.0], [1540.0, 1140.0]]]


def test_position_none_in_row_when_equal_to_project(session):
    row = session.save("skip")
    assert row.x0 is None and row.y0 is None


def test_recentre_false_when_too_big(session):
    session.set_polygons([[[0.0, 0.0], [900.0, 0.0], [900.0, 10.0]]])
    assert session.recentre_current() is False and session.position() == (758, 1733)


def test_local_settings_round_trip(tmp_path):
    assert load_local(tmp_path) is None
    s = LocalSettings([tmp_path / "a", tmp_path / "b"], "nick")
    save_local(tmp_path, s)
    assert (tmp_path / "project.local.json").exists()
    assert load_local(tmp_path) == s


def test_save_refused_for_missing_frame_writes_nothing(tmp_path):
    proj, local = make(tmp_path, frames=())
    s = open_session(proj, local)
    assert s.lookup().problem == "frame missing"
    for st in ("complete", "skip", "reject"):
        with pytest.raises(SaveRefused, match="frame missing"):
            s.save(st)
    assert not (proj / "review.jsonl").exists()


def test_save_refused_for_size_mismatch(tmp_path):
    proj, local = make(tmp_path)
    Image.new("RGB", (100, 100)).save(
        tmp_path / "root/AU/TESTA/TESTA_20240101T0900/DJI_0001.JPG")
    s = open_session(proj, local)
    assert s.current.slice_id == S2
    s.index = 0
    assert s.lookup().problem == "size mismatch 100x100"
    with pytest.raises(SaveRefused, match="size mismatch"):
        s.save("complete")
    assert not (proj / "review.jsonl").exists()


def test_polygons_returns_copy(session):
    session.polygons()[0][0][0] = -5.0
    session.polygons().append([])
    assert session.polygons()[0][0][0] == 1000.0 and len(session.polygons()) == 2
    assert not session.dirty


def test_lookup_failure_in_resolver_maps_to_unreadable(env, monkeypatch):
    _, _, s = env
    def boom(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(s._resolver, "find", boom)
    s._lookups.clear()
    with pytest.warns(UserWarning, match="disk gone"):
        assert s.lookup().problem == "frame unreadable"


def test_discard_changes_drops_unsaved_edits(session):
    session.set_polygons([])
    assert session.dirty
    session.discard_changes()
    assert not session.dirty and len(session.polygons()) == 2
