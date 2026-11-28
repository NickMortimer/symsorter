import json
from pathlib import Path

import pytest

from symsorter.slices.contract import (
    ContractError,
    ReviewRow,
    append_review,
    load_project,
    load_reviews,
    outline_row,
)

FIX = Path(__file__).parent / "fixtures" / "slices"


def test_load_example_project():
    p = load_project(FIX / "project.json")
    assert (p.project_id, p.slice_px, p.margin_px) == ("example_r00", 624, 64)
    s = p.slices[0]
    assert (s.flight, s.frame_stem, s.x0, s.y0, s.size) == (
        "TESTA_20240101T0900", "DJI_0001", 758, 1733, 624)
    assert len(s.polygons) == 2


def test_last_review_wins_and_missing_file_is_empty(tmp_path):
    r = load_reviews(FIX / "review.jsonl")
    assert r["AU--TESTA--TESTA_20240101T0900__DJI_0001__758_1733"].status == "complete"
    assert load_reviews(tmp_path / "none.jsonl") == {}


def test_append_then_load_round_trips_recentred_row(tmp_path):
    row = ReviewRow("example_r00", "id1", "complete",
                    (((1.0, 2.0), (3.0, 2.0), (2.0, 4.0)),), "nick",
                    "2026-12-01T10:00:00+10:00", x0=700, y0=1700)
    append_review(tmp_path / "r.jsonl", row)
    assert load_reviews(tmp_path / "r.jsonl") == {"id1": row}
    raw = json.loads((tmp_path / "r.jsonl").read_text())
    assert raw["schema"] == "symsorter.slice_review/1" and "reason" not in raw


def test_torn_last_line_is_skipped_but_bad_middle_line_is_an_error(tmp_path):
    good = (FIX / "review.jsonl").read_text()
    (tmp_path / "a.jsonl").write_text(good + '{"schema": "symsorter.sl')
    with pytest.warns(UserWarning):
        assert len(load_reviews(tmp_path / "a.jsonl")) == 2
    lines = good.splitlines()
    (tmp_path / "b.jsonl").write_text(lines[0] + "\n{bad\n" + lines[1] + "\n")
    with pytest.raises(ContractError, match="b.jsonl:2"):
        load_reviews(tmp_path / "b.jsonl")


def test_unknown_project_schema_is_refused(tmp_path):
    d = json.loads((FIX / "project.json").read_text())
    d["schema"] = "marine_count.slice_project/2"
    (tmp_path / "project.json").write_text(json.dumps(d))
    with pytest.raises(ContractError):
        load_project(tmp_path / "project.json")


def test_outline_row_matches_the_example():
    row = outline_row(
        "TESTA_20240101T0900", "DJI_0001",
        [[1000.0, 2000.0], [1040.0, 2000.0], [1040.0, 2040.0], [1000.0, 2040.0]],
        "thumb:example_DJI_0001_det_0_turtle_224px.jpg", (5472, 3648))
    assert row == json.loads((FIX / "outlines.jsonl").read_text().splitlines()[0])
