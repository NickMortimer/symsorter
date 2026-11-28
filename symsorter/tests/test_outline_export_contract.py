from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from PIL import Image

from symsorter.commands.outline_export_contract import export_contract, load_denylist
from symsorter.outline.frames import CropWindow, thumb_to_frame

ROI_HEADER = "view,crop_basename,flight,x1,y1,x2,y2,confidence,class_name,original_image,status"
TA = "050_DJIP4_RGB_AU_TESTA_20240101T090001_0001_det_0_turtle_3_224px.jpg"
TB = "051_DJIP4_RGB_AU_SEAL_20240101T090001_0002_det_0_turtle_3_224px.jpg"
LOCKED = [[10, 10], [50, 10], [50, 50]]
CROP = CropWindow(188, 88, 412, 312)  # box centred (300,200) in 640x480
DENY = frozenset({"AU/SEAL/SEAL_20240101T0900"})


def _thumb(folder: Path, name: str, polys) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224)).save(folder / name, "JPEG")
    shapes = [{"label": "turtle", "points": p, "group_id": None, "shape_type": "polygon",
               "flags": {"locked": lk}} for p, lk in polys]
    (folder / name).with_suffix(".json").write_text(json.dumps({
        "version": "5.0.1", "flags": {}, "shapes": shapes, "imagePath": name,
        "imageData": None, "imageHeight": 224, "imageWidth": 224}))


def _roi(name, flight, stem):
    return (f"surface,{name},{flight},290,190,310,210,0.5,turtle_3,"
            f"/x/AU/T/{flight}/{stem}.JPG,ok\n")


def _tree(tmp_path: Path):
    thumbs = tmp_path / "thumbs"
    _thumb(thumbs, TA, [(LOCKED, True), ([[1, 1], [4, 1], [4, 4]], False)])
    _thumb(thumbs, TB, [(LOCKED, True)])
    csv = tmp_path / "thumb_roi.csv"
    csv.write_text(ROI_HEADER + "\n" + _roi(TA, "TESTA_20240101T0900", "DJI_0001")
                   + _roi(TB, "SEAL_20240101T0900", "DJI_0002"))
    root = tmp_path / "surveys"
    for sub, stem in (("AU/TESTA/TESTA_20240101T0900", "DJI_0001"),
                      ("AU/SEAL/SEAL_20240101T0900", "DJI_0002")):
        f = root / sub / f"{stem}.JPG"
        f.parent.mkdir(parents=True)
        Image.new("RGB", (640, 480)).save(f, "JPEG")
    return thumbs, csv, root


def test_locked_polygons_exported_in_frame_pixels(tmp_path):
    thumbs, csv, root = _tree(tmp_path)
    out = tmp_path / "o.jsonl"
    counts = export_contract([thumbs], csv, [root], DENY, out)
    assert counts == {"ok": 1, "sealed": 1, "no_locked": 0, "no_roi_row": 0, "frame_missing": 0,
                      "frame_unreadable": 0}
    [row] = [json.loads(l) for l in out.read_text().splitlines()]
    assert row["schema"] == "symsorter.outlines/1" and row["label"] == "turtle"
    assert row["flight"] == "TESTA_20240101T0900" and row["frame_stem"] == "DJI_0001"
    assert row["frame_wh"] == [640, 480] and row["source"] == f"thumb:{TA}"
    assert row["points"] == [[float(x), float(y)] for x, y in thumb_to_frame(LOCKED, CROP)]


def test_sealed_flight_never_written(tmp_path):
    thumbs, csv, root = _tree(tmp_path)
    out = tmp_path / "o.jsonl"
    export_contract([thumbs], csv, [root], DENY, out)
    assert "SEAL_" not in out.read_text()


def test_missing_or_empty_denylist_is_refused(tmp_path):
    with pytest.raises(typer.BadParameter):
        load_denylist(tmp_path / "nope.txt")
    empty = tmp_path / "e.txt"
    empty.write_text("# only a comment\n\n")
    with pytest.raises(typer.BadParameter):
        load_denylist(empty)
    ok = tmp_path / "d.txt"
    ok.write_text("# c\nAU/SEAL/SEAL_1\n\n")
    assert load_denylist(ok) == frozenset({"AU/SEAL/SEAL_1"})


def test_missing_frame_and_missing_roi_row_are_counted(tmp_path):
    thumbs, csv, root = _tree(tmp_path)
    no_row = "052_DJIP4_RGB_AU_TESTA_20240101T090001_0003_det_0_turtle_3_224px.jpg"
    no_frame = "053_DJIP4_RGB_AU_TESTA_20240101T090001_0009_det_0_turtle_3_224px.jpg"
    unlocked = "054_DJIP4_RGB_AU_TESTA_20240101T090001_0001_det_5_turtle_3_224px.jpg"
    _thumb(thumbs, no_row, [(LOCKED, True)])
    _thumb(thumbs, no_frame, [(LOCKED, True)])
    _thumb(thumbs, unlocked, [([[1, 1], [4, 1], [4, 4]], False)])
    with csv.open("a") as fh:
        fh.write(_roi(no_frame, "TESTA_20240101T0900", "DJI_0009"))
        fh.write(_roi(unlocked, "TESTA_20240101T0900", "DJI_0001"))
    counts = export_contract([thumbs], csv, [root], DENY, tmp_path / "o.jsonl")
    assert counts == {"ok": 1, "sealed": 1, "no_locked": 1, "no_roi_row": 1, "frame_missing": 1,
                      "frame_unreadable": 0}


def test_rows_sorted_and_output_overwritten(tmp_path):
    thumbs, csv, root = _tree(tmp_path)
    out = tmp_path / "o.jsonl"
    out.write_text("stale\n")
    export_contract([thumbs], csv, [root], frozenset({"X/Y/Z"}), out)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["flight"] for r in rows] == ["SEAL_20240101T0900", "TESTA_20240101T0900"]


def test_cli_prints_counts(tmp_path):
    from typer.testing import CliRunner
    from symsorter.cli import app
    thumbs, csv, root = _tree(tmp_path)
    deny = tmp_path / "sealed.txt"
    deny.write_text("AU/SEAL/SEAL_20240101T0900\n")
    res = CliRunner().invoke(app, [
        "outline-export-contract", "--thumbs", str(thumbs), "--roi-csv", str(csv),
        "--survey-root", str(root), "--denylist", str(deny), "--out", str(tmp_path / "o.jsonl")])
    assert res.exit_code == 0, res.output
    assert "ok=1" in res.output and "sealed=1" in res.output


def test_denylist_trailing_slash_still_seals(tmp_path):
    thumbs, csv, root = _tree(tmp_path)
    out = tmp_path / "o.jsonl"
    counts = export_contract([thumbs], csv, [root], frozenset({"AU/SEAL/SEAL_20240101T0900/"}), out)
    assert counts["sealed"] == 1 and "SEAL_" not in out.read_text()


def test_corrupt_frame_is_counted_not_fatal(tmp_path):
    thumbs, csv, root = _tree(tmp_path)
    (root / "AU/TESTA/TESTA_20240101T0900/DJI_0001.JPG").write_bytes(b"junk")
    out = tmp_path / "o.jsonl"
    counts = export_contract([thumbs], csv, [root], DENY, out)
    assert counts["frame_unreadable"] == 1 and counts["ok"] == 0
    assert out.read_text() == ""
