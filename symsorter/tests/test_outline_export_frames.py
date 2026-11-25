"""Tests for the outline-export-frames CLI."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import NamedTuple

import pytest
from PIL import Image

from symsorter.commands.outline_export_frames import (
    link_frame,
    outline_export_frames_command,
)

THUMB = "050_DJIP4_RGB_AU_MANGB_20230503T113703_2116_det_0_turtle_3_224px.jpg"
STEM = "DJIP4_RGB_AU_MANGB_20230503T113703_2116"
FLIGHT = "MANGB_20230503T1137"
ROI_HEADER = (
    "view,crop_basename,flight,x1,y1,x2,y2,confidence,class_name,original_image,status"
)
# 640x480 frame; box centred at (300, 200) → crop window (188, 88, 412, 312)


class _Tree(NamedTuple):
    surface: Path
    finds: Path
    csv: Path
    out: Path
    frame: Path
    training: Path


def _roi_line(
    view: str,
    name: str,
    flight: str,
    stem: str,
    *,
    status: str = "ok",
) -> str:
    return (
        f"{view},{name},{flight},290,190,310,210,0.5,turtle_3,"
        f"/media/x/Timor2/surveys/AU/MANGB/{flight}/{stem}.JPG,{status}\n"
    )


def _write_thumb(
    folder: Path,
    name: str,
    polys: list[tuple[list[list[int]], bool]],
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224)).save(folder / name, "JPEG")
    shapes = [
        {
            "label": "turtle",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {"locked": locked},
        }
        for points, locked in polys
    ]
    payload = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": name,
        "imageData": None,
        "imageHeight": 224,
        "imageWidth": 224,
    }
    (folder / name).with_suffix(".json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _tree(tmp_path: Path) -> _Tree:
    """Build surface thumb, ROI CSV, and a 640×480 frame."""
    training = tmp_path / "training"
    surface = training / "surface"
    _write_thumb(
        surface,
        THUMB,
        [
            ([[10, 10], [50, 10], [50, 50]], True),
            ([[1, 1], [4, 1], [4, 4]], False),
        ],
    )
    csv_path = training / "thumb_roi.csv"
    csv_path.write_text(
        ROI_HEADER + "\n" + _roi_line("surface", THUMB, FLIGHT, STEM),
        encoding="utf-8",
    )
    finds = tmp_path / "finds"
    frame = finds / FLIGHT / "images" / f"{STEM}.JPG"
    frame.parent.mkdir(parents=True)
    Image.new("RGB", (640, 480)).save(frame, "JPEG")
    return _Tree(
        surface=surface,
        finds=finds,
        csv=csv_path,
        out=tmp_path / "out",
        frame=frame,
        training=training,
    )


def _manifest(out: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_export_ok_writes_link_json_manifest(tmp_path: Path) -> None:
    t = _tree(tmp_path)
    rc = outline_export_frames_command(
        thumbs=[t.surface],
        survey_roots=[t.finds],
        roi_csv=t.csv,
        out=t.out,
    )
    assert rc == 0
    jpg, js = t.out / FLIGHT / f"{STEM}.JPG", t.out / FLIGHT / f"{STEM}.json"
    assert os.path.samefile(jpg, t.frame)
    doc = json.loads(js.read_text())
    assert (doc["imageWidth"], doc["imageHeight"], doc["imagePath"]) == (
        640,
        480,
        f"{STEM}.JPG",
    )
    assert len(doc["shapes"]) == 1
    shape = doc["shapes"][0]
    assert shape["points"] == [[198, 98], [238, 98], [238, 138]]
    assert shape["flags"]["source_thumb"] == THUMB and shape["flags"]["locked"] is True
    assert shape["flags"]["crop_xyxy"] == [188, 88, 412, 312]
    rows = _manifest(t.out)
    assert rows[0]["status"] == "ok" and rows[0]["n_locked"] == 1
    assert json.loads((t.out / "summary.json").read_text())["by_status"] == {"ok": 1}


def test_statuses_no_locked_no_row_missing(tmp_path: Path) -> None:
    t = _tree(tmp_path)
    unlocked = "051_DJIP4_RGB_AU_MANGB_20230503T113703_2116_det_9_turtle_3_224px.jpg"
    no_row = "052_DJIP4_RGB_AU_MANGB_20230503T113703_2116_det_8_turtle_3_224px.jpg"
    missing_stem = "DJIP4_RGB_AU_MANGB_20230503T113703_9999"
    missing = "053_DJIP4_RGB_AU_MANGB_20230503T113703_9999_det_0_turtle_3_224px.jpg"
    _write_thumb(t.surface, unlocked, [([[1, 1], [2, 1], [2, 2]], False)])
    _write_thumb(t.surface, no_row, [([[10, 10], [20, 10], [20, 20]], True)])
    _write_thumb(t.surface, missing, [([[10, 10], [20, 10], [20, 20]], True)])
    with t.csv.open("a", encoding="utf-8") as handle:
        handle.write(_roi_line("surface", unlocked, FLIGHT, STEM))
        handle.write(_roi_line("surface", missing, FLIGHT, missing_stem))
    outline_export_frames_command(
        thumbs=[t.surface],
        survey_roots=[t.finds],
        roi_csv=t.csv,
        out=t.out,
    )
    status = {row["thumb"]: row["status"] for row in _manifest(t.out)}
    assert status[unlocked] == "no_locked"
    assert status[no_row] == "no_roi_row"
    assert status[missing] == "frame_missing"


def test_rerun_without_overwrite_skips_existing(tmp_path: Path) -> None:
    t = _tree(tmp_path)
    assert (
        outline_export_frames_command(
            thumbs=[t.surface],
            survey_roots=[t.finds],
            roi_csv=t.csv,
            out=t.out,
        )
        == 0
    )
    js = t.out / FLIGHT / f"{STEM}.json"
    mtime = js.stat().st_mtime_ns
    rc = outline_export_frames_command(
        thumbs=[t.surface],
        survey_roots=[t.finds],
        roi_csv=t.csv,
        out=t.out,
    )
    assert rc == 0
    assert _manifest(t.out)[0]["status"] == "skipped_existing"
    assert js.stat().st_mtime_ns == mtime


def test_rerun_with_overwrite_rebuilds_and_existing_link_is_ok(tmp_path: Path) -> None:
    t = _tree(tmp_path)
    outline_export_frames_command(
        thumbs=[t.surface],
        survey_roots=[t.finds],
        roi_csv=t.csv,
        out=t.out,
    )
    js = t.out / FLIGHT / f"{STEM}.json"
    doc = json.loads(js.read_text())
    doc["shapes"] = []
    js.write_text(json.dumps(doc), encoding="utf-8")
    rc = outline_export_frames_command(
        thumbs=[t.surface],
        survey_roots=[t.finds],
        roi_csv=t.csv,
        out=t.out,
        overwrite=True,
    )
    assert rc == 0
    assert _manifest(t.out)[0]["status"] == "ok"
    restored = json.loads(js.read_text())
    assert len(restored["shapes"]) == 1
    assert os.path.samefile(t.out / FLIGHT / f"{STEM}.JPG", t.frame)


def test_surface_and_deep_share_frame(tmp_path: Path) -> None:
    t = _tree(tmp_path)
    deep_thumb = "051_DJIP4_RGB_AU_MANGB_20230503T113703_2116_det_1_turtle_3_224px.jpg"
    deep = t.training / "deep"
    _write_thumb(deep, deep_thumb, [([[12, 12], [40, 12], [40, 40]], True)])
    with t.csv.open("a", encoding="utf-8") as handle:
        handle.write(_roi_line("deep", deep_thumb, FLIGHT, STEM))
    rc = outline_export_frames_command(
        thumbs=[t.surface, deep],
        survey_roots=[t.finds],
        roi_csv=t.csv,
        out=t.out,
    )
    assert rc == 0
    flight_dir = t.out / FLIGHT
    assert sorted(path.name for path in flight_dir.iterdir()) == [
        f"{STEM}.JPG",
        f"{STEM}.json",
    ]
    assert os.path.samefile(flight_dir / f"{STEM}.JPG", t.frame)
    doc = json.loads((flight_dir / f"{STEM}.json").read_text())
    assert len(doc["shapes"]) == 2
    assert {shape["flags"]["view"] for shape in doc["shapes"]} == {"surface", "deep"}
    assert [row["status"] for row in _manifest(t.out)] == ["ok", "ok"]


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    t = _tree(tmp_path)
    rc = outline_export_frames_command(
        thumbs=[t.surface],
        survey_roots=[t.finds],
        roi_csv=t.csv,
        out=t.out,
        dry_run=True,
    )
    assert rc == 0
    assert not t.out.exists() or not any(t.out.rglob("*"))


def test_link_frame_exdev_falls_back_to_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "frame.JPG"
    src.write_bytes(b"frame")
    dst = tmp_path / "out" / "frame.JPG"
    dst.parent.mkdir()
    monkeypatch.setattr(
        os, "link", lambda *a: (_ for _ in ()).throw(OSError(errno.EXDEV, "x"))
    )
    assert link_frame(src, dst, "hard") == "sym" and dst.is_symlink()
    assert os.readlink(dst) == str(src.resolve())


def test_link_failed_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        os, "link", lambda *a: (_ for _ in ()).throw(OSError(errno.EACCES, "x"))
    )
    t = _tree(tmp_path)
    rc = outline_export_frames_command(
        thumbs=[t.surface],
        survey_roots=[t.finds],
        roi_csv=t.csv,
        out=t.out,
    )
    assert rc == 1
    assert not (t.out / FLIGHT / f"{STEM}.json").exists()
    assert _manifest(t.out)[0]["status"] == "link_failed"


def test_cli_registered() -> None:
    from typer.testing import CliRunner

    from symsorter.cli import app

    result = CliRunner().invoke(app, ["outline-export-frames", "--help"])
    assert result.exit_code == 0 and "--survey-root" in result.output
