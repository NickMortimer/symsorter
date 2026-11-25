"""Tests for ROI index and thumb/frame geometry."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from symsorter.outline.frames import (
    FrameResolver,
    RoiIndex,
    RoiRow,
    crop_window_for_row,
    frame_size,
    frame_to_thumb,
    iter_flat_thumbs,
    thumb_to_frame,
)

THUMB = "050_DJIP4_RGB_AU_MANGB_20230503T113703_2116_det_0_turtle_3_224px.jpg"
STEM = "DJIP4_RGB_AU_MANGB_20230503T113703_2116"
FLIGHT = "MANGB_20230503T1137"
ROI_HEADER = (
    "view,crop_basename,flight,x1,y1,x2,y2,confidence,class_name,original_image,status"
)


def test_roi_index_get_and_status_filter(tmp_path: Path) -> None:
    csv = tmp_path / "thumb_roi.csv"
    csv.write_text(
        ROI_HEADER + "\n"
        f"surface,{THUMB},{FLIGHT},290,190,310,210,0.5,turtle_3,"
        f"/media/x/Timor2/surveys/AU/MANGB/{FLIGHT}/{STEM}.JPG,ok\n"
        f"deep,bad_224px.jpg,{FLIGHT},0,0,1,1,0.5,turtle_3,/a/{STEM}.JPG,failed\n"
    )
    idx = RoiIndex.load(csv)
    row = idx.get(THUMB)
    assert row is not None
    assert row.original_stem == STEM and row.view == "surface"
    assert idx.get("bad_224px.jpg") is None
    assert idx.get("nope.jpg") is None
    assert [r.crop_basename for r in idx.same_frame(FLIGHT, STEM)] == [THUMB]


def test_crop_window_and_round_trip() -> None:
    row = RoiRow("surface", THUMB, FLIGHT, 290, 190, 310, 210, STEM)
    crop = crop_window_for_row(row, (640, 480))
    assert (crop.left, crop.top, crop.right, crop.bottom) == (188, 88, 412, 312)
    assert crop.method == "roi_csv"
    pts = [[0.0, 0.0], [223.5, 10.25], [-5.0, 230.0]]
    assert frame_to_thumb(thumb_to_frame(pts, crop), crop) == pts


def test_crop_window_clamped_at_frame_border() -> None:
    row = RoiRow("surface", THUMB, FLIGHT, 630, 470, 640, 480, STEM)
    crop = crop_window_for_row(row, (640, 480))
    assert (crop.left, crop.top, crop.right, crop.bottom) == (416, 256, 640, 480)
    assert thumb_to_frame([[224.0, 224.0]], crop) == [[640.0, 480.0]]


def test_frame_size_reads_header(tmp_path: Path) -> None:
    p = tmp_path / "f.JPG"
    Image.new("RGB", (640, 480)).save(p, "JPEG")
    assert frame_size(p) == (640, 480)


def test_iter_flat_thumbs_skips_appledouble(tmp_path: Path) -> None:
    Image.new("RGB", (224, 224)).save(tmp_path / THUMB)
    (tmp_path / f"._{THUMB}").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    Image.new("RGB", (224, 224)).save(tmp_path / "sub" / THUMB)
    assert iter_flat_thumbs(tmp_path) == [tmp_path / THUMB]


def _img(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(p, "JPEG")


def test_resolver_yolo_layout_first_root_wins(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _img(a / FLIGHT / "images" / f"{STEM}.JPG")
    _img(b / FLIGHT / "images" / f"{STEM}.JPG")
    assert FrameResolver([a, b]).find(FLIGHT, STEM) == a / FLIGHT / "images" / f"{STEM}.JPG"


def test_resolver_survey_layout_lowercase_ext(tmp_path: Path) -> None:
    p = tmp_path / "AU" / "MANGB" / FLIGHT / f"{STEM}.jpg"
    _img(p)
    assert FrameResolver([tmp_path]).find(FLIGHT, STEM) == p


def test_resolver_recursive_fallback(tmp_path: Path) -> None:
    p = tmp_path / "odd" / "place" / f"{STEM}.JPG"
    _img(p)
    assert FrameResolver([tmp_path / "missing", tmp_path]).find(FLIGHT, STEM) == p


def test_resolver_ignores_json_and_appledouble(tmp_path: Path) -> None:
    d = tmp_path / FLIGHT / "images"
    d.mkdir(parents=True)
    (d / f"{STEM}.json").write_text("{}")
    (d / f"._{STEM}.JPG").write_bytes(b"x")
    assert FrameResolver([tmp_path]).find(FLIGHT, STEM) is None


def test_resolver_missing(tmp_path: Path) -> None:
    assert FrameResolver([tmp_path]).find(FLIGHT, STEM) is None
