from pathlib import Path

from PIL import Image

from symsorter.outline.frames import FrameResolver
from symsorter.slices.contract import SliceSpec
from symsorter.slices.geometry import (
    FrameLookup,
    inside,
    locate_frame,
    recentre,
    sam_window,
    to_frame,
    to_view,
    touches,
    view_rect,
)

FRAME = (5472, 3648)


def test_view_rect_clamps_margin_at_frame_edge():
    assert view_rect(758, 1733, 624, 64, FRAME) == (694, 1669, 1446, 2421)
    assert view_rect(0, 0, 624, 64, FRAME) == (0, 0, 688, 688)
    assert view_rect(4848, 3024, 624, 64, FRAME) == (4784, 2960, 5472, 3648)


def test_view_round_trip():
    v = (694, 1669, 1446, 2421)
    assert to_view([[1000.0, 2000.0]], v) == [[306.0, 331.0]]
    assert to_frame(to_view([[1000.0, 2000.0]], v), v) == [[1000.0, 2000.0]]


def test_recentre_moves_least_distance():
    poly = [[1350.0, 2000.0], [1420.0, 2000.0], [1420.0, 2040.0]]
    assert recentre([poly], 758, 1733, 624, FRAME) == (796, 1733)


def test_recentre_moves_left_up_and_uses_union():
    a = [[700.0, 1700.0], [800.0, 1750.0]]
    b = [[900.0, 2000.0], [950.0, 2000.0]]
    assert recentre([a, b], 758, 1733, 624, FRAME) == (700, 1700)


def test_recentre_float_points_use_floor_ceil():
    poly = [[1382.4, 2000.0], [1390.2, 2010.0]]
    assert recentre([poly], 758, 1733, 624, FRAME) == (767, 1733)


def test_recentre_clamps_at_frame_edge_and_gives_none_when_too_wide():
    edge = [[5400.0, 100.0], [5472.0, 100.0], [5472.0, 140.0]]
    assert recentre([edge], 4800, 0, 624, FRAME) == (4848, 0)
    wide = [[0.0, 0.0], [700.0, 0.0], [700.0, 10.0]]
    assert recentre([wide], 0, 0, 624, FRAME) is None
    tall = [[0.0, 0.0], [10.0, 700.0]]
    assert recentre([tall], 0, 0, 624, FRAME) is None


def test_recentre_none_when_outside_frame():
    out = [[5400.0, 100.0], [5500.0, 100.0]]
    assert recentre([out], 4848, 0, 624, FRAME) is None


def test_recentre_is_identity_when_all_inside():
    assert recentre([[[800.0, 1800.0], [810.0, 1800.0], [805.0, 1810.0]]],
                    758, 1733, 624, FRAME) == (758, 1733)


def test_inside_and_touches():
    sq = (100, 100, 50)
    assert inside([[100.0, 100.0], [150.0, 150.0]], *sq)
    assert not inside([[100.0, 100.0], [150.1, 150.0]], *sq)
    assert touches([[140.0, 140.0], [200.0, 200.0]], *sq)
    assert touches([[150.0, 150.0], [200.0, 200.0]], *sq)  # edge contact counts
    assert not touches([[151.0, 151.0], [200.0, 200.0]], *sq)
    assert not touches([[0.0, 0.0], [99.0, 300.0]], *sq)


def _spec(frame_wh):
    return SliceSpec("s", "AU/TESTA/TESTA_20240101T0900", "DJI_0001.JPG",
                     frame_wh, 0, 0, 624, "turtle", ())


def test_strict_lookup_finds_survey_layout_and_flags_size(tmp_path):
    d = tmp_path / "AU" / "TESTA" / "TESTA_20240101T0900"
    d.mkdir(parents=True)
    Image.new("RGB", (640, 480)).save(d / "DJI_0001.JPG")
    r = FrameResolver([tmp_path], strict=True)
    assert locate_frame(_spec((640, 480)), r) == FrameLookup(d / "DJI_0001.JPG", None)
    got = locate_frame(_spec((5472, 3648)), r)
    assert got == FrameLookup(d / "DJI_0001.JPG", "size mismatch 640x480")


def test_lookup_missing_when_only_stem_elsewhere(tmp_path):
    (tmp_path / "elsewhere").mkdir()
    Image.new("RGB", (640, 480)).save(tmp_path / "elsewhere" / "DJI_0001.JPG")
    got = locate_frame(_spec((640, 480)), FrameResolver([tmp_path], strict=True))
    assert got == FrameLookup(None, "frame missing")


def test_sam_window_clamps():
    assert sam_window(10, 10, FRAME) == (0, 0, 224, 224)
    assert sam_window(5470, 3640, FRAME) == (5248, 3424, 5472, 3648)
    assert sam_window(1000, 1000, FRAME) == (888, 888, 1112, 1112)
