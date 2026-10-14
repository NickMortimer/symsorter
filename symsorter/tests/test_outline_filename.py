"""Tests for sorted thumb filename parsing."""

from __future__ import annotations

from symsorter.outline.filename import parse_thumb_filename


def test_parse_grav_example() -> None:
    name = "051_DJIP4_RGB_AU_GRAV_20221023T064910_0723_det_10_turtle_3_224px.jpg"
    parsed = parse_thumb_filename(name)
    assert parsed.conf_prefix == "051"
    assert parsed.conf == 0.51
    assert parsed.frame_stem == "DJIP4_RGB_AU_GRAV_20221023T064910_0723"
    assert parsed.det_index == 10
    assert parsed.site == "GRAV"
    assert parsed.thumb_stem == name.replace(".jpg", "")


def test_parse_without_conf_prefix() -> None:
    name = "DJIP4_RGB_AU_NEDS_20190526T125451_0242_det_0_turtle_3_224px.jpg"
    parsed = parse_thumb_filename(name)
    assert parsed.conf_prefix is None
    assert parsed.frame_stem == "DJIP4_RGB_AU_NEDS_20190526T125451_0242"
    assert parsed.det_index == 0
