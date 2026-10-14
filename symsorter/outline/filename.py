"""Parse sorted turtle thumb filenames for provenance fields.

Filenames look like:
  ``051_DJIP4_RGB_AU_GRAV_20221023T064910_0723_det_10_turtle_3_224px.jpg``

Patterns ported from turtledrone ``build_compare_index.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_ORIGINAL_STEM_PATTERN = re.compile(
    r"^(?:\d+_)?"
    r"(?P<original>[A-Z0-9]+_[A-Z]+_[A-Z]{2}_[A-Z0-9]+_"
    r"\d{8}T\d{6}_\d{4})"
    r"(?:_det_\d+_.+)?$",
    re.IGNORECASE,
)
_DET_INDEX_PATTERN = re.compile(r"_det_(?P<det>\d+)", re.IGNORECASE)
_CONF_PREFIX_PATTERN = re.compile(r"^(?P<conf>\d{2,3})_")
# Site token inside DJIP4 stem: DJIP4_RGB_AU_<SITE>_<datetime>_<frame>
_SITE_PATTERN = re.compile(
    r"^[A-Z0-9]+_[A-Z]+_[A-Z]{2}_(?P<site>[A-Z0-9]+)_",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedThumbName:
    """Provenance fields extracted from a sorted thumb basename."""

    thumb_stem: str
    conf_prefix: str | None
    conf: float | None
    frame_stem: str | None
    det_index: int | None
    site: str | None


def parse_thumb_filename(name: str | Path) -> ParsedThumbName:
    """Parse conf / frame stem / det index from a thumb path or basename."""
    stem = Path(name).stem
    conf_prefix: str | None = None
    conf: float | None = None
    conf_match = _CONF_PREFIX_PATTERN.match(stem)
    if conf_match is not None:
        conf_prefix = conf_match.group("conf")
        try:
            conf = int(conf_prefix) / 100.0
        except ValueError:
            conf = None

    frame_stem: str | None = None
    site: str | None = None
    original_match = _ORIGINAL_STEM_PATTERN.match(stem)
    if original_match is not None:
        frame_stem = original_match.group("original")
        site_match = _SITE_PATTERN.match(frame_stem)
        if site_match is not None:
            site = site_match.group("site")

    det_index: int | None = None
    det_match = _DET_INDEX_PATTERN.search(stem)
    if det_match is not None:
        det_index = int(det_match.group("det"))

    return ParsedThumbName(
        thumb_stem=stem,
        conf_prefix=conf_prefix,
        conf=conf,
        frame_stem=frame_stem,
        det_index=det_index,
        site=site,
    )
