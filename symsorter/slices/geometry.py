"""Pure, Qt-free frame lookup and slice geometry for slice review.

All points are full-frame pixels ``[x, y]`` unless stated otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from symsorter.outline.frames import FrameResolver, frame_size
from symsorter.slices.contract import Polygon, SliceSpec

Rect = Tuple[int, int, int, int]  # (left, top, right, bottom)


@dataclass(frozen=True)
class FrameLookup:
    path: Optional[Path]
    problem: Optional[str]  # None, "frame missing" or "size mismatch WxH"


def locate_frame(spec: SliceSpec, resolver: FrameResolver) -> FrameLookup:
    """Find the spec's frame (resolver should be strict) and check its size."""
    path = resolver.find(spec.flight, spec.frame_stem)
    if path is None:
        return FrameLookup(None, "frame missing")
    w, h = frame_size(path)
    if (w, h) != tuple(spec.frame_wh):
        return FrameLookup(path, f"size mismatch {w}x{h}")
    return FrameLookup(path, None)


def view_rect(x0: int, y0: int, size: int, margin: int,
              frame_wh: Tuple[int, int]) -> Rect:
    """Slice square grown by ``margin`` on every side, clamped to the frame."""
    fw, fh = frame_wh
    return (max(0, x0 - margin), max(0, y0 - margin),
            min(fw, x0 + size + margin), min(fh, y0 + size + margin))


def to_view(points: Sequence[Sequence[float]], view: Rect) -> Polygon:
    return [[x - view[0], y - view[1]] for x, y in points]


def to_frame(points: Sequence[Sequence[float]], view: Rect) -> Polygon:
    return [[x + view[0], y + view[1]] for x, y in points]


def _bbox(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def touches(points: Sequence[Sequence[float]], x0: int, y0: int, size: int) -> bool:
    """Bounding box overlaps the slice square; edge contact counts as touching."""
    minx, miny, maxx, maxy = _bbox(points)
    return maxx >= x0 and minx <= x0 + size and maxy >= y0 and miny <= y0 + size


def inside(points: Sequence[Sequence[float]], x0: int, y0: int, size: int) -> bool:
    """Every point satisfies x0 <= x <= x0+size and y0 <= y <= y0+size."""
    return all(x0 <= x <= x0 + size and y0 <= y <= y0 + size for x, y in points)


def _recentre_axis(lo: float, hi: float, start: int, size: int,
                   limit: int) -> Optional[int]:
    lo_i, hi_i = math.floor(lo), math.ceil(hi)
    if hi_i - lo_i > size:
        return None
    pos = start
    if lo_i < pos:
        pos = lo_i
    elif hi_i > pos + size:
        pos = hi_i - size
    pos = max(0, min(pos, limit - size))
    if lo_i < pos or hi_i > pos + size:
        return None
    return pos


def recentre(polygons: List[Polygon], x0: int, y0: int, size: int,
             frame_wh: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Least move per axis so the slice holds every polygon, kept inside the frame.

    Returns None if the union bounding box is larger than ``size`` or cannot
    fit after clamping to the frame; ``(x0, y0)`` unchanged if all fits.
    """
    pts = [p for poly in polygons for p in poly]
    if not pts:
        return (x0, y0)
    minx, miny, maxx, maxy = _bbox(pts)
    nx = _recentre_axis(minx, maxx, x0, size, frame_wh[0])
    ny = _recentre_axis(miny, maxy, y0, size, frame_wh[1])
    if nx is None or ny is None:
        return None
    return (nx, ny)


def sam_window(x: float, y: float, frame_wh: Tuple[int, int],
               size: int = 224) -> Rect:
    """``size`` square centred on the click, clamped to the frame."""
    fw, fh = frame_wh
    left = max(0, min(int(round(x - size / 2)), fw - size))
    top = max(0, min(int(round(y - size / 2)), fh - size))
    return (left, top, left + size, top + size)
