"""Geometry helpers for interactive polygon editing (zoom fit, vertices, lasso)."""

from __future__ import annotations

import math
from typing import Any


def point_in_poly(x: float, y: float, pts: list[list[float]]) -> bool:
    """Ray-cast point-in-polygon."""
    inside = False
    n = len(pts)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def centroid(pts: list[list[float]]) -> tuple[float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def polygons_bbox(
    polygons: list[dict[str, Any]],
) -> tuple[float, float, float, float] | None:
    """Axis-aligned bbox (minx, miny, maxx, maxy) over all polygon points."""
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    found = False
    for poly in polygons:
        for p in poly.get("points") or []:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            found = True
            x, y = float(p[0]), float(p[1])
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)
    if not found:
        return None
    return minx, miny, maxx, maxy


def padded_bbox(
    bbox: tuple[float, float, float, float],
    *,
    pad_frac: float = 0.10,
    img_w: float,
    img_h: float,
) -> tuple[float, float, float, float]:
    """Expand bbox by ``pad_frac`` of its size and clamp to image bounds."""
    minx, miny, maxx, maxy = bbox
    bw = max(1.0, maxx - minx)
    bh = max(1.0, maxy - miny)
    px = bw * pad_frac
    py = bh * pad_frac
    return (
        max(0.0, minx - px),
        max(0.0, miny - py),
        min(float(img_w), maxx + px),
        min(float(img_h), maxy + py),
    )


def fit_view(
    *,
    view_w: float,
    view_h: float,
    img_w: float,
    img_h: float,
    polygons: list[dict[str, Any]] | None = None,
    pad_frac: float = 0.10,
) -> tuple[float, float, float]:
    """Return ``(zoom, offset_x, offset_y)`` fitting polys (+pad) or full image.

    Widget coords: ``wx = offset_x + ix * zoom``.
    """
    view_w = max(1.0, view_w)
    view_h = max(1.0, view_h)
    img_w = max(1.0, img_w)
    img_h = max(1.0, img_h)

    bbox = polygons_bbox(polygons or [])
    if bbox is None:
        minx, miny, maxx, maxy = 0.0, 0.0, img_w, img_h
    else:
        minx, miny, maxx, maxy = padded_bbox(
            bbox, pad_frac=pad_frac, img_w=img_w, img_h=img_h
        )

    bw = max(1.0, maxx - minx)
    bh = max(1.0, maxy - miny)
    zoom = min(view_w / bw, view_h / bh)
    # Center the target rect in the view
    ox = (view_w - bw * zoom) / 2.0 - minx * zoom
    oy = (view_h - bh * zoom) / 2.0 - miny * zoom
    return zoom, ox, oy


def zoom_at_pointer(
    *,
    zoom: float,
    offset_x: float,
    offset_y: float,
    pointer_x: float,
    pointer_y: float,
    factor: float,
    min_zoom: float = 0.2,
    max_zoom: float = 40.0,
) -> tuple[float, float, float]:
    """Zoom keeping the image point under the pointer fixed."""
    ix = (pointer_x - offset_x) / max(1e-9, zoom)
    iy = (pointer_y - offset_y) / max(1e-9, zoom)
    new_zoom = max(min_zoom, min(max_zoom, zoom * factor))
    ox = pointer_x - ix * new_zoom
    oy = pointer_y - iy * new_zoom
    return new_zoom, ox, oy


def nearest_vertex(
    pts: list[list[float]],
    x: float,
    y: float,
    *,
    max_dist: float,
) -> int | None:
    """Index of nearest vertex within ``max_dist``, else None."""
    best_i = None
    best_d = max_dist
    for i, p in enumerate(pts):
        d = math.hypot(float(p[0]) - x, float(p[1]) - y)
        if d <= best_d:
            best_d = d
            best_i = i
    return best_i


def _project_on_segment(
    x: float, y: float, x0: float, y0: float, x1: float, y1: float
) -> tuple[float, float, float, float]:
    """Return (proj_x, proj_y, t, dist) for projection onto segment."""
    dx, dy = x1 - x0, y1 - y0
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return x0, y0, 0.0, math.hypot(x - x0, y - y0)
    t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / denom))
    px, py = x0 + t * dx, y0 + t * dy
    return px, py, t, math.hypot(x - px, y - py)


def nearest_edge(
    pts: list[list[float]],
    x: float,
    y: float,
    *,
    max_dist: float,
) -> tuple[int, float, float] | None:
    """Nearest edge: ``(insert_after_index, proj_x, proj_y)`` within ``max_dist``."""
    n = len(pts)
    if n < 2:
        return None
    best: tuple[int, float, float, float] | None = None  # i, px, py, dist
    for i in range(n):
        x0, y0 = float(pts[i][0]), float(pts[i][1])
        x1, y1 = float(pts[(i + 1) % n][0]), float(pts[(i + 1) % n][1])
        px, py, t, dist = _project_on_segment(x, y, x0, y0, x1, y1)
        # Skip near endpoints (those are vertex hits)
        if t < 0.05 or t > 0.95:
            continue
        if dist > max_dist:
            continue
        if best is None or dist < best[3]:
            best = (i, px, py, dist)
    if best is None:
        return None
    return best[0], best[1], best[2]


def insert_vertex(
    pts: list[list[float]], after_index: int, x: float, y: float
) -> list[list[float]]:
    out = [list(p) for p in pts]
    out.insert(after_index + 1, [float(x), float(y)])
    return out


def delete_points_inside(
    pts: list[list[float]],
    lasso: list[list[float]],
    *,
    min_keep: int = 3,
) -> tuple[list[list[float]], int]:
    """Remove vertices inside ``lasso``. Keep at least ``min_keep``. Returns (pts, n_removed)."""
    if len(pts) <= min_keep or len(lasso) < 3:
        return [list(p) for p in pts], 0
    kept: list[list[float]] = []
    removed = 0
    for p in pts:
        if point_in_poly(float(p[0]), float(p[1]), lasso):
            removed += 1
        else:
            kept.append([float(p[0]), float(p[1])])
    if len(kept) < min_keep:
        # Prefer keeping points outside; if too few, fall back to original
        return [list(p) for p in pts], 0
    return kept, removed


def erase_points_near(
    pts: list[list[float]],
    x: float,
    y: float,
    *,
    radius: float,
    min_keep: int = 3,
) -> tuple[list[list[float]], int]:
    """Remove vertices within ``radius`` of (x,y). Keep at least ``min_keep``."""
    if len(pts) <= min_keep or radius <= 0:
        return [list(p) for p in pts], 0
    kept: list[list[float]] = []
    removed = 0
    r2 = radius * radius
    for p in pts:
        dx = float(p[0]) - x
        dy = float(p[1]) - y
        if dx * dx + dy * dy <= r2 and len(pts) - removed > min_keep:
            removed += 1
        else:
            kept.append([float(p[0]), float(p[1])])
    if len(kept) < min_keep:
        return [list(p) for p in pts], 0
    return kept, removed


def hit_polygon_index(
    polygons: list[dict[str, Any]],
    x: float,
    y: float,
) -> int | None:
    """Index of polygon containing (x,y), else nearest centroid."""
    if not polygons:
        return None
    for i, poly in enumerate(polygons):
        pts = poly.get("points") or []
        if len(pts) >= 3 and point_in_poly(x, y, pts):
            return i
    best_i = 0
    best_d = float("inf")
    for i, poly in enumerate(polygons):
        pts = poly.get("points") or []
        if len(pts) < 1:
            continue
        cx, cy = centroid(pts)
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    return best_i
