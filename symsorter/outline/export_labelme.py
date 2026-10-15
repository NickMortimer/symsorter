"""Read/write LabelMe-style JSON for thumb-space draft polygons."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def labelme_path_for_thumb(thumb_path: Path) -> Path:
    """Sibling LabelMe JSON path for a thumb (``foo.jpg`` → ``foo.json``)."""
    return Path(thumb_path).with_suffix(".json")


def load_labelme_json(path: Path) -> dict[str, Any] | None:
    """Load a LabelMe JSON dict, or None if missing/invalid."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def extract_polygons(
    data: dict[str, Any] | None, *, label: str | None = "turtle"
) -> list[dict[str, Any]]:
    """Return polygon shapes from a LabelMe document (points normalized to list).

    If ``label`` is set, only polygons with that label are returned. Pass
    ``label=None`` to keep every polygon shape.
    """
    if not data:
        return []
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return []
    out: list[dict[str, Any]] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        shape_type = shape.get("shape_type")
        if shape_type is not None and shape_type != "polygon":
            continue
        shape_label = shape.get("label") or "turtle"
        if label is not None and shape_label != label:
            continue
        points = shape.get("points")
        if not isinstance(points, list) or len(points) < 3:
            continue
        cleaned: list[list[float]] = []
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            try:
                cleaned.append([float(pt[0]), float(pt[1])])
            except (TypeError, ValueError):
                continue
        if len(cleaned) < 3:
            continue
        out.append(
            {
                "label": shape_label,
                "points": cleaned,
                "flags": shape.get("flags") if isinstance(shape.get("flags"), dict) else {},
                "shape_type": "polygon",
            }
        )
    return out


def _shape_flags(flags: dict[str, Any] | None) -> dict[str, Any]:
    shape_flags: dict[str, Any] = {
        "draft": True,
        "backend": None,
        "det_index": None,
        "turtle_group_id": None,
        "view": None,
        "blur": None,
    }
    if flags:
        shape_flags.update(flags)
    return shape_flags


def _blank_doc(
    *,
    image_path: str,
    image_height: int,
    image_width: int,
) -> dict[str, Any]:
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }


def _finalize_doc(
    path: Path,
    existing: dict[str, Any],
    *,
    image_path: str,
    image_height: int,
    image_width: int,
) -> dict[str, Any]:
    existing["imagePath"] = image_path
    existing["imageHeight"] = image_height
    existing["imageWidth"] = image_width
    existing.setdefault("version", "5.0.1")
    existing.setdefault("flags", {})
    existing["imageData"] = existing.get("imageData", None)
    write_labelme_json(path, existing, overwrite=True)
    return existing


def build_labelme_dict(
    *,
    image_path: str,
    image_height: int,
    image_width: int,
    points: list[list[float]],
    flags: dict[str, Any] | None = None,
    label: str = "turtle",
) -> dict[str, Any]:
    """Build a LabelMe document with a single turtle polygon."""
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": _shape_flags(flags),
            }
        ],
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }


def write_labelme_json(path: Path, data: dict[str, Any], overwrite: bool = False) -> bool:
    """Write LabelMe JSON. Returns False if skipped because file exists."""
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return True


def add_turtle_polygon(
    path: Path,
    *,
    image_path: str,
    image_height: int,
    image_width: int,
    points: list[list[float]],
    flags: dict[str, Any] | None = None,
    label: str = "turtle",
) -> dict[str, Any]:
    """Append a turtle polygon; keep existing turtles and measurement lines."""
    existing = load_labelme_json(path) or _blank_doc(
        image_path=image_path, image_height=image_height, image_width=image_width
    )
    shapes = list(existing.get("shapes") or [])
    shapes.append(
        {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": _shape_flags(flags),
        }
    )
    existing["shapes"] = shapes
    return _finalize_doc(
        path,
        existing,
        image_path=image_path,
        image_height=image_height,
        image_width=image_width,
    )


def upsert_turtle_polygon(
    path: Path,
    *,
    image_path: str,
    image_height: int,
    image_width: int,
    points: list[list[float]],
    flags: dict[str, Any] | None = None,
    label: str = "turtle",
    replace_all: bool = False,
) -> dict[str, Any]:
    """Add a turtle polygon (default) or replace all turtle polygons.

    ``replace_all=True`` keeps the old single-outline behaviour. Measurement
    lines and other labels are always preserved.
    """
    if not replace_all:
        return add_turtle_polygon(
            path,
            image_path=image_path,
            image_height=image_height,
            image_width=image_width,
            points=points,
            flags=flags,
            label=label,
        )

    existing = load_labelme_json(path) or _blank_doc(
        image_path=image_path, image_height=image_height, image_width=image_width
    )
    kept: list[dict[str, Any]] = []
    for shape in existing.get("shapes") or []:
        if not isinstance(shape, dict):
            continue
        if shape.get("shape_type") == "polygon" and (shape.get("label") or "") == label:
            continue
        kept.append(shape)
    kept.append(
        {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": _shape_flags(flags),
        }
    )
    existing["shapes"] = kept
    return _finalize_doc(
        path,
        existing,
        image_path=image_path,
        image_height=image_height,
        image_width=image_width,
    )


def replace_turtle_polygon_at(
    path: Path,
    index: int,
    *,
    image_path: str,
    image_height: int,
    image_width: int,
    points: list[list[float]],
    flags: dict[str, Any] | None = None,
    label: str = "turtle",
) -> dict[str, Any] | None:
    """Replace the Nth turtle polygon (0-based among turtle polygons only)."""
    existing = load_labelme_json(path)
    if not existing:
        return None
    turtle_i = -1
    shapes = list(existing.get("shapes") or [])
    for i, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        if shape.get("shape_type") == "polygon" and (shape.get("label") or "") == label:
            turtle_i += 1
            if turtle_i == index:
                shapes[i] = {
                    "label": label,
                    "points": points,
                    "group_id": shape.get("group_id"),
                    "shape_type": "polygon",
                    "flags": _shape_flags({**(shape.get("flags") or {}), **(flags or {})}),
                }
                existing["shapes"] = shapes
                return _finalize_doc(
                    path,
                    existing,
                    image_path=image_path,
                    image_height=image_height,
                    image_width=image_width,
                )
    return None


def delete_turtle_polygon_at(
    path: Path,
    index: int,
    *,
    label: str = "turtle",
) -> bool:
    """Delete the Nth turtle polygon. Returns True if something was removed."""
    existing = load_labelme_json(path)
    if not existing:
        return False
    turtle_i = -1
    shapes = list(existing.get("shapes") or [])
    for i, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        if shape.get("shape_type") == "polygon" and (shape.get("label") or "") == label:
            turtle_i += 1
            if turtle_i == index:
                shapes.pop(i)
                existing["shapes"] = shapes
                write_labelme_json(path, existing, overwrite=True)
                return True
    return False


def set_turtle_polygons(
    path: Path,
    polygons: list[dict[str, Any]],
    *,
    image_path: str,
    image_height: int,
    image_width: int,
    label: str = "turtle",
) -> dict[str, Any]:
    """Replace all turtle polygons with ``polygons``; keep other shapes."""
    existing = load_labelme_json(path) or _blank_doc(
        image_path=image_path, image_height=image_height, image_width=image_width
    )
    kept: list[dict[str, Any]] = []
    for shape in existing.get("shapes") or []:
        if not isinstance(shape, dict):
            continue
        if shape.get("shape_type") == "polygon" and (shape.get("label") or "") == label:
            continue
        kept.append(shape)
    for poly in polygons:
        kept.append(
            {
                "label": poly.get("label") or label,
                "points": poly["points"],
                "group_id": poly.get("group_id"),
                "shape_type": "polygon",
                "flags": _shape_flags(poly.get("flags")),
            }
        )
    existing["shapes"] = kept
    return _finalize_doc(
        path,
        existing,
        image_path=image_path,
        image_height=image_height,
        image_width=image_width,
    )


def delete_all_turtle_polygons(path: Path, *, label: str = "turtle") -> int:
    """Remove every turtle polygon; keep other shapes. Returns count deleted."""
    existing = load_labelme_json(path)
    if not existing:
        return 0
    kept: list[dict[str, Any]] = []
    deleted = 0
    for shape in existing.get("shapes") or []:
        if (
            isinstance(shape, dict)
            and shape.get("shape_type") == "polygon"
            and (shape.get("label") or "") == label
        ):
            deleted += 1
            continue
        kept.append(shape)
    if deleted:
        existing["shapes"] = kept
        write_labelme_json(path, existing, overwrite=True)
    return deleted


def _point_in_poly(x: float, y: float, pts: list[list[float]]) -> bool:
    """Ray-cast point-in-polygon."""
    inside = False
    n = len(pts)
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


def _centroid(pts: list[list[float]]) -> tuple[float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def find_turtle_polygon_index_at(
    path: Path,
    x: float,
    y: float,
    *,
    label: str = "turtle",
) -> int | None:
    """Index of turtle polygon containing (x,y), else nearest centroid, else None."""
    polys = extract_polygons(load_labelme_json(path), label=label)
    if not polys:
        return None
    for i, poly in enumerate(polys):
        if _point_in_poly(x, y, poly["points"]):
            return i
    # Nearest centroid fallback
    best_i = 0
    best_d = float("inf")
    for i, poly in enumerate(polys):
        cx, cy = _centroid(poly["points"])
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def simplify_polyline(points: list[list[float]], max_points: int = 64) -> list[list[float]]:
    """Uniformly subsample a freehand path (no heavy deps)."""
    if len(points) <= max_points:
        return points
    if len(points) < 3:
        return points
    idxs = [round(i * (len(points) - 1) / (max_points - 1)) for i in range(max_points)]
    return [points[i] for i in idxs]
