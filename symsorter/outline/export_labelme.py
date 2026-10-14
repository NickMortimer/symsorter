"""Read/write LabelMe-style JSON for thumb-space draft polygons."""

from __future__ import annotations

import json
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


def extract_polygons(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return polygon shapes from a LabelMe document (points normalized to list)."""
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
                "label": shape.get("label") or "turtle",
                "points": cleaned,
                "flags": shape.get("flags") if isinstance(shape.get("flags"), dict) else {},
                "shape_type": "polygon",
            }
        )
    return out


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

    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": shape_flags,
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
