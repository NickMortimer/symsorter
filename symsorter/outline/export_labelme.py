"""Write LabelMe-style JSON for thumb-space draft polygons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
