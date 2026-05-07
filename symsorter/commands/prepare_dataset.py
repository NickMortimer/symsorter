"""Commands for preparing ROI thumbnails and syncing updates back to JSON."""

from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from PIL import Image
import typer


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _iter_annotation_files(root: Path) -> list[Path]:
    """Return JSON annotation files under root, excluding generated files."""
    json_files: list[Path] = []
    for path in root.rglob("*.json"):
        if path.name in {"classifications.json", "symsorter_prepare_map.json"}:
            continue
        json_files.append(path)
    return sorted(json_files)


def _load_json(path: Path) -> dict[str, Any] | None:
    """Safely load JSON and return dict data when valid."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _resolve_image_path(json_path: Path, data: dict[str, Any]) -> Path | None:
    """Resolve image path for an annotation file."""
    image_path = data.get("imagePath")
    if isinstance(image_path, str) and image_path.strip():
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = (json_path.parent / candidate).resolve()
        if candidate.exists():
            return candidate

    for ext in IMAGE_EXTENSIONS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _shape_bbox(shape: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Compute a bounding box (xmin, ymin, xmax, ymax) from shape points."""
    points = shape.get("points")
    if not isinstance(points, list) or not points:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if not isinstance(point, list) or len(point) < 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            continue
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _centered_roi_crop(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    thumb_size: int,
    context_scale: float,
) -> Image.Image:
    """Create a square crop centered on bbox and resized to thumb_size."""
    img_w, img_h = image.size
    xmin, ymin, xmax, ymax = bbox

    roi_w = max(1.0, xmax - xmin)
    roi_h = max(1.0, ymax - ymin)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    side = max(roi_w, roi_h) * max(1.0, context_scale)
    side = max(2.0, min(side, float(min(img_w, img_h))))

    left = cx - (side / 2.0)
    top = cy - (side / 2.0)
    right = left + side
    bottom = top + side

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > img_w:
        left -= (right - img_w)
        right = img_w
    if bottom > img_h:
        top -= (bottom - img_h)
        bottom = img_h

    left = max(0, int(round(left)))
    top = max(0, int(round(top)))
    right = min(img_w, int(round(right)))
    bottom = min(img_h, int(round(bottom)))

    if right <= left or bottom <= top:
        # Fallback to full image if bbox is degenerate after clamping
        cropped = image
    else:
        cropped = image.crop((left, top, right, bottom))

    return cropped.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)


def _safe_stem(path: Path) -> str:
    """Create a safe stem string for generated thumbnail names."""
    text = str(path.with_suffix(""))
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    compact = "".join(safe).strip("_")
    return compact or "annotation"


def prepare_dataset_command(
    root: Path,
    out_dir: Path,
    thumb_size: int = 224,
    context_scale: float = 1.6,
) -> None:
    """Create ROI-centered thumbnails from Labelme/X-AnyLabeling JSON files."""
    if not root.exists():
        typer.echo(f"Root not found: {root}", err=True)
        raise typer.Exit(1)
    if thumb_size <= 0:
        typer.echo("thumb-size must be > 0", err=True)
        raise typer.Exit(1)
    if context_scale <= 0:
        typer.echo("context-scale must be > 0", err=True)
        raise typer.Exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    skipped = 0
    created = 0

    for json_path in _iter_annotation_files(root):
        data = _load_json(json_path)
        if not data:
            skipped += 1
            continue

        shapes = data.get("shapes")
        if not isinstance(shapes, list) or not shapes:
            skipped += 1
            continue

        image_path = _resolve_image_path(json_path, data)
        if not image_path:
            skipped += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        json_rel = json_path.resolve().relative_to(root.resolve())
        image_rel = image_path.resolve().relative_to(root.resolve())
        base = _safe_stem(json_rel)

        for shape_index, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                continue
            bbox = _shape_bbox(shape)
            if bbox is None:
                continue

            thumb = _centered_roi_crop(
                image=image,
                bbox=bbox,
                thumb_size=thumb_size,
                context_scale=context_scale,
            )
            thumb_name = f"{base}__roi_{shape_index:04d}.jpg"
            thumb_path = out_dir / thumb_name
            thumb.save(thumb_path, format="JPEG", quality=95)

            entries.append(
                {
                    "thumb_file": thumb_name,
                    "source_json_rel": str(json_rel),
                    "source_image_rel": str(image_rel),
                    "shape_index": int(shape_index),
                    "shape_label": str(shape.get("label", "")),
                    "bbox": [float(v) for v in bbox],
                }
            )
            created += 1

    map_data = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_root": str(root.resolve()),
        "thumb_dir": str(out_dir.resolve()),
        "thumb_size": int(thumb_size),
        "context_scale": float(context_scale),
        "entries": entries,
    }

    map_path = out_dir / "symsorter_prepare_map.json"
    with map_path.open("w", encoding="utf-8") as f:
        json.dump(map_data, f, indent=2, ensure_ascii=False)

    typer.echo(f"Prepared thumbnails: {created}")
    typer.echo(f"Mapping file: {map_path}")
    if skipped:
        typer.echo(f"Skipped JSON files: {skipped}")


def _load_class_assignments(classifications_path: Path) -> dict[str, dict[str, Any]]:
    """Load COCO classifications.json into thumb filename -> class info mapping."""
    with classifications_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images") or []
    annotations = data.get("annotations") or []
    categories = data.get("categories") or []

    id_to_name = {int(c.get("id")): str(c.get("name", "")) for c in categories if "id" in c}
    id_to_file: dict[int, str] = {}
    for image in images:
        try:
            image_id = int(image.get("id"))
        except (TypeError, ValueError):
            continue
        fname = image.get("file_name") or image.get("filename")
        if isinstance(fname, str) and fname:
            id_to_file[image_id] = fname

    out: dict[str, dict[str, Any]] = {}
    for ann in annotations:
        try:
            image_id = int(ann.get("image_id"))
            class_id = int(ann.get("category_id"))
        except (TypeError, ValueError):
            continue
        fname = id_to_file.get(image_id)
        if not fname:
            continue
        out[fname] = {
            "class_id": class_id,
            "class_name": id_to_name.get(class_id, str(class_id)),
        }
    return out


def update_json_from_symsorter_command(
    prepared_dir: Path,
    source_root: Path | None = None,
    replace_label: bool = False,
) -> None:
    """Apply SymSorter classifications on prepared thumbnails back to source JSON files."""
    map_path = prepared_dir / "symsorter_prepare_map.json"
    cls_path = prepared_dir / "classifications.json"

    if not map_path.exists():
        typer.echo(f"Mapping file not found: {map_path}", err=True)
        raise typer.Exit(1)
    if not cls_path.exists():
        typer.echo(f"Classifications file not found: {cls_path}", err=True)
        raise typer.Exit(1)

    with map_path.open("r", encoding="utf-8") as f:
        map_data = json.load(f)

    entries = map_data.get("entries") or []
    if not isinstance(entries, list):
        typer.echo("Invalid mapping file format", err=True)
        raise typer.Exit(1)

    root = source_root or Path(map_data.get("source_root", ""))
    if not root or not root.exists():
        typer.echo("Valid source root is required", err=True)
        raise typer.Exit(1)

    assignments = _load_class_assignments(cls_path)
    if not assignments:
        typer.echo("No assigned classes found in classifications.json")
        return

    json_cache: dict[Path, dict[str, Any]] = {}
    changed_files: set[Path] = set()
    updated_shapes = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        thumb_file = entry.get("thumb_file")
        if not isinstance(thumb_file, str) or thumb_file not in assignments:
            continue

        source_json_rel = entry.get("source_json_rel")
        shape_index = entry.get("shape_index")
        if not isinstance(source_json_rel, str):
            continue
        if not isinstance(shape_index, int):
            continue

        json_path = (root / source_json_rel).resolve()
        if not json_path.exists():
            continue

        if json_path not in json_cache:
            data = _load_json(json_path)
            if not data:
                continue
            json_cache[json_path] = data

        data = json_cache[json_path]
        shapes = data.get("shapes")
        if not isinstance(shapes, list):
            continue
        if shape_index < 0 or shape_index >= len(shapes):
            continue
        shape = shapes[shape_index]
        if not isinstance(shape, dict):
            continue

        assigned = assignments[thumb_file]
        if replace_label:
            shape["label"] = assigned["class_name"]

        shape["symsorter"] = {
            "class_name": assigned["class_name"],
            "class_id": assigned["class_id"],
            "thumb_file": thumb_file,
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        changed_files.add(json_path)
        updated_shapes += 1

    for path in changed_files:
        with path.open("w", encoding="utf-8") as f:
            json.dump(json_cache[path], f, indent=2, ensure_ascii=False)

    typer.echo(f"Updated JSON files: {len(changed_files)}")
    typer.echo(f"Updated shapes: {updated_shapes}")
