"""CLI: copy YOLO / slice boxes onto sorted thumbs for SAM prompting.

Reuses turtledrone geometry (``expand`` 224 window + ``bbox_to_patch`` clamp) and
the ``segment.py`` pattern of LabelMe rectangles → SAM polygons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

from symsorter.commands.outline_draft import DEFAULT_CLASSES, _parse_classes, iter_thumb_paths
from symsorter.outline.backends import DEFAULT_SAM_MODEL, DEFAULT_THUMB_SIZE, load_sam_model, sam_polygon
from symsorter.outline.export_labelme import (
    add_turtle_polygon,
    extract_rectangles,
    labelme_path_for_thumb,
    load_labelme_json,
    upsert_turtle_boxes,
)
from symsorter.outline.filename import parse_thumb_filename
from symsorter.outline.mapping import boxes_for_thumb, rectangles_to_labelme_shapes


def _thumb_hw(thumb_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(thumb_path) as im:
            return int(im.height or DEFAULT_THUMB_SIZE), int(im.width or DEFAULT_THUMB_SIZE)
    except Exception:
        return DEFAULT_THUMB_SIZE, DEFAULT_THUMB_SIZE


def outline_import_boxes_command(
    *,
    thumbs: Path,
    yolo_finds: Path,
    metadata_root: Path | None = None,
    classes: str = "turtle-surface,turtle-deep",
    limit: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    run_sam: bool = False,
    sam_model: str = DEFAULT_SAM_MODEL,
    device: str | None = None,
    min_match_score: float = 0.5,
    out: Path | None = None,
) -> int:
    class_names = _parse_classes(classes)
    thumb_list = list(iter_thumb_paths(thumbs, class_names))
    if limit is not None:
        thumb_list = thumb_list[: max(0, limit)]

    if not thumb_list:
        typer.echo("No thumbs found.", err=True)
        return 1
    if not Path(yolo_finds).is_dir():
        typer.echo(f"yolo_finds not found: {yolo_finds}", err=True)
        return 1

    typer.echo(
        f"Importing boxes for {len(thumb_list)} thumbs "
        f"(yolo_finds={yolo_finds}, sam={run_sam})"
    )

    sam = None
    if run_sam and not dry_run:
        sam = load_sam_model(sam_model, device=device)

    counts: dict[str, int] = {
        "ok": 0,
        "skipped_existing": 0,
        "no_flight": 0,
        "no_boxes": 0,
        "no_crop": 0,
        "low_match": 0,
        "sam_ok": 0,
        "sam_fail": 0,
        "boxes_written": 0,
    }
    manifest_rows: list[dict[str, Any]] = []

    for thumb_path, class_root in tqdm(thumb_list, desc="outline-import-boxes"):
        json_path = labelme_path_for_thumb(thumb_path)
        existing = load_labelme_json(json_path)
        existing_boxes = extract_rectangles(existing)
        if existing_boxes and not overwrite and not run_sam:
            counts["skipped_existing"] += 1
            continue

        thumb_boxes, crop, status = boxes_for_thumb(
            thumb_path,
            yolo_finds=Path(yolo_finds),
            metadata_root=Path(metadata_root) if metadata_root else None,
            min_match_score=min_match_score,
        )
        if status != "ok":
            counts[status] = counts.get(status, 0) + 1
            continue

        shapes = rectangles_to_labelme_shapes(thumb_boxes, crop=crop)
        h, w = _thumb_hw(thumb_path)
        parsed = parse_thumb_filename(thumb_path)

        if dry_run:
            counts["ok"] += 1
            counts["boxes_written"] += len(shapes)
            continue

        if overwrite or not existing_boxes:
            upsert_turtle_boxes(
                json_path,
                shapes,
                image_path=thumb_path.name,
                image_height=h,
                image_width=w,
            )
            counts["boxes_written"] += len(shapes)

        if run_sam and sam is not None:
            for shape in shapes:
                (x1, y1), (x2, y2) = shape["points"]
                try:
                    draft = sam_polygon(
                        thumb_path,
                        model=sam,
                        bboxes=[x1, y1, x2, y2],
                        sam_model=sam_model,
                        device=device,
                    )
                    flags = dict(draft.flags)
                    flags["det_index"] = parsed.det_index
                    flags["source"] = "yolo_box_sam"
                    flags["draft"] = True
                    add_turtle_polygon(
                        json_path,
                        image_path=thumb_path.name,
                        image_height=h,
                        image_width=w,
                        points=draft.points,
                        flags=flags,
                        label="turtle",
                    )
                    counts["sam_ok"] += 1
                except Exception as exc:  # noqa: BLE001
                    counts["sam_fail"] += 1
                    typer.echo(f"SAM failed {thumb_path.name}: {exc}", err=True)

        counts["ok"] += 1
        manifest_rows.append(
            {
                "thumb": str(thumb_path),
                "class": class_root.name,
                "n_boxes": len(shapes),
                "crop": (
                    [crop.left, crop.top, crop.right, crop.bottom] if crop else None
                ),
                "crop_method": crop.method if crop else None,
                "crop_score": crop.score if crop else None,
            }
        )

    if out is not None and not dry_run:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        man = out / "import_boxes_manifest.jsonl"
        with man.open("w", encoding="utf-8") as f:
            for row in manifest_rows:
                f.write(json.dumps(row) + "\n")
        typer.echo(f"Wrote {man}")

    typer.echo("Counts: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    return 0 if counts["ok"] or counts["skipped_existing"] else 1
