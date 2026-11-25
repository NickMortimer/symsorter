"""Export locked thumb outlines onto original frames."""

from __future__ import annotations

import errno
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer

from symsorter.outline.export_labelme import (
    extract_polygons,
    is_polygon_locked,
    labelme_path_for_thumb,
    load_labelme_json,
    write_labelme_json,
)
from symsorter.outline.filename import parse_thumb_filename
from symsorter.outline.frames import (
    FrameResolver,
    RoiIndex,
    crop_window_for_row,
    frame_size,
    iter_flat_thumbs,
    thumb_to_frame,
)

_LINK_MODES = {"hard", "sym", "copy"}
_STATUS_ORDER = (
    "ok",
    "skipped_existing",
    "no_locked",
    "no_roi_row",
    "frame_missing",
    "link_failed",
)


@dataclass
class _Item:
    thumb_name: str
    view: str
    flight: str | None
    stem: str | None
    crop: list[int] | None
    crop_method: str | None
    n_locked: int
    status: str
    frame_src: str | None
    group_key: tuple[str, str] | None = None
    frame: Path | None = None
    shapes: list[dict[str, Any]] = field(default_factory=list)
    image_size: tuple[int, int] | None = None


def link_frame(src: Path, dst: Path, mode: str) -> str:
    """Hardlink, symlink, or copy ``src`` to ``dst``; return the method used."""
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        return "existing"
    if mode == "hard":
        try:
            os.link(src, dst)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            os.symlink(src.resolve(), dst)
            return "sym"
        return "hard"
    if mode == "sym":
        os.symlink(src.resolve(), dst)
        return "sym"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    raise ValueError(f"link mode must be hard, sym, or copy, not {mode!r}")


def outline_export_frames_command(
    *,
    thumbs: list[Path],
    survey_roots: list[Path],
    roi_csv: Path,
    out: Path,
    link: str = "hard",
    overwrite: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
) -> int:
    """Write locked thumb outlines onto original frames."""
    if link not in _LINK_MODES:
        typer.echo("Error: --link must be hard, sym, or copy", err=True)
        raise typer.Exit(2)

    collected: list[Path] = []
    for folder in thumbs:
        collected.extend(iter_flat_thumbs(Path(folder)))
    if limit is not None:
        collected = collected[: max(0, limit)]

    roi = RoiIndex.load(Path(roi_csv))
    resolver = FrameResolver([Path(root) for root in survey_roots])
    out = Path(out)
    items = [_classify(path, roi, resolver) for path in collected]
    _apply_groups(items, out=out, link=link, overwrite=overwrite, dry_run=dry_run)

    by_status, by_view = _counts(items)
    if not dry_run:
        _write_reports(out, items, by_status, by_view)
    summary = ", ".join(f"{key}={value}" for key, value in by_status.items())
    typer.echo(f"Counts: {summary}")
    return 0 if by_status.get("ok") or by_status.get("skipped_existing") else 1


def _classify(thumb: Path, roi: RoiIndex, resolver: FrameResolver) -> _Item:
    """Status a thumb, or attach the frame shapes that should be exported."""
    name = thumb.name
    row = roi.get(name)
    item = _Item(
        thumb_name=name,
        view=row.view if row is not None else thumb.parent.name,
        flight=row.flight if row is not None else None,
        stem=row.original_stem if row is not None else None,
        crop=None,
        crop_method=None,
        n_locked=0,
        status="",
        frame_src=None,
    )
    locked = [
        poly
        for poly in extract_polygons(load_labelme_json(labelme_path_for_thumb(thumb)))
        if is_polygon_locked(poly)
    ]
    item.n_locked = len(locked)
    if not locked:
        item.status = "no_locked"
        return item
    if row is None:
        item.status = "no_roi_row"
        return item
    frame = resolver.find(row.flight, row.original_stem)
    if frame is None:
        item.status = "frame_missing"
        return item

    image_size = frame_size(frame)
    crop = crop_window_for_row(row, image_size)
    crop_xyxy = [crop.left, crop.top, crop.right, crop.bottom]
    parsed = parse_thumb_filename(thumb)
    shapes: list[dict[str, Any]] = []
    for poly in locked:
        flags = dict(poly.get("flags") or {})
        flags.update(
            {
                "source_thumb": name,
                "view": row.view,
                "det_index": parsed.det_index,
                "crop_xyxy": list(crop_xyxy),
                "crop_method": "roi_csv",
            }
        )
        shapes.append(
            {
                "label": "turtle",
                "points": thumb_to_frame(poly["points"], crop),
                "group_id": None,
                "shape_type": "polygon",
                "flags": flags,
            }
        )
    item.crop = crop_xyxy
    item.crop_method = "roi_csv"
    item.frame_src = str(frame)
    item.group_key = (row.flight, row.original_stem)
    item.frame = frame
    item.shapes = shapes
    item.image_size = image_size
    return item


def _apply_groups(
    items: list[_Item],
    *,
    out: Path,
    link: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """Write one frame link and LabelMe doc per group, or mark the skip."""
    groups: dict[tuple[str, str], list[_Item]] = {}
    for item in items:
        if item.group_key is None:
            continue
        groups.setdefault(item.group_key, []).append(item)

    preexisting = {
        key
        for key in groups
        if (out / key[0] / f"{key[1]}.json").is_file()
    }
    for key, members in groups.items():
        if key in preexisting and not overwrite:
            for member in members:
                member.status = "skipped_existing"
            continue
        if dry_run:
            for member in members:
                member.status = "ok"
            continue
        _write_group(key, members, out=out, link=link)


def _write_group(
    key: tuple[str, str],
    members: list[_Item],
    *,
    out: Path,
    link: str,
) -> None:
    """Link the frame and write a fresh LabelMe document for one group."""
    frame = members[0].frame
    image_size = members[0].image_size
    if frame is None or image_size is None:
        for member in members:
            member.status = "frame_missing"
        return
    flight, stem = key
    destination = out / flight / frame.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        link_frame(frame, destination, link)
    except OSError:
        for member in members:
            member.status = "link_failed"
        return
    width, height = image_size
    shapes: list[dict[str, Any]] = []
    for member in members:
        shapes.extend(member.shapes)
    document = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": frame.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    write_labelme_json(out / flight / f"{stem}.json", document, overwrite=True)
    for member in members:
        member.status = "ok"


def _counts(
    items: list[_Item],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Non-zero status totals, overall and per view."""
    raw: dict[str, int] = {}
    by_view: dict[str, dict[str, int]] = {}
    for item in items:
        raw[item.status] = raw.get(item.status, 0) + 1
        view_counts = by_view.setdefault(item.view, {})
        view_counts[item.status] = view_counts.get(item.status, 0) + 1
    by_status = {name: raw[name] for name in _STATUS_ORDER if raw.get(name)}
    for name, count in raw.items():
        if name not in by_status and count:
            by_status[name] = count
    return by_status, by_view


def _write_reports(
    out: Path,
    items: list[_Item],
    by_status: dict[str, int],
    by_view: dict[str, dict[str, int]],
) -> None:
    """Write ``manifest.jsonl`` and ``summary.json``."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            row = {
                "thumb": item.thumb_name,
                "view": item.view,
                "flight": item.flight,
                "stem": item.stem,
                "crop": item.crop,
                "crop_method": item.crop_method,
                "n_locked": item.n_locked,
                "status": item.status,
                "frame_src": item.frame_src,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"by_status": by_status, "by_view": by_view}
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
