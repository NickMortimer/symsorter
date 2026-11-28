"""Export locked thumb outlines as ``outlines.jsonl`` (full-frame pixels) for marine_objects."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import typer

from symsorter.outline.export_labelme import (
    extract_polygons,
    is_polygon_locked,
    labelme_path_for_thumb,
    load_labelme_json,
)
from symsorter.outline.frames import (
    FrameResolver,
    RoiIndex,
    crop_window_for_row,
    frame_size,
    iter_flat_thumbs,
    thumb_to_frame,
)
from symsorter.slices.contract import outline_row


def load_denylist(path: Path) -> FrozenSet[str]:
    """Read sealed survey ids, one per line; blank and ``#`` lines are ignored."""
    path = Path(path)
    if not path.is_file():
        raise typer.BadParameter(f"denylist not found: {path}")
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(line)
    if not ids:
        raise typer.BadParameter(f"denylist has no survey ids: {path}")
    return frozenset(ids)


def export_contract(
    thumbs: List[Path],
    roi_csv: Path,
    survey_roots: List[Path],
    denylist: FrozenSet[str],
    out: Path,
) -> Dict[str, int]:
    """Write locked outlines in frame pixels to ``out``; return per-reason counts."""
    sealed_flights = {sid.strip().rstrip("/").split("/")[-1] for sid in denylist}
    sealed_flights.discard("")
    roi = RoiIndex.load(Path(roi_csv))
    resolver = FrameResolver([Path(r) for r in survey_roots])
    counts = {"ok": 0, "sealed": 0, "no_locked": 0, "no_roi_row": 0, "frame_missing": 0,
              "frame_unreadable": 0}
    rows: List[Tuple[Tuple[str, str, str], dict]] = []

    paths: List[Path] = []
    for folder in thumbs:
        paths.extend(iter_flat_thumbs(Path(folder)))

    for thumb in paths:
        row = roi.get(thumb.name)
        if row is not None and row.flight in sealed_flights:
            counts["sealed"] += 1
            continue
        locked = [
            p
            for p in extract_polygons(load_labelme_json(labelme_path_for_thumb(thumb)))
            if is_polygon_locked(p)
        ]
        if not locked:
            counts["no_locked"] += 1
            continue
        if row is None:
            counts["no_roi_row"] += 1
            continue
        frame = resolver.find(row.flight, row.original_stem)
        if frame is None:
            counts["frame_missing"] += 1
            continue
        try:
            wh = frame_size(frame)
        except Exception:  # noqa: BLE001 - one corrupt frame must not abort the export
            counts["frame_unreadable"] += 1
            continue
        crop = crop_window_for_row(row, wh)
        source = f"thumb:{thumb.name}"
        for poly in locked:
            pts = thumb_to_frame(poly["points"], crop)
            key = (row.flight, row.original_stem, source)
            rows.append((key, outline_row(row.flight, row.original_stem, pts, source, wh)))
            counts["ok"] += 1

    rows.sort(key=lambda kr: kr[0])  # stable: ties keep thumb JSON order
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for _, r in rows:
            fh.write(json.dumps(r) + "\n")
    return counts


def outline_export_contract_command(
    *,
    thumbs: List[Path],
    roi_csv: Path,
    survey_roots: List[Path],
    denylist: Path,
    out: Path,
) -> int:
    ids = load_denylist(denylist)
    counts = export_contract(thumbs, roi_csv, survey_roots, ids, out)
    typer.echo("Counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0
