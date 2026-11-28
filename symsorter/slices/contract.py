"""Shared file contract for the slice-review loop (see board spec 2026-11-28-slice-review-design).

Pure Python: reads project.json, reads/appends review.jsonl, builds outlines.jsonl rows.
All coordinates are full-frame pixels, [x, y].
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OUTLINES_SCHEMA = "symsorter.outlines/1"
PROJECT_SCHEMA = "marine_count.slice_project/1"
REVIEW_SCHEMA = "symsorter.slice_review/1"
STATUSES = ("complete", "reject", "skip")
LABEL = "turtle"

Polygon = List[List[float]]
PolyT = Tuple[Tuple[float, float], ...]


class ContractError(ValueError):
    """A contract file is malformed or has an unknown schema."""


@dataclass(frozen=True)
class SliceSpec:
    slice_id: str
    survey_id: str
    frame_id: str
    frame_wh: Tuple[int, int]
    x0: int
    y0: int
    size: int
    kind: str
    polygons: Tuple[PolyT, ...]

    @property
    def flight(self) -> str:
        return self.survey_id.split("/")[-1]

    @property
    def frame_stem(self) -> str:
        return Path(self.frame_id).stem


@dataclass(frozen=True)
class Project:
    project_id: str
    slice_px: int
    margin_px: int
    slices: Tuple[SliceSpec, ...]


@dataclass(frozen=True)
class ReviewRow:
    project_id: str
    slice_id: str
    status: str
    polygons: Tuple[PolyT, ...]
    reviewer: str
    time: str
    x0: Optional[int] = None
    y0: Optional[int] = None
    reason: Optional[str] = None


def _poly(d: Any, where: str) -> PolyT:
    if not isinstance(d, dict):
        raise ContractError(f"{where}: polygon entry must be an object")
    if d.get("label") != LABEL:
        raise ContractError(f"{where}: label must be {LABEL!r}, got {d.get('label')!r}")
    try:
        return tuple((float(x), float(y)) for x, y in d["points"])
    except (KeyError, TypeError, ValueError) as e:
        raise ContractError(f"{where}: bad polygon points ({e})") from e


def load_project(path: Path) -> Project:
    path = Path(path)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ContractError(f"{path}: invalid JSON ({e})") from e
    if not isinstance(d, dict):
        raise ContractError(f"{path}: top level must be an object")
    if d.get("schema") != PROJECT_SCHEMA:
        raise ContractError(f"{path}: unknown schema {d.get('schema')!r}, expected {PROJECT_SCHEMA!r}")
    try:
        slices = tuple(
            SliceSpec(
                slice_id=s["slice_id"],
                survey_id=s["survey_id"],
                frame_id=s["frame_id"],
                frame_wh=(int(s["frame_wh"][0]), int(s["frame_wh"][1])),
                x0=int(s["x0"]), y0=int(s["y0"]), size=int(s["size"]),
                kind=s["kind"],
                polygons=tuple(_poly(p, f"{path}: slice {s['slice_id']}") for p in s["polygons"]),
            )
            for s in d["slices"]
        )
        return Project(d["project_id"], int(d["slice_px"]), int(d["margin_px"]), slices)
    except (KeyError, TypeError, ValueError, IndexError) as e:
        if isinstance(e, ContractError):
            raise
        raise ContractError(f"{path}: malformed project ({e!r})") from e


def _row_from_dict(d: Any, where: str) -> ReviewRow:
    if not isinstance(d, dict):
        raise ContractError(f"{where}: row must be a JSON object")
    if d.get("schema") != REVIEW_SCHEMA:
        raise ContractError(f"{where}: unknown schema {d.get('schema')!r}")
    if d.get("status") not in STATUSES:
        raise ContractError(f"{where}: bad status {d.get('status')!r}")
    try:
        return ReviewRow(
            project_id=d["project_id"], slice_id=d["slice_id"], status=d["status"],
            polygons=tuple(_poly(p, where) for p in d["polygons"]),
            reviewer=d["reviewer"], time=d["time"],
            x0=d.get("x0"), y0=d.get("y0"), reason=d.get("reason"),
        )
    except KeyError as e:
        raise ContractError(f"{where}: missing field {e}") from e


def load_reviews(path: Path) -> Dict[str, ReviewRow]:
    """Last row per slice_id wins. Missing file -> {}. A torn final line is skipped with a warning."""
    path = Path(path)
    if not path.exists():
        return {}
    lines = [(i, ln) for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if ln.strip()]
    out: Dict[str, ReviewRow] = {}
    for n, (i, ln) in enumerate(lines):
        where = f"{path}:{i}"
        try:
            d = json.loads(ln)
        except json.JSONDecodeError as e:
            if n == len(lines) - 1:
                warnings.warn(f"{where}: ignoring torn final line", UserWarning)
                continue
            raise ContractError(f"{where}: invalid JSON ({e})") from e
        row = _row_from_dict(d, where)
        out[row.slice_id] = row
    return out


def append_review(path: Path, row: ReviewRow) -> None:
    d: Dict[str, Any] = {
        "schema": REVIEW_SCHEMA,
        "project_id": row.project_id,
        "slice_id": row.slice_id,
        "status": row.status,
        "polygons": [{"label": LABEL, "points": [list(p) for p in poly]} for poly in row.polygons],
        "reviewer": row.reviewer,
        "time": row.time,
    }
    for k in ("x0", "y0", "reason"):
        v = getattr(row, k)
        if v is not None:
            d[k] = v
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        data = path.read_bytes()
        if not data.endswith(b"\n"):
            # Torn final line from a crash: drop the fragment so the log stays valid.
            warnings.warn(f"{path}: dropping torn final line before append", UserWarning)
            with open(path, "r+b") as tf:
                tf.truncate(data.rfind(b"\n") + 1)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(d) + "\n")
        f.flush()
        os.fsync(f.fileno())


def outline_row(flight: str, frame_stem: str, points: Polygon, source: str,
                frame_wh: Tuple[int, int]) -> Dict[str, Any]:
    return {
        "schema": OUTLINES_SCHEMA,
        "flight": flight,
        "frame_stem": frame_stem,
        "label": LABEL,
        "points": [[float(x), float(y)] for x, y in points],
        "source": source,
        "frame_wh": [int(frame_wh[0]), int(frame_wh[1])],
    }
