"""Qt-free model behind the Slice review window (spec 2026-11-28-slice-review-design)."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from symsorter.outline.frames import FrameResolver
from symsorter.slices.contract import (
    STATUSES,
    Polygon,
    Project,
    ReviewRow,
    SliceSpec,
    append_review,
    load_project,
    load_reviews,
)
from symsorter.slices.geometry import (
    FrameLookup,
    inside,
    locate_frame,
    recentre,
    touches,
    view_rect,
)

LOCAL_NAME = "project.local.json"


@dataclass
class LocalSettings:
    survey_roots: List[Path]
    reviewer: str


def load_local(project_dir: Path) -> Optional[LocalSettings]:
    p = Path(project_dir) / LOCAL_NAME
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return LocalSettings([Path(r) for r in d.get("survey_roots", [])], str(d.get("reviewer", "")))


def save_local(project_dir: Path, s: LocalSettings) -> None:
    p = Path(project_dir) / LOCAL_NAME
    p.write_text(json.dumps({"survey_roots": [str(r) for r in s.survey_roots],
                             "reviewer": s.reviewer}, indent=2) + "\n", encoding="utf-8")


class SaveRefused(Exception):
    """Save not allowed; the message is shown to the user."""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _to_lists(polys) -> List[Polygon]:
    return [[[float(x), float(y)] for x, y in poly] for poly in polys]


def _to_tuples(polys: List[Polygon]):
    return tuple(tuple((float(x), float(y)) for x, y in poly) for poly in polys)


def _rect_touches(points, rect) -> bool:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) >= rect[0] and min(xs) <= rect[2]
            and max(ys) >= rect[1] and min(ys) <= rect[3])


class ReviewSession:
    def __init__(self, project_dir: Path, local: LocalSettings, *,
                 clock: Callable[[], str] = _now) -> None:
        self.project_dir = Path(project_dir)
        self.local = local
        self._clock = clock
        self.project: Project = load_project(self.project_dir / "project.json")
        self._review_path = self.project_dir / "review.jsonl"
        loaded = load_reviews(self._review_path)
        self._reviews: Dict[str, ReviewRow] = {
            k: r for k, r in loaded.items() if r.project_id == self.project.project_id}
        self.foreign_reviews = len(loaded) - len(self._reviews)
        if self.foreign_reviews:
            others = sorted({r.project_id for r in loaded.values()} - {self.project.project_id})
            warnings.warn(
                f"{self._review_path}: ignoring {self.foreign_reviews} review row(s) from other "
                f"project(s) {others}; expected {self.project.project_id!r}", UserWarning)
        self._resolver = FrameResolver(local.survey_roots, strict=True)
        self._lookups: Dict[str, FrameLookup] = {}
        self._polys: Optional[List[Polygon]] = None
        self._pos: Optional[Tuple[int, int]] = None
        self._dirty = False
        self.index = 0
        for i in range(len(self.project.slices)):
            if self.status(i) is None and self.lookup(i).problem is None:
                self.index = i
                break

    # -- basics -----------------------------------------------------------
    @property
    def current(self) -> SliceSpec:
        return self.project.slices[self.index]

    @property
    def dirty(self) -> bool:
        return self._dirty

    def _spec(self, i: Optional[int]) -> SliceSpec:
        return self.current if i is None else self.project.slices[i]

    def _reset_working(self) -> None:
        self._polys = None
        self._pos = None
        self._dirty = False

    def lookup(self, i: Optional[int] = None) -> FrameLookup:
        spec = self._spec(i)
        got = self._lookups.get(spec.slice_id)
        if got is None:
            try:
                got = locate_frame(spec, self._resolver)
            except Exception as e:
                warnings.warn(f"{spec.slice_id}: frame lookup failed: {e!r}", UserWarning)
                got = FrameLookup(None, "frame unreadable")
            self._lookups[spec.slice_id] = got
        return got

    def status(self, i: Optional[int] = None) -> Optional[str]:
        row = self._reviews.get(self._spec(i).slice_id)
        return row.status if row else None

    def _saved_polys(self, spec: SliceSpec) -> List[Polygon]:
        row = self._reviews.get(spec.slice_id)
        return _to_lists(row.polygons if row else spec.polygons)

    def position(self) -> Tuple[int, int]:
        if self._pos is not None:
            return self._pos
        spec = self.current
        row = self._reviews.get(spec.slice_id)
        if row is not None and row.x0 is not None and row.y0 is not None:
            return (row.x0, row.y0)
        return (spec.x0, spec.y0)

    def polygons(self) -> List[Polygon]:
        if self._polys is None:
            self._polys = self._saved_polys(self.current)
        return [[list(p) for p in poly] for poly in self._polys]

    def set_polygons(self, polys: List[Polygon]) -> None:
        self._polys = [[list(p) for p in poly] for poly in polys]
        self._dirty = True

    def discard_changes(self) -> None:
        """Drop unsaved polygon and position edits for the current slice."""
        self._reset_working()

    # -- neighbours -------------------------------------------------------
    def _others(self) -> List[Polygon]:
        cur = self.current
        out: List[Polygon] = []
        for s in self.project.slices:
            if (s.slice_id != cur.slice_id and s.survey_id == cur.survey_id
                    and s.frame_id == cur.frame_id):
                out.extend(self._saved_polys(s))
        return [p for p in out if len(p) >= 3]

    def neighbours(self) -> List[Polygon]:
        cur = self.current
        x0, y0 = self.position()
        rect = view_rect(x0, y0, cur.size, self.project.margin_px, cur.frame_wh)
        return [p for p in self._others() if _rect_touches(p, rect)]

    def adopt_neighbours(self) -> int:
        cur = self.current
        x0, y0 = self.position()
        have = [[tuple(p) for p in poly] for poly in self.polygons()]
        added = []
        for p in self._others():
            if not touches(p, x0, y0, cur.size):
                continue
            t = [tuple(q) for q in p]
            if t in have:
                continue
            have.append(t)
            added.append(p)
        if added:
            self.set_polygons(self.polygons() + added)
        return len(added)

    def recentre_current(self) -> bool:
        cur = self.current
        x0, y0 = self.position()
        res = recentre(self.polygons(), x0, y0, cur.size, cur.frame_wh)
        if res is None:
            return False
        if res != (x0, y0):
            self._pos = res
            self._dirty = True
        return True

    # -- saving -----------------------------------------------------------
    def save(self, status: str, reason: Optional[str] = None) -> ReviewRow:
        if status not in STATUSES:
            raise SaveRefused(f"Unknown status {status!r}.")
        cur = self.current
        prob = self.lookup().problem
        if prob:
            raise SaveRefused(f"Cannot save: {prob}.")
        x0, y0 = self.position()
        kept = [p for p in self.polygons()
                if len(p) >= 3 and touches(p, x0, y0, cur.size)]
        if status == "complete":
            if any(not inside(p, x0, y0, cur.size) for p in kept):
                raise SaveRefused(
                    "An outline crosses the slice edge. Re-centre the slice or edit the outline.")
        same = (x0, y0) == (cur.x0, cur.y0)
        row = ReviewRow(
            project_id=self.project.project_id, slice_id=cur.slice_id, status=status,
            polygons=_to_tuples(kept), reviewer=self.local.reviewer, time=self._clock(),
            x0=None if same else x0, y0=None if same else y0, reason=reason)
        append_review(self._review_path, row)
        self._reviews[cur.slice_id] = row
        self._reset_working()
        return row

    # -- navigation -------------------------------------------------------
    def next_unreviewed(self) -> bool:
        if self._dirty:
            return False
        for i in range(self.index + 1, len(self.project.slices)):
            if self.status(i) is None and self.lookup(i).problem is None:
                self.index = i
                self._reset_working()
                return True
        return False

    def prev(self) -> bool:
        if self._dirty:
            return False
        for i in range(self.index - 1, -1, -1):
            if self.lookup(i).problem is None:
                self.index = i
                self._reset_working()
                return True
        return False

    def progress(self) -> Dict[str, int]:
        out = {"total": len(self.project.slices), "complete": 0, "reject": 0,
               "skip": 0, "unreviewed": 0, "missing": 0}
        for i in range(len(self.project.slices)):
            if self.lookup(i).problem is not None:
                out["missing"] += 1
                continue
            st = self.status(i)
            out[st if st else "unreviewed"] += 1
        return out
