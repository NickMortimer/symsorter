"""Map YOLO / slice boxes onto sorted 224×224 thumbs (turtle-repo patterns).

Ports geometry from turtledrone:
- ``detect_and_crop.expand`` (fixed 224 window centered on a box)
- ``build_training_set.bbox_to_patch_yolo`` (clamp box into patch pixel space)
- Preferred provenance: ``*_crops_metadata.csv`` via stem + ``det_N``
- Fallback: template-match thumb → full frame (``cv2.matchTemplate``), then
  project every overlapping frame/slice box into thumb coords.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from symsorter.outline.filename import ParsedThumbName, parse_thumb_filename

_SLICE_XYXY = re.compile(
    r"_(?P<x1>\d+)_(?P<y1>\d+)_(?P<x2>\d+)_(?P<y2>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FrameBox:
    """Axis-aligned box in full-frame (or already thumb-local) pixels."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "turtle_3_yolo"
    source: str = "image"  # image | slice | metadata

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.as_xyxy()
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass(frozen=True)
class CropWindow:
    left: int
    top: int
    right: int
    bottom: int
    score: float = 1.0
    method: str = "metadata"

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def fixed_size_crop_window(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    img_w: int,
    img_h: int,
    fixed_size: int = 224,
) -> CropWindow:
    """Center a ``fixed_size`` window on a box (turtledrone ``expand`` math)."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = fixed_size / 2.0
    cx = min(max(cx, half), img_w - half)
    cy = min(max(cy, half), img_h - half)
    left = int(round(cx - half))
    top = int(round(cy - half))
    right = left + fixed_size
    bottom = top + fixed_size
    if right > img_w:
        left = max(0, img_w - fixed_size)
        right = left + fixed_size
    if bottom > img_h:
        top = max(0, img_h - fixed_size)
        bottom = top + fixed_size
    if left < 0:
        left = 0
        right = min(img_w, fixed_size)
    if top < 0:
        top = 0
        bottom = min(img_h, fixed_size)
    return CropWindow(left, top, right, bottom, score=1.0, method="expand")


def project_box_to_crop(
    box: FrameBox,
    crop: CropWindow,
) -> FrameBox | None:
    """Clamp ``box`` into crop pixel space (``bbox_to_patch_yolo`` geometry)."""
    x1, y1, x2, y2 = box.as_xyxy()
    ix1 = max(x1, float(crop.left))
    iy1 = max(y1, float(crop.top))
    ix2 = min(x2, float(crop.right))
    iy2 = min(y2, float(crop.bottom))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return FrameBox(
        x1=ix1 - crop.left,
        y1=iy1 - crop.top,
        x2=ix2 - crop.left,
        y2=iy2 - crop.top,
        label=box.label,
        source=box.source,
    )


def find_flight_dirs(yolo_finds: Path, parsed: ParsedThumbName) -> list[Path]:
    """Candidate flight folders under ``yolo_finds`` for a parsed thumb."""
    yolo_finds = Path(yolo_finds)
    if not yolo_finds.is_dir() or parsed.frame_stem is None:
        return []
    site = parsed.site
    # DJIP4_RGB_AU_GRAV_20221023T064910_0723 → date 20221023
    m = re.search(r"_(\d{8})T\d{6}_\d{4}$", parsed.frame_stem, re.IGNORECASE)
    date = m.group(1) if m else None
    cands: list[Path] = []
    if site and date:
        cands.extend(sorted(yolo_finds.glob(f"{site}_{date}T*")))
    if not cands and site:
        cands.extend(sorted(yolo_finds.glob(f"{site}_*")))
    # Keep only dirs that actually contain this frame
    stem = parsed.frame_stem
    out: list[Path] = []
    for flight in cands:
        img_json = flight / "images" / f"{stem}.json"
        img_jpg = flight / "images" / f"{stem}.JPG"
        slices = flight / "slices"
        if img_json.is_file() or img_jpg.is_file():
            out.append(flight)
            continue
        if slices.is_dir() and any(slices.glob(f"{stem}_*.json")):
            out.append(flight)
    return out


def _rect_from_shape(shape: dict[str, Any], *, source: str) -> FrameBox | None:
    if shape.get("shape_type") not in (None, "rectangle"):
        # Some exports omit shape_type; require 2 points for rectangle
        pts = shape.get("points") or []
        if len(pts) != 2:
            return None
    pts = shape.get("points") or []
    if len(pts) < 2:
        return None
    try:
        x1, y1 = float(pts[0][0]), float(pts[0][1])
        x2, y2 = float(pts[1][0]), float(pts[1][1])
    except (TypeError, ValueError, IndexError):
        return None
    label = str(shape.get("label") or "turtle_3_yolo")
    if "turtle" not in label.lower():
        return None
    return FrameBox(x1, y1, x2, y2, label=label, source=source)


def load_boxes_from_image_json(path: Path) -> list[FrameBox]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: list[FrameBox] = []
    for shape in data.get("shapes") or []:
        if not isinstance(shape, dict):
            continue
        box = _rect_from_shape(shape, source="image")
        if box is not None:
            out.append(box)
    return out


def load_boxes_from_slices(slices_dir: Path, frame_stem: str) -> list[FrameBox]:
    """Lift slice-local rectangles into full-frame coords via filename xyxy."""
    if not slices_dir.is_dir():
        return []
    out: list[FrameBox] = []
    seen: set[tuple[float, float, float, float]] = set()
    for path in sorted(slices_dir.glob(f"{frame_stem}_*.json")):
        m = _SLICE_XYXY.search(path.stem)
        if m is None:
            continue
        ox, oy = float(m.group("x1")), float(m.group("y1"))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for shape in data.get("shapes") or []:
            if not isinstance(shape, dict):
                continue
            local = _rect_from_shape(shape, source="slice")
            if local is None:
                continue
            x1, y1, x2, y2 = local.as_xyxy()
            full = FrameBox(
                x1 + ox,
                y1 + oy,
                x2 + ox,
                y2 + oy,
                label=local.label,
                source="slice",
            )
            key = tuple(round(v, 2) for v in full.as_xyxy())
            if key in seen:
                continue
            seen.add(key)
            out.append(full)
    return out


def load_frame_boxes(flight: Path, frame_stem: str) -> list[FrameBox]:
    """Prefer full-frame image JSON; fall back to lifted slice boxes."""
    img_json = flight / "images" / f"{frame_stem}.json"
    boxes = load_boxes_from_image_json(img_json)
    if boxes:
        return boxes
    return load_boxes_from_slices(flight / "slices", frame_stem)


def load_metadata_box(
    metadata_root: Path | None,
    parsed: ParsedThumbName,
) -> FrameBox | None:
    """Lookup tight YOLO box from ``*_crops_metadata.csv`` (turtle build_compare_index)."""
    if metadata_root is None or not Path(metadata_root).is_dir():
        return None
    if parsed.frame_stem is None:
        return None
    stem_key = parsed.frame_stem.casefold()
    det = parsed.det_index
    thumb_key = parsed.thumb_stem.casefold().replace("_224px", "")
    for csv_path in Path(metadata_root).rglob("*_crops_metadata.csv"):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "x1" not in reader.fieldnames:
                    continue
                for row in reader:
                    crop_stem = Path(row.get("crop_path") or "").stem.casefold()
                    crop_stem = crop_stem.replace("expanded_224px_", "").replace(
                        "_224px", ""
                    )
                    orig = Path(row.get("original_image") or "").stem.casefold()
                    if orig and orig != stem_key and stem_key not in crop_stem:
                        continue
                    matched = False
                    if crop_stem == thumb_key or thumb_key.endswith(crop_stem):
                        matched = True
                    elif det is not None and f"_det_{det}_" in crop_stem and stem_key in crop_stem:
                        matched = True
                    if not matched:
                        continue
                    try:
                        return FrameBox(
                            float(row["x1"]),
                            float(row["y1"]),
                            float(row["x2"]),
                            float(row["y2"]),
                            label=str(row.get("class_name") or "turtle_3"),
                            source="metadata",
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
        except OSError:
            continue
    return None


def match_crop_window_cv2(
    thumb_path: Path,
    full_image_path: Path,
    *,
    fixed_size: int = 224,
) -> CropWindow | None:
    """Locate the 224 crop on the full frame with OpenCV matchTemplate."""
    try:
        import cv2
    except ImportError:
        return None
    thumb = cv2.imread(str(thumb_path), cv2.IMREAD_GRAYSCALE)
    full = cv2.imread(str(full_image_path), cv2.IMREAD_GRAYSCALE)
    if thumb is None or full is None:
        return None
    if thumb.shape[0] != fixed_size or thumb.shape[1] != fixed_size:
        thumb = cv2.resize(thumb, (fixed_size, fixed_size), interpolation=cv2.INTER_AREA)
    if full.shape[0] < fixed_size or full.shape[1] < fixed_size:
        return None
    result = cv2.matchTemplate(full, thumb, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    left, top = int(max_loc[0]), int(max_loc[1])
    return CropWindow(
        left,
        top,
        left + fixed_size,
        top + fixed_size,
        score=float(max_val),
        method="matchTemplate",
    )


def match_crop_window_via_slices(
    thumb_path: Path,
    slices_dir: Path,
    frame_stem: str,
    *,
    fixed_size: int = 224,
) -> CropWindow | None:
    """Match thumb against SHAI slice PNGs; return full-frame crop window."""
    try:
        import cv2
    except ImportError:
        return None
    if not slices_dir.is_dir():
        return None
    thumb = cv2.imread(str(thumb_path), cv2.IMREAD_GRAYSCALE)
    if thumb is None:
        return None
    if thumb.shape[0] != fixed_size or thumb.shape[1] != fixed_size:
        thumb = cv2.resize(thumb, (fixed_size, fixed_size), interpolation=cv2.INTER_AREA)
    best: CropWindow | None = None
    for png in sorted(slices_dir.glob(f"{frame_stem}_*.png")):
        m = _SLICE_XYXY.search(png.stem)
        if m is None:
            continue
        ox, oy = int(m.group("x1")), int(m.group("y1"))
        tile = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if tile is None or tile.shape[0] < fixed_size or tile.shape[1] < fixed_size:
            continue
        result = cv2.matchTemplate(tile, thumb, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        left = ox + int(max_loc[0])
        top = oy + int(max_loc[1])
        cand = CropWindow(
            left,
            top,
            left + fixed_size,
            top + fixed_size,
            score=float(max_val),
            method="sliceMatch",
        )
        if best is None or cand.score > best.score:
            best = cand
    return best


def resolve_full_image(flight: Path, frame_stem: str) -> Path | None:
    for ext in (".JPG", ".jpg", ".JPEG", ".jpeg", ".png"):
        p = flight / "images" / f"{frame_stem}{ext}"
        if p.is_file():
            return p
    return None


def boxes_for_thumb(
    thumb_path: Path,
    *,
    yolo_finds: Path,
    metadata_root: Path | None = None,
    fixed_size: int = 224,
    min_match_score: float = 0.5,
) -> tuple[list[FrameBox], CropWindow | None, str]:
    """Return thumb-local boxes for ``thumb_path``.

    Returns ``(thumb_boxes, crop_window, status)`` where status is
    ``ok`` | ``no_flight`` | ``no_boxes`` | ``no_crop`` | ``low_match``.
    """
    parsed = parse_thumb_filename(thumb_path)
    if parsed.frame_stem is None:
        return [], None, "no_flight"

    flights = find_flight_dirs(yolo_finds, parsed)
    if not flights:
        return [], None, "no_flight"
    flight = flights[0]

    meta_box = load_metadata_box(metadata_root, parsed)
    frame_boxes = load_frame_boxes(flight, parsed.frame_stem)
    full_image = resolve_full_image(flight, parsed.frame_stem)

    crop: CropWindow | None = None
    if meta_box is not None and full_image is not None:
        try:
            import cv2

            im = cv2.imread(str(full_image), cv2.IMREAD_GRAYSCALE)
            img_h, img_w = im.shape[:2] if im is not None else (3648, 5472)
        except Exception:
            img_w, img_h = 5472, 3648
        x1, y1, x2, y2 = meta_box.as_xyxy()
        crop = fixed_size_crop_window(
            x1, y1, x2, y2, img_w=img_w, img_h=img_h, fixed_size=fixed_size
        )
        if meta_box not in frame_boxes:
            frame_boxes = [meta_box, *frame_boxes]
    elif full_image is not None and Path(thumb_path).is_file():
        crop = match_crop_window_cv2(
            Path(thumb_path), full_image, fixed_size=fixed_size
        )
        if crop is None:
            return [], None, "no_crop"
        if crop.score < min_match_score:
            return [], crop, "low_match"
    elif meta_box is not None:
        x1, y1, x2, y2 = meta_box.as_xyxy()
        crop = fixed_size_crop_window(
            x1, y1, x2, y2, img_w=5472, img_h=3648, fixed_size=fixed_size
        )
        frame_boxes = [meta_box, *frame_boxes]
    elif Path(thumb_path).is_file():
        # No full JPG — match against SHAI slices (same boxes, tile-local)
        crop = match_crop_window_via_slices(
            Path(thumb_path),
            flight / "slices",
            parsed.frame_stem,
            fixed_size=fixed_size,
        )
        if crop is None:
            return [], None, "no_crop"
        if crop.score < min_match_score:
            return [], crop, "low_match"
        if not frame_boxes:
            frame_boxes = load_boxes_from_slices(flight / "slices", parsed.frame_stem)
    else:
        return [], None, "no_crop"

    if not frame_boxes:
        return [], crop, "no_boxes"

    assert crop is not None
    thumb_boxes: list[FrameBox] = []
    for box in frame_boxes:
        local = project_box_to_crop(box, crop)
        if local is not None:
            thumb_boxes.append(local)
    if not thumb_boxes:
        return [], crop, "no_boxes"
    return thumb_boxes, crop, "ok"


def rectangles_to_labelme_shapes(
    boxes: Iterable[FrameBox],
    *,
    crop: CropWindow | None = None,
) -> list[dict[str, Any]]:
    """LabelMe rectangle shapes for thumb-local boxes (SAM prompts)."""
    shapes: list[dict[str, Any]] = []
    for box in boxes:
        x1, y1, x2, y2 = box.as_xyxy()
        shapes.append(
            {
                "label": "turtle_box",
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {
                    "draft": True,
                    "source": box.source,
                    "prompt_kind": "box",
                    "crop_method": crop.method if crop else None,
                    "crop_score": crop.score if crop else None,
                },
            }
        )
    return shapes
