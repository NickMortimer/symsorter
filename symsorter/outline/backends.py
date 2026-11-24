"""Draft polygon backends in 224×224 thumb space.

``box`` / ``threshold`` need only numpy + Pillow. ``sam`` uses Ultralytics SAM
(same pattern as turtledrone ``utils/segment.py``): prompt with an inset box on
the thumb and take ``masks.xy`` as the draft polygon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_THUMB_SIZE = 224
DEFAULT_BOX_INSET = 8
DEFAULT_SAM_MODEL = "sam_b.pt"


@dataclass(frozen=True)
class DraftPolygon:
    """A draft polygon plus soft quality flags."""

    points: list[list[float]]
    backend: str
    flags: dict[str, Any]


def box_polygon(
    size: int = DEFAULT_THUMB_SIZE,
    inset: int = DEFAULT_BOX_INSET,
) -> DraftPolygon:
    """Inset rectangle covering most of the thumb (tight crop seed)."""
    inset = max(0, min(inset, (size // 2) - 1))
    x0 = float(inset)
    y0 = float(inset)
    x1 = float(size - inset)
    y1 = float(size - inset)
    points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return DraftPolygon(
        points=points,
        backend="box",
        flags={"draft": True, "backend": "box", "inset": inset},
    )


def _otsu_threshold(gray: np.ndarray) -> int:
    """Otsu threshold for uint8 grayscale."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    if total == 0:
        return 128
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return int(threshold)


def _largest_component_mask(binary: np.ndarray) -> np.ndarray:
    """Keep the largest connected foreground component.

    Uses ``scipy.ndimage`` when available (pulled in via scikit-learn);
    otherwise returns the full binary mask.
    """
    foreground = binary.astype(bool)
    if not foreground.any():
        return foreground
    try:
        from scipy import ndimage  # type: ignore
    except ImportError:
        return foreground

    labeled, n_labels = ndimage.label(foreground)
    if n_labels <= 0:
        return np.zeros_like(foreground)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # background
    best = int(counts.argmax())
    return labeled == best


def _mask_boundary_polygon(mask: np.ndarray, max_points: int = 32) -> list[list[float]]:
    """Sample a coarse polygon from the mask boundary (no OpenCV)."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return box_polygon(size=mask.shape[0]).points

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    if x_max <= x_min or y_max <= y_min:
        return box_polygon(size=mask.shape[0]).points

    # Vectorized edge pixels: foreground with a background 4-neighbor.
    up = np.zeros_like(mask)
    down = np.zeros_like(mask)
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)
    up[1:, :] = mask[:-1, :]
    down[:-1, :] = mask[1:, :]
    left[:, 1:] = mask[:, :-1]
    right[:, :-1] = mask[:, 1:]
    edge = mask & ~(up & down & left & right)
    ey, ex = np.where(edge)
    if len(ex) < 4:
        return [
            [float(x_min), float(y_min)],
            [float(x_max), float(y_min)],
            [float(x_max), float(y_max)],
            [float(x_min), float(y_max)],
        ]

    cx = float(xs.mean())
    cy = float(ys.mean())
    angles = np.arctan2(ey.astype(np.float64) - cy, ex.astype(np.float64) - cx)
    order = np.argsort(angles)
    ordered_x = ex[order]
    ordered_y = ey[order]
    step = max(1, len(ordered_x) // max_points)
    sampled_x = ordered_x[::step]
    sampled_y = ordered_y[::step]
    if len(sampled_x) < 3:
        return [
            [float(x_min), float(y_min)],
            [float(x_max), float(y_min)],
            [float(x_max), float(y_max)],
            [float(x_min), float(y_max)],
        ]
    return [[float(x), float(y)] for x, y in zip(sampled_x.tolist(), sampled_y.tolist())]


def threshold_polygon(image_path: Path, size: int = DEFAULT_THUMB_SIZE) -> DraftPolygon:
    """Simple silhouette via Otsu + largest blob (numpy/Pillow only).

    Soft-flags ``glint_suspect`` when the mask covers mostly bright pixels.
    Falls back to the inset box if the silhouette is degenerate.
    """
    with Image.open(image_path) as im:
        rgb = im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
        arr = np.asarray(rgb, dtype=np.uint8)

    # Prefer darker animal-on-bright-sand: invert if mean is high (typical surface).
    mean_val = float(gray.mean())
    work = gray if mean_val < 128 else (255 - gray)
    thr = _otsu_threshold(work)
    binary = work > thr

    # Drop tiny speckles by requiring a minimum area; use largest CC.
    mask = _largest_component_mask(binary)
    area = int(mask.sum())
    min_area = max(64, (size * size) // 50)
    if area < min_area:
        draft = box_polygon(size=size)
        return DraftPolygon(
            points=draft.points,
            backend="threshold",
            flags={
                "draft": True,
                "backend": "threshold",
                "fallback": "box",
                "reason": "small_mask",
                "glint_suspect": False,
            },
        )

    points = _mask_boundary_polygon(mask)
    bright_frac = float((gray[mask] > 220).mean()) if area else 0.0
    glint_suspect = bright_frac > 0.45 or float((arr.max(axis=2)[mask] > 240).mean()) > 0.4

    return DraftPolygon(
        points=points,
        backend="threshold",
        flags={
            "draft": True,
            "backend": "threshold",
            "glint_suspect": glint_suspect,
            "mask_area": area,
            "bright_frac": round(bright_frac, 4),
        },
    )


def load_sam_model(sam_model: str = DEFAULT_SAM_MODEL, device: str | None = None) -> Any:
    """Load Ultralytics SAM once for a batch. Requires ``pip install ultralytics``."""
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise ImportError(
            "SAM backend requires ultralytics. Install with: "
            "pip install 'symsorter[sam]'  (or: pip install ultralytics)"
        ) from exc
    model = SAM(sam_model)
    # Stash preferred device for predict calls (None → ultralytics default).
    model._symsorter_device = device
    return model


def _prompt_bbox_xyxy(size: int = DEFAULT_THUMB_SIZE, inset: int = DEFAULT_BOX_INSET) -> list[float]:
    """Inset box prompt covering the thumb (tight crop → almost whole frame)."""
    inset = max(0, min(inset, (size // 2) - 1))
    return [float(inset), float(inset), float(size - inset), float(size - inset)]


def _points_from_sam_result(result: Any) -> list[list[float]] | None:
    """Extract the first mask polygon from an Ultralytics Results object."""
    masks = getattr(result, "masks", None)
    if masks is None:
        return None
    xy = getattr(masks, "xy", None)
    if xy is None or len(xy) == 0:
        return None
    poly = xy[0]
    if poly is None:
        return None
    # numpy array or list
    if hasattr(poly, "tolist"):
        points = poly.tolist()
    else:
        points = list(poly)
    if not points or len(points) < 3:
        return None
    # Ensure [[x,y], ...] of floats
    cleaned: list[list[float]] = []
    for pt in points:
        if len(pt) < 2:
            continue
        cleaned.append([float(pt[0]), float(pt[1])])
    return cleaned if len(cleaned) >= 3 else None


def _sam_source(image_path: Path) -> Any:
    """Prefer BGR ndarray (turtledrone path); fall back to file path."""
    try:
        import cv2  # type: ignore

        im = cv2.imread(str(image_path))
        if im is not None:
            return im
    except ImportError:
        pass
    return str(image_path)


def sam_polygon(
    image_path: Path,
    model: Any | None = None,
    *,
    size: int = DEFAULT_THUMB_SIZE,
    inset: int = DEFAULT_BOX_INSET,
    sam_model: str = DEFAULT_SAM_MODEL,
    device: str | None = None,
    bboxes: list[float] | list[list[float]] | None = None,
    points: list[list[float]] | None = None,
    point_labels: list[int] | None = None,
) -> DraftPolygon:
    """Ultralytics SAM draft in thumb space.

    Prompt priority:
    1. Explicit ``points`` (click seeds; labels default to foreground=1)
    2. Explicit ``bboxes`` (user box, XYXY)
    3. Fallback inset full-thumb box (batch CLI default)
    """
    if model is None:
        model = load_sam_model(sam_model=sam_model, device=device)
    device = device if device is not None else getattr(model, "_symsorter_device", None)

    predict_kwargs: dict[str, Any] = {
        "verbose": False,
        "save": False,
    }
    if device is not None:
        predict_kwargs["device"] = device

    prompt_kind = "inset_box"
    used_bboxes: list[float] | list[list[float]] | None = None
    used_points: list[list[float]] | None = None

    if points:
        used_points = [[float(p[0]), float(p[1])] for p in points if len(p) >= 2]
        if not used_points:
            raise ValueError("points prompt was empty")
        labels = point_labels
        if labels is None:
            labels = [1] * len(used_points)
        if len(labels) != len(used_points):
            raise ValueError("point_labels length must match points")
        predict_kwargs["points"] = used_points
        predict_kwargs["labels"] = labels
        prompt_kind = "points"
    elif bboxes is not None:
        # Accept flat [x1,y1,x2,y2] or [[x1,y1,x2,y2], ...]
        if bboxes and isinstance(bboxes[0], (int, float)):
            used_bboxes = [float(v) for v in bboxes]  # type: ignore[arg-type]
        else:
            used_bboxes = bboxes
        predict_kwargs["bboxes"] = used_bboxes
        prompt_kind = "box"
    else:
        used_bboxes = _prompt_bbox_xyxy(size=size, inset=inset)
        predict_kwargs["bboxes"] = used_bboxes
        prompt_kind = "inset_box"

    source = _sam_source(image_path)
    results = model(source, **predict_kwargs)
    flags_base: dict[str, Any] = {
        "draft": True,
        "backend": "sam",
        "sam_model": sam_model,
        "prompt_kind": prompt_kind,
    }
    if used_bboxes is not None:
        flags_base["prompt_bbox"] = used_bboxes
    if used_points is not None:
        flags_base["prompt_points"] = used_points

    if not results:
        draft = box_polygon(size=size, inset=inset)
        flags_base.update({"fallback": "box", "reason": "empty_results"})
        return DraftPolygon(points=draft.points, backend="sam", flags=flags_base)

    poly_points = _points_from_sam_result(results[0])
    if poly_points is None:
        draft = box_polygon(size=size, inset=inset)
        flags_base.update({"fallback": "box", "reason": "no_mask"})
        return DraftPolygon(points=draft.points, backend="sam", flags=flags_base)

    flags_base["n_points"] = len(poly_points)
    return DraftPolygon(points=poly_points, backend="sam", flags=flags_base)


def draft_polygon(
    backend: str,
    image_path: Path | None = None,
    size: int = DEFAULT_THUMB_SIZE,
    inset: int = DEFAULT_BOX_INSET,
    *,
    sam_model_obj: Any | None = None,
    sam_model: str = DEFAULT_SAM_MODEL,
    device: str | None = None,
) -> DraftPolygon:
    """Dispatch to a draft backend: ``sam`` (default preference), ``box``, ``threshold``."""
    name = (backend or "sam").strip().lower()
    if name == "box":
        return box_polygon(size=size, inset=inset)
    if name == "threshold":
        if image_path is None:
            raise ValueError("threshold backend requires image_path")
        return threshold_polygon(image_path, size=size)
    if name == "sam":
        if image_path is None:
            raise ValueError("sam backend requires image_path")
        return sam_polygon(
            image_path,
            model=sam_model_obj,
            size=size,
            inset=inset,
            sam_model=sam_model,
            device=device,
        )
    raise ValueError(f"Unknown outline backend: {backend!r} (expected sam|box|threshold)")
