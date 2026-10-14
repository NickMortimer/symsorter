"""Draft polygon backends in 224×224 thumb space (no heavy ML deps)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_THUMB_SIZE = 224
DEFAULT_BOX_INSET = 8


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


def draft_polygon(
    backend: str,
    image_path: Path | None = None,
    size: int = DEFAULT_THUMB_SIZE,
    inset: int = DEFAULT_BOX_INSET,
) -> DraftPolygon:
    """Dispatch to a draft backend. ``sam`` is intentionally unimplemented."""
    name = (backend or "box").strip().lower()
    if name == "box":
        return box_polygon(size=size, inset=inset)
    if name == "threshold":
        if image_path is None:
            raise ValueError("threshold backend requires image_path")
        return threshold_polygon(image_path, size=size)
    if name == "sam":
        raise NotImplementedError(
            "SAM backend is not part of the MVP; use --backend box or threshold"
        )
    raise ValueError(f"Unknown outline backend: {backend!r} (expected box|threshold)")
