"""Draw LabelMe polygons onto QPixmaps (viewer + grid thumbs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF

from symsorter.outline.export_labelme import (
    extract_polygons,
    labelme_path_for_thumb,
    load_labelme_json,
)


def load_polygons_for_thumb(thumb_path: str | Path) -> list[dict[str, Any]]:
    """Load polygon shapes from the sibling LabelMe JSON, if present."""
    path = Path(thumb_path)
    data = load_labelme_json(labelme_path_for_thumb(path))
    return extract_polygons(data)


def draw_polygons_on_pixmap(
    pixmap: QPixmap,
    polygons: list[dict[str, Any]],
    *,
    source_width: int,
    source_height: int,
    outline_only: bool = False,
    selected_index: int | None = None,
) -> QPixmap:
    """Return a copy of ``pixmap`` with polygons scaled from source image coords."""
    if pixmap is None or pixmap.isNull() or not polygons:
        return pixmap

    canvas = QPixmap(pixmap)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing, True)

    sx = canvas.width() / max(1, source_width)
    sy = canvas.height() / max(1, source_height)

    # Thicker relative stroke on tiny grid icons
    pen_w = 2 if canvas.width() >= 160 else 1

    for i, shape in enumerate(polygons):
        pts = shape.get("points") or []
        if len(pts) < 3:
            continue
        selected = selected_index is not None and i == selected_index
        if selected:
            pen = QPen(QColor(255, 80, 80, 240))
            brush = QColor(255, 80, 80, 60)
        else:
            pen = QPen(QColor(0, 255, 200, 230))
            brush = QColor(0, 220, 180, 50)
        pen.setWidth(pen_w + (1 if selected else 0))
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush if outline_only and not selected else brush)
        poly = QPolygonF([QPointF(float(p[0]) * sx, float(p[1]) * sy) for p in pts])
        painter.drawPolygon(poly)

    painter.end()
    return canvas


def shift_polygons(
    polygons: list[dict[str, Any]],
    *,
    dx: float,
    dy: float,
) -> list[dict[str, Any]]:
    """Translate polygon points (e.g. into a center-crop coordinate frame)."""
    out: list[dict[str, Any]] = []
    for poly in polygons:
        pts = [[float(p[0]) - dx, float(p[1]) - dy] for p in (poly.get("points") or [])]
        if len(pts) < 3:
            continue
        shifted = dict(poly)
        shifted["points"] = pts
        out.append(shifted)
    return out


def apply_outline_overlay_if_present(
    pixmap: QPixmap,
    thumb_path: str | Path,
    *,
    source_width: int | None = None,
    source_height: int | None = None,
    outline_only: bool = True,
    crop_xyxy: tuple[int, int, int, int] | None = None,
    selected_index: int | None = None,
) -> QPixmap:
    """If sibling LabelMe has polygons, draw them on a copy of ``pixmap``.

    ``crop_xyxy`` is ``(left, top, right, bottom)`` in full-thumb pixels when the
    pixmap is a center crop — polygons are shifted into crop space first.
    """
    if pixmap is None or pixmap.isNull():
        return pixmap
    polygons = load_polygons_for_thumb(thumb_path)
    if not polygons:
        return pixmap
    if crop_xyxy is not None:
        left, top, right, bottom = crop_xyxy
        polygons = shift_polygons(polygons, dx=float(left), dy=float(top))
        sw = max(1, right - left)
        sh = max(1, bottom - top)
    else:
        sw = source_width if source_width is not None else pixmap.width()
        sh = source_height if source_height is not None else pixmap.height()
    return draw_polygons_on_pixmap(
        pixmap,
        polygons,
        source_width=sw,
        source_height=sh,
        outline_only=outline_only,
        selected_index=selected_index,
    )
