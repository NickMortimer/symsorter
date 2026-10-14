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
    pen = QPen(QColor(0, 255, 200, 230))
    pen.setWidth(pen_w)
    painter.setPen(pen)
    if outline_only:
        painter.setBrush(Qt.NoBrush)
    else:
        painter.setBrush(QColor(0, 220, 180, 50))

    for shape in polygons:
        pts = shape.get("points") or []
        if len(pts) < 3:
            continue
        poly = QPolygonF([QPointF(float(p[0]) * sx, float(p[1]) * sy) for p in pts])
        painter.drawPolygon(poly)

    painter.end()
    return canvas


def apply_outline_overlay_if_present(
    pixmap: QPixmap,
    thumb_path: str | Path,
    *,
    source_width: int | None = None,
    source_height: int | None = None,
    outline_only: bool = True,
) -> QPixmap:
    """If sibling LabelMe has polygons, draw them on a copy of ``pixmap``."""
    if pixmap is None or pixmap.isNull():
        return pixmap
    polygons = load_polygons_for_thumb(thumb_path)
    if not polygons:
        return pixmap
    sw = source_width if source_width is not None else pixmap.width()
    sh = source_height if source_height is not None else pixmap.height()
    # Prefer LabelMe image size when available via first load — caller may pass
    # raw pixmap size (pre-scale). For scaled icons, pass original w/h.
    return draw_polygons_on_pixmap(
        pixmap,
        polygons,
        source_width=sw,
        source_height=sh,
        outline_only=outline_only,
    )
