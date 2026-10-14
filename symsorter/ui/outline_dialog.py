"""Display LabelMe polygons over sorted thumbs (view-first outline UI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF, QFont
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from symsorter.outline.export_labelme import (
    extract_polygons,
    labelme_path_for_thumb,
    load_labelme_json,
)

DEFAULT_DISPLAY_SIZE = 672  # 3× 224 thumbs


def render_thumb_with_polygons(
    image_path: Path,
    polygons: list[dict[str, Any]],
    display_size: int = DEFAULT_DISPLAY_SIZE,
) -> QPixmap:
    """Scale thumb to ``display_size`` and draw closed polygons on top."""
    pix = QPixmap(str(image_path))
    if pix.isNull():
        blank = QPixmap(display_size, display_size)
        blank.fill(QColor(40, 40, 40))
        return blank

    scaled = pix.scaled(
        display_size,
        display_size,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )
    # Draw on a copy so we don't mutate the scaled cache unexpectedly.
    canvas = QPixmap(scaled)
    if not polygons:
        return canvas

    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing, True)

    sx = canvas.width() / max(1, pix.width())
    sy = canvas.height() / max(1, pix.height())

    fill = QColor(0, 220, 180, 55)
    pen = QPen(QColor(0, 255, 200, 230))
    pen.setWidth(2)
    painter.setPen(pen)
    painter.setBrush(fill)

    for shape in polygons:
        pts = shape.get("points") or []
        if len(pts) < 3:
            continue
        poly = QPolygonF([QPointF(p[0] * sx, p[1] * sy) for p in pts])
        painter.drawPolygon(poly)

    painter.end()
    return canvas


class OutlineViewerDialog(QDialog):
    """Modal viewer: show selected thumbs with sibling LabelMe polygons."""

    def __init__(
        self,
        parent: QWidget | None,
        entries: list[tuple[Path, str]],
        start_index: int = 0,
        display_size: int = DEFAULT_DISPLAY_SIZE,
    ):
        """
        ``entries``: list of ``(absolute_thumb_path, display_name)``.
        """
        super().__init__(parent)
        self.entries = entries
        self.index = max(0, min(start_index, len(entries) - 1)) if entries else 0
        self.display_size = display_size

        self.setWindowTitle("Outline viewer")
        self.resize(display_size + 40, display_size + 120)

        layout = QVBoxLayout(self)

        self.title_label = QLabel("")
        self.title_label.setWordWrap(True)
        title_font = QFont()
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(display_size, display_size)
        layout.addWidget(self.image_label, stretch=1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("← Prev")
        self.prev_btn.setShortcut("Left")
        self.prev_btn.clicked.connect(self.show_prev)
        nav.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next →")
        self.next_btn.setShortcut("Right")
        self.next_btn.clicked.connect(self.show_next)
        nav.addWidget(self.next_btn)

        nav.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setShortcut("Esc")
        close_btn.clicked.connect(self.accept)
        nav.addWidget(close_btn)
        layout.addLayout(nav)

        if not self.entries:
            self.title_label.setText("No images to show")
            self.status_label.setText("Select a thumb in the grid first.")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
        else:
            self.refresh()

    def show_prev(self) -> None:
        if not self.entries:
            return
        self.index = (self.index - 1) % len(self.entries)
        self.refresh()

    def show_next(self) -> None:
        if not self.entries:
            return
        self.index = (self.index + 1) % len(self.entries)
        self.refresh()

    def refresh(self) -> None:
        thumb_path, display_name = self.entries[self.index]
        json_path = labelme_path_for_thumb(thumb_path)
        data = load_labelme_json(json_path)
        polygons = extract_polygons(data)

        self.title_label.setText(
            f"{self.index + 1}/{len(self.entries)}  —  {display_name}"
        )

        if not thumb_path.exists():
            self.image_label.setPixmap(QPixmap())
            self.status_label.setText(f"Missing image: {thumb_path}")
            return

        pix = render_thumb_with_polygons(
            thumb_path, polygons, display_size=self.display_size
        )
        self.image_label.setPixmap(pix)

        if data is None:
            self.status_label.setText(
                f"No outline JSON beside thumb.\nExpected: {json_path.name}"
            )
        elif not polygons:
            self.status_label.setText(
                f"Found {json_path.name} but no polygon shapes."
            )
        else:
            flags = polygons[0].get("flags") or {}
            backend = flags.get("backend")
            draft = flags.get("draft")
            label = polygons[0].get("label")
            n_pts = len(polygons[0].get("points") or [])
            parts = [
                f"{len(polygons)} polygon(s)",
                f"label={label}",
                f"points={n_pts}",
            ]
            if backend is not None:
                parts.append(f"backend={backend}")
            if draft is not None:
                parts.append(f"draft={draft}")
            parts.append(f"file={json_path.name}")
            self.status_label.setText(" · ".join(parts))

        self.prev_btn.setEnabled(len(self.entries) > 1)
        self.next_btn.setEnabled(len(self.entries) > 1)


def open_outline_viewer_for_browser(browser: QWidget) -> None:
    """Collect selected (or current) thumbs from ImageBrowser and open viewer."""
    folder = getattr(browser, "folder", "") or ""
    if not folder:
        QMessageBox.information(
            browser,
            "Outline viewer",
            "Open a folder or embeddings NPZ first.",
        )
        return

    folder_path = Path(folder)
    view = getattr(browser, "view", None)
    model = getattr(browser, "model", None)
    if view is None or model is None:
        return

    indexes = view.selectedIndexes()
    filenames: list[str] = []
    for index in indexes:
        fn = index.data(Qt.UserRole)
        if fn and fn not in filenames:
            filenames.append(fn)

    if not filenames:
        # Fall back to current index
        current = view.currentIndex()
        if current.isValid():
            fn = current.data(Qt.UserRole)
            if fn:
                filenames = [fn]

    if not filenames:
        QMessageBox.information(
            browser,
            "Outline viewer",
            "Select one or more thumbs in the grid, then press O.",
        )
        return

    entries: list[tuple[Path, str]] = []
    for fn in filenames:
        path = folder_path / fn
        entries.append((path, fn))

    dlg = OutlineViewerDialog(browser, entries=entries, start_index=0)
    dlg.exec()
