"""Outline viewer: multi-turtle polygons, SAM point/box, lasso cleanup, delete."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from symsorter.outline.backends import DEFAULT_THUMB_SIZE, load_sam_model, sam_polygon
from symsorter.outline.export_labelme import (
    extract_polygons,
    labelme_path_for_thumb,
    load_labelme_json,
    set_turtle_polygons,
    simplify_polyline,
)
from symsorter.outline.filename import parse_thumb_filename
from symsorter.outline.overlay import draw_polygons_on_pixmap

DEFAULT_DISPLAY_SIZE = 672  # 3× 224 thumbs


def render_thumb_with_polygons(
    image_path: Path,
    polygons: list[dict[str, Any]],
    display_size: int = DEFAULT_DISPLAY_SIZE,
    selected_index: int | None = None,
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
    if not polygons:
        return scaled
    return draw_polygons_on_pixmap(
        scaled,
        polygons,
        source_width=pix.width(),
        source_height=pix.height(),
        outline_only=False,
        selected_index=selected_index,
    )


class OutlineCanvas(QLabel):
    """Point / box / lasso / delete interactions in image pixel coords."""

    pointSeeded = Signal(float, float)
    boxSeeded = Signal(float, float, float, float)
    lassoFinished = Signal(object)  # list[[x,y], ...]
    deleteAt = Signal(float, float)
    selectAt = Signal(float, float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.mode = "point"
        self._base_pix: QPixmap | None = None
        self._img_w = DEFAULT_THUMB_SIZE
        self._img_h = DEFAULT_THUMB_SIZE
        self._drag_origin: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._lasso: list[tuple[float, float]] = []
        self._drawing_lasso = False

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._drag_origin = None
        self._drag_current = None
        self._lasso = []
        self._drawing_lasso = False
        self._redraw()

    def set_image(
        self,
        pixmap: QPixmap,
        *,
        image_width: int,
        image_height: int,
    ) -> None:
        self._base_pix = QPixmap(pixmap)
        self._img_w = max(1, image_width)
        self._img_h = max(1, image_height)
        self._drag_origin = None
        self._drag_current = None
        self._lasso = []
        self._drawing_lasso = False
        self._redraw()

    def _content_rect(self) -> QRect:
        if self._base_pix is None or self._base_pix.isNull():
            return QRect()
        pw, ph = self._base_pix.width(), self._base_pix.height()
        x = (self.width() - pw) // 2
        y = (self.height() - ph) // 2
        return QRect(x, y, pw, ph)

    def _label_to_image(self, pos: QPoint) -> tuple[float, float] | None:
        rect = self._content_rect()
        if not rect.contains(pos) or self._base_pix is None:
            return None
        lx = pos.x() - rect.x()
        ly = pos.y() - rect.y()
        sx = self._img_w / max(1, rect.width())
        sy = self._img_h / max(1, rect.height())
        ix = max(0.0, min(float(self._img_w - 1), lx * sx))
        iy = max(0.0, min(float(self._img_h - 1), ly * sy))
        return ix, iy

    def _redraw(self) -> None:
        if self._base_pix is None or self._base_pix.isNull():
            self.setPixmap(QPixmap())
            return
        canvas = QPixmap(self._base_pix)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing, True)
        sx = canvas.width() / self._img_w
        sy = canvas.height() / self._img_h

        if self._lasso and len(self._lasso) >= 2:
            painter.setPen(QPen(QColor(255, 220, 0), 2))
            painter.setBrush(QColor(255, 220, 0, 35))
            poly = QPolygonF([QPointF(x * sx, y * sy) for x, y in self._lasso])
            painter.drawPolyline(poly)

        if self._drag_origin is not None and self._drag_current is not None:
            content = self._content_rect()
            x0 = self._drag_origin.x() - content.x()
            y0 = self._drag_origin.y() - content.y()
            x1 = self._drag_current.x() - content.x()
            y1 = self._drag_current.y() - content.y()
            painter.setPen(QPen(QColor(255, 220, 0), 2, Qt.DashLine))
            painter.setBrush(QColor(255, 220, 0, 40))
            painter.drawRect(QRect(QPoint(x0, y0), QPoint(x1, y1)).normalized())

        painter.end()
        self.setPixmap(canvas)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or self.mode == "view":
            return super().mousePressEvent(event)
        pos = event.position().toPoint()
        mapped = self._label_to_image(pos)

        if self.mode == "point":
            if mapped is None:
                return
            self.pointSeeded.emit(mapped[0], mapped[1])
            return
        if self.mode == "delete":
            if mapped is None:
                return
            self.deleteAt.emit(mapped[0], mapped[1])
            return
        if self.mode == "select":
            if mapped is None:
                return
            self.selectAt.emit(mapped[0], mapped[1])
            return
        if self.mode == "box":
            self._drag_origin = pos
            self._drag_current = pos
            self._redraw()
            return
        if self.mode == "lasso":
            if mapped is None:
                return
            self._drawing_lasso = True
            self._lasso = [mapped]
            self._redraw()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()
        if self.mode == "box" and self._drag_origin is not None:
            self._drag_current = pos
            self._redraw()
            return
        if self.mode == "lasso" and self._drawing_lasso:
            mapped = self._label_to_image(pos)
            if mapped is not None:
                self._lasso.append(mapped)
                self._redraw()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return super().mouseReleaseEvent(event)

        if self.mode == "box" and self._drag_origin is not None:
            p0 = self._label_to_image(self._drag_origin)
            p1 = self._label_to_image(self._drag_current or self._drag_origin)
            self._drag_origin = None
            self._drag_current = None
            self._redraw()
            if p0 is None or p1 is None:
                return
            x1, y1 = min(p0[0], p1[0]), min(p0[1], p1[1])
            x2, y2 = max(p0[0], p1[0]), max(p0[1], p1[1])
            if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                self.boxSeeded.emit(x1, y1, x2, y2)
            return

        if self.mode == "lasso" and self._drawing_lasso:
            self._drawing_lasso = False
            pts = [[float(x), float(y)] for x, y in self._lasso]
            self._lasso = []
            self._redraw()
            if len(pts) >= 3:
                self.lassoFinished.emit(simplify_polyline(pts, max_points=64))
            return

        super().mouseReleaseEvent(event)


class OutlineViewerDialog(QDialog):
    """Multi-polygon outline editor for selected thumbs."""

    def __init__(
        self,
        parent: QWidget | None,
        entries: list[tuple[Path, str]],
        start_index: int = 0,
        display_size: int = DEFAULT_DISPLAY_SIZE,
        sam_model: str = "sam_b.pt",
        device: str | None = None,
    ):
        super().__init__(parent)
        self.entries = entries
        self.index = max(0, min(start_index, len(entries) - 1)) if entries else 0
        self.display_size = display_size
        self.sam_model_name = sam_model
        self.device = device
        self._sam = None
        self._polygons: list[dict[str, Any]] = []
        self._selected: int | None = None
        self._dirty = False
        self._img_w = DEFAULT_THUMB_SIZE
        self._img_h = DEFAULT_THUMB_SIZE

        self.setWindowTitle("Outline viewer")
        self.resize(display_size + 80, display_size + 220)

        layout = QVBoxLayout(self)

        self.title_label = QLabel("")
        self.title_label.setWordWrap(True)
        title_font = QFont()
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Tool:"))
        self.mode_group = QButtonGroup(self)
        self.radio_point = QRadioButton("SAM point (+)")
        self.radio_box = QRadioButton("SAM box (+)")
        self.radio_lasso = QRadioButton("Lasso edit")
        self.radio_delete = QRadioButton("Delete")
        self.radio_view = QRadioButton("View")
        self.radio_point.setChecked(True)
        for btn in (
            self.radio_point,
            self.radio_box,
            self.radio_lasso,
            self.radio_delete,
            self.radio_view,
        ):
            self.mode_group.addButton(btn)
            mode_row.addWidget(btn)
            btn.toggled.connect(self._on_mode_changed)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        self.canvas = OutlineCanvas()
        self.canvas.setMinimumSize(display_size, display_size)
        self.canvas.pointSeeded.connect(self._on_point_seed)
        self.canvas.boxSeeded.connect(self._on_box_seed)
        self.canvas.lassoFinished.connect(self._on_lasso)
        self.canvas.deleteAt.connect(self._on_delete_at)
        layout.addWidget(self.canvas, stretch=1)

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

        self.del_btn = QPushButton("Delete selected")
        self.del_btn.setShortcut("Delete")
        self.del_btn.clicked.connect(self.delete_selected)
        nav.addWidget(self.del_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.setShortcut("Ctrl+S")
        self.save_btn.clicked.connect(self.save_current)
        nav.addWidget(self.save_btn)

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
            self.save_btn.setEnabled(False)
        else:
            self._on_mode_changed()
            self.refresh()

    def _on_mode_changed(self) -> None:
        if self.radio_point.isChecked():
            self.canvas.set_mode("point")
            tip = "Click to ADD a SAM polygon (multi-turtle OK)."
        elif self.radio_box.isChecked():
            self.canvas.set_mode("box")
            tip = "Drag a box to ADD a SAM polygon."
        elif self.radio_lasso.isChecked():
            self.canvas.set_mode("lasso")
            tip = "Draw a freehand lasso to replace the selected polygon (or add if none)."
        elif self.radio_delete.isChecked():
            self.canvas.set_mode("delete")
            tip = "Click a polygon to delete it."
        else:
            self.canvas.set_mode("view")
            tip = "View only."
        if self.entries:
            self.status_label.setText(tip)

    def _ensure_sam(self) -> Any:
        if self._sam is None:
            self.status_label.setText(f"Loading SAM ({self.sam_model_name})…")
            self.repaint()
            self._sam = load_sam_model(self.sam_model_name, device=self.device)
        return self._sam

    def _current_paths(self) -> tuple[Path, str]:
        return self.entries[self.index]

    def _confirm_leave_if_dirty(self) -> bool:
        if not self._dirty:
            return True
        ans = QMessageBox.question(
            self,
            "Unsaved outline",
            "Save polygon changes before leaving?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if ans == QMessageBox.Cancel:
            return False
        if ans == QMessageBox.Save:
            return self.save_current()
        self._dirty = False
        return True

    def show_prev(self) -> None:
        if not self.entries or not self._confirm_leave_if_dirty():
            return
        self.index = (self.index - 1) % len(self.entries)
        self.refresh()

    def show_next(self) -> None:
        if not self.entries or not self._confirm_leave_if_dirty():
            return
        self.index = (self.index + 1) % len(self.entries)
        self.refresh()

    def _repaint_canvas(self) -> None:
        thumb_path, _ = self._current_paths()
        display = render_thumb_with_polygons(
            thumb_path,
            self._polygons,
            display_size=self.display_size,
            selected_index=self._selected,
        )
        self.canvas.set_image(
            display, image_width=self._img_w, image_height=self._img_h
        )

    def refresh(self) -> None:
        thumb_path, display_name = self._current_paths()
        data = load_labelme_json(labelme_path_for_thumb(thumb_path))
        self._polygons = extract_polygons(data)
        self._selected = len(self._polygons) - 1 if self._polygons else None
        self._dirty = False

        self.title_label.setText(
            f"{self.index + 1}/{len(self.entries)}  —  {display_name}  "
            f"({len(self._polygons)} turtle polygon(s))"
        )

        if not thumb_path.exists():
            self.canvas.set_image(QPixmap(), image_width=1, image_height=1)
            self.status_label.setText(f"Missing image: {thumb_path}")
            return

        pix = QPixmap(str(thumb_path))
        self._img_w = pix.width() or DEFAULT_THUMB_SIZE
        self._img_h = pix.height() or DEFAULT_THUMB_SIZE
        self._repaint_canvas()
        self._on_mode_changed()
        self.prev_btn.setEnabled(len(self.entries) > 1)
        self.next_btn.setEnabled(len(self.entries) > 1)

    def _on_point_seed(self, x: float, y: float) -> None:
        thumb_path, _ = self._current_paths()
        try:
            draft = sam_polygon(
                thumb_path,
                model=self._ensure_sam(),
                points=[[x, y]],
                point_labels=[1],
                device=self.device,
                sam_model=self.sam_model_name,
            )
            self._polygons.append(
                {
                    "label": "turtle",
                    "points": draft.points,
                    "flags": dict(draft.flags),
                    "shape_type": "polygon",
                }
            )
            self._selected = len(self._polygons) - 1
            self._dirty = True
            self._repaint_canvas()
            self.status_label.setText(
                f"Added SAM polygon ({len(self._polygons)} total). Save when done."
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "SAM failed", str(exc))

    def _on_box_seed(self, x1: float, y1: float, x2: float, y2: float) -> None:
        thumb_path, _ = self._current_paths()
        try:
            draft = sam_polygon(
                thumb_path,
                model=self._ensure_sam(),
                bboxes=[x1, y1, x2, y2],
                device=self.device,
                sam_model=self.sam_model_name,
            )
            self._polygons.append(
                {
                    "label": "turtle",
                    "points": draft.points,
                    "flags": dict(draft.flags),
                    "shape_type": "polygon",
                }
            )
            self._selected = len(self._polygons) - 1
            self._dirty = True
            self._repaint_canvas()
            self.status_label.setText(
                f"Added SAM polygon ({len(self._polygons)} total). Save when done."
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "SAM failed", str(exc))

    def _on_lasso(self, points: list) -> None:
        if len(points) < 3:
            return
        flags = {
            "draft": True,
            "backend": "lasso",
            "prompt_kind": "lasso",
        }
        if self._selected is not None and 0 <= self._selected < len(self._polygons):
            old = self._polygons[self._selected]
            flags = {**(old.get("flags") or {}), **flags}
            self._polygons[self._selected] = {
                "label": "turtle",
                "points": points,
                "flags": flags,
                "shape_type": "polygon",
            }
            self.status_label.setText(
                f"Replaced polygon #{self._selected + 1} with lasso. Save when done."
            )
        else:
            self._polygons.append(
                {
                    "label": "turtle",
                    "points": points,
                    "flags": flags,
                    "shape_type": "polygon",
                }
            )
            self._selected = len(self._polygons) - 1
            self.status_label.setText("Added lasso polygon. Save when done.")
        self._dirty = True
        self._repaint_canvas()

    def _on_delete_at(self, x: float, y: float) -> None:
        # Use in-memory polygons; map click to index
        from symsorter.outline.export_labelme import _point_in_poly, _centroid

        if not self._polygons:
            return
        hit = None
        for i, poly in enumerate(self._polygons):
            if _point_in_poly(x, y, poly["points"]):
                hit = i
                break
        if hit is None:
            best_d = float("inf")
            for i, poly in enumerate(self._polygons):
                cx, cy = _centroid(poly["points"])
                d = (cx - x) ** 2 + (cy - y) ** 2
                if d < best_d:
                    best_d = d
                    hit = i
        if hit is None:
            return
        self._polygons.pop(hit)
        self._selected = min(hit, len(self._polygons) - 1) if self._polygons else None
        self._dirty = True
        self._repaint_canvas()
        self.status_label.setText(
            f"Deleted polygon. {len(self._polygons)} remaining. Save when done."
        )

    def delete_selected(self) -> None:
        if self._selected is None or not self._polygons:
            # fall back: delete last
            if not self._polygons:
                return
            self._selected = len(self._polygons) - 1
        self._polygons.pop(self._selected)
        self._selected = (
            min(self._selected, len(self._polygons) - 1) if self._polygons else None
        )
        self._dirty = True
        self._repaint_canvas()
        self.status_label.setText(
            f"Deleted. {len(self._polygons)} polygon(s) left. Save when done."
        )

    def save_current(self) -> bool:
        if not self.entries:
            return False
        thumb_path, _ = self._current_paths()
        parsed = parse_thumb_filename(thumb_path.name)
        polys = []
        for poly in self._polygons:
            flags = dict(poly.get("flags") or {})
            flags["det_index"] = parsed.det_index
            flags.setdefault("turtle_group_id", None)
            flags.setdefault("draft", True)
            polys.append(
                {
                    "label": "turtle",
                    "points": poly["points"],
                    "flags": flags,
                    "shape_type": "polygon",
                }
            )
        set_turtle_polygons(
            labelme_path_for_thumb(thumb_path),
            polys,
            image_path=thumb_path.name,
            image_height=self._img_h,
            image_width=self._img_w,
        )
        self._dirty = False
        self.status_label.setText(
            f"Saved {len(polys)} turtle polygon(s) → {thumb_path.with_suffix('.json').name}"
        )
        parent = self.parent()
        if parent is not None and hasattr(parent, "refresh_outline_icon"):
            parent.refresh_outline_icon(thumb_path.name)
        return True


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
    if view is None:
        return

    indexes = view.selectedIndexes()
    filenames: list[str] = []
    for index in indexes:
        fn = index.data(Qt.UserRole)
        if fn and fn not in filenames:
            filenames.append(fn)

    if not filenames:
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

    entries: list[tuple[Path, str]] = [(folder_path / fn, fn) for fn in filenames]
    dlg = OutlineViewerDialog(browser, entries=entries, start_index=0)
    dlg.exec()
