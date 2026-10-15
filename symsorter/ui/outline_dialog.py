"""Outline viewer: zoomable multi-turtle editor (vertices, lasso, eraser, SAM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
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
from symsorter.outline.edit_geom import (
    delete_points_inside,
    erase_points_near,
    fit_view,
    hit_polygon_index,
    insert_vertex,
    nearest_edge,
    nearest_vertex,
    zoom_at_pointer,
)
from symsorter.outline.export_labelme import (
    extract_polygons,
    labelme_path_for_thumb,
    load_labelme_json,
    set_turtle_polygons,
    simplify_polyline,
)
from symsorter.outline.filename import parse_thumb_filename

DEFAULT_DISPLAY_SIZE = 720
FIT_PAD_FRAC = 0.10
MIN_VERTICES = 3
# Hit thresholds in *screen* pixels (converted via / zoom)
VERTEX_HIT_PX = 10.0
EDGE_HIT_PX = 8.0
ERASER_RADIUS_PX = 14.0


def render_thumb_with_polygons(
    image_path: Path,
    polygons: list[dict[str, Any]],
    display_size: int = DEFAULT_DISPLAY_SIZE,
    selected_index: int | None = None,
) -> QPixmap:
    """Backward-compatible helper used by tests: scale + draw filled polys."""
    from symsorter.outline.overlay import draw_polygons_on_pixmap

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


class OutlineCanvas(QWidget):
    """Zoomable image canvas with vertex edit / lasso / eraser / SAM tools."""

    pointSeeded = Signal(float, float)
    boxSeeded = Signal(float, float, float, float)
    polygonsChanged = Signal()  # after in-place vertex edits
    statusMessage = Signal(str)
    selectionChanged = Signal(object)  # int | None

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 320)

        self.mode = "edit"
        self._src: QPixmap | None = None
        self._img_w = DEFAULT_THUMB_SIZE
        self._img_h = DEFAULT_THUMB_SIZE
        self._polygons: list[dict[str, Any]] = []
        self._selected: int | None = None

        self._zoom = 1.0
        self._ox = 0.0
        self._oy = 0.0
        self._fit_pending = True

        # Interaction state
        self._drag_vertex: int | None = None
        self._box_origin: QPoint | None = None
        self._box_current: QPoint | None = None
        self._lasso: list[tuple[float, float]] = []
        self._drawing_lasso = False
        self._panning = False
        self._pan_last: QPoint | None = None
        self._erasing = False
        self._hover: QPoint | None = None

    # ----- public API -----

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._cancel_transient()
        cursors = {
            "edit": Qt.ArrowCursor,
            "add_point": Qt.CrossCursor,
            "lasso": Qt.CrossCursor,
            "eraser": Qt.CrossCursor,
            "point": Qt.CrossCursor,
            "box": Qt.CrossCursor,
            "delete": Qt.PointingHandCursor,
            "view": Qt.OpenHandCursor,
        }
        self.setCursor(cursors.get(mode, Qt.ArrowCursor))
        self.update()

    def set_scene(
        self,
        pixmap: QPixmap,
        polygons: list[dict[str, Any]],
        *,
        selected_index: int | None,
        image_width: int,
        image_height: int,
        refit: bool = False,
    ) -> None:
        self._src = QPixmap(pixmap) if pixmap is not None else None
        self._img_w = max(1, image_width)
        self._img_h = max(1, image_height)
        self._polygons = polygons
        self._selected = selected_index
        self._cancel_transient()
        if refit or self._fit_pending:
            self._fit_pending = True
            self.fit_to_polygons()
        self.update()

    def set_selected(self, index: int | None) -> None:
        self._selected = index
        self.update()

    def fit_to_polygons(self) -> None:
        if self.width() < 2 or self.height() < 2:
            self._fit_pending = True
            return
        z, ox, oy = fit_view(
            view_w=float(self.width()),
            view_h=float(self.height()),
            img_w=float(self._img_w),
            img_h=float(self._img_h),
            polygons=self._polygons,
            pad_frac=FIT_PAD_FRAC,
        )
        self._zoom, self._ox, self._oy = z, ox, oy
        self._fit_pending = False
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._fit_pending:
            self.fit_to_polygons()

    # ----- coordinate transforms -----

    def _img_to_widget(self, ix: float, iy: float) -> QPointF:
        return QPointF(self._ox + ix * self._zoom, self._oy + iy * self._zoom)

    def _widget_to_img(self, pos: QPoint | QPointF) -> tuple[float, float]:
        ix = (float(pos.x()) - self._ox) / max(1e-9, self._zoom)
        iy = (float(pos.y()) - self._oy) / max(1e-9, self._zoom)
        return ix, iy

    def _hit_radius_img(self, screen_px: float) -> float:
        return screen_px / max(1e-9, self._zoom)

    def _selected_points(self) -> list[list[float]] | None:
        if self._selected is None or not (0 <= self._selected < len(self._polygons)):
            return None
        return self._polygons[self._selected].get("points")

    def _set_selected_points(self, pts: list[list[float]]) -> None:
        if self._selected is None or not (0 <= self._selected < len(self._polygons)):
            return
        self._polygons[self._selected]["points"] = pts
        self.polygonsChanged.emit()

    def _cancel_transient(self) -> None:
        self._drag_vertex = None
        self._box_origin = None
        self._box_current = None
        self._lasso = []
        self._drawing_lasso = False
        self._panning = False
        self._pan_last = None
        self._erasing = False

    # ----- painting -----

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(28, 28, 32))
        if self._src is None or self._src.isNull():
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image")
            painter.end()
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.Antialiasing, True)

        target = QRectF(
            self._ox,
            self._oy,
            self._img_w * self._zoom,
            self._img_h * self._zoom,
        )
        painter.drawPixmap(target, self._src, QRectF(self._src.rect()))

        for i, shape in enumerate(self._polygons):
            pts = shape.get("points") or []
            if len(pts) < 2:
                continue
            selected = i == self._selected
            pen = QPen(QColor(255, 80, 80, 240) if selected else QColor(0, 255, 200, 220))
            pen.setWidthF(2.0 if selected else 1.5)
            painter.setPen(pen)
            brush = QColor(255, 80, 80, 45) if selected else QColor(0, 220, 180, 35)
            painter.setBrush(brush)
            poly = QPolygonF([self._img_to_widget(float(p[0]), float(p[1])) for p in pts])
            painter.drawPolygon(poly)

            if selected:
                # Vertex handles
                for p in pts:
                    wp = self._img_to_widget(float(p[0]), float(p[1]))
                    painter.setBrush(QColor(255, 255, 255, 230))
                    painter.setPen(QPen(QColor(220, 40, 40), 1.5))
                    r = 4.0
                    painter.drawEllipse(wp, r, r)

        # Lasso preview
        if self._lasso and len(self._lasso) >= 2:
            painter.setPen(QPen(QColor(255, 220, 0), 2))
            painter.setBrush(QColor(255, 220, 0, 40))
            poly = QPolygonF(
                [self._img_to_widget(x, y) for x, y in self._lasso]
            )
            painter.drawPolyline(poly)

        # SAM box preview
        if self._box_origin is not None and self._box_current is not None:
            painter.setPen(QPen(QColor(255, 220, 0), 2, Qt.DashLine))
            painter.setBrush(QColor(255, 220, 0, 40))
            painter.drawRect(
                QRectF(self._box_origin, self._box_current).normalized()
            )

        # Eraser cursor circle
        if self.mode == "eraser" and self._hover is not None:
            painter.setPen(QPen(QColor(255, 120, 80, 220), 1.5))
            painter.setBrush(QColor(255, 120, 80, 40))
            painter.drawEllipse(
                QPointF(self._hover), ERASER_RADIUS_PX, ERASER_RADIUS_PX
            )

        painter.end()

    # ----- input -----

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        pos = event.position()
        self._zoom, self._ox, self._oy = zoom_at_pointer(
            zoom=self._zoom,
            offset_x=self._ox,
            offset_y=self._oy,
            pointer_x=float(pos.x()),
            pointer_y=float(pos.y()),
            factor=factor,
        )
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()

        # Pan: middle button, or right button, or Space+left, or view mode left
        if event.button() == Qt.MiddleButton or event.button() == Qt.RightButton or (
            event.button() == Qt.LeftButton
            and (self.mode == "view" or bool(event.modifiers() & Qt.AltModifier))
        ):
            self._panning = True
            self._pan_last = pos
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)

        ix, iy = self._widget_to_img(pos)

        if self.mode == "point":
            self.pointSeeded.emit(ix, iy)
            return

        if self.mode == "box":
            self._box_origin = pos
            self._box_current = pos
            self.update()
            return

        if self.mode == "delete":
            hit = hit_polygon_index(self._polygons, ix, iy)
            if hit is not None:
                self._polygons.pop(hit)
                self._selected = (
                    min(hit, len(self._polygons) - 1) if self._polygons else None
                )
                self.selectionChanged.emit(self._selected)
                self.polygonsChanged.emit()
                self.statusMessage.emit(
                    f"Deleted polygon. {len(self._polygons)} remaining."
                )
                self.update()
            return

        if self.mode == "lasso":
            self._drawing_lasso = True
            self._lasso = [(ix, iy)]
            self.update()
            return

        if self.mode == "eraser":
            self._erasing = True
            self._erase_at(ix, iy)
            return

        if self.mode in {"edit", "add_point"}:
            self._edit_press(ix, iy)
            return

        super().mousePressEvent(event)

    def _edit_press(self, ix: float, iy: float) -> None:
        # Prefer vertex on selected poly
        pts = self._selected_points()
        if pts is not None:
            vi = nearest_vertex(
                pts, ix, iy, max_dist=self._hit_radius_img(VERTEX_HIT_PX)
            )
            if vi is not None and self.mode == "edit":
                self._drag_vertex = vi
                return
            # Insert on edge
            if self.mode in {"edit", "add_point"}:
                edge = nearest_edge(
                    pts, ix, iy, max_dist=self._hit_radius_img(EDGE_HIT_PX)
                )
                if edge is not None:
                    after, px, py = edge
                    # In add_point mode use click location; edit uses projection
                    ax, ay = (ix, iy) if self.mode == "add_point" else (px, py)
                    new_pts = insert_vertex(pts, after, ax, ay)
                    self._set_selected_points(new_pts)
                    self._drag_vertex = after + 1 if self.mode == "edit" else None
                    self.statusMessage.emit(
                        f"Added vertex (now {len(new_pts)}). Save when done."
                    )
                    self.update()
                    return

        # Select another polygon
        hit = hit_polygon_index(self._polygons, ix, iy)
        if hit is not None and hit != self._selected:
            self._selected = hit
            self.selectionChanged.emit(hit)
            self.statusMessage.emit(f"Selected polygon #{hit + 1}")
            self.update()
            return

        # add_point with no near-edge: insert at nearest edge regardless of dist
        # (or append if user clicks freely near selected)
        if self.mode == "add_point" and pts is not None and len(pts) >= 2:
            edge = nearest_edge(pts, ix, iy, max_dist=1e9)
            if edge is not None:
                after, _, _ = edge
                new_pts = insert_vertex(pts, after, ix, iy)
                self._set_selected_points(new_pts)
                self.statusMessage.emit(
                    f"Added vertex (now {len(new_pts)}). Save when done."
                )
                self.update()

    def _erase_at(self, ix: float, iy: float) -> None:
        pts = self._selected_points()
        if pts is None:
            # Auto-select under eraser
            hit = hit_polygon_index(self._polygons, ix, iy)
            if hit is None:
                return
            self._selected = hit
            self.selectionChanged.emit(hit)
            pts = self._selected_points()
            if pts is None:
                return
        radius = self._hit_radius_img(ERASER_RADIUS_PX)
        new_pts, n = erase_points_near(
            pts, ix, iy, radius=radius, min_keep=MIN_VERTICES
        )
        if n:
            self._set_selected_points(new_pts)
            self.statusMessage.emit(
                f"Erased {n} vertex(es); {len(new_pts)} left. Save when done."
            )
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()
        self._hover = pos
        if self.mode == "eraser" and not self._erasing and not self._panning:
            self.update()

        if self._panning and self._pan_last is not None:
            dx = pos.x() - self._pan_last.x()
            dy = pos.y() - self._pan_last.y()
            self._ox += dx
            self._oy += dy
            self._pan_last = pos
            self.update()
            return

        if self.mode == "box" and self._box_origin is not None:
            self._box_current = pos
            self.update()
            return

        if self.mode == "lasso" and self._drawing_lasso:
            ix, iy = self._widget_to_img(pos)
            self._lasso.append((ix, iy))
            self.update()
            return

        if self.mode == "eraser" and self._erasing:
            ix, iy = self._widget_to_img(pos)
            self._erase_at(ix, iy)
            return

        if self._drag_vertex is not None and self.mode == "edit":
            pts = self._selected_points()
            if pts is not None and 0 <= self._drag_vertex < len(pts):
                ix, iy = self._widget_to_img(pos)
                ix = max(0.0, min(float(self._img_w - 1), ix))
                iy = max(0.0, min(float(self._img_h - 1), iy))
                pts = [list(p) for p in pts]
                pts[self._drag_vertex] = [ix, iy]
                self._set_selected_points(pts)
                self.update()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning and event.button() in (
            Qt.MiddleButton,
            Qt.RightButton,
            Qt.LeftButton,
        ):
            self._panning = False
            self._pan_last = None
            self.set_mode(self.mode)  # restore cursor
            return

        if event.button() != Qt.LeftButton:
            return super().mouseReleaseEvent(event)

        if self._drag_vertex is not None:
            self._drag_vertex = None
            self.statusMessage.emit("Moved vertex. Save when done.")
            return

        if self._erasing:
            self._erasing = False
            return

        if self.mode == "box" and self._box_origin is not None:
            p0 = self._widget_to_img(self._box_origin)
            p1 = self._widget_to_img(self._box_current or self._box_origin)
            self._box_origin = None
            self._box_current = None
            self.update()
            x1, y1 = min(p0[0], p1[0]), min(p0[1], p1[1])
            x2, y2 = max(p0[0], p1[0]), max(p0[1], p1[1])
            if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                self.boxSeeded.emit(x1, y1, x2, y2)
            return

        if self.mode == "lasso" and self._drawing_lasso:
            self._drawing_lasso = False
            raw = [[float(x), float(y)] for x, y in self._lasso]
            self._lasso = []
            self.update()
            if len(raw) >= 3:
                lasso = simplify_polyline(raw, max_points=96)
                self._apply_lasso_delete(lasso)
            return

        super().mouseReleaseEvent(event)

    def _apply_lasso_delete(self, lasso: list[list[float]]) -> None:
        pts = self._selected_points()
        if pts is None:
            # Try hit by lasso centroid / first point
            if lasso:
                hit = hit_polygon_index(self._polygons, lasso[0][0], lasso[0][1])
                if hit is not None:
                    self._selected = hit
                    self.selectionChanged.emit(hit)
                    pts = self._selected_points()
        if pts is None:
            self.statusMessage.emit("Select a polygon first, then lasso vertices to remove.")
            return
        new_pts, n = delete_points_inside(pts, lasso, min_keep=MIN_VERTICES)
        if n == 0:
            self.statusMessage.emit(
                "No vertices removed (need ≥3 left, or none were inside lasso)."
            )
            return
        self._set_selected_points(new_pts)
        self.statusMessage.emit(
            f"Lasso removed {n} vertex(es); {len(new_pts)} left. Save when done."
        )
        self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_F:
            self.fit_to_polygons()
            self.statusMessage.emit("Fit view to polygons (+10%).")
            return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.mode == "edit":
            # Delete nearest vertex to… we don't track hover; delete last selected drag — skip
            # Or delete whole poly via dialog button
            pass
        super().keyPressEvent(event)


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
        self._src_pix = QPixmap()

        self.setWindowTitle("Outline viewer")
        self.resize(display_size + 100, display_size + 240)

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
        self.radio_edit = QRadioButton("Edit")
        self.radio_add = QRadioButton("Add point")
        self.radio_lasso = QRadioButton("Lasso (−pts)")
        self.radio_eraser = QRadioButton("Eraser")
        self.radio_point = QRadioButton("SAM +")
        self.radio_box = QRadioButton("SAM box +")
        self.radio_delete = QRadioButton("Del poly")
        self.radio_view = QRadioButton("Pan")
        self.radio_edit.setChecked(True)
        for btn in (
            self.radio_edit,
            self.radio_add,
            self.radio_lasso,
            self.radio_eraser,
            self.radio_point,
            self.radio_box,
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
        self.canvas.polygonsChanged.connect(self._on_canvas_polys_changed)
        self.canvas.statusMessage.connect(self._set_status)
        self.canvas.selectionChanged.connect(self._on_selection_changed)
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

        self.fit_btn = QPushButton("Fit (F)")
        self.fit_btn.setShortcut("F")
        self.fit_btn.clicked.connect(self._fit_view)
        nav.addWidget(self.fit_btn)

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

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_mode_changed(self) -> None:
        tips = {
            self.radio_edit: (
                "edit",
                "Drag vertices · click edge to insert · Alt/right-drag pan · wheel zoom",
            ),
            self.radio_add: (
                "add_point",
                "Click near an edge (or anywhere on selected poly) to insert a vertex.",
            ),
            self.radio_lasso: (
                "lasso",
                "Draw a freehand lasso — vertices INSIDE are deleted (keeps ≥3).",
            ),
            self.radio_eraser: (
                "eraser",
                "Brush-erase vertices under the cursor (keeps ≥3).",
            ),
            self.radio_point: ("point", "Click to ADD a SAM polygon (multi-turtle OK)."),
            self.radio_box: ("box", "Drag a box to ADD a SAM polygon."),
            self.radio_delete: ("delete", "Click a polygon to delete the whole shape."),
            self.radio_view: ("view", "Drag to pan. Wheel zooms around pointer. F = fit."),
        }
        for radio, (mode, tip) in tips.items():
            if radio.isChecked():
                self.canvas.set_mode(mode)
                if self.entries:
                    self.status_label.setText(tip)
                break

    def _fit_view(self) -> None:
        self.canvas.fit_to_polygons()
        self.status_label.setText("Fit view to polygons (+10% pad). Wheel zooms at pointer.")

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

    def _push_scene(self, *, refit: bool) -> None:
        self.canvas.set_scene(
            self._src_pix,
            self._polygons,
            selected_index=self._selected,
            image_width=self._img_w,
            image_height=self._img_h,
            refit=refit,
        )

    def _on_canvas_polys_changed(self) -> None:
        self._dirty = True
        # polygons already mutated in-place on the shared list
        self._update_title()

    def _on_selection_changed(self, index: object) -> None:
        self._selected = index if isinstance(index, int) else None
        self._update_title()

    def _update_title(self) -> None:
        if not self.entries:
            return
        _, display_name = self._current_paths()
        sel = (
            f", selected #{self._selected + 1}"
            if self._selected is not None
            else ""
        )
        dirty = " *" if self._dirty else ""
        self.title_label.setText(
            f"{self.index + 1}/{len(self.entries)}  —  {display_name}  "
            f"({len(self._polygons)} turtle polygon(s){sel}){dirty}"
        )

    def refresh(self) -> None:
        thumb_path, display_name = self._current_paths()
        data = load_labelme_json(labelme_path_for_thumb(thumb_path))
        self._polygons = extract_polygons(data)
        self._selected = len(self._polygons) - 1 if self._polygons else None
        self._dirty = False

        if not thumb_path.exists():
            self._src_pix = QPixmap()
            self._img_w = 1
            self._img_h = 1
            self._push_scene(refit=True)
            self.status_label.setText(f"Missing image: {thumb_path}")
            self._update_title()
            return

        self._src_pix = QPixmap(str(thumb_path))
        self._img_w = self._src_pix.width() or DEFAULT_THUMB_SIZE
        self._img_h = self._src_pix.height() or DEFAULT_THUMB_SIZE
        self._push_scene(refit=True)
        self._on_mode_changed()
        self._update_title()
        self.prev_btn.setEnabled(len(self.entries) > 1)
        self.next_btn.setEnabled(len(self.entries) > 1)
        tip = (
            "Wheel = zoom at pointer · F = fit (+10%) · Alt/right-drag = pan · "
            "Edit = drag vertices"
        )
        self.status_label.setText(tip)

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
            self._push_scene(refit=False)
            self.canvas.set_selected(self._selected)
            self._update_title()
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
            self._push_scene(refit=False)
            self.canvas.set_selected(self._selected)
            self._update_title()
            self.status_label.setText(
                f"Added SAM polygon ({len(self._polygons)} total). Save when done."
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "SAM failed", str(exc))

    def delete_selected(self) -> None:
        if self._selected is None or not self._polygons:
            if not self._polygons:
                return
            self._selected = len(self._polygons) - 1
        self._polygons.pop(self._selected)
        self._selected = (
            min(self._selected, len(self._polygons) - 1) if self._polygons else None
        )
        self._dirty = True
        self._push_scene(refit=False)
        self.canvas.set_selected(self._selected)
        self._update_title()
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
        self._update_title()
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
