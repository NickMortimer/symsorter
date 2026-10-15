"""Outline viewer with interactive SAM point / box prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
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
    upsert_turtle_polygon,
)
from symsorter.outline.filename import parse_thumb_filename
from symsorter.outline.overlay import draw_polygons_on_pixmap

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
    if not polygons:
        return scaled
    return draw_polygons_on_pixmap(
        scaled,
        polygons,
        source_width=pix.width(),
        source_height=pix.height(),
        outline_only=False,
    )


class OutlineCanvas(QLabel):
    """Click for a SAM seed point, or drag for a prompt box (image pixel coords)."""

    pointSeeded = Signal(float, float)  # image XY
    boxSeeded = Signal(float, float, float, float)  # image XYXY

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.mode = "point"  # point | box | view
        self._base_pix: QPixmap | None = None
        self._img_w = DEFAULT_THUMB_SIZE
        self._img_h = DEFAULT_THUMB_SIZE
        self._drag_origin: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._seed_points: list[tuple[float, float]] = []

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._drag_origin = None
        self._drag_current = None
        self._redraw()

    def set_image(
        self,
        pixmap: QPixmap,
        *,
        image_width: int,
        image_height: int,
        clear_seeds: bool = True,
    ) -> None:
        self._base_pix = QPixmap(pixmap)
        self._img_w = max(1, image_width)
        self._img_h = max(1, image_height)
        if clear_seeds:
            self._seed_points = []
        self._drag_origin = None
        self._drag_current = None
        self._redraw()

    def clear_seeds(self) -> None:
        self._seed_points = []
        self._drag_origin = None
        self._drag_current = None
        self._redraw()

    def _content_rect(self) -> QRect:
        """Rect of the pixmap inside the label (centered)."""
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
        rect = QRect(0, 0, canvas.width(), canvas.height())
        sx = canvas.width() / self._img_w
        sy = canvas.height() / self._img_h

        # Seed points
        painter.setPen(QPen(QColor(255, 64, 64), 2))
        painter.setBrush(QColor(255, 64, 64, 180))
        for ix, iy in self._seed_points:
            painter.drawEllipse(QPoint(int(ix * sx), int(iy * sy)), 5, 5)

        # Drag box preview in label space → map into canvas
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
        if self.mode == "box":
            self._drag_origin = event.position().toPoint()
            self._drag_current = self._drag_origin
            self._redraw()
        elif self.mode == "point":
            mapped = self._label_to_image(event.position().toPoint())
            if mapped is None:
                return
            self._seed_points = [mapped]  # single foreground seed for now
            self._redraw()
            self.pointSeeded.emit(mapped[0], mapped[1])
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.mode == "box" and self._drag_origin is not None:
            self._drag_current = event.position().toPoint()
            self._redraw()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if (
            event.button() == Qt.LeftButton
            and self.mode == "box"
            and self._drag_origin is not None
            and self._drag_current is not None
        ):
            p0 = self._label_to_image(self._drag_origin)
            p1 = self._label_to_image(self._drag_current)
            self._drag_origin = None
            self._drag_current = None
            self._redraw()
            if p0 is None or p1 is None:
                return
            x1, y1 = min(p0[0], p1[0]), min(p0[1], p1[1])
            x2, y2 = max(p0[0], p1[0]), max(p0[1], p1[1])
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                self.statusTip()  # no-op; tiny box ignored
                return
            self.boxSeeded.emit(x1, y1, x2, y2)
        else:
            super().mouseReleaseEvent(event)


class OutlineViewerDialog(QDialog):
    """Show thumbs + polygons; refine with SAM point/box prompts and save."""

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
        self._dirty = False
        self._img_w = DEFAULT_THUMB_SIZE
        self._img_h = DEFAULT_THUMB_SIZE

        self.setWindowTitle("Outline viewer")
        self.resize(display_size + 80, display_size + 200)

        layout = QVBoxLayout(self)

        self.title_label = QLabel("")
        self.title_label.setWordWrap(True)
        title_font = QFont()
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("SAM prompt:"))
        self.mode_group = QButtonGroup(self)
        self.radio_point = QRadioButton("Click point")
        self.radio_box = QRadioButton("Draw box")
        self.radio_view = QRadioButton("View only")
        self.radio_point.setChecked(True)
        for btn in (self.radio_point, self.radio_box, self.radio_view):
            self.mode_group.addButton(btn)
            mode_row.addWidget(btn)
        self.radio_point.toggled.connect(self._on_mode_changed)
        self.radio_box.toggled.connect(self._on_mode_changed)
        self.radio_view.toggled.connect(self._on_mode_changed)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        self.canvas = OutlineCanvas()
        self.canvas.setMinimumSize(display_size, display_size)
        self.canvas.pointSeeded.connect(self._on_point_seed)
        self.canvas.boxSeeded.connect(self._on_box_seed)
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

        self.clear_btn = QPushButton("Clear prompt")
        self.clear_btn.clicked.connect(self.canvas.clear_seeds)
        nav.addWidget(self.clear_btn)

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
            tip = "Click the turtle body to seed SAM."
        elif self.radio_box.isChecked():
            self.canvas.set_mode("box")
            tip = "Drag a box around the turtle for SAM."
        else:
            self.canvas.set_mode("view")
            tip = "View only — no prompting."
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
            "Save the current SAM outline before leaving?",
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

    def refresh(self) -> None:
        thumb_path, display_name = self._current_paths()
        json_path = labelme_path_for_thumb(thumb_path)
        data = load_labelme_json(json_path)
        self._polygons = extract_polygons(data)
        self._dirty = False

        self.title_label.setText(
            f"{self.index + 1}/{len(self.entries)}  —  {display_name}"
        )

        if not thumb_path.exists():
            self.canvas.set_image(QPixmap(), image_width=1, image_height=1)
            self.status_label.setText(f"Missing image: {thumb_path}")
            return

        pix = QPixmap(str(thumb_path))
        self._img_w = pix.width() or DEFAULT_THUMB_SIZE
        self._img_h = pix.height() or DEFAULT_THUMB_SIZE
        display = render_thumb_with_polygons(
            thumb_path, self._polygons, display_size=self.display_size
        )
        self.canvas.set_image(
            display, image_width=self._img_w, image_height=self._img_h, clear_seeds=True
        )

        if data is None:
            extra = f"No outline JSON yet ({json_path.name})."
        elif not self._polygons:
            extra = f"{json_path.name} has no turtle polygon (measurement lines ignored)."
        else:
            flags = self._polygons[0].get("flags") or {}
            extra = (
                f"polygon pts={len(self._polygons[0].get('points') or [])} · "
                f"backend={flags.get('backend')} · prompt={flags.get('prompt_kind')}"
            )
        mode_hint = (
            "Click the turtle."
            if self.radio_point.isChecked()
            else "Drag a box around the turtle."
            if self.radio_box.isChecked()
            else "View only."
        )
        self.status_label.setText(f"{mode_hint}  {extra}")

        self.prev_btn.setEnabled(len(self.entries) > 1)
        self.next_btn.setEnabled(len(self.entries) > 1)

    def _apply_sam_draft(self, draft: Any) -> None:
        self._polygons = [
            {
                "label": "turtle",
                "points": draft.points,
                "flags": dict(draft.flags),
                "shape_type": "polygon",
            }
        ]
        self._dirty = True
        thumb_path, _name = self._current_paths()
        display = render_thumb_with_polygons(
            thumb_path, self._polygons, display_size=self.display_size
        )
        self.canvas.set_image(
            display,
            image_width=self._img_w,
            image_height=self._img_h,
            clear_seeds=False,
        )
        self.status_label.setText(
            f"SAM ok ({draft.flags.get('prompt_kind')}, "
            f"{len(draft.points)} pts). Press Save to write JSON."
        )

    def _on_point_seed(self, x: float, y: float) -> None:
        thumb_path, _ = self._current_paths()
        try:
            model = self._ensure_sam()
            draft = sam_polygon(
                thumb_path,
                model=model,
                points=[[x, y]],
                point_labels=[1],
                device=self.device,
                sam_model=self.sam_model_name,
            )
            self._apply_sam_draft(draft)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "SAM failed", str(exc))

    def _on_box_seed(self, x1: float, y1: float, x2: float, y2: float) -> None:
        thumb_path, _ = self._current_paths()
        try:
            model = self._ensure_sam()
            draft = sam_polygon(
                thumb_path,
                model=model,
                bboxes=[x1, y1, x2, y2],
                device=self.device,
                sam_model=self.sam_model_name,
            )
            self._apply_sam_draft(draft)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "SAM failed", str(exc))

    def save_current(self) -> bool:
        if not self.entries or not self._polygons:
            QMessageBox.information(self, "Save", "No polygon to save yet.")
            return False
        thumb_path, _ = self._current_paths()
        poly = self._polygons[0]
        parsed = parse_thumb_filename(thumb_path.name)
        flags = dict(poly.get("flags") or {})
        flags["det_index"] = parsed.det_index
        flags.setdefault("turtle_group_id", None)
        flags["draft"] = True
        flags["backend"] = flags.get("backend") or "sam"

        upsert_turtle_polygon(
            labelme_path_for_thumb(thumb_path),
            image_path=thumb_path.name,
            image_height=self._img_h,
            image_width=self._img_w,
            points=poly["points"],
            flags=flags,
            label="turtle",
        )
        self._dirty = False
        self.status_label.setText(f"Saved {thumb_path.with_suffix('.json').name}")
        # Nudge parent grid to rebuild overlays if present
        parent = self.parent()
        if parent is not None and hasattr(parent, "toggle_outline_overlays"):
            # Force icon cache refresh without flipping the toggle state
            show = getattr(parent, "show_outline_overlays", True)
            if show and hasattr(parent, "_cache"):
                parent._cache.image_cache.clear()
                if hasattr(parent, "reload_images_with_new_size"):
                    parent.reload_images_with_new_size()
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
