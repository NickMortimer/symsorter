"""Interactive SAM point/box prompts directly on the SymSorter grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QMessageBox, QRubberBand

from symsorter.outline.backends import load_sam_model, sam_polygon
from symsorter.outline.export_labelme import upsert_turtle_polygon
from symsorter.outline.filename import parse_thumb_filename


class GridOutlinePromptFilter(QObject):
    """Event filter for QListView viewport: click seed or drag box → SAM → save."""

    def __init__(self, browser: Any):
        super().__init__(browser)
        self.browser = browser
        self.mode = "off"  # off | point | box
        self._press_index = None
        self._press_pos: QPoint | None = None
        self._rubber: QRubberBand | None = None
        self._sam = None

    def set_mode(self, mode: str) -> None:
        mode = (mode or "off").lower()
        if mode not in {"off", "point", "box"}:
            mode = "off"
        self.mode = mode
        view = getattr(self.browser, "view", None)
        if view is not None:
            if mode == "off":
                view.viewport().setCursor(Qt.ArrowCursor)
            elif mode == "point":
                view.viewport().setCursor(Qt.CrossCursor)
            else:
                view.viewport().setCursor(Qt.CrossCursor)
        self._cancel_drag()
        sb = getattr(self.browser, "statusBar", None)
        if callable(sb) and self.browser.statusBar():
            tips = {
                "off": "Outline prompt off",
                "point": "Grid outline: click a turtle to seed SAM",
                "box": "Grid outline: drag a box around a turtle for SAM",
            }
            self.browser.statusBar().showMessage(tips.get(mode, ""), 5000)

    def ensure_installed(self) -> None:
        view = getattr(self.browser, "view", None)
        if view is None:
            return
        viewport = view.viewport()
        viewport.installEventFilter(self)

    def _cancel_drag(self) -> None:
        self._press_index = None
        self._press_pos = None
        if self._rubber is not None:
            self._rubber.hide()

    def _icon_rect_for_index(self, index) -> QRect:
        view = self.browser.view
        vr = view.visualRect(index)
        icon_size = int(view.iconSize().width() or getattr(self.browser, "icon_size", 128))
        # IconMode: decoration centered horizontally at top of item rect.
        x = vr.x() + max(0, (vr.width() - icon_size) // 2)
        y = vr.y() + 2
        return QRect(x, y, icon_size, icon_size)

    def _thumb_path_for_index(self, index) -> Path | None:
        fn = index.data(Qt.UserRole)
        if not fn:
            return None
        folder = getattr(self.browser, "folder", "") or ""
        if not folder:
            return None
        return Path(folder) / fn

    def _map_viewport_to_image(
        self, index, viewport_pos: QPoint, thumb_path: Path
    ) -> tuple[float, float] | None:
        icon_rect = self._icon_rect_for_index(index)
        if not icon_rect.contains(viewport_pos):
            return None
        # Resolve native image size (thumbs are usually 224×224).
        from PySide6.QtGui import QPixmap

        pix = QPixmap(str(thumb_path))
        img_w = pix.width() or 224
        img_h = pix.height() or 224
        lx = viewport_pos.x() - icon_rect.x()
        ly = viewport_pos.y() - icon_rect.y()
        ix = max(0.0, min(float(img_w - 1), lx / max(1, icon_rect.width()) * img_w))
        iy = max(0.0, min(float(img_h - 1), ly / max(1, icon_rect.height()) * img_h))
        return ix, iy

    def _ensure_sam(self):
        if self._sam is None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._sam = load_sam_model(
                    getattr(self.browser, "outline_sam_model", "sam_b.pt"),
                    device=getattr(self.browser, "outline_sam_device", None),
                )
            finally:
                QApplication.restoreOverrideCursor()
        return self._sam

    def _run_sam_and_save(
        self,
        thumb_path: Path,
        *,
        points=None,
        bboxes=None,
    ) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            model = self._ensure_sam()
            draft = sam_polygon(
                thumb_path,
                model=model,
                points=points,
                bboxes=bboxes,
                point_labels=[1] * len(points) if points else None,
                sam_model=getattr(self.browser, "outline_sam_model", "sam_b.pt"),
                device=getattr(self.browser, "outline_sam_device", None),
            )
            parsed = parse_thumb_filename(thumb_path.name)
            flags = dict(draft.flags)
            flags["det_index"] = parsed.det_index
            flags.setdefault("turtle_group_id", None)
            flags["draft"] = True
            flags["backend"] = "sam"

            from PySide6.QtGui import QPixmap

            pix = QPixmap(str(thumb_path))
            upsert_turtle_polygon(
                thumb_path.with_suffix(".json"),
                image_path=thumb_path.name,
                image_height=pix.height() or 224,
                image_width=pix.width() or 224,
                points=draft.points,
                flags=flags,
                label="turtle",
            )
        except Exception as exc:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self.browser, "SAM outline failed", str(exc))
            return
        QApplication.restoreOverrideCursor()

        if hasattr(self.browser, "refresh_outline_icon"):
            self.browser.refresh_outline_icon(thumb_path.name)
        tip = f"Saved outline for {thumb_path.name} ({len(draft.points)} pts)"
        if hasattr(self.browser, "statusBar") and self.browser.statusBar():
            self.browser.statusBar().showMessage(tip, 4000)

    def eventFilter(self, obj, event):  # noqa: N802
        if self.mode == "off":
            return False
        view = getattr(self.browser, "view", None)
        if view is None or obj is not view.viewport():
            return False

        et = event.type()
        if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            index = view.indexAt(pos)
            if not index.isValid():
                return False
            icon_rect = self._icon_rect_for_index(index)
            if not icon_rect.contains(pos):
                return False
            thumb = self._thumb_path_for_index(index)
            if thumb is None or not thumb.exists():
                return False

            # Select the item so the user sees which thumb is being edited.
            view.setCurrentIndex(index)

            if self.mode == "point":
                mapped = self._map_viewport_to_image(index, pos, thumb)
                if mapped is None:
                    return True
                self._run_sam_and_save(thumb, points=[[mapped[0], mapped[1]]])
                return True

            # box mode — start rubber band in viewport coords
            self._press_index = index
            self._press_pos = pos
            if self._rubber is None:
                self._rubber = QRubberBand(QRubberBand.Rectangle, view.viewport())
            self._rubber.setGeometry(QRect(pos, pos))
            self._rubber.show()
            return True

        if et == QEvent.MouseMove and self._press_pos is not None and self.mode == "box":
            if self._rubber is not None:
                self._rubber.setGeometry(QRect(self._press_pos, event.position().toPoint()).normalized())
            return True

        if et == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self.mode == "box":
            if self._press_index is None or self._press_pos is None:
                return False
            release = event.position().toPoint()
            index = self._press_index
            thumb = self._thumb_path_for_index(index)
            origin = self._press_pos
            self._cancel_drag()
            if thumb is None or not thumb.exists():
                return True
            p0 = self._map_viewport_to_image(index, origin, thumb)
            p1 = self._map_viewport_to_image(index, release, thumb)
            if p0 is None or p1 is None:
                # Allow release slightly outside icon — clamp via icon rect corners
                icon_rect = self._icon_rect_for_index(index)
                clamped = QPoint(
                    min(max(release.x(), icon_rect.left()), icon_rect.right()),
                    min(max(release.y(), icon_rect.top()), icon_rect.bottom()),
                )
                p1 = self._map_viewport_to_image(index, clamped, thumb)
                if p0 is None or p1 is None:
                    return True
            x1, y1 = min(p0[0], p1[0]), min(p0[1], p1[1])
            x2, y2 = max(p0[0], p1[0]), max(p0[1], p1[1])
            if (x2 - x1) < 3 or (y2 - y1) < 3:
                return True
            self._run_sam_and_save(thumb, bboxes=[x1, y1, x2, y2])
            return True

        return False
