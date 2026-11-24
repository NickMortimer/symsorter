"""Interactive SAM point/box prompts + delete directly on the SymSorter grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QRubberBand

from symsorter.outline.backends import load_sam_model, sam_polygon
from symsorter.outline.edit_geom import map_icon_local_to_image
from symsorter.outline.export_labelme import (
    add_turtle_polygon,
    delete_all_turtle_polygons,
    delete_turtle_polygon_at,
    extract_polygons,
    find_turtle_polygon_index_at,
    is_polygon_locked,
    labelme_path_for_thumb,
    load_labelme_json,
)
from symsorter.outline.filename import parse_thumb_filename


class GridOutlinePromptFilter(QObject):
    """Event filter for QListView viewport: click/box SAM or delete a polygon."""

    def __init__(self, browser: Any):
        super().__init__(browser)
        self.browser = browser
        self.mode = "off"  # off | point | box | delete
        self._press_index = None
        self._press_pos: QPoint | None = None
        self._rubber: QRubberBand | None = None
        self._sam = None

    def set_mode(self, mode: str) -> None:
        mode = (mode or "off").lower()
        if mode not in {"off", "point", "box", "delete"}:
            mode = "off"
        self.mode = mode
        view = getattr(self.browser, "view", None)
        if view is not None:
            view.viewport().setCursor(
                Qt.ArrowCursor if mode == "off" else Qt.CrossCursor
            )
        self._cancel_drag()
        tips = {
            "off": "Outline prompt off",
            "point": "Grid: click turtle → add SAM polygon (multi-OK)",
            "box": "Grid: drag box → add SAM polygon (multi-OK)",
            "delete": "Grid: click a polygon to delete it (Shift+Del = clear all on thumb)",
        }
        if hasattr(self.browser, "statusBar") and self.browser.statusBar():
            self.browser.statusBar().showMessage(tips.get(mode, ""), 6000)

    def ensure_installed(self) -> None:
        view = getattr(self.browser, "view", None)
        if view is None:
            return
        view.viewport().installEventFilter(self)
        # Also watch the view for Delete key when an item is selected
        view.installEventFilter(self)

    def _cancel_drag(self) -> None:
        self._press_index = None
        self._press_pos = None
        if self._rubber is not None:
            self._rubber.hide()

    def _icon_rect_for_index(self, index) -> QRect:
        view = self.browser.view
        vr = view.visualRect(index)
        icon_size = int(view.iconSize().width() or getattr(self.browser, "icon_size", 128))
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
        """Map grid click → full-thumb image pixels (respects center-crop zoom)."""
        icon_rect = self._icon_rect_for_index(index)
        if not icon_rect.contains(viewport_pos):
            return None
        from PySide6.QtGui import QPixmap

        pix = QPixmap(str(thumb_path))
        img_w = pix.width() or 224
        img_h = pix.height() or 224
        lx = float(viewport_pos.x() - icon_rect.x())
        ly = float(viewport_pos.y() - icon_rect.y())
        crop_size = getattr(self.browser, "crop_size", None)
        return map_icon_local_to_image(
            lx,
            ly,
            icon_w=float(icon_rect.width()),
            icon_h=float(icon_rect.height()),
            img_w=float(img_w),
            img_h=float(img_h),
            crop_size=crop_size,
        )

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

    def _run_sam_and_add(
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
            add_turtle_polygon(
                thumb_path.with_suffix(".json"),
                image_path=thumb_path.name,
                image_height=pix.height() or 224,
                image_width=pix.width() or 224,
                points=draft.points,
                flags=flags,
                label="turtle",
            )
            n = len(
                extract_polygons(load_labelme_json(labelme_path_for_thumb(thumb_path)))
            )
        except Exception as exc:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self.browser, "SAM outline failed", str(exc))
            return
        QApplication.restoreOverrideCursor()

        if hasattr(self.browser, "refresh_outline_icon"):
            self.browser.refresh_outline_icon(thumb_path.name)
        tip = f"Added polygon ({n} total) on {thumb_path.name}"
        if hasattr(self.browser, "statusBar") and self.browser.statusBar():
            self.browser.statusBar().showMessage(tip, 4000)

    def _delete_at_click(self, thumb_path: Path, x: float, y: float) -> None:
        json_path = labelme_path_for_thumb(thumb_path)
        idx = find_turtle_polygon_index_at(json_path, x, y)
        if idx is None:
            if hasattr(self.browser, "statusBar") and self.browser.statusBar():
                self.browser.statusBar().showMessage("No turtle polygon to delete", 2500)
            return
        polys = extract_polygons(load_labelme_json(json_path))
        if 0 <= idx < len(polys) and is_polygon_locked(polys[idx]):
            if hasattr(self.browser, "statusBar") and self.browser.statusBar():
                self.browser.statusBar().showMessage(
                    "Polygon is quality-locked — unlock in Outline viewer (K) first",
                    4000,
                )
            return
        if delete_turtle_polygon_at(json_path, idx):
            if hasattr(self.browser, "refresh_outline_icon"):
                self.browser.refresh_outline_icon(thumb_path.name)
            if hasattr(self.browser, "statusBar") and self.browser.statusBar():
                self.browser.statusBar().showMessage(
                    f"Deleted polygon #{idx + 1} on {thumb_path.name}", 3000
                )

    def delete_last_on_selected(self) -> None:
        """Delete the last turtle polygon on the current grid selection."""
        view = getattr(self.browser, "view", None)
        if view is None:
            return
        index = view.currentIndex()
        if not index.isValid():
            return
        thumb = self._thumb_path_for_index(index)
        if thumb is None:
            return
        polys = extract_polygons(load_labelme_json(labelme_path_for_thumb(thumb)))
        if not polys:
            return
        if delete_turtle_polygon_at(labelme_path_for_thumb(thumb), len(polys) - 1):
            if hasattr(self.browser, "refresh_outline_icon"):
                self.browser.refresh_outline_icon(thumb.name)
            if hasattr(self.browser, "statusBar") and self.browser.statusBar():
                self.browser.statusBar().showMessage(
                    f"Deleted last polygon on {thumb.name}", 3000
                )

    def clear_all_on_selected(self) -> None:
        view = getattr(self.browser, "view", None)
        if view is None:
            return
        index = view.currentIndex()
        if not index.isValid():
            return
        thumb = self._thumb_path_for_index(index)
        if thumb is None:
            return
        n = delete_all_turtle_polygons(labelme_path_for_thumb(thumb))
        if n and hasattr(self.browser, "refresh_outline_icon"):
            self.browser.refresh_outline_icon(thumb.name)
        if hasattr(self.browser, "statusBar") and self.browser.statusBar():
            self.browser.statusBar().showMessage(
                f"Cleared {n} turtle polygon(s) on {thumb.name}", 3000
            )

    def eventFilter(self, obj, event):  # noqa: N802
        view = getattr(self.browser, "view", None)
        if view is None:
            return False

        # Keyboard delete on the view (selected thumb) — only when outlines exist
        if obj is view and event.type() == QEvent.KeyPress:
            key = event.key()
            mods = event.modifiers()
            if key in (Qt.Key_Delete, Qt.Key_Backspace):
                index = view.currentIndex()
                thumb = self._thumb_path_for_index(index) if index.isValid() else None
                has_poly = False
                if thumb is not None:
                    has_poly = bool(
                        extract_polygons(
                            load_labelme_json(labelme_path_for_thumb(thumb))
                        )
                    )
                if has_poly:
                    if mods & Qt.ShiftModifier:
                        self.clear_all_on_selected()
                    else:
                        self.delete_last_on_selected()
                    return True

        if self.mode == "off":
            return False
        if obj is not view.viewport():
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

            view.setCurrentIndex(index)

            if self.mode == "delete":
                mapped = self._map_viewport_to_image(index, pos, thumb)
                if mapped is None:
                    return True
                self._delete_at_click(thumb, mapped[0], mapped[1])
                return True

            if self.mode == "point":
                mapped = self._map_viewport_to_image(index, pos, thumb)
                if mapped is None:
                    return True
                self._run_sam_and_add(thumb, points=[[mapped[0], mapped[1]]])
                return True

            # box mode
            self._press_index = index
            self._press_pos = pos
            if self._rubber is None:
                self._rubber = QRubberBand(QRubberBand.Rectangle, view.viewport())
            self._rubber.setGeometry(QRect(pos, pos))
            self._rubber.show()
            return True

        if et == QEvent.MouseMove and self._press_pos is not None and self.mode == "box":
            if self._rubber is not None:
                self._rubber.setGeometry(
                    QRect(self._press_pos, event.position().toPoint()).normalized()
                )
            return True

        if (
            et == QEvent.MouseButtonRelease
            and event.button() == Qt.LeftButton
            and self.mode == "box"
        ):
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
            self._run_sam_and_add(thumb, bboxes=[x1, y1, x2, y2])
            return True

        return False
