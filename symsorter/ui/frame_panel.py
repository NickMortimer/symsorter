"""Frame panel: a thumbnail's outlines drawn on the original image."""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QVBoxLayout, QWidget

from symsorter.outline.export_labelme import (
    extract_polygons,
    is_polygon_locked,
    load_labelme_json,
)
from symsorter.outline.frames import (
    FrameResolver,
    RoiIndex,
    RoiRow,
    crop_window_for_row,
    frame_size,
    frame_to_thumb,
    thumb_to_frame,
)
from symsorter.outline.mapping import CropWindow
from symsorter.ui.frame_settings import FrameSettings, roi_csv_for
from symsorter.ui.outline_dialog import OutlineCanvas

_CACHE_SIZE = 4
_REASON_TEXT = {
    "no_roi_row": "No ROI row for this thumbnail (no_roi_row).",
    "frame_missing": "Original frame was not found (frame_missing).",
    "no_survey_roots": "No survey roots are configured (no_survey_roots).",
}


class _FrameLoadSignals(QObject):
    """Deliver a decoded frame from a worker thread to the UI thread."""

    loaded = Signal(int, str, QImage)


class _FrameLoadRunnable(QRunnable):
    """Decode one frame to a QImage away from the UI thread."""

    def __init__(self, token: int, path: str, signals: _FrameLoadSignals) -> None:
        super().__init__()
        self.token = token
        self._path = path
        self._signals = signals

    def run(self) -> None:
        image = QImage(self._path)
        self._signals.loaded.emit(self.token, self._path, image)


def _frame_copies(
    polygons: list[dict[str, Any]], crop: CropWindow
) -> list[dict[str, Any]]:
    copies: list[dict[str, Any]] = []
    for poly in polygons:
        copied = copy.deepcopy(poly)
        copied["points"] = thumb_to_frame(list(copied.get("points") or []), crop)
        copies.append(copied)
    return copies


class FramePanel(QWidget):
    """Second outline canvas that shows a thumbnail on its original frame."""

    polygonsEdited = Signal()
    selectionChanged = Signal(object)

    def __init__(
        self,
        settings: FrameSettings,
        parent: QWidget | None = None,
        *,
        load_async: bool = True,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._load_async = load_async
        self.reason: str | None = None
        self._crop: CropWindow | None = None
        self._token = 0
        self._roi_cache: dict[str, RoiIndex] = {}
        self._images: OrderedDict[str, QImage] = OrderedDict()
        self._workers: set[_FrameLoadRunnable] = set()
        self._resolver = FrameResolver(settings.survey_roots)
        self._signals = _FrameLoadSignals(self)
        self._signals.loaded.connect(self._on_loaded)

        self.canvas = OutlineCanvas(self)
        self.canvas.polygonsChanged.connect(self.polygonsEdited)
        self.canvas.selectionChanged.connect(self.selectionChanged)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def show_thumb(
        self,
        thumb_path: Path,
        polygons: list[dict[str, Any]],
        selected: int | None,
    ) -> None:
        """Show ``thumb_path`` on its original frame, or a reason if it cannot."""
        self._token += 1
        token = self._token
        thumb_path = Path(thumb_path)
        if not self._settings.survey_roots:
            self._show_reason("no_survey_roots")
            return

        csv_path = roi_csv_for(thumb_path.parent, self._settings)
        if csv_path is None or not csv_path.is_file():
            self._show_reason("no_roi_row")
            return
        try:
            roi = self._roi_index(csv_path)
        except OSError:
            self._show_reason("no_roi_row")
            return
        row = roi.get(thumb_path.name)
        if row is None:
            self._show_reason("no_roi_row")
            return

        frame_path = self._resolver.find(row.flight, row.original_stem)
        if frame_path is None or not frame_path.is_file():
            self._show_reason("frame_missing")
            return
        try:
            frame_wh = frame_size(frame_path)
        except OSError:
            self._show_reason("frame_missing")
            return

        crop = crop_window_for_row(row, frame_wh)
        self._crop = crop
        self.reason = None
        self.canvas.set_message(None)
        self.canvas.set_crop_rect((crop.left, crop.top, crop.right, crop.bottom))
        self.canvas.set_overlays(
            self._neighbour_overlays(roi, row, frame_wh, thumb_path)
        )
        frame_polys = _frame_copies(polygons, crop)
        image = self._cached_image(frame_path)
        if image is None and not self._load_async:
            image = QImage(str(frame_path))
            if image.isNull():
                self._show_reason("frame_missing")
                return
            self._remember(frame_path, image)
        if image is not None:
            self._show_image(image, frame_polys, selected)
            return
        self._show_image(None, frame_polys, selected, fallback_wh=frame_wh)
        self._start_load(frame_path, token)

    def sync_from_thumb(
        self, polygons: list[dict[str, Any]], selected: int | None
    ) -> None:
        """Rebuild frame-space polygon copies and repaint without reloading."""
        if self._crop is None:
            return
        src = self.canvas._src if self.canvas._src is not None else QPixmap()
        self.canvas.set_scene(
            src,
            _frame_copies(polygons, self._crop),
            selected_index=selected,
            image_width=self.canvas._img_w,
            image_height=self.canvas._img_h,
            refit=False,
        )

    def thumb_polygons(self) -> list[dict[str, Any]]:
        """Return canvas polygons converted back to thumbnail coordinates."""
        if self._crop is None:
            return []
        converted: list[dict[str, Any]] = []
        for poly in self.canvas._polygons:
            copied = copy.deepcopy(poly)
            copied["points"] = frame_to_thumb(
                list(copied.get("points") or []), self._crop
            )
            converted.append(copied)
        return converted

    def set_tool_mode(self, mode: str) -> None:
        """Set the canvas tool, mapping SAM point and box modes to view."""
        if mode in {"point", "box"}:
            mode = "view"
        self.canvas.set_mode(mode)

    def _roi_index(self, csv_path: Path) -> RoiIndex:
        key = str(Path(csv_path))
        cached = self._roi_cache.get(key)
        if cached is None:
            cached = RoiIndex.load(Path(csv_path))
            self._roi_cache[key] = cached
        return cached

    def _neighbour_overlays(
        self,
        roi: RoiIndex,
        row: RoiRow,
        frame_wh: tuple[int, int],
        thumb_path: Path,
    ) -> list[list[list[float]]]:
        overlays: list[list[list[float]]] = []
        training_root = thumb_path.parent.parent
        for neighbour in roi.same_frame(row.flight, row.original_stem):
            if neighbour.crop_basename == thumb_path.name:
                continue
            json_path = (
                training_root
                / neighbour.view
                / Path(neighbour.crop_basename).with_suffix(".json")
            )
            window = crop_window_for_row(neighbour, frame_wh)
            for poly in extract_polygons(load_labelme_json(json_path)):
                if not is_polygon_locked(poly):
                    continue
                overlays.append(thumb_to_frame(poly["points"], window))
        return overlays

    def _show_reason(self, reason: str) -> None:
        self.reason = reason
        self._crop = None
        self.canvas.set_overlays([])
        self.canvas.set_crop_rect(None)
        self.canvas.set_message(_REASON_TEXT[reason])
        self.canvas.set_scene(
            QPixmap(),
            [],
            selected_index=None,
            image_width=1,
            image_height=1,
            refit=False,
        )

    def _show_image(
        self,
        image: QImage | None,
        polygons: list[dict[str, Any]],
        selected: int | None,
        *,
        fallback_wh: tuple[int, int] | None = None,
    ) -> None:
        if image is None:
            width, height = fallback_wh or (1, 1)
            pixmap = QPixmap()
        else:
            width, height = image.width(), image.height()
            pixmap = QPixmap.fromImage(image)
        self.canvas.set_scene(
            pixmap,
            polygons,
            selected_index=selected,
            image_width=max(1, width),
            image_height=max(1, height),
            refit=image is not None,
        )

    def _cached_image(self, path: Path) -> QImage | None:
        key = str(Path(path))
        image = self._images.get(key)
        if image is None:
            return None
        self._images.move_to_end(key)
        return image

    def _remember(self, path: Path, image: QImage) -> None:
        key = str(Path(path))
        self._images.pop(key, None)
        self._images[key] = image
        while len(self._images) > _CACHE_SIZE:
            self._images.popitem(last=False)

    def _start_load(self, path: Path, token: int) -> None:
        worker = _FrameLoadRunnable(token, str(path), self._signals)
        self._workers.add(worker)
        QThreadPool.globalInstance().start(worker)

    def _on_loaded(self, token: int, path: str, image: QImage) -> None:
        self._workers = {worker for worker in self._workers if worker.token != token}
        if token != self._token:
            return
        if image.isNull():
            self._show_reason("frame_missing")
            return
        frame_path = Path(path)
        self._remember(frame_path, image)
        self._show_image(image, self.canvas._polygons, self.canvas._selected)
