"""Slice review window: a thin Qt shell over ``symsorter.slices.session.ReviewSession``."""
from __future__ import annotations

import getpass
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QRadioButton, QVBoxLayout,
)

from symsorter.outline.backends import load_sam_model, sam_polygon
from symsorter.slices.geometry import sam_window, to_frame, to_view, view_rect
from symsorter.slices.session import (
    LocalSettings, ReviewSession, SaveRefused, load_local, save_local,
)
from symsorter.ui.outline_dialog import OutlineCanvas

_CACHE_SIZE = 4
_KEYS = {Qt.Key_C: "_complete", Qt.Key_X: "_reject", Qt.Key_Space: "_skip",
         Qt.Key_N: "_next", Qt.Key_P: "_prev", Qt.Key_R: "_recentre", Qt.Key_A: "_adopt",
         Qt.Key_Delete: "delete_selected", Qt.Key_Backspace: "delete_selected"}
_TOOL_KEYS = {Qt.Key_E: "edit", Qt.Key_L: "lasso", Qt.Key_S: "point", Qt.Key_D: "delete",
              Qt.Key_H: "view"}
_TOOLS = [("Edit", "edit"), ("Add point", "add_point"), ("Lasso (-pts)", "lasso"),
          ("Eraser", "eraser"), ("SAM +", "point"), ("Del poly", "delete"), ("Pan", "view")]
HELP = ("C complete · X reject · Space skip · N next · P prev · R re-centre · "
        "A adopt neighbours · E/L/S/D/H tools")


def _load_crop(path: str, rect: tuple) -> QImage:
    """Decode the view rect of a frame with Pillow (runs off the UI thread)."""
    try:
        with Image.open(path) as im:
            crop = im.convert("RGB").crop(rect)
        w, h = crop.size
        return QImage(crop.tobytes(), w, h, w * 3, QImage.Format_RGB888).copy()
    except Exception:  # noqa: BLE001
        return QImage()


class _Signals(QObject):
    loaded = Signal(str, QImage)


class _Load(QRunnable):
    def __init__(self, key: str, path: str, rect: tuple, signals: _Signals) -> None:
        super().__init__()
        self.key, self._path, self._rect, self._signals = key, path, rect, signals

    def run(self) -> None:
        self._signals.loaded.emit(self.key, _load_crop(self._path, self._rect))


class SliceReviewDialog(QDialog):
    def __init__(self, session: ReviewSession, parent=None, *, load_async: bool = True,
                 sam_model: str = "sam_b.pt", device: Optional[str] = None) -> None:
        super().__init__(parent)
        self.session, self._load_async = session, load_async
        self.sam_model_name, self.device, self._sam = sam_model, device, None
        self._polys: list[dict[str, Any]] = []
        self._view: tuple = (0, 0, 1, 1)
        self._want: Optional[str] = None
        self._images: OrderedDict[str, QImage] = OrderedDict()
        self._pending: set[str] = set()
        self._workers: set[_Load] = set()
        self._signals = _Signals(self)
        self._signals.loaded.connect(self._on_loaded)
        self.resize(900, 940)

        layout = QVBoxLayout(self)
        self.title_label = QLabel("")
        bold = QFont()
        bold.setBold(True)
        self.title_label.setFont(bold)
        layout.addWidget(self.title_label)
        row = QHBoxLayout()
        row.addWidget(QLabel("Tool:"))
        self.mode_group = QButtonGroup(self)
        self.radios: dict[str, QRadioButton] = {}
        for text, mode in _TOOLS:
            r = QRadioButton(text)
            r.setFocusPolicy(Qt.NoFocus)  # keep Space/letters for the dialog
            r.toggled.connect(lambda on, m=mode: on and self.canvas.set_mode(m))
            self.mode_group.addButton(r)
            self.radios[mode] = r
            row.addWidget(r)
        row.addStretch()
        layout.addLayout(row)
        self.canvas = OutlineCanvas()
        self.canvas.setMinimumSize(640, 640)
        self.canvas.polygonsChanged.connect(self._on_canvas_changed)
        self.canvas.pointSeeded.connect(self._on_point_seed)
        self.canvas.statusMessage.connect(self._set_status)
        layout.addWidget(self.canvas, stretch=1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%v / %m reviewed")
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel(HELP)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        nav = QHBoxLayout()
        for text, slot in (("Complete (C)", "_complete"), ("Reject (X)", "_reject"),
                           ("Skip (Space)", "_skip"), ("Prev (P)", "_prev"),
                           ("Next (N)", "_next"), ("Re-centre (R)", "_recentre"),
                           ("Adopt (A)", "_adopt")):
            b = QPushButton(text)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(getattr(self, slot))
            nav.addWidget(b)
        layout.addLayout(nav)
        self.radios["edit"].setChecked(True)
        self.canvas.setFocus()

    # -- small helpers ----------------------------------------------------
    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _ask_reason(self) -> Optional[str]:
        text, ok = QInputDialog.getText(self, "Reject slice", "Reason (optional):")
        return text.strip() or "" if ok else None

    def _ask_leave(self) -> str:
        box = QMessageBox(QMessageBox.Question, "Unsaved edits",
                          "This slice has unsaved edits.", parent=self)
        save = box.addButton("Save as complete", QMessageBox.AcceptRole)
        discard = box.addButton("Discard", QMessageBox.DestructiveRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec()
        return {save: "save", discard: "discard"}.get(box.clickedButton(), "cancel")

    def _leave_ok(self) -> bool:
        if not self.session.dirty:
            return True
        choice = self._ask_leave()
        if choice == "discard":
            self.session.discard_changes()
            return True
        if choice == "save":
            try:
                self.session.save("complete")
            except SaveRefused as e:
                self._set_status(str(e))
                return False
            return True
        return False

    # -- display ----------------------------------------------------------
    def show_current(self, refit: bool = True) -> None:
        s, cur = self.session, self.session.current
        lk = s.lookup()
        x0, y0 = s.position()
        self._view = view_rect(x0, y0, cur.size, s.project.margin_px, cur.frame_wh)
        self._update_title()
        self._polys = [{"label": "turtle", "points": p, "flags": {}, "shape_type": "polygon"}
                       for p in (to_view(q, self._view) for q in s.polygons())]
        if lk.problem or lk.path is None:
            self._want = None
            self._push(QPixmap(), 1, 1, False)
            self.canvas.set_message(f"{lk.problem or 'frame missing'}: {cur.frame_stem}")
            self.canvas.set_crop_rect(None)
            self.canvas.set_overlays([])
            return
        v = self._view
        self.canvas.set_crop_rect((x0 - v[0], y0 - v[1], x0 - v[0] + cur.size, y0 - v[1] + cur.size))
        self.canvas.set_overlays([to_view(p, v) for p in s.neighbours()])
        key = self._key(cur.slice_id, v)
        self._want = key
        img = self._images.get(key)
        if img is None and not self._load_async:
            img = _load_crop(str(lk.path), v)
            if not img.isNull():
                self._remember(key, img)
        if img is not None and not img.isNull():
            self._push(QPixmap.fromImage(img), v[2] - v[0], v[3] - v[1], refit)
        else:
            self._push(QPixmap(), v[2] - v[0], v[3] - v[1], False)
            self.canvas.set_message("Loading…" if self._load_async else "frame unreadable")
            if self._load_async:
                self._start(key, str(lk.path), v)
        if self._load_async:
            self._prefetch()

    def _push(self, pix: QPixmap, w: int, h: int, refit: bool) -> None:
        self.canvas.set_message(None)
        self.canvas.set_scene(pix, self._polys, selected_index=None,
                              image_width=w, image_height=h, refit=refit)

    @staticmethod
    def _key(slice_id: str, rect: tuple) -> str:
        return f"{slice_id}|{rect}"

    def _remember(self, key: str, img: QImage) -> None:
        self._images.pop(key, None)
        self._images[key] = img
        while len(self._images) > _CACHE_SIZE:
            self._images.popitem(last=False)

    def _start(self, key: str, path: str, rect: tuple) -> None:
        if key in self._pending:
            return
        self._pending.add(key)
        w = _Load(key, path, rect, self._signals)
        self._workers.add(w)
        QThreadPool.globalInstance().start(w)

    def _prefetch(self) -> None:
        s = self.session
        for i in range(s.index + 1, len(s.project.slices)):
            lk = s.lookup(i)
            if s.status(i) is None and lk.problem is None and lk.path is not None:
                sp = s.project.slices[i]
                rect = view_rect(sp.x0, sp.y0, sp.size, s.project.margin_px, sp.frame_wh)
                key = self._key(sp.slice_id, rect)
                if key not in self._images:
                    self._start(key, str(lk.path), rect)
                return

    def _on_loaded(self, key: str, img: QImage) -> None:
        self._pending.discard(key)
        self._workers = {w for w in self._workers if w.key != key}
        if not img.isNull():
            self._remember(key, img)
        if key != self._want:
            return
        if img.isNull():
            self.canvas.set_message("frame unreadable")
            return
        v = self._view
        self._push(QPixmap.fromImage(img), v[2] - v[0], v[3] - v[1], True)

    def _update_title(self) -> None:
        s = self.session
        p = s.progress()
        done = p["complete"] + p["reject"] + p["skip"]
        self.setWindowTitle(
            f"{s.project.project_id} · slice {s.index + 1}/{p['total']} · "
            f"{s.status() or 'unreviewed'} · complete {p['complete']} · reject {p['reject']} · "
            f"skip {p['skip']} · unreviewed {p['unreviewed']} · missing {p['missing']}")
        self.title_label.setText(self.windowTitle())
        self.progress_bar.setRange(0, max(1, p["total"]))
        self.progress_bar.setValue(done)

    # -- edits ------------------------------------------------------------
    def _on_canvas_changed(self) -> None:
        self.session.set_polygons([to_frame(p["points"], self._view) for p in self._polys])
        self._update_title()

    def delete_selected(self) -> None:
        i = self.canvas._selected
        if i is None or not (0 <= i < len(self._polys)):
            return
        del self._polys[i]
        self.canvas.set_selected(None)
        self._on_canvas_changed()
        self.canvas.update()

    def _ensure_sam(self) -> Any:
        if self._sam is None:
            self._set_status(f"Loading SAM ({self.sam_model_name})…")
            self.repaint()
            self._sam = load_sam_model(self.sam_model_name, device=self.device)
        return self._sam

    def _on_point_seed(self, x: float, y: float) -> None:
        s = self.session
        fx, fy = x + self._view[0], y + self._view[1]
        left, top, right, bottom = sam_window(fx, fy, s.current.frame_wh)
        try:
            with Image.open(s.lookup().path) as im, tempfile.TemporaryDirectory() as tmp:
                tmp_jpg = Path(tmp) / "sam_window.jpg"
                im.convert("RGB").crop((left, top, right, bottom)).save(tmp_jpg, quality=95)
                draft = sam_polygon(tmp_jpg, model=self._ensure_sam(),
                                    points=[[fx - left, fy - top]], point_labels=[1], size=224,
                                    device=self.device, sam_model=self.sam_model_name)
            pts = [[px + left, py + top] for px, py in draft.points]
            s.set_polygons(s.polygons() + [pts])
            self.show_current(refit=False)
            self._set_status(f"Added SAM outline ({len(s.polygons())} total).")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "SAM failed", str(exc))

    # -- actions ----------------------------------------------------------
    def _save(self, status: str, reason: Optional[str] = None) -> None:
        try:
            self.session.save(status, reason)
        except SaveRefused as e:
            self._set_status(str(e))
            return
        moved = self.session.next_unreviewed()
        self.show_current()
        self._set_status(f"Saved {status}." + ("" if moved else " No more unreviewed slices."))

    def _complete(self) -> None:
        self._save("complete")

    def _skip(self) -> None:
        self._save("skip")

    def _reject(self) -> None:
        reason = self._ask_reason()
        if reason is not None:
            self._save("reject", reason or None)

    def _move(self, fn, none_msg: str) -> None:
        if not self._leave_ok():
            return
        if fn():
            self.show_current()
            self._set_status("")
        else:
            self.show_current(refit=False)
            self._set_status(none_msg)

    def _next(self) -> None:
        self._move(self.session.next_unreviewed, "No more unreviewed slices.")

    def _prev(self) -> None:
        self._move(self.session.prev, "No earlier slice.")

    def _recentre(self) -> None:
        if self.session.recentre_current():
            self.show_current()
            self._set_status("Re-centred. Check the slice, then press C.")
        else:
            self._set_status("No 624 slice can hold every outline. Reject this slice.")

    def _adopt(self) -> None:
        n = self.session.adopt_neighbours()
        self.show_current(refit=False)
        self._set_status(f"Adopted {n} neighbour outline(s).")

    # -- keys / closing ---------------------------------------------------
    def keyPressEvent(self, event) -> None:  # noqa: N802
        k = event.key()
        if not event.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
            if k in _KEYS:
                getattr(self, _KEYS[k])()
                return
            if k in _TOOL_KEYS:
                self.radios[_TOOL_KEYS[k]].setChecked(True)
                return
        super().keyPressEvent(event)

    def reject(self) -> None:
        if self._leave_ok():
            super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._leave_ok():
            event.accept()
        else:
            event.ignore()


def _ask_settings(parent, roots: list) -> Optional[tuple]:
    """Ask for survey roots and a reviewer name; None if cancelled."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("Slice project settings")
    edit = QPlainTextEdit("\n".join(str(r) for r in roots))
    add = QPushButton("Add folder…")

    def _add() -> None:
        d = QFileDialog.getExistingDirectory(dlg, "Add survey root")
        if d:
            edit.setPlainText((edit.toPlainText().rstrip() + "\n" + d).strip())

    add.clicked.connect(_add)
    reviewer = QLineEdit(getpass.getuser())
    box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    box.accepted.connect(dlg.accept)
    box.rejected.connect(dlg.reject)
    lay = QVBoxLayout(dlg)
    for w in (QLabel("Survey roots (one per line)"), edit, add, QLabel("Reviewer"), reviewer, box):
        lay.addWidget(w)
    if dlg.exec() != QDialog.Accepted:
        return None
    lines = [ln.strip() for ln in edit.toPlainText().splitlines() if ln.strip()]
    return [Path(ln) for ln in lines], reviewer.text().strip()


def open_slice_project_dialog(parent, project_dir: Optional[Path] = None) -> Optional[SliceReviewDialog]:
    if project_dir is None:
        chosen = QFileDialog.getExistingDirectory(parent, "Open slice project")
        if not chosen:
            return None
        project_dir = Path(chosen)
    project_dir = Path(project_dir)
    try:
        local = load_local(project_dir)
        if local is None:
            from symsorter.ui.frame_settings import load_frame_settings
            got = _ask_settings(parent, load_frame_settings().survey_roots)
            if got is None:
                return None
            local = LocalSettings(got[0], got[1])
            save_local(project_dir, local)
        dlg = SliceReviewDialog(ReviewSession(project_dir, local), parent)
    except Exception as exc:  # noqa: BLE001
        QMessageBox.warning(parent, "Cannot open slice project", str(exc))
        return None
    dlg.show_current()
    dlg.show()
    return dlg
