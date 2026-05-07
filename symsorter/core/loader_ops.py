from PySide6.QtWidgets import QMessageBox, QProgressDialog
from PySide6.QtCore import Qt, QTimer, QProcess
from pathlib import Path
import re
import sys
import shutil
import json
from ..clip_encode import load_existing_embeddings
from ..io.classifications import build_coco_data as io_build_coco
import os
import math
from PySide6.QtCore import QTimer, QSize, Qt
from PySide6.QtWidgets import QListView
from .image_worker import ImageLoadWorker


FILLER_ROLE = Qt.UserRole + 42


class LoaderOpsMixin:
    def calculate_visible_range(self):
        if not self.image_files:
            return 0, 0
        vw = self.view.viewport().width() or 800
        vh = self.view.viewport().height() or 600
        gw = self.view.gridSize().width() or (self.icon_size + 10)
        gh = self.view.gridSize().height() or (self.icon_size + 10)
        cols = max(1, vw // gw)
        rows_visible = max(1, vh // gh) + 4
        scroll = self.view.verticalScrollBar()
        ratio = (scroll.value() / scroll.maximum()) if scroll.maximum() > 0 else 0
        total_rows = math.ceil(len(self.image_files) / cols)
        start_row = max(0, int(ratio * (total_rows - rows_visible)))
        start_idx = start_row * cols
        end_idx = min(start_idx + (rows_visible * cols), len(self.image_files))
        return start_idx, end_idx

    def load_image_batch(self, start_idx, end_idx):
        print(f"load_image_batch called with range {start_idx} to {end_idx}")
        to_load = [i for i in range(start_idx, end_idx)
                   if i not in self.loaded_images and i < len(self.image_files)]
        if not to_load:
            print(f"No new images to load in range {start_idx}-{end_idx}")
            return
        print(f"Loading {len(to_load)} new images: indices {to_load}")
        for i in to_load:
            self.loaded_images.add(i)
        for i in to_load:
            file_path = os.path.join(self.folder, self.image_files[i])
            worker = ImageLoadWorker(i, file_path, self.icon_size, self.crop_size, cache_manager=self)
            worker.signals.imageLoaded.connect(self.update_image)
            worker.signals.error.connect(self.handle_worker_error)
            self.active_workers.append(worker)
            self.thread_pool.start(worker)

    def handle_worker_error(self, error_msg):
        print(error_msg)

    def load_visible_images(self):
        s, e = self.calculate_visible_range()
        if s < e:
            self.load_image_batch(s, e)
        self.start_background_loading()

    def start_background_loading(self):
        if not self.background_load_timer.isActive():
            self.background_load_timer.start(250)

    def stop_background_loading(self):
        self.background_load_timer.stop()

    def load_next_batch_background(self):
        if not self.image_files:
            self.stop_background_loading()
            return
        batch = 50
        start_idx = None
        for i in range(0, len(self.image_files), batch):
            if any(j not in self.loaded_images for j in range(i, min(i + batch, len(self.image_files)))):
                start_idx = i
                break
        if start_idx is not None:
            self.load_image_batch(start_idx, min(start_idx + batch, len(self.image_files)))
        else:
            self.stop_background_loading()
            print("Background loading complete - all images loaded")

    def update_image(self, index, icon):
        if index >= self.model.rowCount():
            return
        item = self.model.item(index)
        if not item:
            return
        item.setIcon(icon)
        filename = item.data(self.Qt.UserRole) if hasattr(self, 'Qt') else item.data(0x0100)  # safe fallback
        # Use standard path if Qt.UserRole is available in current scope
        filename = item.data(Qt.UserRole)
        if filename:
            import os
            display_name = os.path.basename(filename)
            if filename in self.image_categories:
                info = self.image_categories[filename]
                tip = f"File: {display_name}\nClass: {info['class_name']} (ID: {info['class_id']})"
            else:
                tip = f"File: {display_name}\nClass: Unallocated"
            item.setToolTip(tip)

    def reload_images_with_new_size(self):
        """Reload all images after a thumbnail-size change (clears old-size cache)."""
        if not self.model.rowCount():
            return
        self.stop_background_loading()
        # Purge processed-thumbnail cache (wrong size) but keep raw image cache
        if hasattr(self, "_cache"):
            self._cache.image_cache.clear()
            self._cache.cache_access_order = [
                k for k in self._cache.cache_access_order
                if not isinstance(k, tuple)
            ]
        elif hasattr(self, "image_cache"):
            self.image_cache.clear()
        # Reset all item icons to keep placeholder in sync with new grid size
        ph = getattr(self, "placeholder_icon", None)
        if ph:
            for row in range(self.model.rowCount()):
                item = self.model.item(row)
                if item:
                    item.setIcon(ph)
        # Clear loaded-index set so images get re-dispatched at new size
        self.loaded_images.clear()
        # Stop stale workers
        for w in self.active_workers:
            w.stop()
        self.active_workers.clear()
        self.thread_pool.clear()
        # Restart loading
        self.load_visible_images()

    def reload_images_with_new_crop(self):
        """Reload all images after a crop change (clears processed-thumbnail cache)."""
        if not self.model.rowCount():
            return
        self.stop_background_loading()
        # Purge processed-thumbnail cache (wrong crop) but keep raw image cache
        if hasattr(self, "_cache"):
            self._cache.image_cache.clear()
            self._cache.cache_access_order = [
                k for k in self._cache.cache_access_order
                if not isinstance(k, tuple)
            ]
        elif hasattr(self, "image_cache"):
            self.image_cache.clear()
        # Reset icons so new crop is shown
        ph = getattr(self, "placeholder_icon", None)
        if ph:
            for row in range(self.model.rowCount()):
                item = self.model.item(row)
                if item:
                    item.setIcon(ph)
        self.loaded_images.clear()
        for w in self.active_workers:
            w.stop()
        self.active_workers.clear()
        self.thread_pool.clear()
        self.load_visible_images()

    # Alias used by crop handlers in view_controls
    def reload_visible_images(self):
        self.reload_images_with_new_crop()

    def on_scroll(self):
        self.load_visible_images()

    def on_view_resize(self, event):
        QListView.resizeEvent(self.view, event)
        if hasattr(self, '_resize_timer'):
            self._resize_timer.stop()
        self._resize_timer = QTimer()
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self.load_visible_images)
        self._resize_timer.start(100)

    def _choose_npz_in_folder(self, folder: Path) -> Path | None:
        npzs = sorted(folder.glob("*.npz"))
        if not npzs:
            return None
        pat = re.compile(r'[_\-](\d+)\.npz$', re.IGNORECASE)
        numbered = []
        for p in npzs:
            m = pat.search(p.name)
            if m:
                numbered.append((int(m.group(1)), p))
        preferred_224 = [p for (n, p) in numbered if n == 224]
        if preferred_224:
            return max(preferred_224, key=lambda x: x.stat().st_mtime)
        if numbered:
            max_n = max(n for n, _ in numbered)
            best = [p for (n, p) in numbered if n == max_n]
            return max(best, key=lambda x: x.stat().st_mtime)
        for name in ("embeddings.npz", "symsorter_embeddings.npz"):
            cand = folder / name
            if cand.exists():
                return cand
        return max(npzs, key=lambda p: p.stat().st_mtime)

    def _encode_folder_and_then_load(self, folder: Path):
        dlg = QProgressDialog("Encoding embeddings...", "Cancel", 0, 0, self)
        dlg.setWindowTitle("Encoding")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(True)
        dlg.setAutoReset(True)
        dlg.show()

        proc = QProcess(self)
        self._encode_process = proc
        self._encode_progress = dlg
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self._on_encode_stdout(proc, dlg))

        def on_finished(exitCode, _status):
            dlg.close()
            dlg.deleteLater()
            self._encode_process = None
            self._encode_progress = None
            if exitCode == 0:
                npz = self._choose_npz_in_folder(folder)
                if npz and npz.exists():
                    self.load_folder_from_path(str(npz))
                else:
                    QMessageBox.warning(self, "Encode finished", "Encoding finished, but no NPZ was found.")
            else:
                QMessageBox.critical(self, "Encode failed", f"Encoding failed with exit code {exitCode}.")

        proc.finished.connect(on_finished)

        def on_cancel():
            if proc.state() != QProcess.NotRunning:
                proc.terminate()
                QTimer.singleShot(2000, lambda: proc.kill() if proc.state() != QProcess.NotRunning else None)

        dlg.canceled.connect(on_cancel)

        # Find the symsorter CLI command (Poetry entry point)
        sibling_cli = shutil.which("symsorter")
        if not sibling_cli:
            # Fallback to current Python interpreter if symsorter not found
            sibling_cli = sys.executable

        proc.start(str(sibling_cli), ["encode", str(folder)])

        if not proc.waitForStarted(3000):
            dlg.reset()
            self._encode_process = None
            self._encode_progress = None
            QMessageBox.critical(self, "Encode failed", "Could not start encoder process.")

    def _on_encode_stdout(self, proc: QProcess, dlg: QProgressDialog):
        try:
            data = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        except Exception:
            return
        if not data:
            return
        lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
        if lines:
            dlg.setLabelText(f"Encoding… {lines[-1]}")

    def load_folder_from_path(self, path_str):
        """Moved from ImageBrowser. No behavior change."""
        p = Path(path_str)
        if p.is_dir():
            npz = self._choose_npz_in_folder(p)
            if not npz:
                reply = QMessageBox.question(
                    self,
                    "Encode embeddings?",
                    f"No embeddings (.npz) found in:\n{p}\n\nEncode this folder now?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply != QMessageBox.Yes:
                    return
                return self._encode_folder_and_then_load(p)
            npz_path = npz
        else:
            npz_path = p

        if not npz_path.exists():
            print(f"NPZ file not found: {npz_path}")
            return

        self.folder = str(npz_path.parent)
        self.npz_file_path = npz_path

        print(f"Loading embeddings from {npz_path}")
        self.embeddings, self.cached_dimensions, self.embedding_metadata = load_existing_embeddings(npz_path)

        # reset state
        self.hidden_flags = {}
        self.image_categories = {}

        # load classifications.json if present (expects self.load_classifications_json in Lifecycle/IO)
        json_loaded = False
        if hasattr(self, "load_classifications_json"):
            json_loaded = self.load_classifications_json()
        if not json_loaded and hasattr(self, "check_and_offer_backup_restore"):
            self.check_and_offer_backup_restore()
            print("No classifications JSON file found - starting with empty classifications")

        self.has_unsaved_changes = False
        if hasattr(self, "update_window_title"):
            self.update_window_title()
        if hasattr(self, "start_auto_backup"):
            self.start_auto_backup()

        if not self.embeddings:
            print("No embeddings loaded!")
            return

        # stop workers / clear pools if present
        if hasattr(self, "model"):
            self.model.clear()
        if hasattr(self, "loaded_images"):
            self.loaded_images.clear()
        for worker in getattr(self, "active_workers", []):
            try:
                worker.stop()
            except Exception:
                pass
        if hasattr(self, "active_workers"):
            self.active_workers.clear()
        if hasattr(self, "thread_pool"):
            try:
                self.thread_pool.clear()
            except Exception:
                pass

        selected_filter = self.class_filter_combo.currentText() if hasattr(self, "class_filter_combo") else "All Images"
        folder_path = Path(self.folder)

        if selected_filter == "All Images":
            self.image_files = [
                fn for fn in self.embeddings.keys()
                if (folder_path / fn).exists() and not self.hidden_flags.get(fn, False)
            ]
        elif selected_filter == "Unallocated":
            self.image_files = [
                fn for fn in self.embeddings.keys()
                if (folder_path / fn).exists() and not self.hidden_flags.get(fn, False)
                and fn not in self.image_categories
            ]
        else:
            self.image_files = [
                fn for fn in self.embeddings.keys()
                if (folder_path / fn).exists()
                and not self.hidden_flags.get(fn, False)
                and fn in self.image_categories
                and self.image_categories[fn]['class_name'] == selected_filter
            ]

        self.original_order = list(self.image_files)
        print(f"Found {len(self.image_files)} images with embeddings (filter: {selected_filter})")

        # Ensure the filter combo reflects loaded JSON/classes (do it after UI is ready)
        if hasattr(self, "schedule_filter_combo_refresh"):
            self.schedule_filter_combo_refresh()

        # build placeholder items
        for fn in self.image_files:
            from PySide6.QtGui import QStandardItem
            item = QStandardItem()
            item.setIcon(self.placeholder_icon)
            item.setEditable(False)
            item.setData(fn, Qt.UserRole)
            if hasattr(self, "update_item_with_classification"):
                self.update_item_with_classification(item, fn)
            self.model.appendRow(item)

        print(f"Added {self.model.rowCount()} items to model")

        if hasattr(self, "update_grid_layout"):
            self.update_grid_layout()

        if self.image_files and hasattr(self, "load_image_batch"):
            initial = min(100, len(self.image_files))
            print(f"Loading initial batch of {initial} images")
            self.load_image_batch(0, initial)
            if hasattr(self, "update_model_with_categories"):
                self.update_model_with_categories()
            QTimer.singleShot(100, getattr(self, "load_visible_images", lambda: None))

    # Optional: keep COCO builder here if you want it out of ImageBrowser
    def build_coco_data(self):
        return io_build_coco(
            self.folder,
            self.embeddings,
            self.hidden_flags,
            self.classes,
            self.class_keystrokes,
            self.class_descriptions,
            self.image_categories,
            getattr(self, 'embedding_metadata', {}),
            getattr(self, 'cached_dimensions', {}),
            getattr(self, 'class_ids', {}),
        )