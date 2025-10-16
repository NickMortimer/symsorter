import os
import math
from PySide6.QtCore import QTimer, QSize, Qt
from PySide6.QtWidgets import QListView
from .image_worker import ImageLoadWorker


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
            self.background_load_timer.start(150)

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