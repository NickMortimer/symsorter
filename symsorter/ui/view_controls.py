from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QPixmap, QIcon


class ViewControlsMixin:
    def create_size_handler(self, size_name):
        def handler():
            self.set_thumbnail_size(size_name)
        return handler

    def create_crop_handler(self, crop_name):
        def handler():
            self.set_crop_size(crop_name)
        return handler

    def set_thumbnail_size(self, size_name):
        if size_name == self.current_size_name:
            return
        for action in self.size_actions:
            action.setChecked(action.text().replace('&', '') == size_name)
        self.on_thumbnail_size_changed(size_name)

    def increase_thumbnail_size(self):
        names = list(self.icon_sizes.keys())
        i = names.index(self.current_size_name)
        if i < len(names) - 1:
            self.set_thumbnail_size(names[i + 1])

    def decrease_thumbnail_size(self):
        names = list(self.icon_sizes.keys())
        i = names.index(self.current_size_name)
        if i > 0:
            self.set_thumbnail_size(names[i - 1])

    def assign_to_last_used_class(self):
        if self.last_used_class_idx is not None:
            self.assign_class_to_selected(self.last_used_class_idx)

    def increase_crop_zoom(self):
        names = list(self.crop_sizes.keys())
        i = names.index(self.current_crop_name)
        if i < len(names) - 1:
            self.set_crop_size(names[i + 1])

    def decrease_crop_zoom(self):
        names = list(self.crop_sizes.keys())
        i = names.index(self.current_crop_name)
        if i > 0:
            self.set_crop_size(names[i - 1])

    def set_crop_size(self, crop_name):
        if crop_name == self.current_crop_name:
            return
        old = self.current_crop_name
        self.current_crop_name = crop_name
        self.crop_size = self.crop_sizes[crop_name]
        if hasattr(self, 'crop_actions'):
            for action in self.crop_actions:
                action.setChecked(action.text().replace('&', '') == crop_name)
        if self.model.rowCount() > 0:
            self.reload_images_with_new_crop()
        self.update_class_info_label()

    def reload_images_with_new_crop(self):
        if not self.model:
            return
        self.stop_background_loading()
        self.image_cache.clear()
        self.cache_access_order = [k for k in self.cache_access_order if isinstance(k, str)]
        for row in range(self.model.rowCount()):
            item = self.model.item(row)
            if item:
                item.setIcon(self.placeholder_icon)
        self.loaded_images.clear()
        for w in self.active_workers:
            w.stop()
        self.active_workers.clear()
        self.thread_pool.clear()
        self.load_visible_images()

    def update_placeholder_icon(self):
        pm = QPixmap(self.icon_size, self.icon_size)
        pm.fill(Qt.lightGray)
        self.placeholder_icon = QIcon(pm)

    def on_thumbnail_size_changed(self, size_name):
        if size_name not in self.icon_sizes:
            return
        old = self.icon_size
        self.current_size_name = size_name
        self.icon_size = self.icon_sizes[size_name]
        self.update_placeholder_icon()
        if self.model.rowCount() > 0:
            padding = max(10, self.icon_size // 20)
            grid = self.icon_size + padding
            self.view.setGridSize(QSize(grid, grid))
            self.view.setIconSize(QSize(self.icon_size, self.icon_size))
            spacing = max(2, self.icon_size // 50)
            self.view.setSpacing(spacing)
            self.reload_images_with_new_size()
        self.update_class_info_label()

    def reload_images_with_new_size(self):
        if not self.model:
            return
        self.stop_background_loading()
        self.image_cache.clear()
        self.cache_access_order = [k for k in self.cache_access_order if isinstance(k, str)]
        for row in range(self.model.rowCount()):
            item = self.model.item(row)
            if item:
                item.setIcon(self.placeholder_icon)
        self.loaded_images.clear()
        for w in self.active_workers:
            w.stop()
        self.active_workers.clear()
        self.thread_pool.clear()
        self.load_visible_images()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_main_resize_timer'):
            self._main_resize_timer.stop()
        self._main_resize_timer = QTimer()
        self._main_resize_timer.setSingleShot(True)
        self._main_resize_timer.timeout.connect(self.load_visible_images)
        self._main_resize_timer.start(150)

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        if mods & Qt.ControlModifier and key in (Qt.Key_Plus, Qt.Key_Equal):
            self.increase_thumbnail_size()
            event.accept()
            return
        if mods & Qt.ControlModifier and key == Qt.Key_Minus:
            self.decrease_thumbnail_size()
            event.accept()
            return
        if mods & (Qt.ShiftModifier | Qt.ControlModifier) and key in (Qt.Key_Plus, Qt.Key_Equal):
            self.increase_crop_zoom()
            event.accept()
            return
        if mods & (Qt.ShiftModifier | Qt.ControlModifier) and key == Qt.Key_Minus:
            self.decrease_crop_zoom()
            event.accept()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if self.last_used_class_idx is not None:
                self.assign_class_to_selected(self.last_used_class_idx)
            event.accept()
            return
        super().keyPressEvent(event)