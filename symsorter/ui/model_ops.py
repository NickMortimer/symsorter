import os
from pathlib import Path
from PySide6.QtCore import Qt, QSize, QTimer, QItemSelectionModel
from PySide6.QtWidgets import QAbstractItemView
from PySide6.QtGui import QStandardItem


class ModelOpsMixin:
    def update_model_with_categories(self):
        for row in range(self.model.rowCount()):
            item = self.model.item(row)
            if not item:
                continue
            filename = item.data(Qt.UserRole)
            if not filename:
                continue
            display_name = os.path.basename(filename)
            if filename in self.image_categories:
                info = self.image_categories[filename]
                tooltip = f"File: {display_name}\nClass: {info['class_name']} (ID: {info['class_id']})"
            else:
                tooltip = f"File: {display_name}\nClass: Unallocated"
            item.setToolTip(tooltip)

    def rebuild_model(self, scroll_to_filename=None):
        existing_icons = {}
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            if not item:
                continue
            filename = item.data(Qt.UserRole)
            icon = item.icon()
            if not filename or icon is None or icon.isNull():
                continue
            # Skip placeholder icons so they will be reloaded
            if icon.cacheKey() == self.placeholder_icon.cacheKey():
                continue
            existing_icons[filename] = icon

        self.model.clear()
        new_loaded = set()
        scroll_to_index = None
        for i, file in enumerate(self.image_files):
            item = QStandardItem()
            if file in existing_icons:
                item.setIcon(existing_icons[file])
                new_loaded.add(i)  # only mark as loaded for real icons
            else:
                item.setIcon(self.placeholder_icon)
            item.setEditable(False)
            item.setData(file, Qt.UserRole)

            display_name = os.path.basename(file)
            if file in self.image_categories:
                info = self.image_categories[file]
                tip = f"File: {display_name}\nClass: {info['class_name']} (ID: {info['class_id']})"
            else:
                tip = f"File: {display_name}\nClass: Unallocated"
            item.setToolTip(tip)

            self.model.appendRow(item)
            if scroll_to_filename and file == scroll_to_filename:
                scroll_to_index = i

        self.loaded_images = new_loaded
        self.load_visible_images()
        if scroll_to_index is not None:
            QTimer.singleShot(50, lambda: self.scroll_to_and_select_index(scroll_to_index))

    def scroll_to_and_select_index(self, index):
        if 0 <= index < self.model.rowCount():
            model_index = self.model.index(index, 0)
            self.view.clearSelection()
            self.view.setCurrentIndex(model_index)
            self.view.selectionModel().select(model_index, QItemSelectionModel.Select)
            self.view.scrollTo(model_index, QAbstractItemView.PositionAtTop)

    def reset_order(self):
        if not self.original_order:
            return
        self.image_files = self.original_order.copy()
        self.rebuild_model()

    def show_classified_images(self):
        if not self.image_categories:
            print("No classified images to show")
            return
        temp_categories = self.image_categories.copy()
        self.image_categories.clear()
        self.reload_current_folder()
        self.image_categories = temp_categories
        self.update_model_with_categories()

    def reload_current_folder(self):
        if not self.npz_file_path:
            return
        folder_path = Path(self.folder)
        self.image_files = [fn for fn in self.embeddings.keys()
                            if (folder_path / fn).exists()
                            and not self.hidden_flags.get(fn, False)
                            and fn not in self.image_categories]
        self.original_order = self.image_files.copy()
        self.model.clear()
        self.loaded_images.clear()
        for file in self.image_files:
            item = QStandardItem()
            item.setIcon(self.placeholder_icon)
            item.setEditable(False)
            item.setData(file, Qt.UserRole)
            display_name = os.path.basename(file)
            if file in self.image_categories:
                info = self.image_categories[file]
                tip = f"File: {display_name}\nClass: {info['class_name']} (ID: {info['class_id']})"
            else:
                tip = f"File: {display_name}\nClass: Unallocated"
            item.setToolTip(tip)
            self.model.appendRow(item)
        if self.image_files:
            self.load_image_batch(0, min(40, len(self.image_files)))

    def apply_class_filter(self, filter_text):
        if not self.embeddings:
            return
        folder_path = Path(self.folder)
        if filter_text == "All Images":
            visible = [fn for fn in self.embeddings.keys()
                       if (folder_path / fn).exists()
                       and not self.hidden_flags.get(fn, False)]
        elif filter_text == "Unallocated":
            visible = [fn for fn in self.embeddings.keys()
                       if (folder_path / fn).exists()
                       and not self.hidden_flags.get(fn, False)
                       and fn not in self.image_categories]
        else:
            visible = [fn for fn in self.embeddings.keys()
                       if (folder_path / fn).exists()
                       and not self.hidden_flags.get(fn, False)
                       and fn in self.image_categories
                       and self.image_categories[fn]['class_name'] == filter_text]
        self.image_files = visible
        self.rebuild_model_after_filter()

    def rebuild_model_after_filter(self):
        existing_icons = {}
        for row in range(self.model.rowCount()):
            item = self.model.item(row)
            if not item:
                continue
            filename = item.data(Qt.UserRole)
            icon = item.icon()
            if not filename or icon is None or icon.isNull():
                continue
            # Skip placeholder icons
            if icon.cacheKey() == self.placeholder_icon.cacheKey():
                continue
            existing_icons[filename] = icon

        self.stop_background_loading()
        self.model.clear()
        self.loaded_images.clear()

        for i, file in enumerate(self.image_files):
            item = QStandardItem()
            if file in existing_icons:
                item.setIcon(existing_icons[file])
                self.loaded_images.add(i)  # only real icons count as loaded
            else:
                item.setIcon(self.placeholder_icon)
            item.setEditable(False)
            item.setData(file, Qt.UserRole)
            display_name = os.path.basename(file)
            if file in self.image_categories:
                info = self.image_categories[file]
                tip = f"File: {display_name}\nClass: {info['class_name']} (ID: {info['class_id']})"
            else:
                tip = f"File: {display_name}\nClass: Unallocated"
            item.setToolTip(tip)
            self.model.appendRow(item)

        if self.model.rowCount() > 0:
            padding = max(10, self.icon_size // 20)
            grid = self.icon_size + padding
            self.view.setGridSize(QSize(grid, grid))
            self.view.setIconSize(QSize(self.icon_size, self.icon_size))
            spacing = max(2, self.icon_size // 50)
            self.view.setSpacing(spacing)

        if self.image_files:
            QTimer.singleShot(50, self.load_visible_images)

    def cosine_similarity(self, a, b):
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def on_double_click(self, index):
        if not self.embeddings:
            return
        filename = index.data(Qt.UserRole)
        if filename not in self.embeddings:
            return
        ref = self.embeddings[filename]
        sims = []
        for img_file in self.image_files:
            if img_file in self.embeddings:
                sims.append((img_file, self.cosine_similarity(ref, self.embeddings[img_file])))
        sims.sort(key=lambda x: x[1], reverse=True)
        self.image_files = [f for f, _ in sims]
        self.rebuild_model(filename)