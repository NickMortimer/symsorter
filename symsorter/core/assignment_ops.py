import os
from PySide6.QtCore import Qt


class AssignmentOpsMixin:
    def assign_class_to_selected(self, class_idx):
        if not self.classes or class_idx >= len(self.classes):
            print(f"Class index {class_idx} not available (have {len(self.classes)} classes)")
            return

        indexes = self.view.selectedIndexes()
        if not indexes:
            print("No images selected")
            return

        self.last_used_class_idx = class_idx
        class_name = self.classes[class_idx]
        persistent_id = int(self.class_ids.get(class_idx, class_idx))
        assigned_count = 0

        for index in indexes:
            filename = index.data(Qt.UserRole)
            if filename:
                self.image_categories[filename] = {
                    'class_id': persistent_id,
                    'class_name': class_name
                }
                assigned_count += 1

        if assigned_count > 0:
            self.has_unsaved_changes = True
            self.update_window_title()
            self.start_auto_backup()

        self.hide_classified_images(indexes)
        self.update_model_with_categories()

    def hide_classified_images(self, indexes):
        hidden_count = 0
        current_filter = self.class_filter_combo.currentText()
        for index in sorted(indexes, key=lambda x: x.row(), reverse=True):
            filename = index.data(Qt.UserRole)
            if not filename or filename not in self.image_categories:
                continue
            assigned_class = self.image_categories[filename]['class_name']

            should_hide = False
            if current_filter == "Unallocated":
                should_hide = True
            elif current_filter == "All Images":
                should_hide = False
            elif current_filter in self.classes:
                should_hide = (assigned_class != current_filter)

            if should_hide:
                if filename in self.image_files:
                    self.image_files.remove(filename)
                self.model.removeRow(index.row())
                hidden_count += 1
            else:
                item = self.model.item(index.row())
                if item:
                    display_name = os.path.basename(filename)
                    tip = f"File: {display_name}\nClass: {assigned_class} (ID: {self.image_categories[filename]['class_id']})"
                    item.setToolTip(tip)

        if hidden_count > 0:
            self.loaded_images = set(i for i in self.loaded_images if i < len(self.loaded_images))