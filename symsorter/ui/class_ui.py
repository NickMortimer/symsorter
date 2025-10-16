from PySide6.QtGui import QAction, QKeySequence


class ClassUIMixin:
    def update_class_info_label(self):
        cache_text = f"Cache: {len(self.image_cache)} thumbnails, {len(self.raw_image_cache)} raw images"
        size_text = f"Thumbnail size: {self.current_size_name} (Ctrl+/-)"
        crop_text = f"Crop: {self.current_crop_name} (Shift+Ctrl+/-)"

        if self.classes:
            class_text = f"Classes loaded: {len(self.classes)} (Shift+F1-F{min(len(self.classes), 12)}, Enter=repeat last)"
            if len(self.classes) > 12:
                class_text += f" [showing first 12 of {len(self.classes)}]"
            self.class_info_label.setText(f"{class_text}\n{size_text} | {crop_text} | {cache_text}")
            self.update_class_filter_combo()
            self.update_classes_menu()
        else:
            self.class_info_label.setText(f"No classes loaded\n{size_text} | {crop_text} | {cache_text}")

    def update_class_filter_combo(self):
        current_text = self.class_filter_combo.currentText()
        self.class_filter_combo.blockSignals(True)
        self.class_filter_combo.clear()
        self.class_filter_combo.addItem("All Images")
        self.class_filter_combo.addItem("Unallocated")
        for class_name in self.classes:
            self.class_filter_combo.addItem(class_name)
        index = self.class_filter_combo.findText(current_text)
        if index >= 0:
            self.class_filter_combo.setCurrentIndex(index)
        self.class_filter_combo.blockSignals(False)

    def update_classes_menu(self):
        self.classes_menu.clear()
        if not self.classes:
            a = QAction('No classes loaded', self)
            a.setEnabled(False)
            self.classes_menu.addAction(a)
            self.classes_menu.setEnabled(False)
            return

        self.classes_menu.setEnabled(True)
        for i, class_name in enumerate(self.classes):
            description = self.class_descriptions.get(i, "")
            description_suffix = f" - {description}" if description else ""
            if i in self.class_keystrokes:
                keystroke = self.class_keystrokes[i]
                action = QAction(f"&{class_name}", self)
                action.setShortcut(QKeySequence(keystroke))
                action.setStatusTip(f'Assign selected images to class "{class_name}" ({keystroke}){description_suffix}')
                action.triggered.connect(self.create_class_assignment_handler(i))
                self.classes_menu.addAction(action)
            elif i < 12:
                f_key = f"Shift+F{i+1}"
                action = QAction(f"&{class_name}", self)
                action.setShortcut(QKeySequence(f_key))
                action.setStatusTip(f'Assign selected images to class "{class_name}" ({f_key}){description_suffix}')
                action.triggered.connect(self.create_class_assignment_handler(i))
                self.classes_menu.addAction(action)
            else:
                action = QAction(f"{class_name}", self)
                action.setStatusTip(f'Assign selected images to class "{class_name}" (no shortcut){description_suffix}')
                action.triggered.connect(self.create_class_assignment_handler(i))
                self.classes_menu.addAction(action)

        self.classes_menu.addSeparator()
        enter_action = QAction('Assign to &Last Used Class', self)
        enter_action.setShortcut(QKeySequence('Return'))
        enter_action.setStatusTip('Assign selected images to last used class (Enter)')
        enter_action.triggered.connect(self.assign_to_last_used_class)
        self.classes_menu.addAction(enter_action)

        classes_without_shortcuts = []
        for i, class_name in enumerate(self.classes):
            if i not in self.class_keystrokes and i >= 12:
                classes_without_shortcuts.append((i + 1, class_name))
        if classes_without_shortcuts:
            nums = [str(n) for n, _ in classes_without_shortcuts]
            display_range = f"{nums[0]}-{nums[-1]}" if len(nums) > 3 else ", ".join(nums)
            info_action = QAction(f'Classes {display_range} have no shortcuts', self)
            info_action.setEnabled(False)
            self.classes_menu.addAction(info_action)

    def on_class_filter_changed(self, filter_text):
        self.apply_class_filter(filter_text)