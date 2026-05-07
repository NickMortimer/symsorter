from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtCore import Qt, QTimer
from pathlib import Path
import json
import os

FILLER_ROLE = Qt.UserRole + 42


class ClassUIMixin:
    def update_class_info_label(self):
        label = getattr(self, "class_info_label", None)
        if not label:
            return
        size_name = getattr(self, "current_size_name", "Medium")
        crop_name = getattr(self, "current_crop_name", "none")
        cache_len = len(getattr(self, "image_cache", {})) if hasattr(self, "image_cache") else 0
        raw_len = len(getattr(self, "raw_image_cache", {})) if hasattr(self, "raw_image_cache") else 0
        if getattr(self, "classes", []):
            label.setText(
                f"Classes loaded: {len(self.classes)}\n"
                f"Thumbnail size: {size_name} (Ctrl+/-) | "
                f"Crop: {crop_name} (Shift+Ctrl+/-) | "
                f"Cache: {cache_len} thumbs, {raw_len} raw"
            )
        else:
            label.setText(
                f"No classes loaded\n"
                f"Thumbnail size: {size_name} (Ctrl+/-) | "
                f"Crop: {crop_name} (Shift+Ctrl+/-) | "
                f"Cache: {cache_len} thumbs, {raw_len} raw"
            )

    def update_classes_menu(self):
        menu = getattr(self, "classes_menu", None)
        if not menu:
            return
        # remove old class actions
        for act in list(menu.actions()):
            if getattr(act, "_is_class_action", False):
                menu.removeAction(act)

        if not getattr(self, "classes", None):
            return

        items = []
        for idx, name in enumerate(self.classes):
            ks = getattr(self, "class_keystrokes", {}).get(idx, "")
            items.append((idx, name, ks))

        def sort_key(entry):
            _, _, ks = entry
            if not ks:
                return (2, "")
            if ks.isdigit():
                return (0, int(ks))
            return (1, ks.lower())

        items.sort(key=sort_key)

        if menu.actions():
            sep = menu.addSeparator()
            sep._is_class_action = True

        for idx, name, ks in items:
            text = f"&{ks}: {name}" if ks else name
            action = QAction(text, self)
            action._is_class_action = True
            if ks:
                action.setShortcut(QKeySequence(ks))
            action.triggered.connect(lambda _=False, i=idx: self.assign_class_to_selected(i))
            menu.addAction(action)

    def schedule_filter_combo_refresh(self):
        QTimer.singleShot(0, self.update_class_filter_combo)

    def force_update_class_filter_combo(self):
        self.update_class_filter_combo()

    def update_class_filter_combo(self):
        combo = getattr(self, "class_filter_combo", None)
        if not combo:
            return

        combo.blockSignals(True)
        current = combo.currentText()
        combo.clear()
        combo.addItem("All Images")
        combo.addItem("Unallocated")

        json_cats = []
        folder = getattr(self, "folder", "")
        if folder:
            jpath = Path(folder) / "classifications.json"
            if jpath.exists():
                try:
                    with open(jpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    json_cats = [c["name"] for c in data.get("categories", []) if "name" in c]
                except Exception:
                    pass

        file_classes = getattr(self, "classes", []) or []

        merged = []
        for n in json_cats:
            if n not in merged:
                merged.append(n)
        for n in file_classes:
            if n not in merged:
                merged.append(n)

        for name in sorted(merged):
            combo.addItem(name)

        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentText("All Images")

        combo.blockSignals(False)

    def apply_class_filter(self, filter_text):
        if not getattr(self, "embeddings", None):
            return

        folder_path = Path(getattr(self, "folder", ""))
        hidden = getattr(self, "hidden_flags", {})
        cats = getattr(self, "image_categories", {})

        if filter_text == "All Images":
            visible = [fn for fn in self.embeddings.keys()
                       if (folder_path / fn).exists() and not hidden.get(fn, False)]
        elif filter_text == "Unallocated":
            visible = [fn for fn in self.embeddings.keys()
                       if (folder_path / fn).exists() and not hidden.get(fn, False) and fn not in cats]
        else:
            visible = [fn for fn in self.embeddings.keys()
                       if (folder_path / fn).exists() and not hidden.get(fn, False)
                       and fn in cats and cats[fn].get("class_name") == filter_text]

        self.image_files = visible
        self.original_order = list(self.image_files)
        if hasattr(self, "rebuild_model_after_filter"):
            self.rebuild_model_after_filter()

    def update_item_with_classification(self, item, filename):
        if item.data(FILLER_ROLE):
            item.setText("")
            item.setToolTip("")
            return

        display_name = os.path.basename(filename)
        cats = getattr(self, "image_categories", {})
        show_labels = getattr(self, "show_classification_labels", False)

        if filename in cats:
            info = cats[filename]
            cname = info.get("class_name", "Unknown")
            cid = info.get("class_id", "N/A")
            item.setToolTip(f"File: {display_name}\nClass: {cname} (ID: {cid})")
            item.setText(cname if show_labels else "")
        else:
            item.setToolTip(f"File: {display_name}\nClass: Unallocated")
            item.setText("Unallocated" if show_labels else "")

    def show_classified_images(self):
        """Show only images that have a classification, grouped by class."""
        if not getattr(self, "embeddings", None):
            return
        folder_path = Path(getattr(self, "folder", "")) if getattr(self, "folder", "") else None
        hidden = getattr(self, "hidden_flags", {})
        cats = getattr(self, "image_categories", {})

        def exists(fn: str) -> bool:
            return True if not folder_path else (folder_path / fn).exists()

        visible = [
            fn for fn in self.embeddings.keys()
            if exists(fn) and not hidden.get(fn, False) and fn in cats
        ]

        self.image_files = visible
        self.original_order = list(self.image_files)

        # Prefer grouped view
        if hasattr(self, "sort_by_class"):
            self.sort_by_class()
        elif hasattr(self, "rebuild_model_after_filter"):
            self.current_sort_mode = None
            self.rebuild_model_after_filter()