from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog
from PySide6.QtGui import QStandardItem, QIcon
from PySide6.QtCore import Qt, QSize
import os
import json


class FolderFlowMixin:
    def load_folder(self):
        npz_file, _ = QFileDialog.getOpenFileName(
            self, "Select Embeddings File", "", "NPZ files (*.npz);;All files (*)"
        )
        if npz_file:
            self.load_folder_from_path(npz_file)

    def load_folder_from_path(self, npz_file):
        npz_path = Path(npz_file)
        if not npz_path.exists():
            print(f"NPZ file not found: {npz_path}")
            return

        self.folder = str(npz_path.parent)
        self.npz_file_path = npz_path
        print(f"Loading embeddings from {npz_path}")
        self.embeddings, self.cached_dimensions, self.embedding_metadata = self._load_embeddings(npz_path)

        self.hidden_flags = {}
        self.image_categories = {}

        json_loaded = self.load_classifications_json()
        if not json_loaded:
            self.check_and_offer_backup_restore()
            print("No classifications JSON file found - starting with empty classifications")

        self.has_unsaved_changes = False
        self.update_window_title()
        self.start_auto_backup()

        if not self.embeddings:
            print("No embeddings loaded!")
            return

        self.model.clear()
        self.loaded_images.clear()
        for w in self.active_workers:
            w.stop()
        self.active_workers.clear()
        self.thread_pool.clear()

        self.class_filter_combo.setCurrentText("Unallocated")

        folder_path = Path(self.folder)
        self.image_files = [fn for fn in self.embeddings.keys()
                            if (folder_path / fn).exists()
                            and not self.hidden_flags.get(fn, False)
                            and fn not in self.image_categories]
        self.original_order = self.image_files.copy()

        print(f"Found {len(self.image_files)} images with embeddings")

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

        print(f"Added {self.model.rowCount()} items to model")

        padding = max(10, self.icon_size // 20)
        grid_size = self.icon_size + padding
        self.view.setGridSize(QSize(grid_size, grid_size))
        self.view.setIconSize(QSize(self.icon_size, self.icon_size))
        spacing = max(2, self.icon_size // 50)
        self.view.setSpacing(spacing)

        if self.image_files:
            initial_batch_size = min(100, len(self.image_files))
            print(f"Loading initial batch of {initial_batch_size} images")
            self.load_image_batch(0, initial_batch_size)
            self.update_model_with_categories()
            QTimer.singleShot(100, self.load_visible_images)

    def save_classifications(self):
        if not self.npz_file_path:
            print("No npz file loaded")
            return
        try:
            self.save_classifications_json()
            hidden_count = sum(1 for hidden in self.hidden_flags.values() if hidden)
            category_count = len(self.image_categories)
            print(f"Saved classifications to JSON: {self.get_classifications_json_path()}")
            print(f"  - {category_count} image classifications")
            print(f"  - {hidden_count} hidden images")
            self.has_unsaved_changes = False
            self.update_window_title()
            self.cleanup_backup_on_successful_save()
        except Exception as e:
            print(f"Error saving data: {e}")

    def get_classifications_json_path(self):
        if not self.npz_file_path:
            return None
        return self.npz_file_path.parent / "classifications.json"

    def save_classifications_json(self):
        if not self.npz_file_path:
            return
        json_path = self.get_classifications_json_path()
        coco_data = self.build_coco_data()
        with open(str(json_path), 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        print(f"Saved COCO-style classifications to {json_path}")
        return str(json_path)

    def load_classifications_json(self, json_path=None):
        if json_path is None:
            json_path = self.get_classifications_json_path()
        if not json_path or not json_path.exists():
            print("No classifications JSON file found")
            return False
        try:
            coco = self._io_load_coco(json_path)
            self.image_categories = {}
            self.hidden_flags = {}
            if not self.classes and 'categories' in coco:
                self.classes, self.class_keystrokes, self.class_descriptions = [], {}, {}
                self.class_ids = {}
                for idx, cat in enumerate(coco['categories']):
                    self.classes.append(cat['name'])
                    if cat.get('keystroke'):
                        self.class_keystrokes[idx] = cat['keystroke']
                    if cat.get('description'):
                        self.class_descriptions[idx] = cat['description']
                    self.class_ids[idx] = int(cat.get('id', idx))
                self.update_class_info_label()
                self.update_classes_menu()
            id_by_filename = {}
            for img in coco.get('images', []):
                id_by_filename[img['file_name']] = img['id']
                if 'hidden' in img:
                    self.hidden_flags[img['file_name']] = bool(img['hidden'])
            cats = {c['id']: c['name'] for c in coco.get('categories', [])}
            for ann in coco.get('annotations', []):
                filename = next((fn for fn, iid in id_by_filename.items() if iid == ann['image_id']), None)
                if not filename:
                    continue
                cid = int(ann['category_id'])
                cname = cats.get(cid, self.classes[cid] if cid < len(self.classes) else str(cid))
                self.image_categories[filename] = {'class_id': cid, 'class_name': cname}
            print(f"Loaded from JSON: {len(self.image_categories)} classifications, {sum(1 for v in self.hidden_flags.values() if v)} hidden images")
            return True
        except Exception as e:
            print(f"Error loading classifications JSON: {e}")
            return False

    # Adapter to keep imports local where originally used
    def _load_embeddings(self, npz_path):
        from ..clip_encode import load_existing_embeddings
        return load_existing_embeddings(npz_path)

    def _io_load_coco(self, json_path):
        from ..io.classifications import load_coco_json
        return load_coco_json(json_path)