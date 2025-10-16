from pathlib import Path
from PySide6.QtWidgets import QFileDialog, QMessageBox


class ExportDialogsMixin:
    def export_yolo_annotations(self):
        if not self.image_categories or not self.classes:
            print("No categories assigned or classes loaded")
            return

        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return
        export_path = Path(export_dir)
        labels_dir = export_path / "labels"
        labels_dir.mkdir(exist_ok=True)

        exported_count = 0
        for filename, category_info in self.image_categories.items():
            txt_path = labels_dir / (Path(filename).stem + ".txt")
            class_id = category_info['class_id']
            with open(txt_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            exported_count += 1

        classes_file = export_path / "classes.txt"
        with open(classes_file, 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")

        print(f"Exported {exported_count} YOLO annotations to {export_path}")
        print(f"Classes file saved to {classes_file}")

    def export_classifications_json_dialog(self):
        if not self.image_categories:
            QMessageBox.information(self, "Export Classifications", "No classifications to export.")
            return

        json_file, _ = QFileDialog.getSaveFileName(
            self,
            "Export Classifications as JSON",
            str(self.get_classifications_json_path() or "classifications.json"),
            "JSON files (*.json);;All files (*)"
        )
        if not json_file:
            return

        try:
            original_path = self.npz_file_path
            self.npz_file_path = Path(json_file).with_suffix('.npz')  # for path calc
            temp_path = Path(json_file)
            coco_data = self.build_coco_data()
            with open(temp_path, 'w', encoding='utf-8') as f:
                import json as _json
                _json.dump(coco_data, f, indent=2, ensure_ascii=False)
            self.npz_file_path = original_path
            QMessageBox.information(self, "Export Successful", f"Classifications exported to:\n{temp_path}")
            print(f"Exported classifications to {temp_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export classifications:\n{str(e)}")
            print(f"Error exporting classifications: {e}")

    def import_classifications_json_dialog(self):
        json_file, _ = QFileDialog.getOpenFileName(
            self,
            "Import Classifications from JSON",
            "",
            "JSON files (*.json);;All files (*)"
        )
        if not json_file:
            return

        reply = QMessageBox.question(
            self, "Import Classifications",
            "This will replace current classifications. Continue?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        try:
            success = self.load_classifications_json(Path(json_file))
            if success:
                self.has_unsaved_changes = True
                self.update_window_title()
                self.update_model_with_categories()
                QMessageBox.information(self, "Import Successful", f"Classifications imported from:\n{json_file}")
            else:
                QMessageBox.warning(self, "Import Failed", "Failed to import classifications from JSON file.")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import classifications:\n{str(e)}")
            print(f"Error importing classifications: {e}")