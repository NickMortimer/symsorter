from pathlib import Path
import pickle
from datetime import datetime


class BackupMixin:
    def get_backup_path(self):
        if not self.npz_file_path:
            return None
        return self.npz_file_path.parent / "classifications_backup.pickle"

    def start_auto_backup(self):
        if self.npz_file_path and not self.auto_backup_timer.isActive():
            self.auto_backup_timer.start(self.auto_backup_interval)
            print("Auto-backup started - will backup every 4 minutes")

    def stop_auto_backup(self):
        if self.auto_backup_timer.isActive():
            self.auto_backup_timer.stop()
            print("Auto-backup stopped")

    def auto_backup_classifications(self):
        if not self.npz_file_path or not self.has_unsaved_changes:
            return
        backup_path = self.get_backup_path()
        if not backup_path:
            return
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'image_categories': self.image_categories,
                'hidden_flags': self.hidden_flags,
                'classes': self.classes,
                'class_keystrokes': self.class_keystrokes,
                'class_descriptions': self.class_descriptions,
                'npz_file_path': str(self.npz_file_path),
                'version': '1.0'
            }
            with open(backup_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Auto-backup saved: {len(self.image_categories)} classifications")
        except Exception as e:
            print(f"Auto-backup failed: {e}")

    def check_and_offer_backup_restore(self):
        backup_path = self.get_backup_path()
        if not backup_path or not backup_path.exists():
            return
        try:
            with open(backup_path, 'rb') as f:
                backup_data = pickle.load(f)
            ts = backup_data.get('timestamp', 'Unknown time')
            try:
                from datetime import datetime as dt
                time_str = dt.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = ts
            count = len(backup_data.get('image_categories', {}))
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                'Backup Found',
                f'A backup from {time_str} with {count} classifications was found.\nRestore it?',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.restore_from_backup(backup_data)
            elif reply == QMessageBox.No:
                backup_path.unlink()
                print("Backup removed - starting fresh")
        except Exception as e:
            print(f"Error checking backup: {e}")

    def restore_from_backup(self, backup_data):
        try:
            self.image_categories = backup_data.get('image_categories', {})
            self.hidden_flags = backup_data.get('hidden_flags', {})
            if not self.classes and 'classes' in backup_data:
                self.classes = backup_data.get('classes', [])
                self.class_keystrokes = backup_data.get('class_keystrokes', {})
                self.class_descriptions = backup_data.get('class_descriptions', {})
                self.update_class_info_label()
                self.update_classes_menu()
            self.has_unsaved_changes = True
            self.update_window_title()
            print("Backup restored")
        except Exception as e:
            print(f"Error restoring backup: {e}")

    def cleanup_backup_on_successful_save(self):
        backup_path = self.get_backup_path()
        if backup_path and backup_path.exists():
            try:
                backup_path.unlink()
                print("Backup file cleaned up after successful save")
            except Exception as e:
                print(f"Warning: Could not remove backup file: {e}")