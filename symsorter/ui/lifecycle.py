class LifecycleMixin:
    def closeEvent(self, event):
        self.stop_auto_backup()
        if self.has_unsaved_changes:
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                'Unsaved Changes',
                'You have unsaved classifications and hidden images.\n'
                'Do you want to save your work before exiting?\n\n'
                'Click "Save" to save your work,\n'
                '"Discard" to exit without saving,\n'
                'or "Cancel" to continue working.',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            if reply == QMessageBox.Save:
                self.save_classifications()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return

        self.stop_background_loading()
        for worker in self.active_workers:
            worker.stop()
        if not self.thread_pool.waitForDone(3000):
            print("Warning: Some image loading threads did not finish cleanly")
        self.clear_caches()
        event.accept()

        backup_path = self.get_backup_path()
        if backup_path and backup_path.exists():
            try:
                backup_path.unlink()
                print("Backup file cleaned up after successful save")
            except Exception as e:
                print(f"Warning: Could not remove backup file: {e}")