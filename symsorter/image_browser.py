from PySide6.QtWidgets import (  # add QProgressDialog
    QApplication, QWidget, QListView, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QAbstractItemView, QSlider, QLabel, QComboBox,
    QToolBar, QMenuBar, QMainWindow, QMessageBox, QProgressDialog
)
from PySide6.QtGui import QPixmap, QIcon, QStandardItemModel, QStandardItem, QKeySequence, QShortcut, QAction
from PySide6.QtCore import Qt, QSize, QThread, Signal, QTimer, QThreadPool, QRunnable, QObject, QItemSelectionModel, QProcess
import sys
import os
import math
import numpy as np
import traceback
import argparse
import yaml
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from .clip_encode import load_existing_embeddings
from .core.image_worker import WorkerSignals, ImageLoadWorker
from .core.lazy_loader import LazyImageLoader
from .ui.menus import build_menus
from .core.cache_manager import CacheManager  # new
from .core.class_manager import ClassManager  # new
from .io.classifications import (
    get_classifications_json_path as io_get_json_path,
    build_coco_data as io_build_coco,
    ensure_json_serializable as io_ensure_json,
    save_coco_json as io_save_coco,
    load_coco_json as io_load_coco,
)
from .core.loader_ops import LoaderOpsMixin
from .ui.view_controls import ViewControlsMixin
from .ui.model_ops import ModelOpsMixin
from .io.backup import BackupMixin
from .ui.class_ui import ClassUIMixin
from .core.assignment_ops import AssignmentOpsMixin
from .io.export_dialogs import ExportDialogsMixin
from .core.folder_flow import FolderFlowMixin
from .ui.lifecycle import LifecycleMixin



class ImageBrowser(QMainWindow,
                   LoaderOpsMixin,
                   ViewControlsMixin,
                   ModelOpsMixin,
                   BackupMixin,
                   ClassUIMixin,
                   AssignmentOpsMixin,
                   ExportDialogsMixin,
                   FolderFlowMixin,
                   LifecycleMixin):
    def __init__(self, class_file=None):
        super().__init__()
        self.resize(1000, 700)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Image grid view
        self.view = QListView()
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.setViewMode(QListView.IconMode)
        self.view.setResizeMode(QListView.Adjust)
        self.view.setUniformItemSizes(True)
        self.view.setSpacing(2)
        layout.addWidget(self.view)

        # Model
        self.model = QStandardItemModel()
        self.view.setModel(self.model)
        
        # Store folder path and image files
        self.folder = ""
        self.image_files = []
        self.embeddings = {}  # Store embeddings for each image
        self.hidden_flags = {}  # Store hidden status for each image
        self.loaded_images = set()  # Track which individual images have been loaded
        self.original_order = []  # Store original order for reset
        self.npz_file_path = None  # Store path to npz file for saving
        
        # Smart image caching system optimized for 224x224 images
        # Replace raw cache dicts with manager but keep attributes for compatibility
        self._cache = CacheManager()
        self.image_cache = self._cache.image_cache
        self.raw_image_cache = self._cache.raw_image_cache
        self.max_cache_size = self._cache.max_cache_size
        self.max_raw_cache_size = self._cache.max_raw_cache_size
        self.cache_access_order = self._cache.cache_access_order

        # Class manager
        self._classes = ClassManager()
        if class_file:
            self.load_class_file(class_file)
        
        # Track unsaved changes
        self.has_unsaved_changes = False
        
        # YOLO class management
        self.classes = []  # List of class names
        self.class_keystrokes = {}  # Dictionary mapping class index to keystroke
        self.class_descriptions = {}  # Dictionary mapping class index to description
        self.image_categories = {}  # Store category assignments for each image
        self.class_file_path = class_file
        self.last_used_class_idx = None  # Track last used class for Enter key
        
        # Icon and crop settings optimized for 224x224 source images
        self.icon_sizes = {"Small": 64, "Medium": 112, "Large": 224, "Extra Large": 320}
        self.current_size_name = "Medium"
        self.icon_size = self.icon_sizes[self.current_size_name]  # Current icon size
        self.crop_sizes = {"64": 64, "112": 112, "128": 128, "224": 224, "none": None}
        self.current_crop_name = "none"
        self.crop_size = self.crop_sizes[self.current_crop_name]  # Current crop size (None = no crop)

        # Create a simple placeholder icon (will be updated with zoom)
        self.update_placeholder_icon()
        
        # Thread pool for faster image loading - optimized for small 224x224 images
        self.thread_pool = QThreadPool()
        # Set max threads higher for small images (CPU count * 4, max 24)
        max_threads = min(24, QThreadPool.globalInstance().maxThreadCount() * 4)
        self.thread_pool.setMaxThreadCount(max_threads)
        self.active_workers = []  # Keep references to active workers
        self.background_load_timer = QTimer()  # Timer for background loading
        self.background_load_timer.timeout.connect(self.load_next_batch_background)
        
        # Set up auto-save backup timer (every 4 minutes)
        self.auto_backup_timer = QTimer()
        self.auto_backup_timer.timeout.connect(self.auto_backup_classifications)
        self.auto_backup_interval = 4 * 60 * 1000  # 4 minutes in milliseconds
        
        print(f"Image loading thread pool initialized with {max_threads} threads")
        
        # Connect scroll event for lazy loading and double-click for sorting
        self.view.verticalScrollBar().valueChanged.connect(self.on_scroll)
        self.view.doubleClicked.connect(self.on_double_click)
        
        # Connect resize event to load more images when window grows
        self.view.resizeEvent = self.on_view_resize
        
        # Load class file if provided
        if self.class_file_path:
            self.load_class_file(self.class_file_path)
        
        # Set up menus and toolbar
        self.setup_menus_and_toolbar()
        
        # Set up keyboard shortcuts
        self.setup_shortcuts()

        # Class filter layout
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter by class:")
        filter_layout.addWidget(filter_label)
        
        # Class filter combobox
        self.class_filter_combo = QComboBox()
        self.class_filter_combo.addItem("All Images")
        self.class_filter_combo.addItem("Unallocated")
        self.class_filter_combo.currentTextChanged.connect(self.on_class_filter_changed)
        filter_layout.addWidget(self.class_filter_combo)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Set initial filter only on first app start
        self.class_filter_combo.blockSignals(True)
        self.class_filter_combo.setCurrentText("Unallocated")
        self.class_filter_combo.blockSignals(False)

        # Class info label
        self.class_info_label = QLabel("No classes loaded")
        layout.addWidget(self.class_info_label)

        # Now it's safe to load the class file (UI attrs exist)
        if self.class_file_path:
            self.load_class_file(self.class_file_path)

        # Set initial window title
        self.update_window_title()
    
    def setup_menus_and_toolbar(self):
        """Setup menus and toolbar"""
        return build_menus(self)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts for class assignment"""        
        # Keyboard shortcuts are now handled by QAction shortcuts in the Classes menu
        # This avoids duplicate shortcut registration which causes "Ambiguous shortcut overload"
        self.function_key_shortcuts = []
        print("DEBUG: Keyboard shortcuts will be created by Classes menu actions")
    
    def create_class_assignment_handler(self, class_idx):
        """Create a proper closure for class assignment"""
        def handler():
            print(f"DEBUG: Shortcut activated for class index {class_idx}")
            self.assign_class_to_selected(class_idx)
        return handler
    

    # Cache wrappers (unchanged signatures)
    def get_cache_key(self, filename, icon_size, crop_size):
        return self._cache.get_cache_key(filename, icon_size, crop_size)

    def get_cached_icon(self, filename, icon_size, crop_size):
        return self._cache.get_cached_icon(filename, icon_size, crop_size)

    def cache_icon(self, filename, icon_size, crop_size, icon):
        return self._cache.cache_icon(filename, icon_size, crop_size, icon)

    def get_cached_raw_image(self, filename):
        return self._cache.get_cached_raw_image(filename)

    def cache_raw_image(self, filename, pixmap):
        return self._cache.cache_raw_image(filename, pixmap)

    def clear_caches(self):
        return self._cache.clear()
    
    def update_window_title(self):
        """Update window title to show unsaved changes"""
        base_title = "SymSorter - Image Classification Tool"
        if self.npz_file_path:
            base_title += f" - {self.npz_file_path.name}"
        
        if self.has_unsaved_changes:
            base_title += " *"
        
        self.setWindowTitle(base_title)

    # ... (rest of the methods would continue with the same adaptations)
    # For brevity, I'll include the key methods that need adaptation



    def load_class_file_dialog(self):
        """Open dialog to load class file"""
        class_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Class File",
            "",
            "YAML files (*.yaml *.yml);;Text files (*.txt);;Names files (*.names);;All files (*)"
        )
        if class_file:
            self.load_class_file(class_file)
    
    def load_class_file(self, class_file_path):
        """Load classes via ClassManager and update UI."""
        ok = self._classes.load(class_file_path)
        if not ok:
            return
        # Sync local attributes
        self.classes = self._classes.classes
        self.class_keystrokes = self._classes.class_keystrokes
        self.class_descriptions = self._classes.class_descriptions
        self.class_ids = self._classes.class_ids
        self.class_file_path = self._classes.class_file_path
        # Update UI
        self.update_class_info_label()
        self.update_classes_menu()
    

    def load_folder_from_path(self, path_str):
        """Open either a folder or an NPZ file. If a folder has no NPZ, encode it first with progress."""
        p = Path(path_str)
        if p.is_dir():
            npz = self._choose_npz_in_folder(p)
            if not npz:
                # No NPZ found; ask to encode
                reply = QMessageBox.question(
                    self,
                    "Encode embeddings?",
                    f"No embeddings (.npz) found in:\n{p}\n\nEncode this folder now?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply != QMessageBox.Yes:
                    return
                return self._encode_folder_and_then_load(p)
            # Found NPZ, continue as usual
            npz_path = npz
        else:
            npz_path = p

        if not npz_path.exists():
            print(f"NPZ file not found: {npz_path}")
            return

        self.folder = str(npz_path.parent)
        self.npz_file_path = npz_path

        print(f"Loading embeddings from {npz_path}")
        self.embeddings, self.cached_dimensions, self.embedding_metadata = load_existing_embeddings(npz_path)

        # Load hidden flags and categories from JSON file only
        self.hidden_flags = {}
        self.image_categories = {}
        
        # Load from JSON file (only format for classifications now)
        json_loaded = self.load_classifications_json()
        
        if not json_loaded:
            # Check for backup file and offer to restore
            self.check_and_offer_backup_restore()
            print("No classifications JSON file found - starting with empty classifications")
        
        # Clear unsaved changes flag when loading new data
        self.has_unsaved_changes = False
        self.update_window_title()
        
        # Start auto-backup timer now that we have data loaded
        self.start_auto_backup()
        
        if not self.embeddings:
            print("No embeddings loaded!")
            return
        
        self.model.clear()
        self.loaded_images.clear()
        # Stop any active workers
        for worker in self.active_workers:
            worker.stop()
        self.active_workers.clear()
        self.thread_pool.clear()
        
        # Use the current filter selection to choose which images to list
        selected_filter = self.class_filter_combo.currentText() or "All Images"
        folder_path = Path(self.folder)

        if selected_filter == "All Images":
            self.image_files = [
                fn for fn in self.embeddings.keys()
                if (folder_path / fn).exists()
                and not self.hidden_flags.get(fn, False)
            ]
        elif selected_filter == "Unallocated":
            self.image_files = [
                fn for fn in self.embeddings.keys()
                if (folder_path / fn).exists()
                and not self.hidden_flags.get(fn, False)
                and fn not in self.image_categories
            ]
        else:
            # Specific class filter
            self.image_files = [
                fn for fn in self.embeddings.keys()
                if (folder_path / fn).exists()
                and not self.hidden_flags.get(fn, False)
                and fn in self.image_categories
                and self.image_categories[fn]['class_name'] == selected_filter
            ]

        self.original_order = self.image_files.copy()
        print(f"Found {len(self.image_files)} images with embeddings (filter: {selected_filter})")
        
        # Create placeholder items for all images
        for file in self.image_files:
            item = QStandardItem()
            item.setIcon(self.placeholder_icon)
            item.setEditable(False)
            item.setData(file, Qt.UserRole)  # Store filename in user data
            
            # Set initial tooltip
            display_name = os.path.basename(file)
            if file in self.image_categories:
                category_info = self.image_categories[file]
                tooltip = f"File: {display_name}\nClass: {category_info['class_name']} (ID: {category_info['class_id']})"
            else:
                tooltip = f"File: {display_name}\nClass: Unallocated"
            item.setToolTip(tooltip)
            
            self.model.appendRow(item)
        
        print(f"Added {self.model.rowCount()} items to model")
        
        # Set initial grid and icon sizes with appropriate padding
        # Use larger padding for bigger icons to prevent overlap
        padding = max(10, self.icon_size // 20)  # Minimum 10px, or 5% of icon size
        grid_size = self.icon_size + padding
        self.view.setGridSize(QSize(grid_size, grid_size))
        self.view.setIconSize(QSize(self.icon_size, self.icon_size))
        # Set spacing that scales with icon size
        spacing = max(2, self.icon_size // 50)  # Minimum 2px, scales with icon size
        self.view.setSpacing(spacing)
        
        # Load first batch immediately
        if self.image_files:
            # Load first 100 images immediately (larger batch for initial load with small 224x224 images)
            initial_batch_size = min(100, len(self.image_files))
            print(f"Loading initial batch of {initial_batch_size} images")
            self.load_image_batch(0, initial_batch_size)
            
            # Update model with category information
            self.update_model_with_categories()
            
            # Also try to load visible images after a short delay
            QTimer.singleShot(100, self.load_visible_images)



    def save_classifications_json(self):
        """Save classifications in COCO-style JSON format"""
        if not self.npz_file_path:
            return
            
        json_path = self.get_classifications_json_path()
        coco_data = self.build_coco_data()
        
        # Save to JSON file
        with open(str(json_path), 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved COCO-style classifications to {json_path}")
        return str(json_path)
    

    def build_coco_data(self):
        """Delegate COCO build to io.classifications with persistent class IDs."""
        return io_build_coco(
            self.folder,
            self.embeddings,
            self.hidden_flags,
            self.classes,
            self.class_keystrokes,
            self.class_descriptions,
            self.image_categories,
            getattr(self, 'embedding_metadata', {}),
            getattr(self, 'cached_dimensions', {}),
            getattr(self, 'class_ids', {}),
        )
    
    def _choose_npz_in_folder(self, folder: Path) -> Path | None:
        """Pick the best NPZ in folder:
        1) Prefer *(_|-)224.npz
        2) Else largest trailing number *(_|-)<num>.npz
        3) Else common names
        4) Else most recent .npz
        """
        npzs = sorted(folder.glob("*.npz"))
        if not npzs:
            return None

        # Collect files with a trailing numeric suffix before .npz
        pat = re.compile(r'[_\-](\d+)\.npz$', re.IGNORECASE)
        numbered = []
        for p in npzs:
            m = pat.search(p.name)
            if m:
                numbered.append((int(m.group(1)), p))

        # 1) Prefer 224
        preferred_224 = [p for (n, p) in numbered if n == 224]
        if preferred_224:
            return max(preferred_224, key=lambda x: x.stat().st_mtime)

        # 2) Else largest numeric suffix
        if numbered:
            max_n = max(n for n, _ in numbered)
            best = [p for (n, p) in numbered if n == max_n]
            return max(best, key=lambda x: x.stat().st_mtime)

        # 3) Fallback common names
        for name in ("embeddings.npz", "symsorter_embeddings.npz"):
            cand = folder / name
            if cand.exists():
                return cand

        # 4) Most recent .npz
        return max(npzs, key=lambda p: p.stat().st_mtime)

    def _encode_folder_and_then_load(self, folder: Path):
        """Run `symsorter encode <folder>` (or fall back to `python -m`) with a progress dialog."""
        dlg = QProgressDialog("Encoding embeddings...", "Cancel", 0, 0, self)
        dlg.setWindowTitle("Encoding")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(True)
        dlg.setAutoReset(True)
        dlg.show()

        proc = QProcess(self)
        self._encode_process = proc
        self._encode_progress = dlg

        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self._on_encode_stdout(proc, dlg))

        def on_finished(exitCode, exitStatus):
            dlg.close()
            dlg.deleteLater()
            self._encode_process = None
            self._encode_progress = None
            if exitCode == 0:
                npz = self._choose_npz_in_folder(folder)
                if npz and npz.exists():
                    self.load_folder_from_path(str(npz))
                else:
                    QMessageBox.warning(self, "Encode finished", "Encoding finished, but no NPZ was found.")
            else:
                QMessageBox.critical(self, "Encode failed", f"Encoding failed with exit code {exitCode}.")

        proc.finished.connect(on_finished)

        def on_cancel():
            if proc.state() != QProcess.NotRunning:
                proc.terminate()
                QTimer.singleShot(2000, lambda: proc.kill() if proc.state() != QProcess.NotRunning else None)

        dlg.canceled.connect(on_cancel)

        # Prefer the console script `symsorter`, fallback to `python -m symsorter.cli`
        program = shutil.which("symsorter")
        if program:
            args = ["encode", str(folder)]
            proc.start(program, args)
        else:
            python = sys.executable
            args = ["-m", "symsorter.cli", "encode", str(folder)]
            proc.start(python, args)

        if not proc.waitForStarted(3000):
            dlg.reset()
            self._encode_process = None
            self._encode_progress = None
            QMessageBox.critical(self, "Encode failed", "Could not start encoder process.")

    def _on_encode_stdout(self, proc: QProcess, dlg: QProgressDialog):
        try:
            data = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        except Exception:
            return
        if not data:
            return
        # Show last non-empty line as status
        lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
        if lines:
            dlg.setLabelText(f"Encoding… {lines[-1]}")


