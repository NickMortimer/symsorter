from PySide6.QtCore import Qt, QModelIndex, QTimer, QThreadPool  # add QThreadPool
from PySide6.QtGui import QStandardItemModel, QDesktopServices, QIcon, QPixmap, QColor
from PySide6.QtWidgets import QMainWindow, QListView, QAbstractItemView
from PySide6.QtWidgets import (  # add QProgressDialog
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QAbstractItemView, QSlider, QLabel, QComboBox,
    QToolBar, QMenuBar, QMainWindow, QMessageBox, QProgressDialog
)
import numpy as np
from pathlib import Path
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
from .core.assignment_ops import AssignmentOpsMixin  # use the original core mixin
from .io.export_dialogs import ExportDialogsMixin
from .core.folder_flow import FolderFlowMixin
from .ui.lifecycle import LifecycleMixin
from PySide6.QtCore import QTimer
from collections.abc import Iterable

# Single source of truth for filler role (matches mixins)
FILLER_ROLE = Qt.UserRole + 42  # same role used in ModelOps

class ImageBrowser(QMainWindow,
                   LoaderOpsMixin, ViewControlsMixin, ModelOpsMixin,
                   BackupMixin, ClassUIMixin, AssignmentOpsMixin,  # keep in MRO
                   ExportDialogsMixin, FolderFlowMixin, LifecycleMixin):
    def __init__(self, class_file=None):
        super().__init__()
        self.resize(1000, 700)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Image grid view
        self.view = QListView(self)
        self.view.setViewMode(QListView.IconMode)
        self.view.setResizeMode(QListView.Adjust)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.view.setSelectionRectVisible(True)
        self.view.doubleClicked.connect(self.on_double_click)

        self.model = QStandardItemModel(self.view)
        self.view.setModel(self.model)

        # ADD: put the view into the central layout so it shows up
        layout.addWidget(self.view)

        # Defaults needed by menus/handlers
        # XL/XXL are large enough for in-grid SAM point/box prompting.
        self.icon_sizes = {"Small": 96, "Medium": 128, "Large": 160, "XL": 224, "XXL": 336}
        self.current_size_name = getattr(self, "current_size_name", "XL")
        self.icon_size = self.icon_sizes[self.current_size_name]
        self.update_placeholder_icon()
        self.show_classification_labels = False
        self.current_sort_mode = None

        # Ensure all view defaults exist BEFORE menus/actions use them
        self.ensure_view_defaults()
        # Also set crop_size explicitly in case menus changed it before
        self.crop_size = self.crop_sizes.get(self.current_crop_name, None)

        # REMOVE the duplicate block that re-creates self.view and self.model
        # self.view = QListView(self)
        # ...
        # self.model = QStandardItemModel(self.view)
        # self.view.setModel(self.model)

        # Keep placeholder icon in sync with current size
        self.update_placeholder_icon()

        # Thread pool and caches
        self.thread_pool = QThreadPool.globalInstance()  # Initialize image loading thread pool
        max_threads = max(4, self.thread_pool.maxThreadCount())
        self.thread_pool.setMaxThreadCount(max_threads)
        print(f"Image loading thread pool initialized with {max_threads} threads")

        self.loaded_images = set()
        self.original_order = []

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
        # Defer class loading until UI is initialized
        self.class_file_path = class_file
        
        # Track unsaved changes
        self.has_unsaved_changes = False
        
        # YOLO class management
        self.classes = []
        self.class_keystrokes = {}
        self.class_descriptions = {}
        self.image_categories = {}
        self.last_used_class_idx = None
        self.bboxes = {}
        self.gps = {}
        self.classification_viewer = None  # Classification viewer window
        self.show_classification_labels = False
        self.show_outline_overlays = True  # draw sibling LabelMe polygons on grid thumbs
        self.outline_sam_model = "sam_b.pt"
        self.outline_sam_device = None  # Ultralytics default (mps/cpu)
        self.current_sort_mode = None  # 'class' when sorting by class
        
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

        # Connect scroll event for lazy loading
        self.view.verticalScrollBar().valueChanged.connect(self.on_scroll)
        # REMOVE duplicate double-click connect here (it caused re-entrancy)
        # self.view.doubleClicked.connect(self.on_double_click)

        # Ensure double-click is connected once
        try:
            self.view.doubleClicked.disconnect(self.on_double_click)
        except Exception:
            pass
        self.view.doubleClicked.connect(self.on_double_click)

        # In-grid SAM point/box prompting (works best at XL/XXL)
        from .ui.grid_outline_prompt import GridOutlinePromptFilter

        self._grid_outline_prompt = GridOutlinePromptFilter(self)
        self._grid_outline_prompt.ensure_installed()

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
        self.class_filter_combo.setCurrentText("All Images")
        self.class_filter_combo.blockSignals(False)

        # Class info label
        self.class_info_label = QLabel("No classes loaded")
        layout.addWidget(self.class_info_label)

        # Now it's safe to load the class file (UI attrs exist)
        if self.class_file_path:
            self.load_class_file(self.class_file_path)

        # Ensure the combo is populated after UI idle
        if hasattr(self, "schedule_filter_combo_refresh"):
            QTimer.singleShot(0, self.schedule_filter_combo_refresh)

        # Set initial window title
        self.update_window_title()
    
    def setup_menus_and_toolbar(self):
        from .ui.menus import build_menus
        return build_menus(self)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts for class assignment"""        
        # Class shortcuts come from Classes menu actions
        self.function_key_shortcuts = []

    def open_outline_viewer(self):
        """Show sibling LabelMe polygons for the selected thumb(s)."""
        from .ui.outline_dialog import open_outline_viewer_for_browser

        open_outline_viewer_for_browser(self)
    
    def create_class_assignment_handler(self, class_idx):
        """Create a proper closure for class assignment"""
        def handler():
            print(f"DEBUG: Shortcut activated for class index {class_idx}")
            self.assign_class_to_selected(class_idx)
        return handler
    

    # Cache wrappers
    def _outline_cache_variant(self):
        return "outline" if getattr(self, "show_outline_overlays", False) else None

    def get_cache_key(self, filename, icon_size, crop_size, variant=None):
        if variant is None:
            variant = self._outline_cache_variant()
        return self._cache.get_cache_key(filename, icon_size, crop_size, variant=variant)

    def get_cached_icon(self, filename, icon_size, crop_size, variant=None):
        if variant is None:
            variant = self._outline_cache_variant()
        return self._cache.get_cached_icon(
            filename, icon_size, crop_size, variant=variant
        )

    def cache_icon(self, filename, icon_size, crop_size, icon, variant=None):
        if variant is None:
            variant = self._outline_cache_variant()
        return self._cache.cache_icon(
            filename, icon_size, crop_size, icon, variant=variant
        )

    def get_cached_raw_image(self, filename):
        return self._cache.get_cached_raw_image(filename)

    def cache_raw_image(self, filename, pixmap):
        return self._cache.cache_raw_image(filename, pixmap)

    def toggle_outline_overlays(self, checked=None):
        """Toggle polygon outlines on grid thumbnails and reload icons."""
        if checked is None:
            checked = not getattr(self, "show_outline_overlays", False)
        self.show_outline_overlays = bool(checked)
        if hasattr(self, "show_outline_overlays_action"):
            self.show_outline_overlays_action.setChecked(self.show_outline_overlays)
        # Drop processed icons so workers rebuild with/without overlays.
        if hasattr(self, "_cache"):
            self._cache.image_cache.clear()
            self._cache.cache_access_order = [
                k for k in self._cache.cache_access_order if not isinstance(k, tuple)
            ]
        if hasattr(self, "reload_images_with_new_size"):
            self.reload_images_with_new_size()
        tip = "on" if self.show_outline_overlays else "off"
        if hasattr(self, "statusBar") and self.statusBar():
            self.statusBar().showMessage(f"Outline overlays {tip}", 3000)

    def set_outline_prompt_mode(self, mode: str) -> None:
        """Enable in-grid outline tools: ``off`` | ``point`` | ``box`` | ``delete``."""
        if not hasattr(self, "_grid_outline_prompt"):
            return
        self._grid_outline_prompt.set_mode(mode)
        # Keep menu checks in sync if present
        for name, value in (
            ("outline_prompt_off_action", "off"),
            ("outline_prompt_point_action", "point"),
            ("outline_prompt_box_action", "box"),
            ("outline_prompt_delete_action", "delete"),
        ):
            act = getattr(self, name, None)
            if act is not None:
                act.setChecked(mode == value)

    def refresh_outline_icon(self, filename: str) -> None:
        """Drop cached icons for one thumb and reload it (after SAM save)."""
        import os

        key_fn = os.path.basename(filename)
        if hasattr(self, "_cache"):
            drop = [
                k
                for k in list(self._cache.image_cache.keys())
                if isinstance(k, tuple) and k and k[0] == key_fn
            ]
            for k in drop:
                self._cache.image_cache.pop(k, None)
                if k in self._cache.cache_access_order:
                    self._cache.cache_access_order.remove(k)

        # Find model row and re-queue load
        model = getattr(self, "model", None)
        if not model:
            return
        for row in range(model.rowCount()):
            it = model.item(row)
            if not it:
                continue
            fn = it.data(Qt.UserRole)
            if fn != filename and os.path.basename(str(fn)) != key_fn:
                continue
            # Clear to placeholder then load
            if hasattr(self, "placeholder_icon"):
                it.setIcon(self.placeholder_icon)
            folder = getattr(self, "folder", "") or ""
            file_path = os.path.join(folder, key_fn) if folder else key_fn
            if hasattr(self, "load_image_batch"):
                # Reuse single-index load path
                from .core.image_worker import ImageLoadWorker

                worker = ImageLoadWorker(
                    row, file_path, self.icon_size, self.crop_size, cache_manager=self
                )
                worker.signals.imageLoaded.connect(self.update_image)
                self.thread_pool.start(worker)
            break

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



    def load_class_file(self, class_file_path):
        """Load classes via ClassManager and update UI."""
        print(f"DEBUG: Loading class file: {class_file_path}")
        ok = self._classes.load(class_file_path)
        if not ok:
            print("DEBUG: Failed to load class file")
            return
            
        # Sync local attributes
        self.classes = self._classes.classes
        self.class_keystrokes = self._classes.class_keystrokes
        self.class_descriptions = self._classes.class_descriptions
        self.class_ids = self._classes.class_ids
        self.class_file_path = self._classes.class_file_path
        
        print(f"DEBUG: Classes loaded: {self.classes}")
        
        # Update UI components
        self.update_class_info_label()
        self.update_classes_menu()
        # Refresh filter combo after UI idle
        if hasattr(self, "schedule_filter_combo_refresh"):
            self.schedule_filter_combo_refresh()

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
    
    def load_classes_dialog(self):
        """Show dialog to load class file (alternative method name)"""
        self.load_class_file_dialog()

    def on_class_filter_changed(self, filter_text):
        """Handle class filter combo box changes — updates image_files and reloads the view."""
        self.current_sort_mode = None
        if hasattr(self, "enable_class_group_wrapping"):
            self.enable_class_group_wrapping(False)
        self.apply_class_filter(filter_text)  # sets self.image_files and calls rebuild_model_after_filter

    def sort_by_class(self):
        """Sort images by class without inserting blank tiles."""
        if not self.image_files:
            return

        def key_fn(fn):
            if fn in self.image_categories:
                info = self.image_categories[fn]
                return (0, info.get('class_name', '').lower(), info.get('class_id', 999), fn)
            return (1, 'zzz_unclassified', 999, fn)

        self.image_files.sort(key=key_fn)
        self.current_sort_mode = "class"
        # Disable/ignore grouping wrapper (no blank tiles)
        if hasattr(self, "enable_class_group_wrapping"):
            self.enable_class_group_wrapping(False)
        self.rebuild_model_after_sort()

    def sort_by_outline_needs_work(self):
        """Sort: unlocked/empty outlines first, quality-locked (done) last."""
        if not self.image_files:
            return
        from pathlib import Path

        from symsorter.outline.export_labelme import (
            labelme_path_for_thumb,
            outline_work_status,
        )

        folder = Path(getattr(self, "folder", "") or "")

        def key_fn(fn):
            status = outline_work_status(labelme_path_for_thumb(folder / fn))
            # empty & needs_work first (0), done last (1)
            bucket = 0 if status != "done" else 1
            return (bucket, status, fn)

        self.image_files.sort(key=key_fn)
        self.current_sort_mode = "outline_needs_work"
        if hasattr(self, "enable_class_group_wrapping"):
            self.enable_class_group_wrapping(False)
        self.rebuild_model_after_sort()
        n_todo = sum(
            1
            for fn in self.image_files
            if outline_work_status(labelme_path_for_thumb(folder / fn)) != "done"
        )
        if hasattr(self, "statusBar") and self.statusBar():
            self.statusBar().showMessage(
                f"Sorted by outline needs-work — {n_todo} still need review",
                5000,
            )

    def on_double_click(self, index: QModelIndex):
        """Global similarity sort, then restrict to current filter for the view."""
        if not index.isValid() or not getattr(self, "embeddings", None):
            return

        item = self.model.itemFromIndex(index)
        if item and item.data(FILLER_ROLE):
            return  # ignore filler tiles

        anchor = item.data(Qt.UserRole) if item else None
        if not anchor or anchor not in self.embeddings:
            return

        # Current filtered set to display (preserve filter)
        filtered_set = set(getattr(self, "image_files", []) or [])

        # Candidate pool: all known images. Prefer a canonical list if present.
        all_candidates = (
            list(getattr(self, "all_image_files", []))     # if you maintain a full list
            or list(getattr(self, "original_order", []))   # fallback to original order
            or list(self.embeddings.keys())                # last resort: all with embeddings
        )

        # Compute similarity for all candidates (push missing embeddings to end)
        ref = self.embeddings[anchor]
        sims = []
        for fn in all_candidates:
            vec = self.embeddings.get(fn)
            if vec is None:
                sims.append((fn, -1.0))
            else:
                sims.append((fn, self.cosine_similarity(ref, vec)))

        sims.sort(key=lambda x: x[1], reverse=True)

        # Restrict the view to the current filter (no new images added)
        ordered_view = [fn for fn, _ in sims if fn in filtered_set]

        # If any filtered items didn’t get scored, append them preserving current order
        if len(ordered_view) != len(filtered_set):
            current_order = list(getattr(self, "image_files", []))
            seen = set(ordered_view)
            ordered_view.extend([fn for fn in current_order if fn not in seen])

        # Update the view list only (filter preserved)
        self.image_files = ordered_view

        # Rebuild as plain list (no class fillers), using cache
        self.current_sort_mode = None
        if hasattr(self, "enable_class_group_wrapping"):
            self.enable_class_group_wrapping(False)

        # Preserve scroll and selection
        vsb = self.view.verticalScrollBar()
        pos = vsb.value()
        self.rebuild_model()
        QTimer.singleShot(0, lambda: vsb.setValue(pos))

        def _reselect():
            for row in range(self.model.rowCount()):
                it = self.model.item(row)
                if it and it.data(Qt.UserRole) == anchor:
                    self.view.setCurrentIndex(self.model.index(row, 0))
                    break
        QTimer.singleShot(0, _reselect)