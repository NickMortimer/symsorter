from PySide6.QtCore import QSize, Qt, QObject, QEvent, QTimer
from PySide6.QtGui import QFontMetrics, QPixmap, QIcon, QColor

FILLER_ROLE = Qt.UserRole + 42


class ViewControlsMixin:
    def _sync_size_action_checks(self):
        """Keep thumbnail size menu checks aligned with current_size_name."""
        actions = getattr(self, "size_actions", None)
        if not actions:
            return
        current = getattr(self, "current_size_name", None)
        for action in actions:
            action_size = action.data() if action.data() is not None else action.text().replace("&", "")
            action.setChecked(action_size == current)

    def _sync_crop_action_checks(self):
        """Keep crop menu checks aligned with current_crop_name."""
        actions = getattr(self, "crop_actions", None)
        if not actions:
            return
        current = getattr(self, "current_crop_name", None)
        for action in actions:
            action_crop = action.data() if action.data() is not None else action.text().replace("&", "")
            action.setChecked(action_crop == current)

    def ensure_view_defaults(self):
        """Initialize all view-related defaults used by menus and loaders."""
        if not hasattr(self, "icon_sizes"):
            self.icon_sizes = {"Small": 96, "Medium": 128, "Large": 160, "XL": 196}
        if not hasattr(self, "current_size_name"):
            self.current_size_name = "Medium"
        if not hasattr(self, "icon_size"):
            self.icon_size = int(self.icon_sizes.get(self.current_size_name, 128))

        if not hasattr(self, "crop_sizes"):
            # Values MUST be smaller than the source image size to produce a
            # real centre-crop zoom effect.  224-px embeddings are the common
            # case, so 192 / 128 / 64 give slight / medium / strong zoom-in.
            self.crop_sizes = {"none": None, "192": 192, "128": 128, "64": 64}
        if not hasattr(self, "current_crop_name"):
            self.current_crop_name = "none"
        # Required by loader_ops
        self.crop_size = self.crop_sizes.get(self.current_crop_name, None)

        if not hasattr(self, "show_classification_labels"):
            self.show_classification_labels = False

        if not hasattr(self, "_blank_icon_cache"):
            self._blank_icon_cache = {}
        if not hasattr(self, "placeholder_icon"):
            self.update_placeholder_icon()

    def update_placeholder_icon(self, rgb=(60, 60, 60)):
        """Gray square for unloaded real items (not fillers)."""
        size = int(getattr(self, "icon_size", 128))
        pm = QPixmap(size, size)
        pm.fill(QColor(*rgb))
        self.placeholder_icon = QIcon(pm)

    def get_blank_tile_icon(self) -> QIcon:
        """Black square used only for filler grid padding."""
        size = int(getattr(self, "icon_size", 128))
        cache = getattr(self, "_blank_icon_cache", None) or {}
        self._blank_icon_cache = cache
        if size in cache:
            return cache[size]
        pm = QPixmap(size, size)
        pm.fill(QColor(0, 0, 0))  # pure black only for fillers
        icon = QIcon(pm)
        cache[size] = icon
        return icon

    def refresh_placeholder_icons(self):
        """Reapply placeholder icon to unloaded, non-filler items."""
        model = getattr(self, "model", None)
        if not model:
            return
        loaded = getattr(self, "loaded_images", set())
        for row in range(model.rowCount()):
            it = model.item(row)
            if not it or it.data(FILLER_ROLE):
                continue
            fn = it.data(Qt.UserRole)
            if not fn or fn in loaded:
                continue
            it.setIcon(self.placeholder_icon)

    def update_grid_layout(self):
        """Sync grid cell size to icon size and label visibility."""
        view = getattr(self, "view", None)
        if not view:
            return

        # Keep placeholder in sync with current size
        self.update_placeholder_icon()
        self.refresh_placeholder_icons()

        base_icon = int(getattr(self, "icon_size", 128))
        show_labels = bool(getattr(self, "show_classification_labels", False))

        fm = QFontMetrics(view.font())
        text_h = fm.height() if show_labels else 0
        padding_v = 6 if show_labels else 2

        grid_h = base_icon + text_h + padding_v
        padding_h = max(8, base_icon // 28)
        grid_w = base_icon + padding_h

        view.setIconSize(QSize(base_icon, base_icon))
        view.setGridSize(QSize(grid_w, grid_h))
        view.setSpacing(2)
        view.setTextElideMode(Qt.ElideRight)

        # Reload images at the new size (clears old-size cached thumbnails)
        if hasattr(self, "reload_images_with_new_size") and getattr(self, "model", None) and self.model.rowCount() > 0:
            self.reload_images_with_new_size()
        elif getattr(self, "current_sort_mode", None) == "class" and hasattr(self, "rebuild_model_after_sort"):
            # If grouped by class, regroup after layout tick (only if no reload happened)
            QTimer.singleShot(0, self.rebuild_model_after_sort)

    def get_columns_in_view(self) -> int:
        """Exact column count based on viewport width and spacing."""
        view = getattr(self, "view", None)
        if not view:
            return 1
        grid = view.gridSize()
        spacing = view.spacing()
        vw = view.viewport().contentsRect().width() or view.width() or grid.width()
        cell_w = max(1, grid.width())
        return max(1, (vw + spacing) // (cell_w + spacing))

    def enable_class_group_wrapping(self, enabled: bool):
        """Reflow groups on resize when sorting by class (debounced)."""
        view = getattr(self, "view", None)
        if not view:
            return
        if enabled and not getattr(self, "_group_wrap_enabled", False):
            self._group_wrap_enabled = True
            if not getattr(self, "_group_wrap_filter", None):
                self._group_wrap_filter = _GroupWrapResizeFilter(self)
                view.viewport().installEventFilter(self._group_wrap_filter)
        elif not enabled and getattr(self, "_group_wrap_enabled", False):
            self._group_wrap_enabled = False
            if getattr(self, "_group_wrap_filter", None):
                view.viewport().removeEventFilter(self._group_wrap_filter)
                self._group_wrap_filter = None

    def toggle_classification_labels(self, checked: bool):
        """Show/hide labels and refresh grid/items."""
        self.show_classification_labels = bool(checked)
        self.update_grid_layout()
        model = getattr(self, "model", None)
        if not model:
            return
        for row in range(model.rowCount()):
            item = model.item(row)
            if not item or item.data(FILLER_ROLE):
                continue
            fn = item.data(Qt.UserRole)
            if fn and hasattr(self, "update_item_with_classification"):
                self.update_item_with_classification(item, fn)

    # Thumbnail size handlers
    def increase_thumbnail_size(self):
        sizes = getattr(self, "icon_sizes", {"Small": 96, "Medium": 128, "Large": 160, "XL": 196})
        order = list(sizes.keys())
        cur = getattr(self, "current_size_name", "Medium")
        i = order.index(cur) if cur in order else 1
        if i < len(order) - 1:
            self.current_size_name = order[i + 1]
            self.icon_size = sizes[self.current_size_name]
            self._sync_size_action_checks()
            self.update_grid_layout()
            if hasattr(self, "update_class_info_label"):
                self.update_class_info_label()

    def decrease_thumbnail_size(self):
        sizes = getattr(self, "icon_sizes", {"Small": 96, "Medium": 128, "Large": 160, "XL": 196})
        order = list(sizes.keys())
        cur = getattr(self, "current_size_name", "Medium")
        i = order.index(cur) if cur in order else 1
        if i > 0:
            self.current_size_name = order[i - 1]
            self.icon_size = sizes[self.current_size_name]
            self._sync_size_action_checks()
            self.update_grid_layout()
            if hasattr(self, "update_class_info_label"):
                self.update_class_info_label()

    def create_size_handler(self, size_name: str):
        """Return a callable for menu actions to set a specific thumbnail size."""
        def handler():
            sizes = getattr(self, "icon_sizes", {"Small": 96, "Medium": 128, "Large": 160, "XL": 196})
            if size_name in sizes:
                self.current_size_name = size_name
                self.icon_size = sizes[size_name]
                self._sync_size_action_checks()
                self.update_grid_layout()
                if hasattr(self, "update_class_info_label"):
                    self.update_class_info_label()
        return handler

    # Crop zoom handlers (keep self.crop_size in sync)
    def _apply_crop_change(self):
        """Shared helper called after crop_size changes."""
        self._sync_crop_action_checks()
        if hasattr(self, "update_class_info_label"):
            self.update_class_info_label()
        if hasattr(self, "reload_images_with_new_crop"):
            self.reload_images_with_new_crop()
        elif hasattr(self, "load_visible_images"):
            self.loaded_images.clear()
            QTimer.singleShot(0, self.load_visible_images)

    def increase_crop_zoom(self):
        crops = getattr(self, "crop_sizes", {"none": None, "256": 256, "384": 384, "512": 512})
        names = list(crops.keys())
        cur = getattr(self, "current_crop_name", "none")
        i = names.index(cur) if cur in names else 0
        if i < len(names) - 1:
            self.current_crop_name = names[i + 1]
            self.crop_size = crops[self.current_crop_name]
            self._apply_crop_change()

    def decrease_crop_zoom(self):
        crops = getattr(self, "crop_sizes", {"none": None, "256": 256, "384": 384, "512": 512})
        names = list(crops.keys())
        cur = getattr(self, "current_crop_name", "none")
        i = names.index(cur) if cur in names else 0
        if i > 0:
            self.current_crop_name = names[i - 1]
            self.crop_size = crops[self.current_crop_name]
            self._apply_crop_change()

    def create_crop_handler(self, crop_name: str):
        """Return a callable for menu actions to set a specific crop size."""
        def handler():
            crops = getattr(self, "crop_sizes", {"none": None, "256": 256, "384": 384, "512": 512})
            if crop_name in crops:
                self.current_crop_name = crop_name
                self.crop_size = crops[crop_name]
                self._apply_crop_change()
        return handler


class _GroupWrapResizeFilter(QObject):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(80)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Resize, QEvent.LayoutRequest):
            if getattr(self.owner, "current_sort_mode", None) == "class":
                if hasattr(self.owner, "rebuild_model_after_sort"):
                    # restart debounce timer
                    try:
                        self._debounce.timeout.disconnect()
                    except Exception:
                        pass
                    self._debounce.timeout.connect(self.owner.rebuild_model_after_sort)
                    self._debounce.start()
        return False