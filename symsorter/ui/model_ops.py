from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QIcon, QPixmap
import numpy as np
import os  # NEW

FILLER_ROLE = Qt.UserRole + 42  # kept for compatibility (we no longer create fillers)


class ModelOpsMixin:
    def _reset_item_maps(self):
        self.items_by_filename = {}

    def cosine_similarity(self, a, b) -> float:
        """Cosine similarity between two embedding vectors."""
        va = np.asarray(a, dtype=np.float32).ravel()
        vb = np.asarray(b, dtype=np.float32).ravel()
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0 or va.size != vb.size:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def _placeholder_icon(self):
        return getattr(self, "placeholder_icon", None)

    def _snapshot_icons_before_clear(self) -> dict[str, QIcon]:
        """Capture current icons by filename before clearing the model."""
        snap: dict[str, QIcon] = {}
        model = getattr(self, "model", None)
        if not model:
            return snap
        for row in range(model.rowCount()):
            it = model.item(row)
            if not it or it.data(FILLER_ROLE):
                continue
            fn = it.data(Qt.UserRole)
            if not fn:
                continue
            try:
                ic = it.icon()
                # Keep any QIcon; do NOT filter on availableSizes()
                if isinstance(ic, QIcon):
                    snap[fn] = ic
            except Exception:
                pass
        return snap

    def _get_icon_from_cache_or_raw(self, fn: str):
        """Try exact-size cached icon; else derive from raw cached image (no disk reload)."""
        icon_size = int(getattr(self, "icon_size", 128))
        crop_size = getattr(self, "crop_size", None)
        get_ic = getattr(self, "get_cached_icon", None)
        get_raw = getattr(self, "get_cached_raw_image", None)
        cache_ic = getattr(self, "cache_icon", None)

        # IMPORTANT: cache keys use basename (matches ImageLoadWorker)
        key_fn = os.path.basename(fn)

        # Exact-size icon
        if get_ic:
            ic = get_ic(key_fn, icon_size, crop_size)
            if ic:
                return ic

        # Derive from raw
        if get_raw:
            pm: QPixmap | None = get_raw(key_fn)
            if isinstance(pm, QPixmap):
                # Scale keeping aspect, no crop here (consistent with cache)
                scaled = pm.scaled(icon_size, icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                icon = QIcon(scaled)
                if cache_ic:
                    try:
                        cache_ic(key_fn, icon_size, crop_size, icon)
                    except Exception:
                        pass
                return icon
        return None

    def _set_icon_from_cache_if_available(self, fn, item):
        """Apply cached/derived icon; keep placeholder otherwise."""
        if not item or item.data(FILLER_ROLE):
            return
        icon = self._get_icon_from_cache_or_raw(fn)
        if icon:
            item.setIcon(icon)

    def refresh_icons_from_cache(self):
        """Replace placeholders with cached/derived icons for all non-filler items."""
        model = getattr(self, "model", None)
        if not model:
            return
        for row in range(model.rowCount()):
            it = model.item(row)
            if not it or it.data(FILLER_ROLE):
                continue
            fn = it.data(Qt.UserRole)
            if not fn:
                continue
            self._set_icon_from_cache_if_available(fn, it)

    def _append_item_for_file(self, fn):
        item = QStandardItem()
        item.setEditable(False)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setData(fn, Qt.UserRole)
        item.setData(False, FILLER_ROLE)  # real item, never a filler
        # Gray placeholder first
        ph = self._placeholder_icon()
        if ph:
            item.setIcon(ph)
        # Optional: annotate with classification/labels
        if hasattr(self, "update_item_with_classification"):
            self.update_item_with_classification(item, fn)
        # Track and append
        self.items_by_filename.setdefault(fn, []).append(item)
        self.model.appendRow(item)
        # Try cached icon immediately
        self._set_icon_from_cache_if_available(fn, item)

    def rebuild_model_after_sort(self):
        """Rebuild after sorting — no fillers; reuse cached icons; no reloads."""
        _icon_snapshot = self._snapshot_icons_before_clear()

        self.model.clear()
        self._reset_item_maps()
        self._row_map = []

        for fn in getattr(self, "image_files", []):
            self._row_map.append(self.model.rowCount())
            self._append_item_for_file(fn)
            # Reapply previous icon if cache misses (unconditional attempt)
            try:
                it = self.model.item(self.model.rowCount() - 1)
                snap_ic = _icon_snapshot.get(fn)
                if snap_ic:
                    it.setIcon(snap_ic)
            except Exception:
                pass

        if hasattr(self, "update_grid_layout"):
            self.update_grid_layout()

        # Apply cached/derived icons where available
        self.refresh_icons_from_cache()

        # Ensure any non-cached tiles get loaded now
        if hasattr(self, "load_visible_images"):
            self.load_visible_images()

    def rebuild_model(self, anchor_filename=None):
        """Legacy-compatible rebuild (no fillers); reuse cached icons; no reloads."""
        _icon_snapshot = self._snapshot_icons_before_clear()

        self.model.clear()
        self._reset_item_maps()
        for fn in getattr(self, "image_files", []):
            self._append_item_for_file(fn)
            # Reapply previous icon if cache misses (unconditional attempt)
            try:
                it = self.model.item(self.model.rowCount() - 1)
                snap_ic = _icon_snapshot.get(fn)
                if snap_ic:
                    it.setIcon(snap_ic)
            except Exception:
                pass

        if hasattr(self, "update_grid_layout"):
            self.update_grid_layout()
        self.refresh_icons_from_cache()

        # Ensure any non-cached tiles get loaded now
        if hasattr(self, "load_visible_images"):
            self.load_visible_images()

    def rebuild_model_after_filter(self):
        """Rebuild the model after a filter change; clears processed cache so only
        images matching the new filter are loaded."""
        self.stop_background_loading() if hasattr(self, "stop_background_loading") else None
        if hasattr(self, "_cache"):
            self._cache.image_cache.clear()
            self._cache.cache_access_order = [
                k for k in self._cache.cache_access_order
                if not isinstance(k, tuple)
            ]
        self.loaded_images.clear() if hasattr(self, "loaded_images") else None
        for w in getattr(self, "active_workers", []):
            w.stop()
        if hasattr(self, "active_workers"):
            self.active_workers.clear()
        if hasattr(self, "thread_pool"):
            self.thread_pool.clear()
        self.rebuild_model()

    def reset_order(self):
        """Restore original order and rebuild the model."""
        if not getattr(self, "original_order", None):
            return
        self.image_files = list(self.original_order)
        self.current_sort_mode = None
        if hasattr(self, "enable_class_group_wrapping"):
            self.enable_class_group_wrapping(False)
        self.rebuild_model()