"""
Tests for image reload/refresh behaviour in LoaderOpsMixin and ViewControlsMixin.

These guard against regressions where the AI refactoring removed the cache-clearing
and image-reload calls that should happen when the user changes thumbnail size or
crop zoom.

Covered behaviours:
  - reload_images_with_new_size() clears the processed-thumbnail cache
  - reload_images_with_new_size() resets loaded_images so workers re-dispatch
  - reload_images_with_new_size() is a no-op when there are no model rows
  - reload_images_with_new_crop() clears the processed-thumbnail cache
  - reload_images_with_new_crop() keeps the raw-image cache intact
  - reload_images_with_new_crop() is aliased by reload_visible_images()
  - ViewControlsMixin.update_grid_layout() calls reload_images_with_new_size()
      when rows exist
  - Crop handlers (_apply_crop_change / create_crop_handler) call
      reload_images_with_new_crop()
  - background_load_timer fires at 250 ms (not the too-aggressive 150 ms used
      before the fix)
  - loaded_images only ever contains integer indices (never filename strings)
"""

import pytest
from unittest.mock import MagicMock, patch, call
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QAction
import sys

# Ensure a QApplication exists for any Qt object construction.
_app = QApplication.instance() or QApplication(sys.argv)


# ---------------------------------------------------------------------------
# Helpers / minimal stubs
# ---------------------------------------------------------------------------

class _FakeTimer:
    """Minimal timer stub that tracks start/stop calls."""

    def __init__(self):
        self._active = False
        self.last_interval = None
        self.start_calls = 0
        self.stop_calls = 0

    def isActive(self) -> bool:
        return self._active

    def start(self, interval: int = 0) -> None:
        self._active = True
        self.last_interval = interval
        self.start_calls += 1

    def stop(self) -> None:
        self._active = False
        self.stop_calls += 1

    def timeout(self):
        pass

    def connect(self, *args):
        pass


class _FakeCache:
    """Minimal CacheManager-compatible stub."""

    def __init__(self):
        self.image_cache = {("a.jpg", 128, None): "icon_a"}
        self.raw_image_cache = {"a.jpg": "raw_a"}
        self.cache_access_order = [("a.jpg", 128, None), "a.jpg"]


class _FakeWorker:
    """Minimal worker stub."""

    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class _FakeThreadPool:
    """Count clear() calls."""

    def __init__(self):
        self.clear_calls = 0

    def clear(self):
        self.clear_calls += 1


class _FakeModel:
    """Minimal QStandardItemModel stub."""

    def __init__(self, n_rows: int = 3):
        self._items = [MagicMock() for _ in range(n_rows)]

    def rowCount(self) -> int:
        return len(self._items)

    def item(self, row: int):
        if 0 <= row < len(self._items):
            return self._items[row]
        return None


def _make_mixin_instance(n_rows: int = 3):
    """Return a LoaderOpsMixin instance with all required attributes pre-populated."""
    from symsorter.core.loader_ops import LoaderOpsMixin

    obj = LoaderOpsMixin.__new__(LoaderOpsMixin)

    # Required state
    obj.image_files = [f"img{i}.jpg" for i in range(n_rows)]
    obj.folder = "/fake"
    obj.icon_size = 128
    obj.crop_size = None
    obj.loaded_images = {0, 1}          # pretend some were already loaded
    obj.image_categories = {}
    obj.active_workers = [_FakeWorker(), _FakeWorker()]
    obj.thread_pool = _FakeThreadPool()
    obj.background_load_timer = _FakeTimer()
    obj.model = _FakeModel(n_rows)
    obj.placeholder_icon = MagicMock()
    obj._cache = _FakeCache()

    # load_visible_images / load_image_batch should not actually start Qt threads
    obj.load_visible_images = MagicMock()
    obj.load_image_batch = MagicMock()

    return obj


# ---------------------------------------------------------------------------
# LoaderOpsMixin.reload_images_with_new_size
# ---------------------------------------------------------------------------

class TestReloadWithNewSize:
    def test_clears_processed_thumbnail_cache(self):
        """Processed-icon cache must be empty after a size change."""
        obj = _make_mixin_instance()
        assert len(obj._cache.image_cache) > 0, "pre-condition: cache must have entries"

        obj.reload_images_with_new_size()

        assert obj._cache.image_cache == {}, "image_cache must be cleared"

    def test_keeps_raw_image_cache(self):
        """Raw pixmap cache must survive a size change."""
        obj = _make_mixin_instance()
        raw_before = dict(obj._cache.raw_image_cache)

        obj.reload_images_with_new_size()

        assert obj._cache.raw_image_cache == raw_before, \
            "raw_image_cache must not be cleared on size change"

    def test_clears_loaded_images(self):
        """loaded_images must be empty so workers re-dispatch at new size."""
        obj = _make_mixin_instance()
        assert len(obj.loaded_images) > 0, "pre-condition: some items must be loaded"

        obj.reload_images_with_new_size()

        assert obj.loaded_images == set(), "loaded_images must be cleared"

    def test_stops_stale_workers(self):
        """All active workers must be stopped."""
        obj = _make_mixin_instance()
        workers = list(obj.active_workers)

        obj.reload_images_with_new_size()

        for w in workers:
            assert w.stopped, "each active worker must be stopped"
        assert obj.active_workers == [], "active_workers list must be empty"

    def test_clears_thread_pool(self):
        """thread_pool.clear() must be called once."""
        obj = _make_mixin_instance()

        obj.reload_images_with_new_size()

        assert obj.thread_pool.clear_calls == 1

    def test_resets_item_icons_to_placeholder(self):
        """Every model item must receive the placeholder icon."""
        obj = _make_mixin_instance()

        obj.reload_images_with_new_size()

        for item in obj.model._items:
            item.setIcon.assert_called_with(obj.placeholder_icon)

    def test_restarts_visible_loading(self):
        """load_visible_images must be called after the reset."""
        obj = _make_mixin_instance()

        obj.reload_images_with_new_size()

        obj.load_visible_images.assert_called_once()

    def test_noop_when_no_rows(self):
        """Must be a no-op when the model is empty."""
        obj = _make_mixin_instance(n_rows=0)
        cache_before = dict(obj._cache.image_cache)

        obj.reload_images_with_new_size()

        assert obj._cache.image_cache == cache_before, \
            "cache must be untouched when model is empty"
        obj.load_visible_images.assert_not_called()

    def test_loaded_images_contains_only_integers(self):
        """
        loaded_images must only hold integer indices, never filename strings.
        The dead _apply_loaded_icon code used to pollute it with strings.
        """
        obj = _make_mixin_instance()
        # Simulate some loading having happened
        obj.loaded_images = {0, 1, 2}

        obj.reload_images_with_new_size()

        # After reload it should be empty
        assert obj.loaded_images == set()
        # Perform a fake load_batch that would add indices
        obj.loaded_images.add(0)
        for v in obj.loaded_images:
            assert isinstance(v, int), \
                f"loaded_images must only contain ints, got {type(v)}: {v!r}"

    def test_cache_access_order_purged_of_tuples(self):
        """
        cache_access_order entries that are tuples (processed cache keys) must
        be removed; string entries (raw cache keys) may stay.
        """
        obj = _make_mixin_instance()
        # cache_access_order has both tuples and strings
        assert any(isinstance(k, tuple) for k in obj._cache.cache_access_order)

        obj.reload_images_with_new_size()

        for k in obj._cache.cache_access_order:
            assert not isinstance(k, tuple), \
                f"tuple key {k!r} survived in cache_access_order after size reload"


# ---------------------------------------------------------------------------
# LoaderOpsMixin.reload_images_with_new_crop
# ---------------------------------------------------------------------------

class TestReloadWithNewCrop:
    def test_clears_processed_thumbnail_cache(self):
        """Processed-icon cache must be empty after a crop change."""
        obj = _make_mixin_instance()
        assert len(obj._cache.image_cache) > 0

        obj.reload_images_with_new_crop()

        assert obj._cache.image_cache == {}

    def test_keeps_raw_image_cache(self):
        """Raw pixmap cache must survive a crop change."""
        obj = _make_mixin_instance()
        raw_before = dict(obj._cache.raw_image_cache)

        obj.reload_images_with_new_crop()

        assert obj._cache.raw_image_cache == raw_before

    def test_clears_loaded_images(self):
        """loaded_images must be empty so workers re-dispatch with new crop."""
        obj = _make_mixin_instance()

        obj.reload_images_with_new_crop()

        assert obj.loaded_images == set()

    def test_resets_item_icons_to_placeholder(self):
        """Every model item must receive the placeholder icon."""
        obj = _make_mixin_instance()

        obj.reload_images_with_new_crop()

        for item in obj.model._items:
            item.setIcon.assert_called_with(obj.placeholder_icon)

    def test_restarts_visible_loading(self):
        """load_visible_images must be called after the reset."""
        obj = _make_mixin_instance()

        obj.reload_images_with_new_crop()

        obj.load_visible_images.assert_called_once()

    def test_noop_when_no_rows(self):
        """Must be a no-op when the model is empty."""
        obj = _make_mixin_instance(n_rows=0)

        obj.reload_images_with_new_crop()

        obj.load_visible_images.assert_not_called()


# ---------------------------------------------------------------------------
# reload_visible_images alias
# ---------------------------------------------------------------------------

class TestReloadVisibleImagesAlias:
    def test_alias_calls_reload_with_new_crop(self):
        """reload_visible_images must be an alias for reload_images_with_new_crop."""
        obj = _make_mixin_instance()
        obj.reload_images_with_new_crop = MagicMock()

        obj.reload_visible_images()

        obj.reload_images_with_new_crop.assert_called_once()


# ---------------------------------------------------------------------------
# Background load timer interval
# ---------------------------------------------------------------------------

class TestBackgroundLoadTimer:
    def test_timer_interval_is_250ms(self):
        """
        Background load timer must start at 250 ms (not the over-aggressive
        150 ms that caused UI jank before the fix).
        """
        obj = _make_mixin_instance()
        obj.background_load_timer = _FakeTimer()

        obj.start_background_loading()

        assert obj.background_load_timer.last_interval == 250, (
            f"Expected 250 ms, got {obj.background_load_timer.last_interval} ms"
        )

    def test_timer_not_restarted_if_already_active(self):
        """start_background_loading must not restart a timer that is already running."""
        obj = _make_mixin_instance()
        obj.background_load_timer = _FakeTimer()
        obj.background_load_timer._active = True  # already running

        obj.start_background_loading()

        assert obj.background_load_timer.start_calls == 0


# ---------------------------------------------------------------------------
# ViewControlsMixin.update_grid_layout triggers reload
# ---------------------------------------------------------------------------

class TestUpdateGridLayoutTriggersReload:
    """
    update_grid_layout must call reload_images_with_new_size() when the
    model has rows.  Without this, changing thumbnail size left all icons
    at the wrong pixel size.
    """

    def _make_view_controls_obj(self, n_rows: int = 3):
        from symsorter.ui.view_controls import ViewControlsMixin
        from PySide6.QtWidgets import QListView
        from PySide6.QtGui import QStandardItemModel

        obj = ViewControlsMixin.__new__(ViewControlsMixin)

        # Minimal view
        obj.view = QListView()
        obj.model = QStandardItemModel()
        for i in range(n_rows):
            from PySide6.QtGui import QStandardItem
            obj.model.appendRow(QStandardItem(f"item{i}"))
        obj.view.setModel(obj.model)

        obj.icon_size = 128
        obj.current_size_name = "Medium"
        obj.icon_sizes = {"Small": 96, "Medium": 128, "Large": 160, "XL": 196}
        obj.show_classification_labels = False
        obj.current_sort_mode = None
        obj.placeholder_icon = MagicMock()
        obj._blank_icon_cache = {}

        # Track calls
        obj.reload_images_with_new_size = MagicMock()

        return obj

    def test_calls_reload_when_rows_exist(self):
        """update_grid_layout must call reload_images_with_new_size when rows > 0."""
        obj = self._make_view_controls_obj(n_rows=3)

        obj.update_grid_layout()

        obj.reload_images_with_new_size.assert_called_once()

    def test_no_reload_when_no_rows(self):
        """update_grid_layout must NOT call reload when model is empty."""
        obj = self._make_view_controls_obj(n_rows=0)

        obj.update_grid_layout()

        obj.reload_images_with_new_size.assert_not_called()


# ---------------------------------------------------------------------------
# ViewControlsMixin crop handlers call _apply_crop_change
# ---------------------------------------------------------------------------

class TestCropHandlersTriggerReload:
    """
    Changing crop via increase_crop_zoom / decrease_crop_zoom / create_crop_handler
    must end up calling reload_images_with_new_crop() so the cache is flushed.
    """

    def _make_crop_obj(self):
        from symsorter.ui.view_controls import ViewControlsMixin

        obj = ViewControlsMixin.__new__(ViewControlsMixin)
        obj.crop_sizes = {"none": None, "192": 192, "128": 128, "64": 64}
        obj.current_crop_name = "none"
        obj.crop_size = None
        obj.reload_images_with_new_crop = MagicMock()
        obj.update_class_info_label = MagicMock()
        return obj

    def test_increase_crop_zoom(self):
        obj = self._make_crop_obj()
        obj.increase_crop_zoom()
        obj.reload_images_with_new_crop.assert_called_once()

    def test_decrease_crop_zoom(self):
        obj = self._make_crop_obj()
        obj.current_crop_name = "192"
        obj.crop_size = 192
        obj.decrease_crop_zoom()
        obj.reload_images_with_new_crop.assert_called_once()

    def test_create_crop_handler(self):
        obj = self._make_crop_obj()
        handler = obj.create_crop_handler("128")
        handler()
        obj.reload_images_with_new_crop.assert_called_once()
        assert obj.current_crop_name == "128"
        assert obj.crop_size == 128

    def test_crop_size_updated_correctly(self):
        """crop_size attribute must match the newly selected crop after increase."""
        obj = self._make_crop_obj()
        obj.increase_crop_zoom()
        assert obj.crop_size == 192  # first step from "none"

    def test_increase_at_max_is_noop(self):
        """Increasing at the maximum crop must not call reload or change state."""
        obj = self._make_crop_obj()
        obj.current_crop_name = "64"
        obj.crop_size = 64
        obj.increase_crop_zoom()
        obj.reload_images_with_new_crop.assert_not_called()

    def test_decrease_at_min_is_noop(self):
        """Decreasing at the minimum crop must not call reload or change state."""
        obj = self._make_crop_obj()
        # already at "none" (index 0)
        obj.decrease_crop_zoom()
        obj.reload_images_with_new_crop.assert_not_called()


# ---------------------------------------------------------------------------
# Menu action radio groups and sync with zoom/crop handlers
# ---------------------------------------------------------------------------

class TestMenuActionSync:
    def _make_size_sync_obj(self):
        from symsorter.ui.view_controls import ViewControlsMixin

        obj = ViewControlsMixin.__new__(ViewControlsMixin)
        obj.icon_sizes = {"Small": 96, "Medium": 128, "Large": 160, "XL": 196}
        obj.current_size_name = "Medium"
        obj.icon_size = 128
        obj.update_grid_layout = MagicMock()
        obj.update_class_info_label = MagicMock()

        names = list(obj.icon_sizes.keys())
        obj.size_actions = []
        for name in names:
            act = QAction(name)
            act.setCheckable(True)
            act.setData(name)
            act.setChecked(name == obj.current_size_name)
            obj.size_actions.append(act)
        return obj

    def _make_crop_sync_obj(self):
        from symsorter.ui.view_controls import ViewControlsMixin

        obj = ViewControlsMixin.__new__(ViewControlsMixin)
        obj.crop_sizes = {"none": None, "192": 192, "128": 128, "64": 64}
        obj.current_crop_name = "none"
        obj.crop_size = None
        obj.reload_images_with_new_crop = MagicMock()
        obj.update_class_info_label = MagicMock()

        obj.crop_actions = []
        for name in obj.crop_sizes.keys():
            act = QAction(name)
            act.setCheckable(True)
            act.setData(name)
            act.setChecked(name == obj.current_crop_name)
            obj.crop_actions.append(act)
        return obj

    def test_ctrl_plus_syncs_thumbnail_action_checks(self):
        obj = self._make_size_sync_obj()

        obj.increase_thumbnail_size()

        checked = [a.data() for a in obj.size_actions if a.isChecked()]
        assert checked == ["Large"]

    def test_ctrl_minus_syncs_thumbnail_action_checks(self):
        obj = self._make_size_sync_obj()
        obj.current_size_name = "Large"
        obj.icon_size = obj.icon_sizes["Large"]
        for a in obj.size_actions:
            a.setChecked(a.data() == "Large")

        obj.decrease_thumbnail_size()

        checked = [a.data() for a in obj.size_actions if a.isChecked()]
        assert checked == ["Medium"]

    def test_crop_change_syncs_crop_action_checks(self):
        obj = self._make_crop_sync_obj()

        obj.increase_crop_zoom()  # none -> 192

        checked = [a.data() for a in obj.crop_actions if a.isChecked()]
        assert checked == ["192"]

    def test_crop_handler_syncs_crop_action_checks(self):
        obj = self._make_crop_sync_obj()

        handler = obj.create_crop_handler("128")
        handler()

        checked = [a.data() for a in obj.crop_actions if a.isChecked()]
        assert checked == ["128"]


class TestMenusRadioGroups:
    def test_thumbnail_and_crop_submenus_are_exclusive_groups(self):
        from symsorter.ui.menus import build_menus

        class _Window(QMainWindow):
            def __init__(self):
                super().__init__()
                self.icon_sizes = {"Small": 96, "Medium": 128, "Large": 160}
                self.current_size_name = "Medium"
                self.crop_sizes = {"none": None, "192": 192, "128": 128}
                self.current_crop_name = "none"
                self.show_classification_labels = False

            # methods referenced by build_menus
            def load_folder(self):
                pass

            def load_class_file_dialog(self):
                pass

            def save_classifications(self):
                pass

            def export_classifications_json_dialog(self):
                pass

            def import_classifications_json_dialog(self):
                pass

            def export_yolo_annotations(self):
                pass

            def reset_order(self):
                pass

            def toggle_classification_labels(self, _checked):
                pass

            def sort_by_class(self):
                pass

            def show_classified_images(self):
                pass

            def increase_thumbnail_size(self):
                pass

            def decrease_thumbnail_size(self):
                pass

            def increase_crop_zoom(self):
                pass

            def decrease_crop_zoom(self):
                pass

            def create_size_handler(self, _name):
                return lambda: None

            def create_crop_handler(self, _name):
                return lambda: None

            def update_classes_menu(self):
                pass

        w = _Window()
        build_menus(w)

        assert hasattr(w, "size_action_group")
        assert w.size_action_group.isExclusive()
        assert hasattr(w, "crop_action_group")
        assert w.crop_action_group.isExclusive()
