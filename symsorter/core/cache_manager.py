from PySide6.QtGui import QIcon, QPixmap

class CacheManager:
    def __init__(self, max_cache_size=8000, max_raw_cache_size=1000):
        self.image_cache = {}          # (filename, icon_size, crop_size) -> QIcon
        self.raw_image_cache = {}      # filename -> QPixmap
        self.max_cache_size = max_cache_size
        self.max_raw_cache_size = max_raw_cache_size
        self.cache_access_order = []   # LRU order (mix of cache keys and filenames)

    def get_cache_key(self, filename, icon_size, crop_size):
        return (filename, icon_size, crop_size)

    def get_cached_icon(self, filename, icon_size, crop_size):
        key = self.get_cache_key(filename, icon_size, crop_size)
        if key in self.image_cache:
            if key in self.cache_access_order:
                self.cache_access_order.remove(key)
            self.cache_access_order.append(key)
            return self.image_cache[key]
        return None

    def cache_icon(self, filename, icon_size, crop_size, icon: QIcon):
        key = self.get_cache_key(filename, icon_size, crop_size)
        self.image_cache[key] = icon
        if key in self.cache_access_order:
            self.cache_access_order.remove(key)
        self.cache_access_order.append(key)
        while len(self.image_cache) > self.max_cache_size:
            oldest = self.cache_access_order.pop(0)
            if oldest in self.image_cache:
                del self.image_cache[oldest]

    def get_cached_raw_image(self, filename):
        if filename in self.raw_image_cache:
            if filename in self.cache_access_order:
                self.cache_access_order.remove(filename)
            self.cache_access_order.append(filename)
            return self.raw_image_cache[filename]
        return None

    def cache_raw_image(self, filename, pixmap: QPixmap):
        if pixmap.width() <= 1024 and pixmap.height() <= 1024:
            self.raw_image_cache[filename] = pixmap
            if filename in self.cache_access_order:
                self.cache_access_order.remove(filename)
            self.cache_access_order.append(filename)
            while len(self.raw_image_cache) > self.max_raw_cache_size:
                # Evict oldest raw item
                for old in list(self.cache_access_order):
                    if old in self.raw_image_cache:
                        del self.raw_image_cache[old]
                        self.cache_access_order.remove(old)
                        break

    def clear(self):
        self.image_cache.clear()
        self.raw_image_cache.clear()
        self.cache_access_order.clear()