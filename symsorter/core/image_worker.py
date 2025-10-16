from PySide6.QtCore import QObject, Signal, QRunnable, QSize, Qt
from PySide6.QtGui import QPixmap, QIcon
import os


class WorkerSignals(QObject):
    """Signals for the worker thread"""
    imageLoaded = Signal(int, QIcon)
    error = Signal(str)


class ImageLoadWorker(QRunnable):
    """Worker for loading a single image in thread pool"""
    def __init__(self, index, file_path, icon_size, crop_size, cache_manager=None):
        super().__init__()
        self.index = index
        self.file_path = file_path
        self.icon_size = icon_size
        self.crop_size = crop_size  # None for no crop, or pixel size for center crop
        self.cache_manager = cache_manager  # Reference to main window for cache access
        self.should_stop = False
        self.signals = WorkerSignals()
    
    def stop(self):
        """Set stop flag for graceful shutdown"""
        self.should_stop = True
    
    def run(self):
        """Load and process image"""
        if self.should_stop:
            return
            
        try:
            filename = os.path.basename(self.file_path)
            
            # Check cache first
            if self.cache_manager:
                cached_icon = self.cache_manager.get_cached_icon(filename, self.icon_size, self.crop_size)
                if cached_icon:
                    self.signals.imageLoaded.emit(self.index, cached_icon)
                    return
            
            if self.should_stop:
                return
                
            # Load image
            if not os.path.exists(self.file_path):
                self.signals.error.emit(f"File not found: {self.file_path}")
                return
            
            # Try to get cached raw image first
            raw_pixmap = None
            if self.cache_manager:
                raw_pixmap = self.cache_manager.get_cached_raw_image(filename)
            
            if raw_pixmap is None:
                # Load from disk
                raw_pixmap = QPixmap(self.file_path)
                if raw_pixmap.isNull():
                    self.signals.error.emit(f"Failed to load image: {self.file_path}")
                    return
                
                # Cache the raw image if it's reasonably sized
                if self.cache_manager:
                    self.cache_manager.cache_raw_image(filename, raw_pixmap)
            
            if self.should_stop:
                return
            
            # Process the image (crop if needed, then scale)
            processed_pixmap = raw_pixmap
            
            # Apply crop if specified
            if self.crop_size is not None and self.crop_size > 0:
                # Calculate center crop
                orig_width = processed_pixmap.width()
                orig_height = processed_pixmap.height()
                
                # Calculate crop rectangle (center crop)
                crop_x = max(0, (orig_width - self.crop_size) // 2)
                crop_y = max(0, (orig_height - self.crop_size) // 2)
                crop_width = min(self.crop_size, orig_width)
                crop_height = min(self.crop_size, orig_height)
                
                processed_pixmap = processed_pixmap.copy(crop_x, crop_y, crop_width, crop_height)
            
            # Scale to thumbnail size - always scale to match the requested icon size
            if self.icon_size == processed_pixmap.width() and self.icon_size == processed_pixmap.height():
                # Already the exact size, no scaling needed
                scaled_pixmap = processed_pixmap
            else:
                # Scale to requested thumbnail size
                # Use smooth transformation for upscaling or significant downscaling
                if self.icon_size > processed_pixmap.width() or processed_pixmap.width() > self.icon_size * 2:
                    transformation = Qt.SmoothTransformation  # Smooth for upscaling or major downscaling
                else:
                    transformation = Qt.FastTransformation    # Fast for minor downscaling
                
                scaled_pixmap = processed_pixmap.scaled(
                    QSize(self.icon_size, self.icon_size),
                    Qt.KeepAspectRatio,
                    transformation
                )
            
            if self.should_stop:
                return
            
            # Create icon
            icon = QIcon(scaled_pixmap)
            
            # Cache the processed icon
            if self.cache_manager:
                self.cache_manager.cache_icon(filename, self.icon_size, self.crop_size, icon)
            
            # Emit the result
            self.signals.imageLoaded.emit(self.index, icon)
            
        except Exception as e:
            self.signals.error.emit(f"Error loading image {self.file_path}: {str(e)}")