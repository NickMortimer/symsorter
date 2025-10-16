from PySide6.QtCore import QThread, QSize, Signal, Qt
from PySide6.QtGui import QPixmap, QIcon
import os


class LazyImageLoader(QThread):
    """Thread for lazy loading images as needed"""
    imageLoaded = Signal(int, QIcon)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.images_to_load = []
        self.folder = ""
        self.icon_size = 120
        self.crop_size = None  # No crop by default
        
    def run(self):
        for index, filename in self.images_to_load:
            file_path = os.path.join(self.folder, filename)
            
            # Load and scale image
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # Apply crop if specified
                if self.crop_size is not None and self.crop_size > 0:
                    # Calculate center crop
                    orig_width = pixmap.width()
                    orig_height = pixmap.height()
                    
                    # Calculate crop rectangle (center crop)
                    crop_x = max(0, (orig_width - self.crop_size) // 2)
                    crop_y = max(0, (orig_height - self.crop_size) // 2)
                    crop_width = min(self.crop_size, orig_width)
                    crop_height = min(self.crop_size, orig_height)
                    
                    pixmap = pixmap.copy(crop_x, crop_y, crop_width, crop_height)
                
                # Scale to thumbnail size
                scaled_pixmap = pixmap.scaled(
                    QSize(self.icon_size, self.icon_size),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                icon = QIcon(scaled_pixmap)
                self.imageLoaded.emit(index, icon)