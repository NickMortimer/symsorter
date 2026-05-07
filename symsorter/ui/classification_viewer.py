import json
import os
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QGridLayout, QLabel, QPushButton, QComboBox, 
                           QScrollArea, QFileDialog, QMessageBox, QFrame)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QFont

class ClassificationViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = None
        self.current_images = []
        self.current_page = 0
        self.images_per_page = 20
        self.image_size = 150
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Classification Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Load button
        load_btn = QPushButton("Load Classifications")
        load_btn.clicked.connect(self.load_file)
        controls_layout.addWidget(load_btn)
        
        # Filter combo box
        filter_label = QLabel("Filter by Category:")
        controls_layout.addWidget(filter_label)
        
        self.filter_combo = QComboBox()
        self.filter_combo.setMinimumWidth(200)
        controls_layout.addWidget(self.filter_combo)
        
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)
        
        # Status label
        self.status_label = QLabel("Load a classifications file to begin")
        self.status_label.setFont(QFont("Arial", 10))
        main_layout.addWidget(self.status_label)
        
        # Scroll area for images
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Grid widget
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        
        scroll_area.setWidget(self.grid_widget)
        main_layout.addWidget(scroll_area)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("Previous Page")
        self.prev_btn.clicked.connect(self.prev_page)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        
        self.page_label = QLabel("Page: 0/0")
        nav_layout.addWidget(self.page_label)
        
        self.next_btn = QPushButton("Next Page")
        self.next_btn.clicked.connect(self.next_page)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)
        
        nav_layout.addStretch()
        main_layout.addLayout(nav_layout)
    
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Classifications File", "", 
            "JSON files (*.json);;All files (*)")
        
        if file_path:
            self.load_classifications(file_path)
    
    def load_classifications(self, file_path):
        try:
            with open(file_path, 'r') as f:
                self.data = json.load(f)
            
            # Store base directory for loading images
            self.base_dir = os.path.dirname(file_path)
            
            # Setup filter combo after loading data
            self.setup_filter_combo()
            
            # Load first page
            self.current_images = self.data.get('images', [])
            self.current_page = 0
            self.update_grid()
            
            # Update window title
            self.setWindowTitle(f"Classification Viewer - {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    def setup_filter_combo(self):
        """Setup the filter combo box with category options."""
        self.filter_combo.clear()
        
        # Add "All" option
        self.filter_combo.addItem("All Categories", None)
        
        # Get categories from the data
        if hasattr(self, 'data') and self.data and 'categories' in self.data:
            categories = sorted(self.data['categories'], key=lambda x: x['name'])
            for category in categories:
                self.filter_combo.addItem(category['name'], category['id'])
        
        # Connect the combo box to filter function
        self.filter_combo.currentIndexChanged.connect(self.apply_category_filter)
    
    def apply_category_filter(self):
        """Apply category filter based on combo box selection."""
        if not hasattr(self, 'data') or not self.data:
            return
        
        # Get selected category ID
        selected_category_id = self.filter_combo.currentData()
        
        # Clear current grid
        self.clear_grid()
        
        # Filter images based on selection
        if selected_category_id is None:  # "All Categories" selected
            filtered_images = self.data.get('images', [])
        else:
            # Get annotations for the selected category
            category_image_ids = set()
            for annotation in self.data.get('annotations', []):
                if annotation['category_id'] == selected_category_id:
                    category_image_ids.add(annotation['image_id'])
            
            # Filter images that have annotations in this category
            filtered_images = [img for img in self.data.get('images', []) 
                              if img['id'] in category_image_ids]
        
        # Update grid with filtered images
        self.update_grid_with_images(filtered_images)
        
        # Update status
        total_images = len(self.data.get('images', []))
        filtered_count = len(filtered_images)
        if selected_category_id is None:
            status_text = f"Showing all {total_images} images"
        else:
            category_name = self.filter_combo.currentText()
            status_text = f"Showing {filtered_count} of {total_images} images (Category: {category_name})"
        
        if hasattr(self, 'status_label'):
            self.status_label.setText(status_text)
    
    def update_grid_with_images(self, images):
        """Update the grid with a specific list of images."""
        self.current_images = images
        self.current_page = 0
        self.update_grid()
    
    def clear_grid(self):
        """Clear all items from the grid layout."""
        for i in reversed(range(self.grid_layout.count())):
            child = self.grid_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
    
    def update_grid(self):
        self.clear_grid()
        
        if not self.current_images:
            return
        
        # Calculate pagination
        start_idx = self.current_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, len(self.current_images))
        page_images = self.current_images[start_idx:end_idx]
        
        # Get base directory for images (assuming they're in same dir as json)
        base_dir = getattr(self, 'base_dir', '')
        
        # Create grid of images
        cols = 4
        for i, image_info in enumerate(page_images):
            row = i // cols
            col = i % cols
            
            # Create image widget
            image_widget = self.create_image_widget(image_info, base_dir)
            self.grid_layout.addWidget(image_widget, row, col)
        
        # Update navigation
        self.update_navigation()
    
    def create_image_widget(self, image_info, base_dir):
        """Create a widget for displaying an image with its classification."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setMaximumSize(self.image_size + 20, self.image_size + 60)
        
        layout = QVBoxLayout(frame)
        layout.setSpacing(5)
        
        # Image label
        img_label = QLabel()
        img_label.setFixedSize(self.image_size, self.image_size)
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet("border: 1px solid gray;")
        
        # Try to load image
        img_path = os.path.join(base_dir, image_info['file_name'])
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(self.image_size, self.image_size, 
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_label.setPixmap(pixmap)
            else:
                img_label.setText("Invalid\nImage")
        else:
            img_label.setText("Image\nNot Found")
        
        layout.addWidget(img_label)
        
        # Classification label
        classification = self.get_image_classification(image_info['id'])
        class_label = QLabel(classification)
        class_label.setAlignment(Qt.AlignCenter)
        class_label.setWordWrap(True)
        class_label.setMaximumHeight(30)
        layout.addWidget(class_label)
        
        return frame
    
    def get_image_classification(self, image_id):
        """Get the classification for an image."""
        if not self.data:
            return "Unknown"
        
        # Find annotation for this image
        for annotation in self.data.get('annotations', []):
            if annotation['image_id'] == image_id:
                category_id = annotation['category_id']
                # Find category name
                for category in self.data.get('categories', []):
                    if category['id'] == category_id:
                        return category['name']
        
        return "Unclassified"
    
    def update_navigation(self):
        """Update navigation controls."""
        if not self.current_images:
            self.page_label.setText("Page: 0/0")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return
        
        total_pages = (len(self.current_images) - 1) // self.images_per_page + 1
        current_page_display = self.current_page + 1
        
        self.page_label.setText(f"Page: {current_page_display}/{total_pages}")
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page < total_pages - 1)
    
    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_grid()
    
    def next_page(self):
        total_pages = (len(self.current_images) - 1) // self.images_per_page + 1
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.update_grid()

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    viewer = ClassificationViewer()
    viewer.show()
    sys.exit(app.exec_())