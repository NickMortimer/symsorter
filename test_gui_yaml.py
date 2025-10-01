#!/usr/bin/env python3
"""
Test script to launch SymSorter GUI with YAML classes file
"""
import sys
from pathlib import Path
from symsorter.image_browser import ImageBrowser
from PySide6.QtWidgets import QApplication

def main():
    # Create QApplication
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    
    # Path to YAML classes file
    classes_file = Path(__file__).parent / "classes.yaml"
    
    # Create and show the image browser with YAML classes
    browser = ImageBrowser(class_file=str(classes_file))
    browser.show()
    
    print(f"Launched SymSorter GUI with classes from: {classes_file}")
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
