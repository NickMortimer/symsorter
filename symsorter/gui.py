"""
GUI launcher for SymSorter
"""
import sys
from PySide6.QtWidgets import QApplication
from .image_browser import ImageBrowser

def main():
    """Main entry point for SymSorter GUI"""
    try:
        app = QApplication.instance() or QApplication(sys.argv)
        browser = ImageBrowser()
        browser.show()
        sys.exit(app.exec())
    except ImportError as e:
        print("❌ GUI dependencies missing. Install with: pip install 'PySide6'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
