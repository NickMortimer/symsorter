"""
GUI launcher for SymSorter
"""
import argparse
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from .image_browser import ImageBrowser

def main():
    """Main entry point for SymSorter GUI"""
    try:
        parser = argparse.ArgumentParser(prog="symsorter-gui", add_help=False)
        parser.add_argument("--slice-project", type=Path, default=None)
        args, qt_args = parser.parse_known_args(sys.argv[1:])
        app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
        browser = ImageBrowser()
        browser.show()
        if args.slice_project is not None:
            browser.open_slice_project(args.slice_project)
        sys.exit(app.exec())
    except ImportError as e:
        print("❌ GUI dependencies missing. Install with: pip install 'PySide6'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
