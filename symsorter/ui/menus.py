from PySide6.QtWidgets import QToolBar, QFileDialog
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtCore import Qt


def build_menus(window):
    """Build menus and toolbar for the given window (delegated from ImageBrowser)."""
    menubar = window.menuBar()

    # File menu
    file_menu = menubar.addMenu('&File')

    # Load embeddings action
    load_action = QAction('&Load Embeddings...', window)
    load_action.setShortcut(QKeySequence('Ctrl+O'))
    load_action.setStatusTip('Load embeddings from NPZ file')
    load_action.triggered.connect(window.load_folder)
    file_menu.addAction(load_action)

    # Load class file action
    load_classes_action = QAction('Load &Class File...', window)
    load_classes_action.setShortcut(QKeySequence('Ctrl+Shift+O'))
    load_classes_action.setStatusTip('Load YOLO class names file')
    load_classes_action.triggered.connect(window.load_class_file_dialog)
    file_menu.addAction(load_classes_action)

    file_menu.addSeparator()

    # Save action
    save_action = QAction('&Save Classifications', window)
    save_action.setShortcut(QKeySequence('Ctrl+S'))
    save_action.setStatusTip('Save classifications and hidden flags')
    save_action.triggered.connect(window.save_classifications)
    file_menu.addAction(save_action)

    file_menu.addSeparator()

    # Export JSON action
    export_json_action = QAction('Export &Classifications (JSON)...', window)
    export_json_action.setShortcut(QKeySequence('Ctrl+J'))
    export_json_action.setStatusTip('Export classifications as COCO-style JSON')
    export_json_action.triggered.connect(window.export_classifications_json_dialog)
    file_menu.addAction(export_json_action)

    # Import JSON action
    import_json_action = QAction('&Import Classifications (JSON)...', window)
    import_json_action.setShortcut(QKeySequence('Ctrl+Shift+J'))
    import_json_action.setStatusTip('Import classifications from COCO-style JSON')
    import_json_action.triggered.connect(window.import_classifications_json_dialog)
    file_menu.addAction(import_json_action)

    # Export YOLO action
    export_action = QAction('Export &YOLO Annotations...', window)
    export_action.setShortcut(QKeySequence('Ctrl+E'))
    export_action.setStatusTip('Export classifications as YOLO annotations')
    export_action.triggered.connect(window.export_yolo_annotations)
    file_menu.addAction(export_action)

    # Open folder action
    open_folder_act = QAction("Open &Folder…", window)
    open_folder_act.triggered.connect(lambda: (
        setattr(window, '_d', QFileDialog.getExistingDirectory(window, "Select Image Folder") or None),
        window.load_folder_from_path(window._d) if window._d else None
    ))
    file_menu.addAction(open_folder_act)

    # View menu
    view_menu = menubar.addMenu('&View')

    # Reset order action
    reset_action = QAction('&Reset Order', window)
    reset_action.setShortcut(QKeySequence('R'))
    reset_action.setStatusTip('Reset images to original order')
    reset_action.triggered.connect(window.reset_order)
    view_menu.addAction(reset_action)

    # Show classified action
    show_classified_action = QAction('Show &Classified Images', window)
    show_classified_action.setShortcut(QKeySequence('Ctrl+Shift+C'))
    show_classified_action.setStatusTip('Show all classified images')
    show_classified_action.triggered.connect(window.show_classified_images)
    view_menu.addAction(show_classified_action)

    view_menu.addSeparator()

    # Thumbnail size controls
    increase_size_action = QAction('&Increase Thumbnail Size', window)
    increase_size_action.setShortcut(QKeySequence('Ctrl+='))
    increase_size_action.setStatusTip('Increase thumbnail size (Ctrl++)')
    increase_size_action.triggered.connect(window.increase_thumbnail_size)
    view_menu.addAction(increase_size_action)

    decrease_size_action = QAction('&Decrease Thumbnail Size', window)
    decrease_size_action.setShortcut(QKeySequence('Ctrl+-'))
    decrease_size_action.setStatusTip('Decrease thumbnail size (Ctrl+-)')
    decrease_size_action.triggered.connect(window.decrease_thumbnail_size)
    view_menu.addAction(decrease_size_action)

    view_menu.addSeparator()

    # Crop controls
    increase_crop_action = QAction('Increase &Crop Zoom', window)
    increase_crop_action.setShortcut(QKeySequence('Shift+Ctrl+='))
    increase_crop_action.setStatusTip('Increase crop zoom (Shift+Ctrl++)')
    increase_crop_action.triggered.connect(window.increase_crop_zoom)
    view_menu.addAction(increase_crop_action)

    decrease_crop_action = QAction('Decrease C&rop Zoom', window)
    decrease_crop_action.setShortcut(QKeySequence('Shift+Ctrl+-'))
    decrease_crop_action.setStatusTip('Decrease crop zoom (Shift+Ctrl+-)')
    decrease_crop_action.triggered.connect(window.decrease_crop_zoom)
    view_menu.addAction(decrease_crop_action)

    view_menu.addSeparator()

    # Thumbnail size submenu
    size_menu = view_menu.addMenu('&Thumbnail Size')
    window.size_actions = []
    for size_name in window.icon_sizes.keys():
        action = QAction(f'&{size_name}', window)
        action.setCheckable(True)
        action.setStatusTip(f'Set thumbnail size to {size_name} ({window.icon_sizes[size_name]}px)')
        action.triggered.connect(window.create_size_handler(size_name))
        if size_name == window.current_size_name:
            action.setChecked(True)
        size_menu.addAction(action)
        window.size_actions.append(action)

    # Crop submenu
    crop_menu = view_menu.addMenu('&Crop')
    window.crop_actions = []
    for crop_name in window.crop_sizes.keys():
        action = QAction(f'&{crop_name}', window)
        action.setCheckable(True)
        if crop_name == "none":
            action.setStatusTip('No cropping - show full image')
        else:
            action.setStatusTip(f'Crop to {crop_name}x{crop_name} pixels from center')
        action.triggered.connect(window.create_crop_handler(crop_name))
        if crop_name == window.current_crop_name:
            action.setChecked(True)
        crop_menu.addAction(action)
        window.crop_actions.append(action)

    # Classes menu (will be populated when classes are loaded)
    window.classes_menu = menubar.addMenu('&Classes')
    window.classes_menu.setEnabled(False)  # Disabled until classes are loaded
    window.update_classes_menu()

    # Create toolbar
    toolbar = QToolBar()
    toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    window.addToolBar(toolbar)

    # Add actions to toolbar
    toolbar.addAction(load_action)
    toolbar.addAction(load_classes_action)
    toolbar.addSeparator()
    toolbar.addAction(save_action)
    toolbar.addAction(export_action)
    toolbar.addSeparator()
    toolbar.addAction(reset_action)
    toolbar.addAction(show_classified_action)


