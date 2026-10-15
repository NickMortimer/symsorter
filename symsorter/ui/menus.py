from PySide6.QtWidgets import QToolBar, QFileDialog
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtCore import Qt


def build_menus(window):
    menubar = window.menuBar()

    # File
    file_menu = menubar.addMenu('&File')

    # Load embeddings (NPZ) dialog
    load_action = QAction('&Load Embeddings...', window)
    load_action.setShortcut(QKeySequence('Ctrl+O'))
    load_action.setStatusTip('Load embeddings from NPZ file')
    load_action.triggered.connect(window.load_folder)
    file_menu.addAction(load_action)

    # Load class file
    load_classes_action = QAction('Load &Class File...', window)
    load_classes_action.setShortcut(QKeySequence('Ctrl+Shift+O'))
    load_classes_action.setStatusTip('Load YOLO class names file')
    load_classes_action.triggered.connect(window.load_class_file_dialog)
    file_menu.addAction(load_classes_action)

    file_menu.addSeparator()

    # Save classifications
    save_action = QAction('&Save Classifications', window)
    save_action.setShortcut(QKeySequence('Ctrl+S'))
    save_action.setStatusTip('Save classifications and hidden flags')
    save_action.triggered.connect(window.save_classifications)
    file_menu.addAction(save_action)

    file_menu.addSeparator()

    # Export/Import JSON
    export_json_action = QAction('Export &Classifications (JSON)...', window)
    export_json_action.setShortcut(QKeySequence('Ctrl+J'))
    export_json_action.setStatusTip('Export classifications as COCO-style JSON')
    export_json_action.triggered.connect(window.export_classifications_json_dialog)
    file_menu.addAction(export_json_action)

    import_json_action = QAction('&Import Classifications (JSON)...', window)
    import_json_action.setShortcut(QKeySequence('Ctrl+Shift+J'))
    import_json_action.setStatusTip('Import classifications from COCO-style JSON')
    import_json_action.triggered.connect(window.import_classifications_json_dialog)
    file_menu.addAction(import_json_action)

    # Export YOLO
    export_action = QAction('Export &YOLO Annotations...', window)
    export_action.setShortcut(QKeySequence('Ctrl+E'))
    export_action.setStatusTip('Export classifications as YOLO annotations')
    export_action.triggered.connect(window.export_yolo_annotations)
    file_menu.addAction(export_action)

    # Open folder (images root) dialog
    open_folder_act = QAction('Open &Folder…', window)
    open_folder_act.setStatusTip('Open a folder of images to encode/load')
    open_folder_act.triggered.connect(lambda: (
        setattr(window, '_d', QFileDialog.getExistingDirectory(window, 'Select Image Folder') or None),
        window.load_folder_from_path(window._d) if window._d else None
    ))
    file_menu.addAction(open_folder_act)

    # Classes
    classes_menu = menubar.addMenu('&Classes')
    window.classes_menu = classes_menu
    classes_menu.addAction(load_classes_action)  # reuse action
    if hasattr(window, 'update_classes_menu'):
        window.update_classes_menu()

    # View
    view_menu = menubar.addMenu('&View')

    reset_action = QAction('&Reset Order', window)
    reset_action.setShortcut(QKeySequence('R'))
    reset_action.setStatusTip('Reset images to original order')
    reset_action.triggered.connect(window.reset_order)
    view_menu.addAction(reset_action)

    view_menu.addSeparator()

    # Labels toggle
    show_labels_action = QAction('Show Classification &Labels', window)
    show_labels_action.setShortcut(QKeySequence('L'))
    show_labels_action.setCheckable(True)
    show_labels_action.setChecked(getattr(window, 'show_classification_labels', False))
    show_labels_action.setStatusTip('Toggle classification labels below thumbnails')
    show_labels_action.triggered.connect(window.toggle_classification_labels)
    view_menu.addAction(show_labels_action)
    window.show_labels_action = show_labels_action

    # Sort by class
    sort_by_class_action = QAction('Sort by &Class', window)
    sort_by_class_action.setShortcut(QKeySequence('Shift+C'))
    sort_by_class_action.setStatusTip('Sort images by classification')
    sort_by_class_action.triggered.connect(window.sort_by_class)
    view_menu.addAction(sort_by_class_action)

    # Show only classified images
    show_classified_action = QAction('Show &Classified Images', window)
    show_classified_action.setShortcut(QKeySequence('Ctrl+Shift+C'))
    show_classified_action.setStatusTip('Show all classified images')
    show_classified_action.triggered.connect(window.show_classified_images)
    view_menu.addAction(show_classified_action)

    view_menu.addSeparator()

    # Outline polygon viewer (sibling LabelMe JSON)
    outline_action = QAction('&Outline Viewer…', window)
    outline_action.setShortcut(QKeySequence('O'))
    outline_action.setStatusTip(
        'Show LabelMe polygons on the selected thumb(s) (sibling *.json)'
    )
    outline_action.triggered.connect(window.open_outline_viewer)
    view_menu.addAction(outline_action)
    window.outline_action = outline_action

    # Grid overlay of sibling LabelMe polygons
    outline_overlay_action = QAction('Show Outline &Overlays', window)
    outline_overlay_action.setShortcut(QKeySequence('Shift+O'))
    outline_overlay_action.setCheckable(True)
    outline_overlay_action.setChecked(getattr(window, 'show_outline_overlays', True))
    outline_overlay_action.setStatusTip(
        'Draw turtle polygons from sibling LabelMe JSON onto grid thumbnails'
    )
    outline_overlay_action.triggered.connect(window.toggle_outline_overlays)
    view_menu.addAction(outline_overlay_action)
    window.show_outline_overlays_action = outline_overlay_action

    # In-grid SAM prompting (stay in icon view — best at XL/XXL)
    prompt_menu = view_menu.addMenu('Outline &Prompt (grid)')
    window.outline_prompt_group = QActionGroup(window)
    window.outline_prompt_group.setExclusive(True)

    prompt_off = QAction('&Off', window)
    prompt_off.setCheckable(True)
    prompt_off.setChecked(True)
    prompt_off.setShortcut(QKeySequence('Shift+P'))
    prompt_off.setStatusTip('Disable in-grid SAM prompting')
    prompt_off.triggered.connect(lambda: window.set_outline_prompt_mode('off'))
    window.outline_prompt_group.addAction(prompt_off)
    prompt_menu.addAction(prompt_off)
    window.outline_prompt_off_action = prompt_off

    prompt_point = QAction('Click &Point → SAM', window)
    prompt_point.setCheckable(True)
    prompt_point.setShortcut(QKeySequence('P'))
    prompt_point.setStatusTip('Click on a grid thumb to seed SAM (stay in grid)')
    prompt_point.triggered.connect(lambda: window.set_outline_prompt_mode('point'))
    window.outline_prompt_group.addAction(prompt_point)
    prompt_menu.addAction(prompt_point)
    window.outline_prompt_point_action = prompt_point

    prompt_box = QAction('Drag &Box → SAM', window)
    prompt_box.setCheckable(True)
    prompt_box.setShortcut(QKeySequence('B'))
    prompt_box.setStatusTip('Drag a box on a grid thumb to prompt SAM')
    prompt_box.triggered.connect(lambda: window.set_outline_prompt_mode('box'))
    window.outline_prompt_group.addAction(prompt_box)
    prompt_menu.addAction(prompt_box)
    window.outline_prompt_box_action = prompt_box

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

    # Thumbnail Size submenu
    size_menu = view_menu.addMenu('&Thumbnail Size')
    window.size_action_group = QActionGroup(window)
    window.size_action_group.setExclusive(True)
    window.size_actions = []
    for size_name, px in window.icon_sizes.items():
        act = QAction(f'&{size_name}', window)
        act.setCheckable(True)
        act.setData(size_name)
        act.setStatusTip(f'Set thumbnail size to {px}px')
        act.triggered.connect(window.create_size_handler(size_name))
        if size_name == window.current_size_name:
            act.setChecked(True)
        window.size_action_group.addAction(act)
        size_menu.addAction(act)
        window.size_actions.append(act)

    # Crop submenu
    crop_menu = view_menu.addMenu('&Crop')
    window.crop_action_group = QActionGroup(window)
    window.crop_action_group.setExclusive(True)
    window.crop_actions = []
    for crop_name in window.crop_sizes.keys():
        act = QAction(f'&{crop_name}', window)
        act.setCheckable(True)
        act.setData(crop_name)
        if crop_name == 'none':
            act.setStatusTip('No cropping - show full image')
        else:
            act.setStatusTip(f'Crop to {crop_name}x{crop_name} pixels from center')
        act.triggered.connect(window.create_crop_handler(crop_name))
        if crop_name == window.current_crop_name:
            act.setChecked(True)
        window.crop_action_group.addAction(act)
        crop_menu.addAction(act)
        window.crop_actions.append(act)

    # Toolbar
    toolbar = QToolBar()
    toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    window.addToolBar(toolbar)
    toolbar.addAction(load_action)
    toolbar.addAction(load_classes_action)
    toolbar.addSeparator()
    toolbar.addAction(save_action)
    toolbar.addAction(export_action)
    toolbar.addSeparator()
    toolbar.addAction(reset_action)
    toolbar.addAction(show_classified_action)
    toolbar.addAction(show_labels_action)
    toolbar.addAction(sort_by_class_action)
    toolbar.addAction(outline_action)
    toolbar.addAction(outline_overlay_action)
    toolbar.addAction(prompt_point)
    toolbar.addAction(prompt_box)
    toolbar.addAction(prompt_off)


