"""Survey roots and ROI CSV persisted for the frame panel."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass
class FrameSettings:
    """Survey roots and the optional ROI CSV used to locate frames."""

    survey_roots: list[Path] = field(default_factory=list)
    roi_csv: Path | None = None


def load_frame_settings() -> FrameSettings:
    """Load frame survey roots and ROI CSV from QSettings."""
    settings = QSettings("symsorter", "symsorter")
    settings.sync()
    roots = [
        Path(item) for item in _as_path_strings(settings.value("outline/survey_roots"))
    ]
    csv_path = _csv_path(settings.value("outline/roi_csv", ""))
    return FrameSettings(survey_roots=roots, roi_csv=csv_path)


def save_frame_settings(s: FrameSettings) -> None:
    """Save frame survey roots and ROI CSV to QSettings."""
    settings = QSettings("symsorter", "symsorter")
    settings.setValue("outline/survey_roots", [str(path) for path in s.survey_roots])
    settings.setValue("outline/roi_csv", "" if s.roi_csv is None else str(s.roi_csv))
    settings.sync()


def roi_csv_for(thumb_folder: Path, s: FrameSettings) -> Path | None:
    """Return the configured ROI CSV, else ``thumb_roi.csv`` beside the folder."""
    if s.roi_csv is not None:
        return s.roi_csv
    candidate = Path(thumb_folder).parent / "thumb_roi.csv"
    if candidate.is_file():
        return candidate
    return None


def edit_frame_settings_dialog(parent: QWidget | None) -> FrameSettings | None:
    """Edit survey roots and ROI CSV; save on OK, or return None if cancelled."""
    current = load_frame_settings()
    dialog = QDialog(parent)
    dialog.setWindowTitle("Set survey roots")
    roots_edit = QPlainTextEdit()
    roots_edit.setPlainText("\n".join(str(path) for path in current.survey_roots))
    add_button = QPushButton("Add folder…")
    csv_edit = QLineEdit("" if current.roi_csv is None else str(current.roi_csv))
    browse_button = QPushButton("Browse…")

    def _add_folder() -> None:
        chosen = QFileDialog.getExistingDirectory(dialog, "Add survey root")
        if not chosen:
            return
        existing = roots_edit.toPlainText().rstrip()
        roots_edit.setPlainText(f"{existing}\n{chosen}" if existing else chosen)

    def _browse_csv() -> None:
        chosen, _selected_filter = QFileDialog.getOpenFileName(
            dialog,
            "ROI CSV",
            csv_edit.text(),
            "CSV files (*.csv);;All files (*)",
        )
        if chosen:
            csv_edit.setText(chosen)

    add_button.clicked.connect(_add_folder)
    browse_button.clicked.connect(_browse_csv)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)

    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("Survey roots"))
    layout.addWidget(roots_edit)
    layout.addWidget(add_button)
    layout.addWidget(QLabel("ROI CSV"))
    csv_row = QHBoxLayout()
    csv_row.addWidget(csv_edit)
    csv_row.addWidget(browse_button)
    layout.addLayout(csv_row)
    layout.addWidget(buttons)

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    roots = [
        Path(line.strip())
        for line in roots_edit.toPlainText().splitlines()
        if line.strip()
    ]
    csv_text = csv_edit.text().strip()
    updated = FrameSettings(roots, Path(csv_text) if csv_text else None)
    save_frame_settings(updated)
    return updated


def _as_path_strings(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _csv_path(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)
