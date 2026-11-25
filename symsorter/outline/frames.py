"""ROI CSV index, frame lookup, and thumb ↔ frame geometry."""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image

from symsorter.outline.mapping import CropWindow, fixed_size_crop_window


@dataclass(frozen=True)
class RoiRow:
    """One OK thumb-ROI row, keeping only the original image stem."""

    view: str
    crop_basename: str
    flight: str
    x1: float
    y1: float
    x2: float
    y2: float
    original_stem: str


def _original_stem(original_image: str) -> str:
    """Stem of a Windows or POSIX path recorded on another machine."""
    if "\\" in original_image:
        return PureWindowsPath(original_image).stem
    return PurePosixPath(original_image).stem


class RoiIndex:
    """OK ROI rows keyed by crop basename and by ``(flight, original_stem)``."""

    def __init__(
        self,
        by_crop: dict[str, RoiRow],
        by_frame: dict[tuple[str, str], list[RoiRow]],
    ) -> None:
        self._by_crop = by_crop
        self._by_frame = by_frame

    @classmethod
    def load(cls, csv_path: Path) -> RoiIndex:
        """Load rows whose ``status`` is ``ok``; every other row is omitted."""
        by_crop: dict[str, RoiRow] = {}
        by_frame: dict[tuple[str, str], list[RoiRow]] = {}
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                if raw.get("status") != "ok":
                    continue
                row = RoiRow(
                    view=raw["view"],
                    crop_basename=raw["crop_basename"],
                    flight=raw["flight"],
                    x1=float(raw["x1"]),
                    y1=float(raw["y1"]),
                    x2=float(raw["x2"]),
                    y2=float(raw["y2"]),
                    original_stem=_original_stem(raw["original_image"]),
                )
                by_crop[row.crop_basename] = row
                by_frame.setdefault((row.flight, row.original_stem), []).append(row)
        return cls(by_crop, by_frame)

    def get(self, crop_basename: str) -> RoiRow | None:
        """Return the OK row for ``crop_basename``, or None if missing or not ok."""
        return self._by_crop.get(crop_basename)

    def same_frame(self, flight: str, stem: str) -> list[RoiRow]:
        """OK rows that share ``flight`` and the original-image stem."""
        return list(self._by_frame.get((flight, stem), ()))


def frame_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` from the image header."""
    with Image.open(path) as image:
        return image.size


def crop_window_for_row(row: RoiRow, frame_wh: tuple[int, int]) -> CropWindow:
    """224 window centered on the ROI box, tagged ``method='roi_csv'``."""
    width, height = frame_wh
    window = fixed_size_crop_window(
        row.x1,
        row.y1,
        row.x2,
        row.y2,
        img_w=width,
        img_h=height,
        fixed_size=224,
    )
    return replace(window, method="roi_csv")


def thumb_to_frame(points: list[list[float]], crop: CropWindow) -> list[list[float]]:
    """Shift thumb-local points into frame coordinates."""
    return [[x + crop.left, y + crop.top] for x, y in points]


def frame_to_thumb(points: list[list[float]], crop: CropWindow) -> list[list[float]]:
    """Shift frame points into thumb-local coordinates."""
    return [[x - crop.left, y - crop.top] for x, y in points]


def iter_flat_thumbs(folder: Path) -> list[Path]:
    """Sorted ``*_224px.jpg`` files in ``folder``, skipping names starting ``._``."""
    paths = [
        path
        for path in Path(folder).iterdir()
        if path.is_file()
        and path.name.endswith("_224px.jpg")
        and not path.name.startswith("._")
    ]
    return sorted(paths)


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _is_image_name(name: str) -> bool:
    """True for an image filename that is not an AppleDouble sidecar."""
    return not name.startswith("._") and Path(name).suffix.lower() in IMAGE_EXTS


def _first_named(directory: Path, stem: str) -> Path | None:
    """Return the first image in ``directory`` with this stem, or None."""
    if not directory.is_dir():
        return None
    matches = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.stem == stem and _is_image_name(path.name)
    ]
    if not matches:
        return None
    return sorted(matches)[0]


def _survey_match(root: Path, flight: str, stem: str) -> Path | None:
    """Return a frame at ``{root}/*/{site}/{flight}/``, or None."""
    site = flight.split("_")[0]
    countries = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("._")
    )
    for country in countries:
        found = _first_named(country / site / flight, stem)
        if found is not None:
            return found
    return None


class FrameResolver:
    """Locate an original frame under survey roots; the first match wins."""

    def __init__(self, roots: Sequence[Path]) -> None:
        self._roots = [Path(root) for root in roots]
        self._by_root: dict[Path, dict[str, Path]] = {}

    def find(self, flight: str, stem: str) -> Path | None:
        """Return the frame for ``flight`` and ``stem``, or None if absent."""
        for root in self._roots:
            if not root.is_dir():
                continue
            found = _first_named(root / flight / "images", stem)
            if found is not None:
                return found
            found = _survey_match(root, flight, stem)
            if found is not None:
                return found
            indexed = self._stem_index(root).get(stem)
            if indexed is not None:
                return indexed
        return None

    def _stem_index(self, root: Path) -> dict[str, Path]:
        """Lazily map stems to the first image in sorted walk order."""
        cached = self._by_root.get(root)
        if cached is not None:
            return cached
        index: dict[str, Path] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(name for name in dirnames if not name.startswith("._"))
            for name in sorted(filenames):
                if not _is_image_name(name):
                    continue
                path = Path(dirpath) / name
                index.setdefault(path.stem, path)
        self._by_root[root] = index
        return index
