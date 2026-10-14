"""CLI: generate draft turtle polygons on sorted 224×224 thumbs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import typer
from tqdm import tqdm

from symsorter.outline.backends import (
    DEFAULT_SAM_MODEL,
    DEFAULT_THUMB_SIZE,
    draft_polygon,
    load_sam_model,
)
from symsorter.outline.export_labelme import build_labelme_dict, write_labelme_json
from symsorter.outline.filename import parse_thumb_filename

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_CLASSES = ("turtle-surface", "turtle-deep")
SUPPORTED_BACKENDS = frozenset({"sam", "box", "threshold"})


def _parse_classes(classes: str | None) -> tuple[str, ...]:
    if not classes or not classes.strip():
        return DEFAULT_CLASSES
    parts = [p.strip() for p in classes.split(",")]
    return tuple(p for p in parts if p)


def _is_nested_duplicate(rel_parts: tuple[str, ...], class_names: set[str]) -> bool:
    """Skip ``…/shard_*/turtle-*`` nested re-copies under a class shard."""
    seen_shard = False
    for part in rel_parts[:-1]:
        if part.startswith("shard_"):
            seen_shard = True
            continue
        if seen_shard and part in class_names:
            return True
    return False


def _class_roots(thumbs: Path, class_names: tuple[str, ...]) -> list[Path]:
    """Resolve class folders from a sorted root or a single class directory."""
    thumbs = thumbs.resolve()
    if not thumbs.exists():
        return []

    # If ``thumbs`` itself is a selected class folder, use it directly.
    if thumbs.name in class_names:
        return [thumbs]

    roots: list[Path] = []
    for name in class_names:
        candidate = thumbs / name
        if candidate.is_dir():
            roots.append(candidate)
    # Allow pointing at a shard or arbitrary folder of thumbs.
    if not roots:
        roots = [thumbs]
    return roots


def iter_thumb_paths(
    thumbs: Path,
    class_names: tuple[str, ...] = DEFAULT_CLASSES,
) -> Iterable[tuple[Path, Path]]:
    """Yield ``(thumb_path, relative_to_thumbs)`` skipping nested duplicates."""
    thumbs = thumbs.resolve()
    class_set = set(class_names) | set(DEFAULT_CLASSES)
    for class_root in _class_roots(thumbs, class_names):
        for path in sorted(class_root.rglob("*")):
            if not path.is_file() and not path.is_symlink():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                rel = path.relative_to(thumbs)
            except ValueError:
                # class_root may equal thumbs; still produce a relative path
                rel = Path(path.name)
            if _is_nested_duplicate(rel.parts, class_set):
                continue
            yield path, rel


def outline_draft_command(
    thumbs: Path,
    out: Path | None = None,
    classes: str = "turtle-surface,turtle-deep",
    backend: str = "sam",
    limit: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    sam_model: str = DEFAULT_SAM_MODEL,
    device: str | None = None,
) -> dict[str, int]:
    """Draft polygons beside each thumb; optional ``manifest.jsonl`` via ``--out``."""
    if not thumbs.exists():
        typer.echo(f"❌ Thumbs root not found: {thumbs}", err=True)
        raise typer.Exit(1)

    class_names = _parse_classes(classes)
    backend_name = (backend or "sam").strip().lower()
    if backend_name not in SUPPORTED_BACKENDS:
        typer.echo(
            f"❌ Unsupported backend {backend!r}. Use sam|box|threshold.",
            err=True,
        )
        raise typer.Exit(1)

    stats = {
        "enumerated": 0,
        "written": 0,
        "skipped_existing": 0,
        "broken_symlink": 0,
        "missing": 0,
        "errors": 0,
    }

    candidates: list[tuple[Path, Path]] = []
    for path, rel in iter_thumb_paths(thumbs, class_names):
        candidates.append((path, rel))
        if limit is not None and len(candidates) >= limit:
            break

    stats["enumerated"] = len(candidates)
    typer.echo(
        f"Found {len(candidates)} thumb(s) under {thumbs} "
        f"(classes={','.join(class_names)}, backend={backend_name}"
        f"{', dry-run' if dry_run else ''})"
    )
    typer.echo("LabelMe JSON will be written beside each thumb (same directory).")

    if dry_run:
        broken = 0
        ok = 0
        for path, _rel in candidates:
            if path.is_symlink() and not path.exists():
                broken += 1
            elif path.exists():
                ok += 1
            else:
                broken += 1
        typer.echo(f"Dry-run: resolvable={ok}, broken_or_missing={broken}")
        stats["broken_symlink"] = broken
        return stats

    sam_model_obj: Any | None = None
    if backend_name == "sam":
        try:
            typer.echo(f"Loading Ultralytics SAM ({sam_model})…")
            sam_model_obj = load_sam_model(sam_model=sam_model, device=device)
        except ImportError as exc:
            typer.echo(f"❌ {exc}", err=True)
            raise typer.Exit(1) from exc

    manifest_lines: list[str] = []

    progress = tqdm(
        candidates,
        desc=f"outline-draft ({backend_name})",
        unit="thumb",
        total=len(candidates),
    )
    for path, rel in progress:
        parsed = parse_thumb_filename(path.name)
        # LabelMe convention: foo.jpg → foo.json in the same folder.
        label_path = path.with_suffix(".json")

        if path.is_symlink() and not path.exists():
            stats["broken_symlink"] += 1
            tqdm.write(f"⚠️  Broken symlink, skipping: {path}")
            manifest_lines.append(
                json.dumps(
                    {
                        "thumb_path": str(path),
                        "thumb_rel": str(rel),
                        "label_path": str(label_path),
                        "thumb_stem": parsed.thumb_stem,
                        "conf": parsed.conf,
                        "det_index": parsed.det_index,
                        "flight": None,
                        "frame_stem": parsed.frame_stem,
                        "site": parsed.site,
                        "slice_path": None,
                        "slice_xyxy_full": None,
                        "polygon_thumb": None,
                        "polygon_slice": None,
                        "polygon_full": None,
                        "backend": backend_name,
                        "status": "broken_symlink",
                        "turtle_group_id": None,
                    },
                    ensure_ascii=False,
                )
            )
            continue

        if not path.exists():
            stats["missing"] += 1
            tqdm.write(f"⚠️  Missing file, skipping: {path}")
            continue

        try:
            draft = draft_polygon(
                backend_name,
                image_path=path,
                sam_model_obj=sam_model_obj,
                sam_model=sam_model,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue batch
            stats["errors"] += 1
            tqdm.write(f"⚠️  Draft failed for {path.name}: {exc}")
            continue

        shape_flags = dict(draft.flags)
        shape_flags["det_index"] = parsed.det_index
        shape_flags["turtle_group_id"] = None
        shape_flags.setdefault("view", None)
        shape_flags.setdefault("blur", None)

        labelme = build_labelme_dict(
            image_path=path.name,
            image_height=DEFAULT_THUMB_SIZE,
            image_width=DEFAULT_THUMB_SIZE,
            points=draft.points,
            flags=shape_flags,
            label="turtle",
        )

        wrote = write_labelme_json(label_path, labelme, overwrite=overwrite)
        if not wrote:
            stats["skipped_existing"] += 1
            status = "skipped_existing"
        else:
            stats["written"] += 1
            status = "drafted"

        manifest_lines.append(
            json.dumps(
                {
                    "thumb_path": str(path),
                    "thumb_rel": str(rel),
                    "label_path": str(label_path),
                    "thumb_stem": parsed.thumb_stem,
                    "conf": parsed.conf,
                    "conf_prefix": parsed.conf_prefix,
                    "det_index": parsed.det_index,
                    "flight": None,
                    "frame_stem": parsed.frame_stem,
                    "site": parsed.site,
                    "slice_path": None,
                    "slice_xyxy_full": None,
                    "polygon_thumb": draft.points,
                    "polygon_slice": None,
                    "polygon_full": None,
                    "backend": draft.backend,
                    "status": status,
                    "turtle_group_id": None,
                    "flags": shape_flags,
                },
                ensure_ascii=False,
            )
        )
        progress.set_postfix(
            written=stats["written"],
            skipped=stats["skipped_existing"],
            errors=stats["errors"],
            refresh=False,
        )

    if out is not None:
        out = out.resolve()
        out.mkdir(parents=True, exist_ok=True)
        manifest_path = out / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for line in manifest_lines:
                f.write(line)
                f.write("\n")
        typer.echo(f"   Manifest: {manifest_path}")
    else:
        # Still useful for the run summary when not writing a separate out tree.
        typer.echo("   Manifest: (not written; pass --out DIR to save manifest.jsonl)")

    typer.echo(
        f"✅ Done: written={stats['written']}, "
        f"skipped_existing={stats['skipped_existing']}, "
        f"broken_symlink={stats['broken_symlink']}, "
        f"errors={stats['errors']}"
    )
    typer.echo("   Labels: beside each thumb (*.json next to *.jpg)")
    return stats
