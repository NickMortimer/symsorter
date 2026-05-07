from pathlib import Path
from typing import Dict, List, Tuple
import os
import json
import shutil
import typer


def _find_manifests(root: Path) -> List[Path]:
    """Recursively find classifications.json under root."""
    manifests: List[Path] = []
    for p in root.rglob("classifications.json"):
        if p.is_file():
            manifests.append(p)
    return manifests


def _safe_class_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_") or "unknown"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_by_class(
    manifests: List[Path],
    exclude_hidden: bool,
) -> Dict[str, List[Tuple[Path, str]]]:
    """Collect (class_name -> [(src_path, filename), ...]) from various schemas."""
    by_class: Dict[str, List[Tuple[Path, str]]] = {}

    for mpath in manifests:
        try:
            data = _load_json(mpath)
        except Exception:
            continue

        folder = mpath.parent

        # Simple mapping schema
        files_map = data.get("files", {})
        if isinstance(files_map, dict) and files_map:
            for fname, info in files_map.items():
                if not isinstance(info, dict):
                    continue
                if exclude_hidden and info.get("hidden") is True:
                    continue
                cname = info.get("class_name")
                if not cname:
                    continue
                src = folder / fname
                if src.exists():
                    by_class.setdefault(cname, []).append((src, fname))
            continue

        # COCO-like schema
        images = data.get("images") or []
        ann = data.get("annotations") or []
        cats = data.get("categories") or []

        if images and ann and cats:
            id_to_file = {
                img.get("id"): img.get("file_name") or img.get("filename")
                for img in images
            }
            id_to_hidden = {img.get("id"): bool(img.get("hidden", False)) for img in images}
            cat_id_to_name = {c.get("id"): c.get("name") for c in cats}

            img_to_cat = {}
            for a in ann:
                iid = a.get("image_id")
                cid = a.get("category_id")
                if iid is None or cid is None:
                    continue
                if iid not in img_to_cat:
                    img_to_cat[iid] = cid

            for iid, cid in img_to_cat.items():
                fname = id_to_file.get(iid)
                if not fname:
                    continue
                if exclude_hidden and id_to_hidden.get(iid, False):
                    continue
                cname = cat_id_to_name.get(cid)
                if not cname:
                    continue
                src = folder / fname
                if src.exists():
                    by_class.setdefault(cname, []).append((src, fname))

    return by_class


def sort_images_command(
    root: Path,
    out: Path,
    shard_size: int = 5000,
    exclude_hidden: bool = False,
    link: bool = False,
    dry_run: bool = False,
):
    """
    Combine images from all subdirectories into `out`, grouped by classification.
    Shard each class into subdirectories of at most `shard_size` images.
    """
    if not root.exists():
        typer.echo(f"❌ Root not found: {root}", err=True)
        raise typer.Exit(1)

    manifests = _find_manifests(root)
    if not manifests:
        typer.echo(f"❌ No classifications.json found under {root}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(manifests)} manifests")
    by_class = _collect_by_class(manifests, exclude_hidden=exclude_hidden)
    if not by_class:
        typer.echo("❌ No classified images found to sort", err=True)
        raise typer.Exit(1)

    typer.echo(f"Classes found: {len(by_class)}")
    for cname, items in by_class.items():
        typer.echo(f" - {cname}: {len(items)} images")

    ops = "link" if link else "copy"
    created = 0
    out.mkdir(parents=True, exist_ok=True) if not dry_run else None

    for cname, items in by_class.items():
        safe = _safe_class_name(cname)
        class_root = out / safe
        if not dry_run:
            class_root.mkdir(parents=True, exist_ok=True)

        shard_idx = 1
        shard_count = 0
        shard_dir = class_root / f"shard_{shard_idx:05d}"
        if not dry_run:
            shard_dir.mkdir(parents=True, exist_ok=True)

        for src, fname in items:
            if shard_count >= shard_size:
                shard_idx += 1
                shard_dir = class_root / f"shard_{shard_idx:05d}"
                shard_count = 0
                if not dry_run:
                    shard_dir.mkdir(parents=True, exist_ok=True)

            dst = shard_dir / fname
            try:
                if dry_run:
                    typer.echo(f"{ops}: {src} -> {dst}")
                else:
                    if link:
                        rel = os.path.relpath(src, shard_dir)
                        try:
                            dst.symlink_to(rel)
                        except FileExistsError:
                            pass
                    else:
                        shutil.copy2(src, dst)
                created += 1
                shard_count += 1
            except Exception as e:
                typer.echo(f"⚠️ Failed to {ops} {src} -> {dst}: {e}", err=True)

    typer.echo(f"✅ Sorted {created} images into {out}")