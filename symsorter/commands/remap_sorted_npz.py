"""Rebuild SymSorter NPZ embeddings under sorted shards from crop-folder NPZs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from tqdm import tqdm

from symsorter.clip_encode import save_folder_embeddings
from symsorter.commands.outline_draft import _is_appledouble_or_junk, _parse_classes

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _iter_source_npzs(source: Path, npz_name: str) -> list[Path]:
    """Find crop NPZs named ``npz_name`` under ``source``."""
    source = source.resolve()
    if not source.exists():
        return []
    # Exact name search keeps joins deterministic (e.g. encoding_dinov3_224.npz).
    return sorted(source.rglob(npz_name))


def build_embedding_index(npz_paths: list[Path]) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, int]], str, int]:
    """Merge source NPZs into filename → embedding / dimensions maps."""
    embeddings: dict[str, np.ndarray] = {}
    dimensions: dict[str, tuple[int, int]] = {}
    model_type = "dinov3"
    crop_size = 224

    for npz_path in tqdm(npz_paths, desc="Index source NPZs", unit="npz"):
        with np.load(npz_path, allow_pickle=True) as data:
            if "embeddings" not in data.files or "filenames" not in data.files:
                typer.echo(f"⚠️  Skipping (missing keys): {npz_path}", err=True)
                continue
            emb = data["embeddings"]
            fns = data["filenames"]
            dims = data["dimensions"] if "dimensions" in data.files else None
            if "model_type" in data.files and len(data["model_type"]):
                model_type = str(data["model_type"][0])
            if "crop_size" in data.files and len(data["crop_size"]):
                crop_size = int(data["crop_size"][0])

            for i, fn in enumerate(fns):
                name = str(fn)
                if name in embeddings:
                    continue
                embeddings[name] = np.asarray(emb[i], dtype=np.float32)
                if dims is not None:
                    d0, d1 = int(dims[i][0]), int(dims[i][1])
                    dimensions[name] = (d0, d1)
                else:
                    dimensions[name] = (crop_size, crop_size)

    return embeddings, dimensions, model_type, crop_size


def _iter_shard_dirs(thumbs: Path, class_names: tuple[str, ...]) -> list[Path]:
    """Return class/shard_* directories under thumbs."""
    thumbs = thumbs.resolve()
    shards: list[Path] = []

    roots: list[Path] = []
    if thumbs.name in class_names or thumbs.name.startswith("shard_"):
        # Pointed at a class folder or a single shard
        if thumbs.name.startswith("shard_"):
            return [thumbs]
        roots = [thumbs]
    else:
        for name in class_names:
            candidate = thumbs / name
            if candidate.is_dir():
                roots.append(candidate)
        if not roots:
            roots = [thumbs]

    for root in roots:
        if root.name.startswith("shard_"):
            shards.append(root)
            continue
        found = sorted(p for p in root.glob("shard_*") if p.is_dir())
        shards.extend(found)
    return shards


def _shard_thumb_names(shard: Path) -> list[str]:
    names: list[str] = []
    for path in sorted(shard.iterdir()):
        if not path.is_file() and not path.is_symlink():
            continue
        if _is_appledouble_or_junk(path):
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        names.append(path.name)
    return names


def remap_sorted_npz_command(
    thumbs: Path,
    source: Path,
    npz_name: str = "encoding_dinov3_224.npz",
    classes: str = "turtle-surface,turtle-deep",
    overwrite: bool = False,
    dry_run: bool = False,
    limit_shards: int | None = None,
) -> dict[str, int]:
    """Copy embeddings from crop NPZs into per-shard NPZs under sorted thumbs."""
    if not thumbs.exists():
        typer.echo(f"❌ Thumbs root not found: {thumbs}", err=True)
        raise typer.Exit(1)
    if not source.exists():
        typer.echo(f"❌ Source root not found: {source}", err=True)
        raise typer.Exit(1)

    class_names = _parse_classes(classes)
    npz_paths = _iter_source_npzs(source, npz_name)
    if not npz_paths:
        typer.echo(f"❌ No NPZs named {npz_name!r} under {source}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(npz_paths)} source NPZ(s) named {npz_name}")
    embeddings, dimensions, model_type, crop_size = build_embedding_index(npz_paths)
    typer.echo(
        f"Indexed {len(embeddings)} unique embeddings "
        f"(model_type={model_type}, crop_size={crop_size})"
    )

    shards = _iter_shard_dirs(thumbs, class_names)
    if limit_shards is not None:
        shards = shards[: max(0, limit_shards)]

    stats = {
        "shards": 0,
        "written": 0,
        "skipped_existing": 0,
        "matched": 0,
        "missing": 0,
    }

    for shard in tqdm(shards, desc="Write sorted NPZs", unit="shard"):
        stats["shards"] += 1
        out_npz = shard / npz_name
        names = _shard_thumb_names(shard)
        if not names:
            continue

        shard_emb: dict[str, np.ndarray] = {}
        shard_dims: dict[str, tuple[int, int]] = {}
        missing = 0
        for name in names:
            vec = embeddings.get(name)
            if vec is None:
                missing += 1
                continue
            shard_emb[name] = vec
            shard_dims[name] = dimensions.get(name, (crop_size, crop_size))

        stats["matched"] += len(shard_emb)
        stats["missing"] += missing

        typer.echo(
            f"{shard}: thumbs={len(names)} matched={len(shard_emb)} missing={missing}"
        )

        if dry_run:
            continue
        if out_npz.exists() and not overwrite:
            stats["skipped_existing"] += 1
            typer.echo(f"  skip existing {out_npz.name} (use --overwrite)")
            continue
        if not shard_emb:
            typer.echo(f"  ⚠️  no matches; not writing {out_npz}", err=True)
            continue

        save_folder_embeddings(
            out_npz,
            shard_emb,
            dimensions_dict=shard_dims,
            crop_size=crop_size,
            model_type=model_type,
        )
        stats["written"] += 1

    typer.echo(
        f"✅ Done: shards={stats['shards']} written={stats['written']} "
        f"skipped_existing={stats['skipped_existing']} "
        f"matched={stats['matched']} missing={stats['missing']}"
    )
    return stats
