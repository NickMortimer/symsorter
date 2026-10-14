"""
Command-line interface for SymSorter
"""
import typer
from pathlib import Path
import sys
import json

app = typer.Typer(
    name="symsorter",
    help="SymSorter - A CLIP-based image classification and similarity tool",
    no_args_is_help=True
)

# Import the encode and inspect commands directly from clip_encode
from .clip_encode import doit_encode as encode, inspect

# Register the commands with the main app
app.command(name="encode")(encode)
app.command(name="inspect")(inspect)

# New: merge per-folder classifications.json into one COCO JSON
@app.command(name="merge-coco")
def merge_coco(
    root: Path = typer.Argument(..., help="Root directory to scan for classifications.json"),
    out: Path = typer.Option(Path("merged_classifications.json"), "--out", "-o", help="Output COCO JSON path"),
    exclude_hidden: bool = typer.Option(False, "--exclude-hidden", help="Exclude images marked hidden=True"),
):
    """
    Merge all classifications.json under ROOT into a single COCO JSON.
    """
    try:
        from .io.aggregate import find_classification_manifests, merge_coco_manifests
        from .io.classifications import ensure_json_serializable
    except Exception as e:
        typer.echo(f"❌ Missing dependencies or modules: {e}", err=True)
        raise typer.Exit(1)

    manifests = find_classification_manifests(root)
    if not manifests:
        typer.echo(f"❌ No classifications.json found under {root}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(manifests)} manifests")
    merged = merge_coco_manifests(root, manifests, include_hidden=not exclude_hidden)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(ensure_json_serializable(merged), f, indent=2, ensure_ascii=False)
    typer.echo(f"✅ Wrote merged COCO to {out}")

@app.command()
def gui(
    embeddings_file: Path = typer.Option(None, help="NPZ file with CLIP embeddings to load"),
    class_file: Path = typer.Option(None, help="Text file with class names (one per line)")
):
    """
    Launch the SymSorter GUI for image classification and sorting.
    """
    try:
        from PySide6.QtWidgets import QApplication
        from .image_browser import ImageBrowser
        
        # Create QApplication
        if not QApplication.instance():
            app_qt = QApplication(sys.argv)
        else:
            app_qt = QApplication.instance()
        
        # Create and show the image browser
        browser = ImageBrowser(class_file=str(class_file) if class_file else None)
        
        # Load embeddings if provided
        if embeddings_file and embeddings_file.exists():
            browser.load_folder_from_path(str(embeddings_file))
        
        browser.show()
        
        # Run the application
        sys.exit(app_qt.exec())
        
    except ImportError as e:
        typer.echo("❌ GUI dependencies not available. Install with:", err=True)
        typer.echo("pip install 'symsorter[gui]'", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error launching GUI: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def similarity(
    embeddings_file: Path = typer.Argument(..., help="NPZ file with CLIP embeddings"),
    query_image: Path = typer.Argument(..., help="Query image to find similar images for"),
    top_k: int = typer.Option(10, help="Number of most similar images to return"),
    output_dir: Path = typer.Option(None, help="Directory to copy similar images to")
):
    """
    Find images most similar to a query image using CLIP embeddings.
    """
    import numpy as np
    import shutil
    from .clip_encode import load_existing_embeddings, load_clip_model, CLIP_AVAILABLE
    
    if not CLIP_AVAILABLE:
        typer.echo("❌ CLIP not installed. Install with: pip install open-clip-torch", err=True)
        raise typer.Exit(1)
    
    try:
        # Load embeddings
        embeddings = load_existing_embeddings(embeddings_file)
        if not embeddings:
            typer.echo(f"❌ No embeddings found in {embeddings_file}")
            raise typer.Exit(1)
        
        # Load CLIP model
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = load_clip_model(device)
        
        # Process query image
        from PIL import Image
        query_img = Image.open(query_image).convert("RGB")
        query_tensor = preprocess(query_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_features = model.encode_image(query_tensor)
            query_features /= query_features.norm(dim=-1, keepdim=True)
        
        query_embedding = query_features.cpu().numpy().flatten()
        
        # Calculate similarities
        similarities = []
        for filename, embedding in embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((filename, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Display results
        typer.echo(f"🔍 Top {top_k} most similar images to {query_image.name}:")
        typer.echo()
        
        image_dir = embeddings_file.parent
        for i, (filename, sim) in enumerate(similarities[:top_k], 1):
            typer.echo(f"{i:2d}. {filename:<30} (similarity: {sim:.4f})")
            
            # Copy to output directory if specified
            if output_dir:
                output_dir.mkdir(exist_ok=True)
                src = image_dir / filename
                dst = output_dir / f"{i:02d}_{filename}"
                if src.exists():
                    shutil.copy2(src, dst)
        
        if output_dir:
            typer.echo(f"\n📁 Similar images copied to {output_dir}")
        
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)
    
from .commands.sort_images import sort_images_command
from .commands.prepare_dataset import (
    prepare_dataset_command,
    update_json_from_symsorter_command,
)
from .commands.outline_draft import outline_draft_command

@app.command(name="sort")
def sort(
    root: Path = typer.Argument(..., help="Root directory to scan for classifications.json"),
    out: Path = typer.Argument(..., help="Output root directory"), 
    shard_size: int = typer.Option(5000, "--shard-size", "-n", help="Max images per shard subdirectory"),
    exclude_hidden: bool = typer.Option(False, "--exclude-hidden", help="Exclude images marked hidden=True"),
    link: bool = typer.Option(False, "--link", help="Use symlinks instead of copying files"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print actions without writing files"),
):
    """Sort and shard images by classification into an output directory."""
    return sort_images_command(
        root=root,
        out=out,
        shard_size=shard_size,
        exclude_hidden=exclude_hidden,
        link=link,
        dry_run=dry_run,
    )    


@app.command(name="prepare")
def prepare(
    root: Path = typer.Argument(..., help="Root directory containing images and Labelme/X-AnyLabeling JSON files"),
    out: Path = typer.Option(None, "--out", "-o", help="Output thumbnail directory (default: <root>/symsorter_prepared)"),
    thumb_size: int = typer.Option(224, "--thumb-size", help="Thumbnail size in pixels"),
    context_scale: float = typer.Option(1.6, "--context-scale", help="ROI context scale around annotation"),
):
    """Create ROI-centered thumbnails for sorting in SymSorter."""
    out_dir = out or (root / "symsorter_prepared")
    prepare_dataset_command(
        root=root,
        out_dir=out_dir,
        thumb_size=thumb_size,
        context_scale=context_scale,
    )


@app.command(name="update-json")
def update_json(
    prepared_dir: Path = typer.Argument(..., help="Directory created by prepare containing thumbnails and mapping file"),
    source_root: Path = typer.Option(None, "--source-root", help="Original dataset root (optional; inferred from mapping if omitted)"),
    replace_label: bool = typer.Option(False, "--replace-label", help="Replace original shape label with SymSorter class name"),
):
    """Write SymSorter classification results back into source JSON files."""
    update_json_from_symsorter_command(
        prepared_dir=prepared_dir,
        source_root=source_root,
        replace_label=replace_label,
    )


@app.command(name="outline-draft")
def outline_draft(
    thumbs: Path = typer.Option(
        ...,
        "--thumbs",
        help="Root of sorted shards (…/sorted or a single class folder)",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output root for draft LabelMe JSON + manifest.jsonl",
    ),
    classes: str = typer.Option(
        "turtle-surface,turtle-deep",
        "--classes",
        help="Comma-separated class folder names to include",
    ),
    backend: str = typer.Option(
        "box",
        "--backend",
        help="Draft backend: box (inset rectangle) or threshold (simple silhouette)",
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        help="Process only the first N thumbs (smoke runs)",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace existing draft LabelMe JSON files",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Enumerate only; print counts; write nothing",
    ),
):
    """Generate draft turtle polygons on 224×224 sorted thumbs (no yolo_finds)."""
    return outline_draft_command(
        thumbs=thumbs,
        out=out,
        classes=classes,
        backend=backend,
        limit=limit,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def main():
    """Main entry point for symsorter CLI"""
    app()

if __name__ == "__main__":
    main()
