# SymSorter

A CLIP/DINO-based image classification and dataset curation tool with a fast desktop GUI.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)

## Features

- Encode image folders with CLIP, DINOv3 (Transformers), or DINOv3 (timm)
- GUI for high-volume review with keyboard-driven class assignment
- Similarity sorting from a selected image (double-click in GUI)
- Class definitions from YAML, TXT, or NAMES-style files
- Save/reload classifications as COCO-style JSON (`classifications.json`)
- Export YOLO labels from GUI
- Prepare ROI thumbnails from Labelme/X-AnyLabeling datasets
- Write SymSorter class decisions back into source annotation JSON
- Merge many `classifications.json` files into one COCO JSON
- Sort and shard classified images into train-ready folder layouts
- Draft turtle outline polygons on sorted 224×224 thumbs (`outline-draft`)

## Installation

```bash
git clone https://github.com/NickMortimer/symsorter.git
cd symsorter

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

Verify:

```bash
symsorter --help
symsorter-gui
```

## Quick Start

### Workflow A: Classify a Folder of Images

1. Encode image folders (creates per-folder NPZ files like `encoding_clip_000.npz`):

```bash
symsorter encode /path/to/images
```

2. Create class definitions (`classes.yaml` example):

```yaml
classes:
  - name: turtle
    class_id: 7
    keystroke: "5"
    description: "Sea turtle"
  - name: waves
    class_id: 8
    keystroke: "2"
    description: "Water surface"
```

3. Start GUI with a generated NPZ:

```bash
symsorter gui --embeddings-file /path/to/images/encoding_clip_000.npz --class-file classes.yaml
```

4. Classify with keyboard shortcuts and save (`Ctrl+S`).
   This writes COCO-style `classifications.json` next to the NPZ file.

### Workflow B: Labelme/X-AnyLabeling Round Trip

1. Prepare ROI thumbnails from annotation JSON:

```bash
symsorter prepare /path/to/source_dataset --out /path/to/source_dataset/symsorter_prepared
```

2. Encode prepared thumbnails:

```bash
symsorter encode /path/to/source_dataset/symsorter_prepared
```

3. Classify in GUI and save:

```bash
symsorter gui --embeddings-file /path/to/source_dataset/symsorter_prepared/encoding_clip_000.npz --class-file classes.yaml
```

4. Apply SymSorter classes back to original annotation JSON:

```bash
symsorter update-json /path/to/source_dataset/symsorter_prepared --replace-label
```

## CLI Reference

SymSorter uses one main CLI with subcommands:

```bash
symsorter --help
```

Available commands:

- `encode INPUT_DIR`
  - Build or incrementally update per-folder embedding files.
  - Key options: `--pattern`, `--batch-size`, `--crop-size`, `--model-type`
  - Supported `--model-type`: `clip`, `dinov3`, `dinov3_timm`

- `inspect --npz-file PATH`
  - Show keys, embedding shape, dimensions, and metadata in an NPZ.

- `gui`
  - Launch GUI.
  - Options: `--embeddings-file`, `--class-file`

- `similarity EMBEDDINGS_FILE QUERY_IMAGE`
  - Find nearest images to a query image embedding.
  - Options: `--top-k`, `--output-dir`

- `prepare ROOT`
  - Create ROI thumbnails from Labelme/X-AnyLabeling JSON + image pairs.
  - Options: `--out`, `--thumb-size`, `--context-scale`

- `update-json PREPARED_DIR`
  - Apply SymSorter classes from prepared output back to source JSON files.
  - Options: `--source-root`, `--replace-label`

- `sort ROOT OUT`
  - Collect classified images and shard by class into an output tree.
  - Options: `--shard-size`, `--exclude-hidden`, `--link`, `--dry-run`

- `merge-coco ROOT`
  - Merge all nested `classifications.json` files into one COCO JSON.
  - Options: `--out`, `--exclude-hidden`

- `outline-draft`
  - Write draft LabelMe polygons **beside each thumb** (`foo.jpg` → `foo.json`; thumb-native; no `yolo_finds`).
  - Default backend: Ultralytics SAM (`pip install 'symsorter[sam]'`). Also `box` and `threshold`.
  - Options: `--thumbs`, `--out` (optional `manifest.jsonl` only), `--classes`, `--backend`, `--sam-model`, `--device`, `--limit`, `--overwrite`, `--dry-run`
  - Example:
    ```bash
    symsorter outline-draft \
      --thumbs /path/to/sorted \
      --out /path/to/outline_manifests \
      --backend sam \
      --limit 50
    ```

## GUI Controls

Common shortcuts:

- `Ctrl+O`: Load embeddings NPZ
- `Ctrl+Shift+O`: Load class file
- `Ctrl+S`: Save classifications
- `Ctrl+E`: Export YOLO annotations
- `Ctrl+J`: Export classifications JSON
- `Ctrl+Shift+J`: Import classifications JSON
- `R`: Reset order
- `L`: Toggle classification labels
- `Shift+C`: Sort by class
- `Ctrl+Shift+C`: Show classified images
- `Ctrl+=` / `Ctrl+-`: Thumbnail size
- `Shift+Ctrl+=` / `Shift+Ctrl+-`: Crop zoom
- Double-click image: sort by visual similarity
- `O`: Outline viewer — show sibling LabelMe polygons on the selected thumb(s)

Class assignment shortcuts come from your class file `keystroke` values.

## Class File Formats

YAML:

```yaml
classes:
  - name: turtle
    class_id: 7
    keystroke: "5"
    description: "Sea turtle"
  - name: sand
    keystroke: "s"
```

Text formats are also supported:

```text
turtle:5
sand:s
```

or one class per line:

```text
turtle
sand
```

## Output Files

- Embeddings: `encoding_<model_type>_<crop>.npz` in each processed folder
- Classification state: `classifications.json` (COCO-style) beside NPZ
- Prepared dataset map: `symsorter_prepare_map.json` in prepared output folder
- YOLO export (GUI): `labels/*.txt` and `classes.txt`

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).

## Support

- Issues: https://github.com/NickMortimer/symsorter/issues
- Email: nick.mortimer@csiro.au
