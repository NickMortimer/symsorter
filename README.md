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

# Outline / SAM work lives on this branch until merged:
git checkout cursor/outline-draft-mvp-8c57

python -m venv .venv
source .venv/bin/activate

pip install -e .
# Optional: Ultralytics SAM for outline-draft / in-viewer SAM
pip install -e ".[sam]"
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

- `outline-draft` / `outline-import-boxes` — see [Outline mode](#outline-mode-turtle-polygons) below.

- `remap-npz`
  - Rebuild per-shard NPZs under `sorted/` by joining original crop embeddings on filename (no re-encode).
  - Options: `--thumbs`, `--source`, `--npz-name`, `--classes`, `--overwrite`, `--dry-run`, `--limit-shards`
  - Example:
    ```bash
    symsorter remap-npz \
      --thumbs /Volumes/turtle_apfs/sorted \
      --source /Volumes/turtle_apfs/turtle_v01_run02 \
      --npz-name encoding_dinov3_224.npz \
      --overwrite
    ```

## Outline mode (turtle polygons)

Thumb-native LabelMe polygons on sorted **224×224** crops (`foo.jpg` → sibling `foo.json`). No `yolo_finds` mount required for day-to-day editing.

### Setup

```bash
git checkout cursor/outline-draft-mvp-8c57
source .venv/bin/activate
pip install -e ".[sam]"   # needed for SAM draft / in-viewer SAM
```

### Launch GUI on a sorted shard

```bash
symsorter gui --embeddings-file /path/to/sorted/turtle-surface/shard_00001/encoding_….npz
# or open a folder / NPZ from the GUI (Ctrl+O)
```

Optional: `Shift+W` sorts the grid by outline **needs-work** (unlocked/empty first, locked last). `Shift+O` toggles outline overlays on thumbs.

### Outline viewer (`O`)

Select a thumb (or several) and press **`O`** (or the Outline menu). The viewer edits polygons in thumb pixel space.

| Tool | Key | What it does |
|------|-----|----------------|
| **Edit** | `E` | Click a **vertex** to drag · click an **edge** to insert a vertex (then drag) · **double-click** a vertex to delete (keeps ≥3). **No colored fill** while editing so the shell edge stays visible. |
| Add point | `A` | Click near an edge (or on the poly) to insert a vertex. Fill also off. |
| **Lasso (−pts)** | `L` | Remove vertices **inside** a region (keeps ≥3). **Drag** freehand *or* **click** vertices one-by-one. Finish click-lasso by clicking near the **start** point or pressing **Enter**. **Esc** cancels an in-progress lasso; Backspace drops the last click point. |
| Eraser | `R` | Brush-erase nearby vertices. |
| SAM + / box | `P` / `B` | Add a new turtle polygon from a point or box prompt. |
| Del poly | `D` | Click a whole polygon to delete it. |
| Pan | `H` | Drag to pan; wheel zooms at pointer; `F` fits (+10% pad). Alt/right-drag also pans. |

**Lock & save**

- **`K`** — lock selected polygon as quality=`good` and **auto-save** LabelMe. Locked polys are green, not editable until unlocked (`Shift+K`).
- **`Ctrl+S`** — save without advancing.
- **`N`** / **`Ctrl+Enter`** — **Save & Next**: save, then jump to the next thumb that still needs work (respects `Shift+W` order / unlocked-first).
- **`S`** / “SAM from boxes” — run SAM on imported `turtle_box` rectangles (after `outline-import-boxes`).

Grid prompts (`P`/`B`/`D` outside the viewer) use crop-relative coords when crop zoom is on. `Shift+P` turns outline prompt mode off.

### Draft polygons from the CLI

```bash
# SAM drafts beside each thumb ( Ultralytics; needs [sam] )
symsorter outline-draft \
  --thumbs /path/to/sorted \
  --out /path/to/outline_manifests \
  --backend sam \
  --limit 50

# Also available: --backend box | threshold
# Use --overwrite carefully: skip_json skips any sibling .json (including old length/width lines)
```

Import YOLO/SHAI boxes onto thumbs (for “SAM from boxes” in the viewer):

```bash
symsorter outline-import-boxes \
  --thumbs /Volumes/turtle_apfs/sorted \
  --yolo-finds /Volumes/turtle_apfs/yolo_finds \
  --overwrite --limit 100
# optional: --sam   or press S in the Outline viewer
```

### Outline quality notes

- Single class `turtle`. Outline the **animal body** only.
- **Exclude** cast shadows, wake, and **sunglint/glitter** — even on the shell edge. Mild glint on a still-readable carapace is OK; skip (or leave unlocked) when heavy wash makes the edge guesswork.
- Prefer clear surface (then clear deep) for the first cleaned batch; wave/double-flipper / heavy blur cases → leave unlocked or skip.
- Lock (`K`) only when the outline is good enough that you do not want to redo it.

## GUI Controls

Common shortcuts:

- `Ctrl+O`: Load embeddings NPZ
- `Ctrl+Shift+O`: Load class file
- `Ctrl+S`: Save classifications (in outline viewer: save polygons)
- `Ctrl+E`: Export YOLO annotations
- `Ctrl+J`: Export classifications JSON
- `Ctrl+Shift+J`: Import classifications JSON
- `R`: Reset order
- `L`: Toggle classification labels (in outline viewer: lasso tool)
- `Shift+C`: Sort by class
- `Ctrl+Shift+C`: Show classified images
- `Ctrl+=` / `Ctrl+-`: Thumbnail size
- `Shift+Ctrl+=` / `Shift+Ctrl+-`: Crop zoom
- Double-click image: sort by visual similarity
- `O`: Outline viewer (see [Outline mode](#outline-mode-turtle-polygons))
- `Shift+O`: Toggle outline overlays (crop-aware; locked polys draw green)
- `Shift+W`: Sort by outline needs-work (unlocked/empty first, locked last)
- `P` / `B` / `D`: In-grid SAM point / box / delete (multi-turtle; Del = last; Shift+Del = clear)
- `Shift+P`: Turn outline prompt mode off

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
