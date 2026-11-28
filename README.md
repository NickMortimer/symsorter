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
- Draft and refine turtle outline polygons on sorted 224×224 thumbs (`outline-draft`, Outline viewer)

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
  - Options: `--embeddings-file`, `--class-file`, `--survey-root` (repeatable; saved as the frame panel's survey roots)

- `outline-export-frames`
  - Write locked thumb outlines onto the original survey frames. See [Outlines on original frames](#outlines-on-original-frames).

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

### Outlines on original frames

`outline-export-frames` writes locked turtle polygons from 224×224 thumbs onto the original survey frames. The thumbnail JSON stays the only file that is edited; frame JSON in the output folder is generated and may be overwritten.

```bash
symsorter outline-export-frames \
  --thumbs /Volumes/turtle_apfs/training/surface \
  --thumbs /Volumes/turtle_apfs/training/deep \
  --survey-root /Volumes/turtle_apfs/yolo_finds \
  --survey-root /Volumes/Timor2/surveys \
  --roi-csv /Volumes/turtle_apfs/training/thumb_roi.csv \
  --out /Volumes/turtle_apfs/outlines_on_frames
# optional:
#   --link hard|sym|copy   # default hard; falls back to sym when source and output are on different volumes
#   --overwrite            # without it, an existing frame JSON is left alone (skipped_existing)
#   --dry-run --limit N
```

Output layout:

```text
out/
  {FLIGHT}/
    {stem}.JPG        # hardlink (or symlink/copy) of the original
    {stem}.json       # LabelMe 5.0.1, imagePath "{stem}.JPG", real width/height
  manifest.jsonl      # one row per thumbnail: thumb, view, flight, stem,
                      # crop [l,t,r,b], crop_method, n_locked, status, frame_src
  summary.json        # counts by status and view
```

Each thumbnail is counted in `manifest.jsonl`. Status values:

- `ok`
- `no_locked`
- `no_roi_row` (no CSV row, or a row whose `status` is not `ok`; there is no template-match fallback)
- `frame_missing`
- `skipped_existing`
- `link_failed`

`--link` defaults to `hard`. A hardlink across volumes fails with `EXDEV`, and the command falls back to a symlink. `sym` and `copy` do not fall back.

In the Outline viewer, **Shift+F** (or the Frame button) opens the original-frame panel. It is hidden until you toggle it. **File → Set Survey Roots…** saves the survey roots and the ROI CSV. `symsorter gui --survey-root PATH` (repeatable) replaces the saved survey roots before the window opens. The ROI CSV defaults to `thumb_roi.csv` next to the thumbnail folder when the dialog has not set one. If the frame cannot be located, the panel shows `frame_missing` or `no_roi_row` instead of an image.

### Outline quality notes

- Single class `turtle`. Outline the **animal body** only.
- **Exclude** cast shadows, wake, and **sunglint/glitter** — even on the shell edge. Mild glint on a still-readable carapace is OK; skip (or leave unlocked) when heavy wash makes the edge guesswork.
- Prefer clear surface (then clear deep) for the first cleaned batch; wave/double-flipper / heavy blur cases → leave unlocked or skip.
- Lock (`K`) only when the outline is good enough that you do not want to redo it.

## Slice review

Review 624 px slices cut from full drone frames (a `marine_count` slice project) and mark each complete, rejected or skipped. File > **Open Slice Project...**, or `symsorter-gui --slice-project DIR`. The first open asks for survey roots (prefilled from Set Survey Roots) and a reviewer name, saved in `project.local.json` (laptop only). Frames are found by flight + stem and must match `frame_wh`; a slice whose frame is missing or the wrong size shows the problem and cannot be saved.

The loop: open the project, check or fix the outlines (add a missed turtle with the SAM + tool), press `C`. Outline the whole turtle even past the slice edge; a complete slice whose outline crosses the edge is refused, and `R` moves the slice the least distance that holds every outline (check it, then `C`). The window resumes at the next unreviewed slice. Skipped slices count as done for `N`; go back to them with `P` (the title shows the skip count). A slice is only saved once its image is on screen, and holding `C`/`X`/`Space` does not repeat. Use **Settings...** in the window to change survey roots or reviewer later (the window keeps its place). Opening a project that is already open raises the existing window.

Keys: `C` complete · `X` reject (optional reason) · `Space` skip · `N` next unreviewed · `P` previous · `R` re-centre · `A` adopt neighbour outlines · `E` edit · `L` lasso · `S` SAM point · `D` delete polygon · `H` pan. The slice border is dashed; neighbours are faint.

Export locked thumb outlines for marine_objects (run on the laptop; needs the sealed-survey list, see Sync):

```bash
symsorter outline-export-contract --thumbs DIR [--thumbs DIR2 ...] --roi-csv thumb_roi.csv \
  --survey-root DIR [--survey-root DIR2 ...] --denylist sealed_surveys.txt --out outlines.jsonl
```

It writes locked polygons in full-frame pixels to `outlines.jsonl` and prints counts: `ok` outlines written; `sealed` thumbs skipped because their survey is in the denylist; `no_locked` thumbs with no locked polygon; `no_roi_row` locked thumbs missing from the ROI CSV; `frame_missing` original frame not found under the survey roots; `frame_unreadable` frame found but its header could not be read.

Sync (run on the laptop; each file has one writer, so rsync never conflicts):

```bash
rsync -av deepthinker:<denylist path on deepthinker> ./sealed_surveys.txt   # once, and whenever the seal changes
rsync -av outlines.jsonl deepthinker:/home/mor582/proj/slices/
rsync -av --exclude project.local.json deepthinker:/home/mor582/proj/slices/slices_r01/project.json ./slices_r01/
rsync -av --exclude project.local.json ./slices_r01/review.jsonl deepthinker:/home/mor582/proj/slices/slices_r01/
```

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
- `Shift+F`: Original-frame panel in the Outline viewer (File → Set Survey Roots…)
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
