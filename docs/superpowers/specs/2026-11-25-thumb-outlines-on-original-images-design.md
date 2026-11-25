# Thumb outlines on original survey images — design

Status: approved (2026-11-25)

## Intent

Turtle outlines are drawn and locked on 224×224 thumbnails in
`training/surface` and `training/deep`. A thumbnail is only a small crop of a
5472×3648 drone frame, so it is hard to judge an outline without seeing the
turtle in its original frame. We also want the outlines as annotations on the
original images, for training and for checking against survey records.

This feature has two parts:

1. A CLI that takes the root of the survey folders and builds an output folder.
   It hardlinks each original frame into the folder and writes a LabelMe JSON
   beside it with the outlines converted to full-frame coordinates.
2. A panel in the existing Outline viewer that shows the current thumbnail's
   original frame, with the crop window and outlines drawn on it. You can also
   edit the outlines there.

## What exists today

- **Thumbnails and outlines.** The thumbnails live in
  `/Volumes/turtle_apfs/training/{surface,deep}/*_224px.jpg`, each with a
  sibling LabelMe `*.json`. Each folder also has an `encoding_dinov3_224.npz`
  for the GUI. On 2026-11-25:
  - `surface`: 1,200 thumbnails with locked outlines (1,205 polygons).
  - `deep`: 22 thumbnails with locked outlines. More are being added.
- **Older copies to ignore.** `sorted/turtle-surface/shard_00001` holds the
  same image files (hardlinks), but its JSON files are separate and stale:
  1,195 of the 1,200 locked thumbnails have no locked polygon there. Most of
  its polygons are unprompted SAM drafts that cover the whole crop. This
  feature does not read or write `sorted/`.
- **Thumbnail to frame mapping.** `training/thumb_roi.csv` has one row per
  training thumbnail: `view, crop_basename, flight, x1, y1, x2, y2, confidence,
  class_name, original_image, status`. Every locked thumbnail has a row.
  - `original_image` is an absolute path on another machine
    (`/media/mor582/Timor2/surveys/AU/{SITE}/{FLIGHT}/{stem}.JPG`). Only its
    stem and the `flight` column can be trusted.
- **Crop geometry.** `symsorter/outline/mapping.py` already has
  `fixed_size_crop_window`, which centres a 224 window on the YOLO box and
  clamps it to the frame. Checked on 40 random thumbnails, the predicted
  window matched the real frame pixel for pixel (template-match score ≥ 0.983,
  offset 0 in every case).
  - The same module has `match_crop_window_cv2` (about 0.5 s per frame) and
    `match_crop_window_via_slices` for thumbnails with no CSV row.
  - Its `load_metadata_box` looks for `*_crops_metadata.csv` files, which do
    not exist on this volume.
- **Where the originals are.** `yolo_finds/{FLIGHT}/images/{stem}.JPG` holds
  some originals on `turtle_apfs`. Locked thumbnails whose original is there:
  746 of 1,200 surface, 2 of 22 deep. The rest are in the Timor2 survey tree,
  laid out as `surveys/AU/{SITE}/{FLIGHT}/{stem}.JPG`, which is not mounted
  right now.
- **Frames shared by several thumbnails.** Of the 746 surface thumbnails above,
  113 frames hold more than one locked thumbnail, and 8 frames have crop
  windows that overlap.
- **Outline viewer.** `symsorter/ui/outline_dialog.py` has
  `OutlineViewerDialog` and `OutlineCanvas`.
  - The canvas draws any pixmap with polygons in that image's own pixel space,
    and supports zoom, pan and vertex editing.
  - The dialog loads and saves the thumbnail's sibling JSON (`refresh`,
    `save_current`) through `symsorter/outline/export_labelme.py`.
- **Other code to reuse.** `symsorter/commands/sort_images.py` already offers
  symlink or copy. `symsorter/commands/outline_import_boxes.py` is the pattern
  for a batch CLI with counts and a manifest.

## Goals

- A single command builds a folder of original frames and LabelMe JSONs that
  LabelMe or X-AnyLabeling can open directly. Each frame contains every locked
  outline from the thumbnails cropped from it, in full-frame coordinates.
- From the Outline viewer you can see the current thumbnail on its original
  frame, with the crop window and outlines on top, and adjust the outlines
  there.
- It works for both `surface` and `deep`, and keeps working as more deep
  outlines are locked.
- It finds originals under any survey root you pass in, in either of the
  layouts above.

## Non-goals

- Running SAM on full frames. SAM prompting stays on the thumbnail
  (`sam_polygon` is built around 224 crops).
- Writing to or syncing `sorted/` sidecars.
- Changing the source survey JSONs (the `turtle_3_yolo` boxes in
  `yolo_finds/.../images/*.json`, or the Timor2 survey files).
- Merging outlines from overlapping crops into one polygon.
- Recovering originals that are not on any mounted root. They are reported as
  missing.

## Anti-goals (failure even if it works)

- Exporting unprompted whole-crop SAM drafts as if they were real outlines.
- Changing or corrupting the original survey images. Output frames are links,
  never re-encoded or rewritten.
- Hiding thumbnails that were skipped. Every skipped thumbnail is counted and
  listed with its reason.
- Two copies of an outline that drift apart (see "Single source of truth").

## Constraints

- **Hardlinks and volumes.** Hardlinks only work when the output folder and the
  originals are on the same volume. `yolo_finds` is on `turtle_apfs`, but
  Timor2 is a separate volume, so the CLI needs a symlink or copy fallback.
- **Coordinates.** LabelMe points are `[x, y]` in image pixels. To get frame
  coordinates, add the crop window's left/top to the thumbnail coordinates; to
  go back, subtract them.
- **Frame size.** Read it from the image header. Do not assume 5472×3648.
- **Shared originals.** Outputs are read-only views of the originals: the
  output folder never modifies the source images or their JSONs.
- **Code style.** Python 3.9+, PySide6, Typer CLI, LabelMe 5.0.1 schema as
  written by `export_labelme.py`, tests in `symsorter/tests/` with pytest.

## Approaches considered

**A. Build-time export plus a live viewer panel, with the thumbnail JSON as the
single source of truth (chosen).** The CLI regenerates frame JSONs from the
thumbnail JSONs. The viewer panel converts coordinates on the fly, and edits
made on the frame are converted back and saved to the thumbnail JSON.

- Pros: one place to edit, so nothing drifts. The CLI can be re-run at any
  time. Existing grid overlays and lock and needs-work logic keep working.
- Cons: thumbnail JSONs may hold points outside 0–224 when an outline is
  extended past the crop on the frame.

**B. The frame JSON as source of truth.** The CLI seeds the frame JSONs, and
from then on the viewer reads and writes the frame JSON.

- Rejected: every existing tool (grid overlay, lock, needs-work sort, Save &
  Next) works on the thumbnail JSON. Two sources would drift, and re-running
  the CLI would overwrite frame edits.

**C. Export only, with no viewer panel.** Check outlines by opening the output
folder in LabelMe.

- Rejected: you asked for the panel and for editing. A separate tool breaks
  the review flow in the Outline viewer.

## Decisions

### Which thumbnails and outlines

- Input folders: one or more flat folders such as `training/surface` and
  `training/deep`, holding `*_224px.jpg` with sibling JSON.
- Export only **locked** turtle polygons (`is_polygon_locked`). Unlocked drafts
  are counted and skipped. *(My call; the alternative exports thousands of
  whole-crop boxes.)*
- The viewer panel shows every polygon in the thumbnail JSON, locked or not,
  as the existing canvas does.

### Finding the original frame

- `thumb_roi.csv` is keyed by the full thumbnail basename (`crop_basename`,
  e.g. `050_DJIP4_RGB_AU_MANGB_20230503T113703_2116_det_0_turtle_3_224px.jpg`).
  - Checked: every one of the 2,361 surface and 4,576 deep thumbnails has
    exactly one row, there are no duplicate basenames, and every row has
    `status` = `ok`.
  - Only rows with `status` = `ok` are used. *(My call, as a guard for future
    CSVs.)*
- The crop window is `fixed_size_crop_window(x1, y1, x2, y2)`, using the frame
  size from the image header.
- No template-match fallback. A thumbnail with no row is reported as
  `no_roi_row` and skipped: without a row there is no `flight`, and every
  training thumbnail has one. *(My call, YAGNI; `low_match` is dropped.)*
- Locate `{stem}.JPG` (case-insensitive on the extension) under each survey
  root, in the order the roots are given, using the CSV `flight`:
  1. `{root}/{FLIGHT}/images/{stem}.*` (yolo_finds layout)
  2. `{root}/{COUNTRY}/{SITE}/{FLIGHT}/{stem}.*` (survey layout; `SITE` is the
     part of `FLIGHT` before `_`)
  3. If neither is found, a recursive search of the root for `{stem}.*`.
     Results are cached per root for the run.
- The first match wins.

### Single source of truth

- The thumbnail JSON is the only place outlines are edited. Frame JSONs in the
  output folder are generated and may be overwritten.
- Polygon points converted back from the frame may fall outside 0–224. They are
  saved as they are, not clipped. *(My call. The grid overlay simply draws past
  the edge.)*

### CLI: `symsorter outline-export-frames`

```text
symsorter outline-export-frames \
  --thumbs /Volumes/turtle_apfs/training/surface \
  --thumbs /Volumes/turtle_apfs/training/deep \
  --survey-root /Volumes/turtle_apfs/yolo_finds \
  --survey-root /Volumes/Timor2/surveys \
  --roi-csv /Volumes/turtle_apfs/training/thumb_roi.csv \
  --out /Volumes/turtle_apfs/outlines_on_frames \
  [--link hard|sym|copy]   # default hard; falls back to sym when across volumes
  [--overwrite] [--dry-run] [--limit N]
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

- Each output polygon has `label: "turtle"`, `shape_type: "polygon"`, points in
  frame pixels, and these flags:
  - `source_thumb` (basename)
  - `view` (`surface` or `deep`)
  - `det_index`
  - `crop_xyxy`
  - `crop_method` (always `roi_csv`; there is no template-match fallback)
  - every original thumbnail flag (`locked`, `quality`, …)
- Several thumbnails from one frame all go into the same `{stem}.json`. Their
  polygons are kept separate, each tagged by `source_thumb` and `det_index`,
  even where crops overlap.
- Without `--overwrite`, an existing frame JSON is left alone and counted as
  `skipped_existing`. With `--overwrite`, the JSON is rebuilt from scratch from
  the thumbnails in this run. *(My call.)*
- Status values:
  - `ok`
  - `no_locked`
  - `no_roi_row`
  - `frame_missing`
  - `skipped_existing`
  - `link_failed`
- At the end, print the counts in the same style as `outline-import-boxes`.

### Viewer panel

- Add a collapsible panel on the right of `OutlineViewerDialog`, toggled with
  `Shift+F` and with a "Frame" button. It is hidden by default.
- The panel holds a second `OutlineCanvas` showing the original frame:
  - the crop window as a dashed rectangle
  - the current thumbnail's polygons, shifted into frame coordinates
  - locked polygons from other locked thumbnails of the same frame,
    read-only and dimmed, so neighbours are visible
    - Neighbours are found through the CSV: every row with the same `flight`
      and `original_image` stem.
    - Each neighbour's JSON is read from `{training_root}/{view}/`, where
      `training_root` is the parent of the current thumbnail folder and `view`
      comes from the CSV. A surface thumbnail can therefore show deep
      neighbours. *(My call.)*
    - Neighbours can't be selected or hit-tested. Clicks only ever reach the
      current thumbnail's polygons, so an edit can't land on another
      thumbnail's outline. *(My call.)*
- It follows Prev/Next and Save & Next. Frames are loaded off the UI thread.
  The last few decoded frames are cached.
- **Editing.** Vertex edit, add point, lasso, eraser and delete work on the
  current thumbnail's polygons in the frame panel.
  - Each change is converted back to thumbnail coordinates and pushed into the
    dialog's `_polygons`.
  - Both canvases repaint and the dialog is marked dirty.
  - Save keeps using the existing `save_current`.
  - SAM tools (point/box) are disabled in the frame panel.
- **Locked polygons:** same rule as the thumbnail canvas. Locked polygons
  can't be edited or deleted in the frame panel; unlock with `Shift+K`, edit,
  re-lock with `K`. *(Decided 2026-11-25.)*
- **Locating the frame.** The panel uses the same resolver as the CLI. Survey
  roots come from a new `--survey-root` option on `symsorter gui` (repeatable)
  and a "Set survey roots…" item in the File menu, saved with `QSettings`. The
  ROI CSV is found as `../thumb_roi.csv` next to the thumbnail folder, or set
  in the same dialog.
- If no frame is found, the panel shows the reason (`frame_missing` or
  `no_roi_row`) instead of an image.

### Shared code

- A new module, `symsorter/outline/frames.py`:
  - `RoiIndex.load(csv) -> RoiIndex`
  - `FrameResolver(roots).find(flight, stem) -> Path | None`
  - `crop_window_for_thumb(thumb, roi, frame_path) -> CropWindow | status`
  - `thumb_to_frame(points, crop)` and `frame_to_thumb(points, crop)`
- The CLI and the viewer both use this module. Its geometry builds on
  `fixed_size_crop_window` in `mapping.py`.

## Left to the builder

- The look of the panel: splitter vs dock, default zoom, colours for
  neighbour polygons.
- The size of the frame cache and how background loading is done.
- How the recursive fallback search is indexed.
- Test fixtures. Use small synthetic frames, not real survey images.
- README section wording.

## Success

- After running the CLI against `training/surface` and `training/deep` with
  `--survey-root yolo_finds`:
  - 748 locked thumbnails export (746 surface, 2 deep).
  - Every other locked thumbnail is listed with a reason.
  - Opening any output frame in LabelMe shows the outlines on the turtles.
- After mounting Timor2 and adding it as a root, the missing frames export
  with no code change.
- In the Outline viewer, `Shift+F` shows the frame. An edit in either canvas
  appears in the other, and is saved to the thumbnail JSON only.
- Round-trip tests: `frame_to_thumb(thumb_to_frame(p)) == p`, and the edges of
  the crop window stay exact at frame borders.
