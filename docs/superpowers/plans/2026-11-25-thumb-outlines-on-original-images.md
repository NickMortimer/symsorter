# Thumb Outlines on Original Frames Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export locked thumbnail outlines onto their original drone frames
(CLI) and show/edit them on the frame inside the Outline viewer.

**Architecture:** A pure-Python module `symsorter/outline/frames.py` owns the
CSV index, frame lookup and thumb↔frame geometry. The CLI
`outline-export-frames` and a new `FramePanel` widget both use it. The
thumbnail JSON stays the single source of truth: the panel edits frame-space
copies and writes them back into the dialog's `_polygons`.

**Tech Stack:** Python 3.9+, PySide6, Typer, Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-11-25-thumb-outlines-on-original-images-design.md`

**Branch:** `feat/thumb-outlines-on-frames` (off `origin/main`, main checkout).

## Global Constraints

- Python 3.9+: every new module starts with `from __future__ import annotations`.
- LabelMe 5.0.1 schema as written by `symsorter/outline/export_labelme.py`.
- Output frames are hardlinks/symlinks/copies; source images and source JSONs are never opened for writing.
- Nothing reads or writes `sorted/`.
- Export only locked turtle polygons (`is_polygon_locked`); every skipped thumbnail is counted and in `manifest.jsonl` with its status.
- Frame size comes from the image header (Pillow `Image.open(...).size`), never assumed 5472×3648.
- Only CSV rows with `status == "ok"` are used. No template-match fallback; missing row → `no_roi_row`.
- Points converted back from the frame are not clipped to 0–224.
- Locked polygons can't be edited or deleted in the frame panel (same rule as the thumbnail canvas; unlock `Shift+K`, re-lock `K`).
- SAM tools are disabled in the frame panel.
- Tests live in `symsorter/tests/`, use small synthetic images (e.g. 640×480), never real survey data.
- Repo Python conventions: type hints and a one-line PEP 257 docstring on public functions.
- Test command: `python -m pytest symsorter/tests -q`. Two pre-existing failures in `test_basic.py` (`test_load_existing_embeddings`, `test_load_nonexistent_embeddings`) are known and out of scope.

## Review Focus

1. **Re-running the CLI into an existing output folder** — an existing link must not count as `link_failed`; without `--overwrite` the frame is `skipped_existing`; a frame JSON written earlier *in the same run* must not count as existing. (Task 3, `test_rerun_*`.)
2. **One frame reached from both `surface` and `deep` thumbnails** — one link, one JSON holding both polygons, tagged with their own `view`. (Task 3, `test_surface_and_deep_share_frame`.)
3. **Recursive fallback picking the wrong file** — a sibling `{stem}.json` or macOS `._{stem}.JPG` must never be returned as the frame. (Task 2, `test_resolver_ignores_json_and_appledouble`.)
4. **Fast Prev/Next while a frame is still loading** — a late result for the previous thumbnail must not appear under the new one. (Task 5, `test_stale_load_is_ignored`.)
5. **Frame-panel edit then Prev/Next** — the existing "Unsaved outline" prompt must fire, because edits set `_dirty`. (Task 6, `test_frame_edit_marks_dirty_and_saves_thumb_coords`.)

---

## File Structure

| File | Responsibility |
|---|---|
| Create `symsorter/outline/frames.py` | `RoiRow`, `RoiIndex`, `FrameResolver`, `frame_size`, `crop_window_for_row`, `thumb_to_frame`, `frame_to_thumb`, `iter_flat_thumbs` |
| Create `symsorter/commands/outline_export_frames.py` | `link_frame`, `outline_export_frames_command` |
| Modify `symsorter/cli.py` | register `outline-export-frames`; add `--survey-root` to `gui` |
| Modify `symsorter/ui/outline_dialog.py` | `OutlineCanvas` overlays/crop rect/message; dialog splitter, `Shift+F`, sync |
| Create `symsorter/ui/frame_settings.py` | `FrameSettings`, QSettings load/save, "Set survey roots…" dialog |
| Create `symsorter/ui/frame_panel.py` | `FramePanel` widget, async frame loader, LRU cache |
| Modify `symsorter/ui/menus.py`, `symsorter/image_browser.py` | File menu item → settings dialog |
| Create `symsorter/tests/test_outline_frames.py` | Tasks 1–2 |
| Create `symsorter/tests/test_outline_export_frames.py` | Task 3 |
| Create `symsorter/tests/test_frame_panel.py` | Tasks 4–6 |
| Modify `README.md` | usage section |

Shared test fixture (define in each test file that needs it, or a
`conftest.py` in `symsorter/tests/` — builder's choice):

```python
THUMB = "050_DJIP4_RGB_AU_MANGB_20230503T113703_2116_det_0_turtle_3_224px.jpg"
STEM = "DJIP4_RGB_AU_MANGB_20230503T113703_2116"
FLIGHT = "MANGB_20230503T1137"
ROI_HEADER = "view,crop_basename,flight,x1,y1,x2,y2,confidence,class_name,original_image,status"
# 640x480 frame; box centred at (300, 200) → crop window (188, 88, 412, 312)
```

---

### Task 1: ROI index and thumb↔frame geometry

**Files:**
- Create: `symsorter/outline/frames.py`
- Test: `symsorter/tests/test_outline_frames.py`

**Interfaces:**
- Consumes: `CropWindow`, `fixed_size_crop_window` from `symsorter/outline/mapping.py`.
- Produces:
  - `@dataclass(frozen=True) RoiRow(view: str, crop_basename: str, flight: str, x1: float, y1: float, x2: float, y2: float, original_stem: str)` — `original_stem = PureWindowsPath/PurePosixPath(original_image).stem` (path is from another machine; only the stem is used).
  - `RoiIndex.load(csv_path: Path) -> RoiIndex`; `RoiIndex.get(crop_basename: str) -> RoiRow | None` (None when missing or `status != "ok"`); `RoiIndex.same_frame(flight: str, stem: str) -> list[RoiRow]`.
  - `frame_size(path: Path) -> tuple[int, int]` — `(width, height)` from header.
  - `crop_window_for_row(row: RoiRow, frame_wh: tuple[int, int]) -> CropWindow` — `fixed_size_crop_window(..., fixed_size=224)` with `method="roi_csv"` (use `dataclasses.replace`).
  - `thumb_to_frame(points: list[list[float]], crop: CropWindow) -> list[list[float]]` (add `left`, `top`).
  - `frame_to_thumb(points: list[list[float]], crop: CropWindow) -> list[list[float]]` (subtract).
  - `iter_flat_thumbs(folder: Path) -> list[Path]` — sorted `*_224px.jpg` in `folder` (not recursive), skipping names starting with `._`.

> Spec lists `crop_window_for_thumb(...) -> CropWindow | status`. The plan splits it: `RoiIndex.get` returning `None` is the `no_roi_row` status, `crop_window_for_row` is the geometry.

- [ ] **Step 1: Write the failing tests**

```python
def test_roi_index_get_and_status_filter(tmp_path):
    csv = tmp_path / "thumb_roi.csv"
    csv.write_text(
        ROI_HEADER + "\n"
        f"surface,{THUMB},{FLIGHT},290,190,310,210,0.5,turtle_3,"
        f"/media/x/Timor2/surveys/AU/MANGB/{FLIGHT}/{STEM}.JPG,ok\n"
        f"deep,bad_224px.jpg,{FLIGHT},0,0,1,1,0.5,turtle_3,/a/{STEM}.JPG,failed\n"
    )
    idx = RoiIndex.load(csv)
    row = idx.get(THUMB)
    assert row.original_stem == STEM and row.view == "surface"
    assert idx.get("bad_224px.jpg") is None
    assert idx.get("nope.jpg") is None
    assert [r.crop_basename for r in idx.same_frame(FLIGHT, STEM)] == [THUMB]

def test_crop_window_and_round_trip():
    row = RoiRow("surface", THUMB, FLIGHT, 290, 190, 310, 210, STEM)
    crop = crop_window_for_row(row, (640, 480))
    assert (crop.left, crop.top, crop.right, crop.bottom) == (188, 88, 412, 312)
    assert crop.method == "roi_csv"
    pts = [[0.0, 0.0], [223.5, 10.25], [-5.0, 230.0]]
    assert frame_to_thumb(thumb_to_frame(pts, crop), crop) == pts

def test_crop_window_clamped_at_frame_border():
    row = RoiRow("surface", THUMB, FLIGHT, 630, 470, 640, 480, STEM)
    crop = crop_window_for_row(row, (640, 480))
    assert (crop.left, crop.top, crop.right, crop.bottom) == (416, 256, 640, 480)
    assert thumb_to_frame([[224.0, 224.0]], crop) == [[640.0, 480.0]]

def test_frame_size_reads_header(tmp_path):
    p = tmp_path / "f.JPG"; Image.new("RGB", (640, 480)).save(p, "JPEG")
    assert frame_size(p) == (640, 480)

def test_iter_flat_thumbs_skips_appledouble(tmp_path):
    Image.new("RGB", (224, 224)).save(tmp_path / THUMB)
    (tmp_path / f"._{THUMB}").write_bytes(b"x")
    (tmp_path / "sub").mkdir(); Image.new("RGB", (224, 224)).save(tmp_path / "sub" / THUMB)
    assert iter_flat_thumbs(tmp_path) == [tmp_path / THUMB]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest symsorter/tests/test_outline_frames.py -q`
Expected: FAIL — `ModuleNotFoundError: symsorter.outline.frames`

- [ ] **Step 3: Implement the Task 1 interfaces in `symsorter/outline/frames.py`**

Use `csv.DictReader`; `same_frame` is a dict keyed by `(flight, original_stem)` built in `load`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest symsorter/tests/test_outline_frames.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add symsorter/outline/frames.py symsorter/tests/test_outline_frames.py
git commit -m "feat(outline): ROI index and thumb/frame geometry"
```

---

### Task 2: Frame resolver

**Files:**
- Modify: `symsorter/outline/frames.py`
- Test: `symsorter/tests/test_outline_frames.py`

**Interfaces:**
- Produces: `FrameResolver(roots: Sequence[Path])`, `FrameResolver.find(flight: str, stem: str) -> Path | None`. `IMAGE_EXTS = {".jpg", ".jpeg", ".png"}` (compare `suffix.lower()`).

Lookup per root, in root order, first match wins:
1. `{root}/{flight}/images/{stem}.*`
2. `{root}/*/{site}/{flight}/{stem}.*` where `site = flight.split("_")[0]` (the `*` is the country folder)
3. Recursive: lazily build `dict[str, Path]` of `stem → path` for the root with one `os.walk` (cached per root for the resolver's life; first path in sorted walk order wins).

In every step, ignore names starting with `._` and suffixes not in `IMAGE_EXTS`. Missing roots are skipped.

- [ ] **Step 1: Write the failing tests**

```python
def _img(p): p.parent.mkdir(parents=True, exist_ok=True); Image.new("RGB", (8, 8)).save(p, "JPEG")

def test_resolver_yolo_layout_first_root_wins(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _img(a / FLIGHT / "images" / f"{STEM}.JPG"); _img(b / FLIGHT / "images" / f"{STEM}.JPG")
    assert FrameResolver([a, b]).find(FLIGHT, STEM) == a / FLIGHT / "images" / f"{STEM}.JPG"

def test_resolver_survey_layout_lowercase_ext(tmp_path):
    p = tmp_path / "AU" / "MANGB" / FLIGHT / f"{STEM}.jpg"; _img(p)
    assert FrameResolver([tmp_path]).find(FLIGHT, STEM) == p

def test_resolver_recursive_fallback(tmp_path):
    p = tmp_path / "odd" / "place" / f"{STEM}.JPG"; _img(p)
    assert FrameResolver([tmp_path / "missing", tmp_path]).find(FLIGHT, STEM) == p

def test_resolver_ignores_json_and_appledouble(tmp_path):
    d = tmp_path / FLIGHT / "images"; d.mkdir(parents=True)
    (d / f"{STEM}.json").write_text("{}"); (d / f"._{STEM}.JPG").write_bytes(b"x")
    assert FrameResolver([tmp_path]).find(FLIGHT, STEM) is None

def test_resolver_missing(tmp_path):
    assert FrameResolver([tmp_path]).find(FLIGHT, STEM) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest symsorter/tests/test_outline_frames.py -q -k resolver`
Expected: FAIL — `ImportError: cannot import name 'FrameResolver'`

- [ ] **Step 3: Implement `FrameResolver` in `symsorter/outline/frames.py`**

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest symsorter/tests/test_outline_frames.py -q`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add symsorter/outline/frames.py symsorter/tests/test_outline_frames.py
git commit -m "feat(outline): resolve original frames under survey roots"
```

---

### Task 3: `outline-export-frames` CLI

**Files:**
- Create: `symsorter/commands/outline_export_frames.py`
- Modify: `symsorter/cli.py` (import next to `outline_import_boxes_command`; new `@app.command(name="outline-export-frames")`)
- Test: `symsorter/tests/test_outline_export_frames.py`

**Interfaces:**
- Consumes: all of Task 1–2; `load_labelme_json`, `extract_polygons`, `is_polygon_locked`, `labelme_path_for_thumb`, `write_labelme_json` from `export_labelme.py`; `parse_thumb_filename` for `det_index`.
- Produces:
  - `link_frame(src: Path, dst: Path, mode: str) -> str` — `mode` in `{"hard", "sym", "copy"}`; returns the method actually used, or `"existing"` if `dst` already exists (left untouched). `hard` falls back to `sym` only on `OSError` with `errno.EXDEV`. Symlinks point at `src.resolve()`. `copy` uses `shutil.copy2`. Other `OSError`s propagate.
  - `outline_export_frames_command(*, thumbs: list[Path], survey_roots: list[Path], roi_csv: Path, out: Path, link: str = "hard", overwrite: bool = False, dry_run: bool = False, limit: int | None = None) -> int` — returns 0 if any `ok` or `skipped_existing`, else 1.

Algorithm:
1. Collect `iter_flat_thumbs` over every `--thumbs` folder (in order), apply `limit`.
2. Per thumb: locked polys = `[p for p in extract_polygons(json) if is_polygon_locked(p)]`; none → `no_locked`. `roi.get(name)` None → `no_roi_row`. `resolver.find(row.flight, row.original_stem)` None → `frame_missing`. Otherwise add to a group keyed `(row.flight, row.original_stem)`.
3. Before writing anything, record which `out/{flight}/{stem}.json` already exist. Per group: existed and not `overwrite` → every thumb in it `skipped_existing`. Else (unless `dry_run`) `link_frame(frame, out/flight/frame.name, link)` — on `OSError` all thumbs in the group get `link_failed` — then write one fresh LabelMe doc (`version "5.0.1"`, `imagePath` = frame name, real `imageWidth/Height`, `imageData None`, `flags {}`) with every thumb's locked polygons in frame coords, `write_labelme_json(..., overwrite=True)`; thumbs → `ok`.
4. Shape: `{"label": "turtle", "shape_type": "polygon", "group_id": None, "points": ..., "flags": {**poly_flags, "source_thumb": name, "view": row.view, "det_index": parsed.det_index, "crop_xyxy": [l, t, r, b], "crop_method": "roi_csv"}}`.
5. Unless `dry_run`: write `out/manifest.jsonl` (one row per thumb: `thumb, view, flight, stem, crop, crop_method, n_locked, status, frame_src`; `view` is the CSV view or the thumbs folder name when there's no row; missing values `None`) and `out/summary.json` = `{"by_status": {...}, "by_view": {view: {status: n}}}`.
6. Echo `Counts: k=v, ...` (non-zero only), same style as `outline_import_boxes_command`.

CLI options (Typer): `--thumbs` (repeatable, required), `--survey-root` (repeatable, required), `--roi-csv` (required), `--out` (required), `--link` (`hard|sym|copy`, default `hard`; reject others with exit 2), `--overwrite`, `--dry-run`, `--limit`.

- [ ] **Step 1: Write the failing tests**

Fixture `_tree(tmp_path)` builds: `training/surface/{THUMB}` + JSON with one locked polygon `[[10,10],[50,10],[50,50]]` and one unlocked; `training/thumb_roi.csv` with the Task 1 row; `finds/{FLIGHT}/images/{STEM}.JPG` (640×480). Returns paths.

```python
def test_export_ok_writes_link_json_manifest(tmp_path):
    t = _tree(tmp_path)
    rc = outline_export_frames_command(thumbs=[t.surface], survey_roots=[t.finds],
                                       roi_csv=t.csv, out=t.out)
    assert rc == 0
    jpg, js = t.out / FLIGHT / f"{STEM}.JPG", t.out / FLIGHT / f"{STEM}.json"
    assert os.path.samefile(jpg, t.frame)
    doc = json.loads(js.read_text())
    assert (doc["imageWidth"], doc["imageHeight"], doc["imagePath"]) == (640, 480, f"{STEM}.JPG")
    assert len(doc["shapes"]) == 1
    s = doc["shapes"][0]
    assert s["points"] == [[198, 98], [238, 98], [238, 138]]
    assert s["flags"]["source_thumb"] == THUMB and s["flags"]["locked"] is True
    assert s["flags"]["crop_xyxy"] == [188, 88, 412, 312]
    rows = [json.loads(l) for l in (t.out / "manifest.jsonl").read_text().splitlines()]
    assert rows[0]["status"] == "ok" and rows[0]["n_locked"] == 1
    assert json.loads((t.out / "summary.json").read_text())["by_status"] == {"ok": 1}

def test_statuses_no_locked_no_row_missing(tmp_path): ...
    # add: thumb with only unlocked → no_locked; thumb absent from CSV → no_roi_row;
    # thumb whose frame file doesn't exist → frame_missing. Assert manifest statuses.

def test_rerun_without_overwrite_skips_existing(tmp_path):
    # run twice; second run: status skipped_existing, rc 0, JSON mtime unchanged

def test_rerun_with_overwrite_rebuilds_and_existing_link_is_ok(tmp_path):
    # run, edit out json shapes to [], run with overwrite=True → status ok, 1 shape back

def test_surface_and_deep_share_frame(tmp_path):
    # second thumb (det_1) in training/deep, same frame, CSV view "deep"
    # → one JSON with 2 shapes, views {"surface", "deep"}, both rows ok

def test_dry_run_writes_nothing(tmp_path):
    # rc 0, out dir absent or empty

def test_link_frame_exdev_falls_back_to_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "link", lambda *a: (_ for _ in ()).throw(OSError(errno.EXDEV, "x")))
    assert link_frame(src, dst, "hard") == "sym" and dst.is_symlink()

def test_link_failed_status(tmp_path, monkeypatch):
    # os.link raises OSError(errno.EACCES) → manifest status link_failed, no JSON

def test_cli_registered():
    from typer.testing import CliRunner
    from symsorter.cli import app
    r = CliRunner().invoke(app, ["outline-export-frames", "--help"])
    assert r.exit_code == 0 and "--survey-root" in r.output
```

Expand the `...`/comment stubs into full tests with the assertions stated.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest symsorter/tests/test_outline_export_frames.py -q`
Expected: FAIL — `ModuleNotFoundError: symsorter.commands.outline_export_frames`

- [ ] **Step 3: Implement `link_frame` and `outline_export_frames_command`, and register the CLI command**

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest symsorter/tests/test_outline_export_frames.py -q`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add symsorter/commands/outline_export_frames.py symsorter/cli.py symsorter/tests/test_outline_export_frames.py
git commit -m "feat(cli): outline-export-frames writes locked outlines onto original frames"
```

---

### Task 4: `OutlineCanvas` overlays, crop rect and message

**Files:**
- Modify: `symsorter/ui/outline_dialog.py` (`OutlineCanvas.__init__`, `paintEvent`, new setters)
- Test: `symsorter/tests/test_frame_panel.py`

**Interfaces:**
- Produces (all trigger `update()`):
  - `OutlineCanvas.set_overlays(polys: list[list[list[float]]]) -> None` — drawn after the image, before `_polygons`, dimmed (e.g. grey/white α≈120, no brush). Never hit-tested: no mouse handler reads them.
  - `OutlineCanvas.set_crop_rect(rect: tuple[float, float, float, float] | None) -> None` — `(l, t, r, b)` in image px, dashed rectangle.
  - `OutlineCanvas.set_message(text: str | None) -> None` — when set and no pixmap, drawn instead of "No image".

- [ ] **Step 1: Write the failing test**

```python
def test_overlays_are_not_hit_testable(qapp, tmp_path):
    canvas = OutlineCanvas(); canvas.resize(400, 400); canvas.set_mode("delete")
    pix = QPixmap(640, 480); pix.fill(Qt.black)
    own = [{"label": "turtle", "points": [[10, 10], [60, 10], [60, 60]], "flags": {}}]
    canvas.set_scene(pix, own, selected_index=0, image_width=640, image_height=480, refit=True)
    canvas.set_overlays([[[300, 300], [400, 300], [400, 400], [300, 400]]])
    canvas.set_crop_rect((188, 88, 412, 312))
    w = canvas._img_to_widget(350, 350)
    canvas.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, w,
                           Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    assert len(canvas._polygons) == 1
    canvas.grab()  # paint path runs without error
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest symsorter/tests/test_frame_panel.py -q`
Expected: FAIL — `AttributeError: 'OutlineCanvas' object has no attribute 'set_overlays'`

- [ ] **Step 3: Implement the three setters and the paint additions**

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest symsorter/tests/test_frame_panel.py symsorter/tests/test_outline_viewer.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add symsorter/ui/outline_dialog.py symsorter/tests/test_frame_panel.py
git commit -m "feat(ui): read-only overlays and crop rect on OutlineCanvas"
```

---

### Task 5: Frame settings and `FramePanel`

**Files:**
- Create: `symsorter/ui/frame_settings.py`, `symsorter/ui/frame_panel.py`
- Test: `symsorter/tests/test_frame_panel.py`

**Interfaces:**
- Consumes: Task 1–2 (`RoiIndex`, `FrameResolver`, `frame_size`, `crop_window_for_row`, `thumb_to_frame`, `frame_to_thumb`); Task 4 canvas setters; `extract_polygons`, `is_polygon_locked`, `labelme_path_for_thumb`, `load_labelme_json`.
- Produces in `frame_settings.py`:
  - `@dataclass FrameSettings(survey_roots: list[Path] = field(default_factory=list), roi_csv: Path | None = None)`
  - `load_frame_settings() -> FrameSettings` / `save_frame_settings(s: FrameSettings) -> None` — `QSettings("symsorter", "symsorter")`, keys `outline/survey_roots` (list of str) and `outline/roi_csv` (str or empty).
  - `roi_csv_for(thumb_folder: Path, s: FrameSettings) -> Path | None` — `s.roi_csv` if set, else `thumb_folder.parent / "thumb_roi.csv"` if it exists, else None.
  - `edit_frame_settings_dialog(parent: QWidget | None) -> FrameSettings | None` — modal: one survey root per line (`QPlainTextEdit`) with "Add folder…" button, ROI CSV line edit with "Browse…"; OK saves and returns, Cancel returns None.
- Produces in `frame_panel.py`, `class FramePanel(QWidget)`:
  - `__init__(self, settings: FrameSettings, parent: QWidget | None = None, *, load_async: bool = True)`
  - Signals: `polygonsEdited = Signal()`, `selectionChanged = Signal(object)`
  - `canvas: OutlineCanvas`; `reason: str | None` (`"no_roi_row"`, `"frame_missing"`, `"no_survey_roots"`, or None when shown)
  - `show_thumb(thumb_path: Path, polygons: list[dict], selected: int | None) -> None` — resolve row/frame (cache `RoiIndex` per CSV path and one `FrameResolver` per settings), set crop rect, overlays from neighbours, frame-space copies of `polygons`, start the image load.
  - `sync_from_thumb(polygons: list[dict], selected: int | None) -> None` — rebuild frame-space copies (deep-copy `flags`) and repaint, no reload.
  - `thumb_polygons() -> list[dict]` — frame copies converted back with `frame_to_thumb`.
  - `set_tool_mode(mode: str) -> None` — `"point"`/`"box"` map to `"view"`.

Neighbours: `roi.same_frame(row.flight, row.original_stem)` minus the current thumb; JSON at `thumb_path.parent.parent / n.view / Path(n.crop_basename).with_suffix(".json")`; locked polygons only, shifted with that neighbour's own `crop_window_for_row`.

Loading: decode to `QImage` (thread-safe) in a `QRunnable` on `QThreadPool.globalInstance()`, deliver via a `QObject` signal; convert to `QPixmap` on the UI thread. Each `show_thumb` bumps a request token; results with an old token are dropped. Keep an LRU (`OrderedDict`) of the last 4 `QImage`s keyed by frame path. `load_async=False` loads inline (tests).

Canvas wiring inside the panel: `canvas.polygonsChanged → polygonsEdited`, `canvas.selectionChanged → selectionChanged`.

- [ ] **Step 1: Write the failing tests**

Fixture `_panel_tree(tmp_path)`: `training/surface/{THUMB}` with one unlocked polygon `[[10,10],[50,10],[50,50]]`; `training/deep/{DEEP_THUMB}` (det_1) with one locked polygon; CSV rows for both (deep box centred at (500, 400)); frame 640×480 in `finds/{FLIGHT}/images/`.

```python
def test_show_thumb_maps_polys_crop_and_neighbours(qapp, tmp_path):
    t = _panel_tree(tmp_path)
    panel = FramePanel(FrameSettings([t.finds], t.csv), load_async=False)
    polys = extract_polygons(load_labelme_json(t.surface_json))
    panel.show_thumb(t.surface_thumb, polys, 0)
    assert panel.reason is None
    assert panel.canvas._polygons[0]["points"] == [[198, 98], [238, 98], [238, 138]]
    assert panel.canvas._crop_rect == (188, 88, 412, 312)
    assert len(panel.canvas._overlays) == 1      # deep neighbour, locked
    assert panel.thumb_polygons()[0]["points"] == polys[0]["points"]

def test_reasons(qapp, tmp_path):
    # no CSV row → "no_roi_row"; frame deleted → "frame_missing"; empty roots → "no_survey_roots"
    # canvas._message is set to a readable string containing the reason

def test_sam_modes_disabled(qapp):
    panel = FramePanel(FrameSettings(), load_async=False)
    panel.set_tool_mode("box"); assert panel.canvas.mode == "view"
    panel.set_tool_mode("lasso"); assert panel.canvas.mode == "lasso"

def test_stale_load_is_ignored(qapp, tmp_path):
    # show_thumb(A) async, then show_thumb(B) before waiting;
    # QThreadPool.globalInstance().waitForDone(); qapp.processEvents()
    # → canvas pixmap size equals B's frame size (make A's frame 800x600, B's 640x480)

def test_settings_round_trip(qapp, tmp_path):
    QSettings("symsorter", "symsorter").remove("outline")
    save_frame_settings(FrameSettings([tmp_path], tmp_path / "r.csv"))
    s = load_frame_settings()
    assert s.survey_roots == [tmp_path] and s.roi_csv == tmp_path / "r.csv"
    QSettings("symsorter", "symsorter").remove("outline")
```

Use `QCoreApplication.setOrganizationName/ApplicationName` with a test-only name in the settings test if you'd rather not touch the user's real settings; either way clean up.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest symsorter/tests/test_frame_panel.py -q`
Expected: FAIL — `ModuleNotFoundError: symsorter.ui.frame_panel`

- [ ] **Step 3: Implement `frame_settings.py` and `frame_panel.py`**

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest symsorter/tests/test_frame_panel.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add symsorter/ui/frame_settings.py symsorter/ui/frame_panel.py symsorter/tests/test_frame_panel.py
git commit -m "feat(ui): FramePanel shows a thumb on its original frame"
```

---

### Task 6: Wire the panel into `OutlineViewerDialog`

**Files:**
- Modify: `symsorter/ui/outline_dialog.py` (`OutlineViewerDialog.__init__`, `_install_shortcuts`, `_on_mode_changed`, `_push_scene`, `_on_canvas_polys_changed`, `_on_selection_changed`, `refresh`)
- Test: `symsorter/tests/test_frame_panel.py`

**Interfaces:**
- Consumes: `FramePanel` (Task 5), `FrameSettings`, `load_frame_settings`.
- Produces:
  - `OutlineViewerDialog.__init__(..., frame_settings: FrameSettings | None = None)` — defaults to `load_frame_settings()`.
  - `self.frame_panel: FramePanel` (hidden by default), in a horizontal `QSplitter` with `self.canvas`.
  - `self.frame_btn` — checkable "Frame" button in the nav row.
  - `toggle_frame_panel() -> None` — bound to `Shift+F` and `frame_btn`; on show calls `frame_panel.show_thumb(current, self._polygons, self._selected)`.

Sync rules (only while the panel is visible):
- `refresh` → `frame_panel.show_thumb(...)`.
- `_push_scene`, `_on_canvas_polys_changed`, `_on_selection_changed` → `frame_panel.sync_from_thumb(self._polygons, self._selected)`.
- `frame_panel.polygonsEdited` → `self._polygons[:] = frame_panel.thumb_polygons()` (slice-assign: the thumb canvas holds the same list), `self._dirty = True`, `self.canvas.update()`, `_update_title()`.
- `frame_panel.selectionChanged` → set `self._selected`, `self.canvas.set_selected(...)`, `_update_title()`.
- `_on_mode_changed` → `frame_panel.set_tool_mode(mode)`.
- Save path is unchanged: `save_current` writes `self._polygons` to the thumbnail JSON only.

- [ ] **Step 1: Write the failing tests**

```python
def test_frame_edit_marks_dirty_and_saves_thumb_coords(qapp, tmp_path):
    t = _panel_tree(tmp_path)
    dlg = OutlineViewerDialog(None, entries=[(t.surface_thumb, THUMB)], display_size=224,
                              frame_settings=FrameSettings([t.finds], t.csv))
    dlg.frame_panel._load_async = False
    dlg.toggle_frame_panel()
    assert dlg.frame_panel.isVisibleTo(dlg)
    fp = dlg.frame_panel.canvas
    fp._polygons[0]["points"][0] = [200.0, 100.0]   # frame px
    fp.polygonsChanged.emit()
    assert dlg._dirty is True
    assert dlg._polygons[0]["points"][0] == [12.0, 12.0]
    dlg.save_current()
    saved = extract_polygons(load_labelme_json(t.surface_json))
    assert saved[0]["points"][0] == [12.0, 12.0]
    assert not (t.finds / FLIGHT / "images" / f"{STEM}.json").exists()
    dlg.close()

def test_thumb_edit_updates_frame_panel(qapp, tmp_path):
    # toggle panel, set dlg._polygons[0]["points"][0] = [20.0, 20.0],
    # dlg.canvas.polygonsChanged.emit() → frame canvas point == [208.0, 108.0]

def test_locked_polygon_blocked_in_frame_panel(qapp, tmp_path):
    # lock polygon via dlg.toggle_lock_selected(); in frame canvas, edit mode,
    # call fp._set_selected_points([...]) → points unchanged, statusMessage emitted

def test_shift_f_hidden_by_default(qapp, tmp_path):
    # new dialog: frame_panel not visible; toggle twice → hidden again
```

Make `_load_async` the attribute name `FramePanel` stores `load_async` under (Task 5), so this test can force inline loads.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest symsorter/tests/test_frame_panel.py -q -k "dlg or frame_edit or thumb_edit or locked or shift_f"`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'frame_settings'`

- [ ] **Step 3: Implement the dialog wiring**

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest symsorter/tests/test_frame_panel.py symsorter/tests/test_outline_viewer.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add symsorter/ui/outline_dialog.py symsorter/tests/test_frame_panel.py
git commit -m "feat(ui): Shift+F frame panel in Outline viewer with two-way sync"
```

---

### Task 7: Survey roots in the GUI, README, end-to-end check

**Files:**
- Modify: `symsorter/cli.py` (`gui` command), `symsorter/ui/menus.py` (File menu), `symsorter/image_browser.py` (handler)
- Modify: `README.md`
- Test: `symsorter/tests/test_frame_panel.py`

**Interfaces:**
- Consumes: `FrameSettings`, `load_frame_settings`, `save_frame_settings`, `edit_frame_settings_dialog` (Task 5).
- Produces:
  - `gui(..., survey_root: List[Path] = typer.Option(None, "--survey-root", help=...))` — when given, replaces `survey_roots` in saved settings (`save_frame_settings`) before the browser starts.
  - File menu action `"Set Survey &Roots…"` → `ImageBrowser.set_survey_roots_dialog()` → `edit_frame_settings_dialog(self)`.
  - `open_outline_viewer_for_browser` keeps calling `OutlineViewerDialog(...)` without `frame_settings`, so it picks up the saved settings.

- [ ] **Step 1: Write the failing test**

```python
def test_gui_help_lists_survey_root():
    r = CliRunner().invoke(app, ["gui", "--help"])
    assert r.exit_code == 0 and "--survey-root" in r.output
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest symsorter/tests/test_frame_panel.py -q -k gui_help`
Expected: FAIL — `--survey-root` not in output

- [ ] **Step 3: Implement the `gui` option, menu action and handler; add a README section** covering `outline-export-frames` (the spec's example command, output layout, statuses, `--link` fallback) and the `Shift+F` panel with "Set Survey Roots…".

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest symsorter/tests -q`
Expected: all pass except the two known `test_basic.py` failures.

- [ ] **Step 5: Manual end-to-end (only if `/Volumes/turtle_apfs` is mounted)**

```bash
symsorter outline-export-frames \
  --thumbs /Volumes/turtle_apfs/training/surface \
  --thumbs /Volumes/turtle_apfs/training/deep \
  --survey-root /Volumes/turtle_apfs/yolo_finds \
  --roi-csv /Volumes/turtle_apfs/training/thumb_roi.csv \
  --out /tmp/outlines_on_frames_check --link sym --dry-run
```

Expected: `ok=748` (746 surface + 2 deep at spec time; deep grows as more are locked), every other locked thumb counted as `frame_missing`. Then open the GUI, press `O`, `Shift+F`, check that the crop rectangle sits on the turtle.

- [ ] **Step 6: Commit**

```bash
git add symsorter/cli.py symsorter/ui/menus.py symsorter/image_browser.py README.md symsorter/tests/test_frame_panel.py
git commit -m "feat(gui): --survey-root and Set Survey Roots; document frame export"
```
