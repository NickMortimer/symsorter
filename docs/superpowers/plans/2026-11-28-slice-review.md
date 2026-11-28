# Slice Review (SymSorter side) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export locked thumb outlines as a text contract, and add a Slice review window that opens a marine_objects slice project, cuts each 624 slice from the laptop's own survey frames, and records Nick's decisions in `review.jsonl`.

**Architecture:** All logic lives in a new Qt-free package, `symsorter/slices/`: the contract, the frame lookup and geometry, and the review session. It is covered by pytest. The window, `symsorter/ui/slice_review.py`, is a thin shell around `OutlineCanvas` that drives a `ReviewSession`. The export command reuses `symsorter/outline/frames.py`.

**Tech Stack:** Python ≥3.9 (`from __future__ import annotations`), PySide6, Typer, Pillow, pytest with `QT_QPA_PLATFORM=offscreen`.

**Spec:** `/home/mor582/code/board/specs/2026-11-28-slice-review-design.md` (cross-repo, outside this repo). The back-end half is `marine_objects/docs/superpowers/plans/2026-11-28-slice-review.md`.

## Global Constraints

- Schemas: `"symsorter.outlines/1"` (written here), `"marine_count.slice_project/1"` (read here, never written), `"symsorter.slice_review/1"` (written here). Refuse unknown schemas with a message naming the file.
- Coordinates in every shared file are full-frame pixels `[x, y]`. The canvas works in view pixels (slice + margin). Convert only at the boundary.
- v1 label is `turtle` only. There is no label picker.
- `review.jsonl` is append-only. Each save writes one row holding the whole polygon set. The last row per `slice_id` wins.
- `complete` is refused unless every kept polygon lies wholly inside the final slice. Polygons that do not touch the final slice are dropped on save.
- `project.local.json` (survey roots, reviewer) sits beside `project.json`, is never synced, and is never written into `review.jsonl`.
- Frame lookup for slices is strict: flight + stem in the two known layouts, no flight-blind stem search, and the size must equal `frame_wh`.
- `outline-export-contract` writes no row for a sealed flight. A missing or empty denylist is an error.
- Branch `feat/slice-review`, worktree `../symsorter-slice-review`. Agent name `symsorter/slice-review`. Board: `/home/mor582/code/board/board.md` (not `~/proj/board`). Claim before starting. On DONE, give the branch and the GUI checks for Nick.

## Review Focus

1. **Quitting mid-slice, or a crash.** Unsaved edits on the current slice are lost, but every saved row survives, and reopening the project resumes at the first unreviewed slice. Test in Task 4.
2. **A torn last line in `review.jsonl`** (power cut during a write). Loading skips a final line that is not valid JSON, with a warning, instead of refusing the whole file. A bad line in the middle is still an error. Test in Task 1.
3. **Survey drive not mounted.** Slices whose frame is missing show "frame missing" and are skipped by next/previous. Nothing is written for them, and the window still opens. Test in Task 4.
4. **Re-centre pushes the slice against the frame edge.** The slice is clamped inside the frame. If the outlines then still don't fit, re-centre reports that and leaves the slice where it was. Test in Task 3.
5. **A project re-synced with new slices while reviews exist.** Old review rows for `slice_id`s no longer in the project are ignored for display and kept in the file. Test in Task 4.

---

### Task 1: Slice contract (`symsorter/slices/contract.py`)

**Files:**
- Create: `symsorter/slices/__init__.py`, `symsorter/slices/contract.py`
- Create: `symsorter/tests/fixtures/slices/{outlines.jsonl,project.json,review.jsonl}` (copied verbatim from `/home/mor582/code/board/specs/slice-review-examples/`)
- Modify: `CLAUDE.md` (board path → `/home/mor582/code/board/board.md`; laptop SSH lines → `ssh deepthinker cat ~/code/board/board.md` and the matching `echo ... >> ~/code/board/board.md`)
- Test: `symsorter/tests/test_slice_contract.py`

**Interfaces:**
- Produces (in `symsorter.slices.contract`):
  - `OUTLINES_SCHEMA`, `PROJECT_SCHEMA`, `REVIEW_SCHEMA`, `STATUSES = ("complete", "reject", "skip")`
  - `class ContractError(ValueError)`
  - `Polygon = list[list[float]]` (type alias)
  - `@dataclass(frozen=True) SliceSpec(slice_id: str, survey_id: str, frame_id: str, frame_wh: tuple[int, int], x0: int, y0: int, size: int, kind: str, polygons: tuple[tuple[tuple[float, float], ...], ...])`, with properties `flight` (last part of `survey_id`) and `frame_stem` (stem of `frame_id`)
  - `@dataclass(frozen=True) Project(project_id: str, slice_px: int, margin_px: int, slices: tuple[SliceSpec, ...])`
  - `@dataclass(frozen=True) ReviewRow(project_id: str, slice_id: str, status: str, polygons: tuple[tuple[tuple[float, float], ...], ...], reviewer: str, time: str, x0: int | None = None, y0: int | None = None, reason: str | None = None)`
  - `load_project(path: Path) -> Project`
  - `load_reviews(path: Path) -> dict[str, ReviewRow]` (missing file → `{}`; a torn final line is skipped with `warnings.warn`)
  - `append_review(path: Path, row: ReviewRow) -> None` (one line, `flush` + `os.fsync`; `x0`/`y0`/`reason` omitted when `None`)
  - `outline_row(flight: str, frame_stem: str, points: Polygon, source: str, frame_wh: tuple[int, int]) -> dict` (with `"schema"` and `"label": "turtle"`)

- [ ] **Step 1: Copy the fixtures and fix the board path in `CLAUDE.md`**

- [ ] **Step 2: Write the failing tests**

```python
FIX = Path(__file__).parent / "fixtures" / "slices"

def test_load_example_project():
    p = load_project(FIX / "project.json")
    assert (p.project_id, p.slice_px, p.margin_px) == ("example_r00", 624, 64)
    s = p.slices[0]
    assert (s.flight, s.frame_stem, s.x0, s.y0, s.size) == ("TESTA_20240101T0900", "DJI_0001", 758, 1733, 624)
    assert len(s.polygons) == 2

def test_last_review_wins_and_missing_file_is_empty(tmp_path):
    r = load_reviews(FIX / "review.jsonl")
    assert r["AU--TESTA--TESTA_20240101T0900__DJI_0001__758_1733"].status == "complete"
    assert load_reviews(tmp_path / "none.jsonl") == {}

def test_append_then_load_round_trips_recentred_row(tmp_path):
    row = ReviewRow("example_r00", "id1", "complete", (((1.0, 2.0), (3.0, 2.0), (2.0, 4.0)),), "nick", "2026-12-01T10:00:00+10:00", x0=700, y0=1700)
    append_review(tmp_path / "r.jsonl", row)
    assert load_reviews(tmp_path / "r.jsonl") == {"id1": row}
    raw = json.loads((tmp_path / "r.jsonl").read_text())
    assert raw["schema"] == "symsorter.slice_review/1" and "reason" not in raw

def test_torn_last_line_is_skipped_but_bad_middle_line_is_an_error(tmp_path):
    good = (FIX / "review.jsonl").read_text()
    (tmp_path / "a.jsonl").write_text(good + '{"schema": "symsorter.sl')
    with pytest.warns(UserWarning):
        assert len(load_reviews(tmp_path / "a.jsonl")) == 2
    lines = good.splitlines()
    (tmp_path / "b.jsonl").write_text(lines[0] + "\n{bad\n" + lines[1] + "\n")
    with pytest.raises(ContractError, match="b.jsonl:2"):
        load_reviews(tmp_path / "b.jsonl")

def test_unknown_project_schema_is_refused(tmp_path): ...   # "marine_count.slice_project/2" -> ContractError
def test_outline_row_matches_the_example():
    row = outline_row("TESTA_20240101T0900", "DJI_0001", [[1000.0, 2000.0], [1040.0, 2000.0], [1040.0, 2040.0], [1000.0, 2040.0]], "thumb:example_DJI_0001_det_0_turtle_224px.jpg", (5472, 3648))
    assert row == json.loads((FIX / "outlines.jsonl").read_text().splitlines()[0])
```

- [ ] **Step 3: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen pytest symsorter/tests/test_slice_contract.py -q`
Expected: FAIL `ModuleNotFoundError: symsorter.slices`

- [ ] **Step 4: Implement `contract.py` with the interfaces above**

- [ ] **Step 5: Run to verify pass**

Run: `QT_QPA_PLATFORM=offscreen pytest symsorter/tests -q`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md symsorter/slices symsorter/tests/test_slice_contract.py symsorter/tests/fixtures/slices
git commit -m "feat(slices): slice project and review contract"
```

---

### Task 2: `symsorter outline-export-contract`

**Files:**
- Create: `symsorter/commands/outline_export_contract.py`
- Modify: `symsorter/cli.py` (register the command beside `outline-export-frames`)
- Test: `symsorter/tests/test_outline_export_contract.py`

**Interfaces:**
- Consumes: `outline_row` (Task 1). `RoiIndex`, `FrameResolver`, `crop_window_for_row`, `frame_size`, `thumb_to_frame`, `iter_flat_thumbs` from `symsorter.outline.frames`. `extract_polygons`, `is_polygon_locked`, `labelme_path_for_thumb`, `load_labelme_json` from `symsorter.outline.export_labelme`.
- Produces:
  - `load_denylist(path: Path) -> frozenset[str]`: blank and `#` lines are ignored. A missing file or one with no ids raises `typer.BadParameter`.
  - `export_contract(thumbs: list[Path], roi_csv: Path, survey_roots: list[Path], denylist: frozenset[str], out: Path) -> dict[str, int]`. It writes `out` (rows sorted by `flight`, `frame_stem`, `source`) and returns counts with exactly the keys `ok` (polygons written), `sealed`, `no_locked`, `no_roi_row`, `frame_missing`.
  - CLI: `symsorter outline-export-contract --thumbs DIR... --roi-csv CSV --survey-root DIR... --denylist FILE --out FILE`. It prints the counts. An existing `--out` is overwritten (the file is regenerated whole each run).

Rules:
- A flight is sealed when it equals the last `/` part of any denylist id. Sealed thumbs are counted per thumb before the frame lookup.
- Use `FrameResolver(roots)` as `outline-export-frames` does. The strict lookup is for slice review only.
- `frame_wh` comes from `frame_size(frame)`.

- [ ] **Step 1: Write the failing tests**

Build fixtures like `symsorter/tests/test_outline_export_frames.py` does: a 640×480 frame at `{root}/AU/TESTA/TESTA_20240101T0900/DJI_0001.JPG`, a thumb with a locked polygon plus an unlocked one, a `thumb_roi.csv` row, and a second thumb from flight `SEAL_20240101T0900`.

```python
def test_locked_polygons_exported_in_frame_pixels(tmp_path):
    counts = export_contract([thumbs], roi_csv, [root], frozenset({"AU/SEAL/SEAL_20240101T0900"}), tmp_path / "o.jsonl")
    assert counts == {"ok": 1, "sealed": 1, "no_locked": 0, "no_roi_row": 0, "frame_missing": 0}
    [row] = [json.loads(l) for l in (tmp_path / "o.jsonl").read_text().splitlines()]
    assert row["flight"] == "TESTA_20240101T0900" and row["frame_wh"] == [640, 480]
    assert row["points"] == thumb_to_frame(locked_points, crop)

def test_sealed_flight_never_written(tmp_path):
    export_contract(...)
    assert "SEAL_" not in (tmp_path / "o.jsonl").read_text()

def test_missing_or_empty_denylist_is_refused(tmp_path): ...
def test_missing_frame_and_missing_roi_row_are_counted(tmp_path): ...
def test_cli_prints_counts(tmp_path): ...   # typer CliRunner, exit 0, "ok" in output
```

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen pytest symsorter/tests/test_outline_export_contract.py -q`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement `outline_export_contract.py` and register it in `cli.py`**

- [ ] **Step 4: Run all tests**

Run: `QT_QPA_PLATFORM=offscreen pytest symsorter/tests -q`
Expected: all PASS

- [ ] **Step 5: Commit and push (this unblocks the back end)**

```bash
git add symsorter/commands/outline_export_contract.py symsorter/cli.py symsorter/tests/test_outline_export_contract.py
git commit -m "feat(cli): outline-export-contract writes locked outlines as outlines.jsonl"
git push -u origin feat/slice-review
```

Post on the board: `NOTE: outline-export-contract on feat/slice-review. Nick: fetch the seal denylist from deepthinker, run the export on the laptop (command in the spec), and rsync outlines.jsonl to deepthinker:~/proj/slices/.`

---

### Task 3: Frame lookup and slice geometry (`symsorter/slices/geometry.py`)

**Files:**
- Modify: `symsorter/outline/frames.py`: `FrameResolver.__init__(self, roots, *, strict: bool = False)`. With `strict=True`, `find` skips the `_stem_index` fallback.
- Create: `symsorter/slices/geometry.py`
- Test: `symsorter/tests/test_slice_geometry.py`, plus one case in `symsorter/tests/test_outline_frames.py`

**Interfaces:**
- Consumes: `SliceSpec` (Task 1), `FrameResolver`, `frame_size`
- Produces (in `symsorter.slices.geometry`):
  - `Rect = tuple[int, int, int, int]` as `(left, top, right, bottom)`
  - `@dataclass(frozen=True) FrameLookup(path: Path | None, problem: str | None)`, where `problem` is `None`, `"frame missing"` or `"size mismatch WxH"`
  - `locate_frame(spec: SliceSpec, resolver: FrameResolver) -> FrameLookup` (the resolver must be strict)
  - `view_rect(x0: int, y0: int, size: int, margin: int, frame_wh: tuple[int, int]) -> Rect`: slice ± margin, clamped to the frame
  - `to_view(points, view: Rect) -> Polygon` and `to_frame(points, view: Rect) -> Polygon`
  - `touches(points, x0: int, y0: int, size: int) -> bool` (bounding-box overlap) and `inside(points, x0: int, y0: int, size: int) -> bool`
  - `recentre(polygons: list[Polygon], x0: int, y0: int, size: int, frame_wh: tuple[int, int]) -> tuple[int, int] | None`. It returns the least move along each axis that holds every polygon's bounding box, clamped inside the frame. It returns `None` when the union is wider or taller than `size`, or doesn't fit after clamping. It returns `(x0, y0)` unchanged when everything already fits.
  - `sam_window(x: float, y: float, frame_wh: tuple[int, int], size: int = 224) -> Rect`: a `size` square centred on the click, clamped to the frame

- [ ] **Step 1: Write the failing tests**

```python
def test_view_rect_clamps_margin_at_frame_edge():
    assert view_rect(758, 1733, 624, 64, (5472, 3648)) == (694, 1669, 1446, 2421)
    assert view_rect(0, 0, 624, 64, (5472, 3648)) == (0, 0, 688, 688)

def test_view_round_trip():
    v = (694, 1669, 1446, 2421)
    assert to_frame(to_view([[1000.0, 2000.0]], v), v) == [[1000.0, 2000.0]]

def test_recentre_moves_least_distance():
    poly = [[1350.0, 2000.0], [1420.0, 2000.0], [1420.0, 2040.0]]  # sticks out right of 758+624=1382
    assert recentre([poly], 758, 1733, 624, (5472, 3648)) == (796, 1733)

def test_recentre_clamps_at_frame_edge_and_gives_none_when_too_wide():
    edge = [[5400.0, 100.0], [5472.0, 100.0], [5472.0, 140.0]]
    assert recentre([edge], 4800, 0, 624, (5472, 3648)) == (4848, 0)
    wide = [[0.0, 0.0], [700.0, 0.0], [700.0, 10.0]]
    assert recentre([wide], 0, 0, 624, (5472, 3648)) is None

def test_recentre_is_identity_when_all_inside():
    assert recentre([[[800.0, 1800.0], [810.0, 1800.0], [805.0, 1810.0]]], 758, 1733, 624, (5472, 3648)) == (758, 1733)

def test_strict_lookup_finds_survey_layout_and_flags_size(tmp_path):
    # frame at root/AU/TESTA/TESTA_20240101T0900/DJI_0001.JPG, 640x480; spec frame_wh (640, 480) -> path, None
    # same with frame_wh (5472, 3648) -> problem "size mismatch 640x480"
    # same stem only under root/elsewhere/DJI_0001.JPG -> problem "frame missing"

def test_sam_window_clamps():
    assert sam_window(10, 10, (5472, 3648)) == (0, 0, 224, 224)
```

In `test_outline_frames.py`: `test_strict_resolver_skips_stem_index`.

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen pytest symsorter/tests/test_slice_geometry.py symsorter/tests/test_outline_frames.py -q`
Expected: FAIL (module missing / unexpected keyword `strict`)

- [ ] **Step 3: Implement `geometry.py` and the `strict` flag**

- [ ] **Step 4: Run all tests**

Run: `QT_QPA_PLATFORM=offscreen pytest symsorter/tests -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add symsorter/outline/frames.py symsorter/slices/geometry.py symsorter/tests/test_slice_geometry.py symsorter/tests/test_outline_frames.py
git commit -m "feat(slices): strict frame lookup and slice geometry"
```

---

### Task 4: Review session (`symsorter/slices/session.py`)

**Files:**
- Create: `symsorter/slices/session.py`
- Test: `symsorter/tests/test_slice_session.py`

**Interfaces:**
- Consumes: Tasks 1 and 3
- Produces (in `symsorter.slices.session`):
  - `@dataclass LocalSettings(survey_roots: list[Path], reviewer: str)`, with `load_local(project_dir: Path) -> LocalSettings | None` and `save_local(project_dir: Path, s: LocalSettings) -> None` (file `project.local.json`)
  - `class SaveRefused(Exception)`, whose message is shown to the user
  - `class ReviewSession`:
    - `__init__(self, project_dir: Path, local: LocalSettings, *, clock: Callable[[], str] = <ISO now with offset>)`. It loads `project.json` and `review.jsonl`, builds a strict `FrameResolver(local.survey_roots, strict=True)`, and starts at the first unreviewed slice whose frame is found.
    - `project: Project`, `index: int`, `current: SliceSpec`
    - `position(self) -> tuple[int, int]`: the current slice's `(x0, y0)`. It starts from the latest review if there is one, else the project, and changes after `recentre_current`.
    - `polygons(self) -> list[Polygon]`: the editable set in frame pixels, seeded from the latest review, else the project
    - `set_polygons(self, polys: list[Polygon]) -> None`, which marks the session dirty
    - `lookup(self, i: int | None = None) -> FrameLookup` (cached per `slice_id`)
    - `status(self, i: int | None = None) -> str | None`: the latest status, or `None`
    - `neighbours(self) -> list[Polygon]`: polygons from other slices of the same frame (their latest review, else the project) that touch the current view rect
    - `adopt_neighbours(self) -> int`: adds the neighbour polygons that touch the current slice, skipping exact duplicates, and returns how many were added
    - `recentre_current(self) -> bool`: applies `geometry.recentre`. It returns `False` and leaves the position alone when that gives `None`.
    - `save(self, status: str, reason: str | None = None) -> ReviewRow`. It drops polygons that don't touch the position. For `complete` it raises `SaveRefused` if any kept polygon is not `inside`, and refuses when the frame lookup has a problem. It appends the row (with `x0`/`y0` only when they differ from the project) and clears dirty.
    - `next_unreviewed(self) -> bool` and `prev(self) -> bool`. Both skip slices with a lookup problem. They return `False` at the end, and never change `index` while the session is dirty (the UI asks first).
    - `progress(self) -> dict[str, int]`, with keys `total`, `complete`, `reject`, `skip`, `unreviewed`, `missing`

- [ ] **Step 1: Write the failing tests**

Fixture: copy `fixtures/slices/project.json` into `tmp_path/proj/`, and make 5472×3648 frames (`Image.new`) for `DJI_0001` and `DJI_0002` under `tmp_path/root/AU/TESTA/TESTA_20240101T0900/`.

```python
def test_starts_at_first_unreviewed_and_saves_complete(session):
    assert session.current.frame_stem == "DJI_0001"
    row = session.save("complete")
    assert row.status == "complete" and len(row.polygons) == 2 and row.x0 is None
    assert load_reviews(proj / "review.jsonl")[row.slice_id] == row

def test_complete_refused_when_polygon_crosses_edge_then_recentre_allows_it(session):
    session.set_polygons(session.polygons() + [[[1350.0, 2000.0], [1420.0, 2000.0], [1420.0, 2040.0]]])
    with pytest.raises(SaveRefused, match="Re-centre"):
        session.save("complete")
    assert session.recentre_current() and session.position() == (796, 1733)
    row = session.save("complete")
    assert (row.x0, row.y0) == (796, 1733)

def test_polygon_wholly_in_margin_is_dropped_on_save(session):
    session.set_polygons(session.polygons() + [[[700.0, 1700.0], [740.0, 1700.0], [720.0, 1720.0]]])
    assert len(session.save("complete").polygons) == 2

def test_reject_keeps_reason_and_ignores_fit(session): ...
def test_resume_after_reopen_starts_at_next_unreviewed(tmp_path): ...   # save one, new ReviewSession -> second slice
def test_missing_frame_is_skipped_and_counted(tmp_path): ...           # delete DJI_0002 -> progress()["missing"] == 1, next_unreviewed False
def test_dirty_session_does_not_move(session): ...
def test_reviews_for_slices_not_in_project_are_ignored(tmp_path): ...   # extra row in review.jsonl -> progress unaffected, row still in file
def test_adopt_neighbour_adds_once(tmp_path): ...                       # two overlapping slices in one frame
def test_local_settings_round_trip(tmp_path): ...
```

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen pytest symsorter/tests/test_slice_session.py -q`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement `session.py`**

- [ ] **Step 4: Run all tests**

Run: `QT_QPA_PLATFORM=offscreen pytest symsorter/tests -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add symsorter/slices/session.py symsorter/tests/test_slice_session.py
git commit -m "feat(slices): review session with re-centre, resume and progress"
```

---

### Task 5: Slice review window

**Files:**
- Create: `symsorter/ui/slice_review.py`
- Modify: `symsorter/ui/menus.py` (File → **Open Slice Project…**), `symsorter/image_browser.py` (`open_slice_project(self, project_dir: Path | None = None)`), `symsorter/gui.py` (`--slice-project DIR` opens the window at start), `README.md` (a "Slice review" section: the loop, the rsync lines from the spec, the keys)
- Test: `symsorter/tests/test_slice_review_ui.py`

**Interfaces:**
- Consumes: `ReviewSession`, `LocalSettings`, `load_local`, `save_local`, `SaveRefused` (Task 4); `view_rect`, `to_view`, `to_frame`, `sam_window` (Task 3); `OutlineCanvas` (`set_scene`, `set_overlays`, `set_crop_rect`, `polygonsChanged`, `pointSeeded`); `sam_polygon` from `symsorter.outline.backends`; `load_frame_settings` from `symsorter.ui.frame_settings`
- Produces:
  - `class SliceReviewDialog(QDialog)` with `__init__(self, session: ReviewSession, parent=None)`, `show_current(self) -> None`, and the slots below
  - `open_slice_project_dialog(parent, project_dir: Path | None = None) -> SliceReviewDialog | None`. It asks for a folder if none is given. On first open it asks for survey roots, prefilled from `load_frame_settings().survey_roots`, and a reviewer name, then saves `project.local.json`.

Behaviour:
- The canvas shows the view crop (slice + `margin_px`) and the slice border (dashed, via `set_crop_rect` in view pixels). Neighbours are drawn faintly (`set_overlays`). Polygons are converted with `to_view`/`to_frame`.
- Load the frame with Pillow off the UI thread (`QThreadPool`, as `frame_panel.py` does). Prefetch the next slice's frame.
- Keys: `C` complete, `X` reject (asks for an optional reason), `Space` skip, `N` next unreviewed, `P` previous, `R` re-centre, `A` adopt neighbours, plus the canvas's existing edit, lasso and SAM-point tools.
- `SaveRefused` is shown in the status bar. Moving with unsaved edits asks "Save as complete / Discard / Cancel".
- SAM point click: crop `sam_window(x, y)` from the frame to a temp JPEG, call `sam_polygon(tmp, points=[[x - left, y - top]], size=224, ...)`, then shift the points back to frame pixels.
- The title shows `project_id`, slice `i/total`, status and the progress counts. A lookup problem shows as a message in place of the image.

- [ ] **Step 1: Write the failing offscreen tests**

```python
def test_dialog_shows_slice_and_completes_with_key(qapp, session):
    dlg = SliceReviewDialog(session); dlg.show_current(); qapp.processEvents()
    QTest.keyClick(dlg, Qt.Key_C)
    assert session.status(0) == "complete" and session.index == 1

def test_crossing_polygon_shows_refusal_and_recentre_fixes_it(qapp, session): ...
    # add crossing polygon via session.set_polygons; key C -> status label contains "Re-centre"; key R then C -> complete

def test_missing_frame_shows_message_not_crash(qapp, tmp_path): ...
def test_open_dialog_writes_local_settings(qapp, tmp_path, monkeypatch): ...   # monkeypatch the roots/reviewer prompts
```

SAM itself is not tested here (no model on CI). Test the coordinate shift with a `sam_polygon` stub that returns fixed points.

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen pytest symsorter/tests/test_slice_review_ui.py -q`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement the dialog, menu item, `--slice-project`, and the README section**

- [ ] **Step 4: Run all tests**

Run: `QT_QPA_PLATFORM=offscreen pytest symsorter/tests -q`
Expected: all PASS

- [ ] **Step 5: Commit, push, post DONE**

```bash
git add symsorter/ui/slice_review.py symsorter/ui/menus.py symsorter/image_browser.py symsorter/gui.py README.md symsorter/tests/test_slice_review_ui.py
git commit -m "feat(ui): Slice review window for marine_count slice projects"
git push
```

Board DONE entry (branch `feat/slice-review`). Nick's GUI checks:
1. Open the sample project. The slices look right against the frame, and the outlines sit on the turtles.
2. Add a turtle and press `C`. A crossing outline is refused, and `R` re-centres the slice.
3. Quit mid-way, reopen, and it resumes at the next unreviewed slice.
4. SAM click on a slice gives an outline in the right place.
5. rsync `review.jsonl` back. The file is small and has one row per save.
