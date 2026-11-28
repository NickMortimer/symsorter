# SymSorter – agent guide

CLIP/DINO image classification and dataset-curation tool with a PySide6 desktop GUI.
CLI entry: `symsorter` (symsorter/cli.py). GUI entry: `symsorter-gui` (symsorter/gui.py).
Tests live in `symsorter/tests/`.

## How this project is worked on

- **Code is written on deepthinker** by Claude Code agents (with the Superpowers plugin).
- **Nick runs and tests the GUI on his laptop.** He pulls branches from GitHub (`origin`) and runs them there.
- So every piece of work must be **committed and pushed to a branch on origin** before it counts as done.
  Report the branch name on the board so Nick can `git fetch && git checkout <branch>`.
- You can't see the GUI. Keep logic out of widgets where possible, cover it with pytest, and run
  Qt code headless with `QT_QPA_PLATFORM=offscreen`. Say clearly what needs a human GUI check.

## Related repo: marine_objects

Object detection lives in `marine_objects` (also on deepthinker, sibling folder). SymSorter is the
front end for curating and labelling; marine_objects is the detection back end.
If you change a file format, CLI contract or JSON schema that the other repo reads or writes
(e.g. `classifications.json`, YOLO labels, outline/ROI JSON), post a NOTE on the board first.

## Shared agent board

Path (on deepthinker): `~/proj/board/board.md`
- Read it before starting a task and after finishing one. Append only; never edit others' entries.
- Format: `YYYY-MM-DD HH:MM [symsorter/<task>] STATUS: message`
  STATUS is one of CLAIM, DONE, BLOCKED, NOTE, QUESTION.
- On DONE, include the pushed branch name and anything Nick should test in the GUI.
- If you are running on the laptop instead, use it over SSH:
  `ssh deepthinker cat ~/proj/board/board.md`
  `ssh deepthinker 'echo "<entry>" >> ~/proj/board/board.md'`

## Worktrees and branches

- Create worktrees as siblings, never inside the repo (no `.worktrees/`):
  `git worktree add ../symsorter-<task> -b feat/<task>`
- Agent name = `symsorter/<task>`; main checkout = `symsorter/main`.
- Remove the worktree and local branch after merge.
- Superpowers specs and plans go in `docs/superpowers/specs/` and `docs/superpowers/plans/`.

## Don't

- Don't commit model weights (`*.pt`, e.g. `sam_b.pt`, are gitignored), `.venv/`, or board files.
- Don't force-push branches Nick may have checked out.
