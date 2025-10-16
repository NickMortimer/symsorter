import argparse
from pathlib import Path
import json
from symsorter.io.aggregate import find_classification_manifests, merge_coco_manifests
from symsorter.io.classifications import ensure_json_serializable


def main():
    ap = argparse.ArgumentParser(description="Merge per-folder SymSorter classifications.json into one COCO JSON")
    ap.add_argument("root", type=Path, help="Root directory to scan for classifications.json")
    ap.add_argument("-o", "--out", type=Path, default=Path("merged_classifications.json"), help="Output COCO JSON path")
    ap.add_argument("--exclude-hidden", action="store_true", help="Exclude images marked hidden=True")
    args = ap.parse_args()

    manifests = find_classification_manifests(args.root)
    if not manifests:
        print(f"No classifications.json found under {args.root}")
        return 1

    print(f"Found {len(manifests)} manifests")
    merged = merge_coco_manifests(args.root, manifests, include_hidden=not args.exclude_hidden)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ensure_json_serializable(merged), f, indent=2, ensure_ascii=False)
    print(f"Wrote merged COCO to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())