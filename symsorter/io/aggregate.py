from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import itertools

from .classifications import load_coco_json, ensure_json_serializable


def find_classification_manifests(root: Path) -> List[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("classifications.json") if p.is_file())


def merge_coco_manifests(root: Path, manifests: List[Path], include_hidden: bool = True) -> dict:
    root = Path(root)

    out = {
        "info": {
            "description": f"SymSorter merged dataset from {len(manifests)} folders",
            "version": "2.0",
            "year": int(datetime.now().year),
            "contributor": "SymSorter",
            "date_created": str(datetime.now().isoformat()),
            "symsorter_version": "1.0",
            "sources": [str(m.parent.relative_to(root)) for m in manifests],
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    # Merge categories by persistent id; ensure consistent names across sources
    categories_by_id: Dict[int, dict] = {}
    next_image_id = 0
    next_ann_id = 0

    for manifest in manifests:
        coco = load_coco_json(manifest)
        cats = coco.get("categories", [])
        for cat in cats:
            cid = int(cat.get("id"))
            name = str(cat.get("name", ""))
            if cid in categories_by_id:
                # Warn on mismatch; keep the first seen
                if name and categories_by_id[cid]["name"] != name:
                    print(f"Warning: category id {cid} has different names: "
                          f"{categories_by_id[cid]['name']} vs {name} in {manifest}")
                # Merge optional fields (keystroke/description) if missing
                for k in ("keystroke", "description", "supercategory"):
                    if k in cat and k not in categories_by_id[cid]:
                        categories_by_id[cid][k] = cat[k]
            else:
                categories_by_id[cid] = {
                    "id": cid,
                    "name": name,
                    "supercategory": str(cat.get("supercategory", "object")),
                }
                if "keystroke" in cat:
                    categories_by_id[cid]["keystroke"] = cat["keystroke"]
                if "description" in cat:
                    categories_by_id[cid]["description"] = cat["description"]

    out["categories"] = [ensure_json_serializable(categories_by_id[cid]) for cid in sorted(categories_by_id.keys())]

    # Build reverse lookup for categories to resolve names if needed
    # Merge images and annotations; remap image ids to be unique and prefix file_name with folder
    for manifest in manifests:
        coco = load_coco_json(manifest)
        base = manifest.parent
        rel_prefix = base.relative_to(root)

        # Map local image_id -> new global id
        local_to_global: Dict[int, int] = {}

        # Build quick map for image_id->filename to speed up ann mapping in this manifest
        id_to_filename = {int(img["id"]): str(img["file_name"]) for img in coco.get("images", []) if "id" in img and "file_name" in img}

        # Images
        for img in coco.get("images", []):
            # Skip hidden if requested
            if not include_hidden and bool(img.get("hidden", False)):
                continue

            new_img = dict(img)  # shallow copy
            new_img_id = next_image_id
            next_image_id += 1

            local_id = int(img.get("id"))
            local_to_global[local_id] = new_img_id

            # Update id
            new_img["id"] = int(new_img_id)
            # Make file_name relative to root with folder prefix
            fn = str(img.get("file_name"))
            new_img["file_name"] = str((rel_prefix / fn).as_posix())
            out["images"].append(ensure_json_serializable(new_img))

        # Annotations
        for ann in coco.get("annotations", []):
            # Keep only annotations whose image survived filtering
            local_image_id = int(ann.get("image_id"))
            if local_image_id not in local_to_global:
                continue
            new_ann = dict(ann)
            new_ann["id"] = int(next_ann_id)
            next_ann_id += 1
            new_ann["image_id"] = int(local_to_global[local_image_id])

            # Keep category_id as-is (persistent)
            # Ensure category_name is consistent if present
            if "category_name" in new_ann:
                cid = int(new_ann.get("category_id"))
                if cid in categories_by_id:
                    new_ann["category_name"] = categories_by_id[cid]["name"]

            out["annotations"].append(ensure_json_serializable(new_ann))

    return ensure_json_serializable(out)