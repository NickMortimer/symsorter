import json
from pathlib import Path
from datetime import datetime


def get_classifications_json_path(npz_file_path: Path | None) -> Path | None:
    if not npz_file_path:
        return None
    return npz_file_path.parent / "classifications.json"


def ensure_json_serializable(obj):
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): ensure_json_serializable(v) for k, v in obj.items()}
    if hasattr(obj, 'tolist'):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, '__fspath__'):
        return str(obj)
    return obj


def build_coco_data(folder,
                    embeddings,
                    hidden_flags,
                    classes,
                    class_keystrokes,
                    class_descriptions,
                    image_categories,
                    embedding_metadata=None,
                    cached_dimensions=None,
                    class_ids=None):
    # class_ids: dict[index -> persistent id], if None fallback to index
    try:
        from PIL import Image
    except Exception:
        Image = None
    from pathlib import Path as _Path

    class_ids = class_ids or {}

    data = {
        "info": {
            "description": "SymSorter Image Classifications",
            "version": "2.0",
            "year": int(datetime.now().year),
            "contributor": "SymSorter",
            "date_created": str(datetime.now().isoformat()),
            "symsorter_version": "1.0",
            "embedding_metadata": embedding_metadata or {}
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": [],
        "images": [],
        "annotations": []
    }

    # Categories use persistent IDs if provided
    for i, name in enumerate(classes):
        cid = int(class_ids.get(i, i))
        cat = {
            "id": cid,
            "name": str(name),
            "supercategory": "object",
            "description": str(class_descriptions.get(i, "")),
            "keystroke": str(class_keystrokes.get(i, "")),
        }
        data["categories"].append(ensure_json_serializable(cat))

    image_id = 0
    ann_id = 0
    folder_path = _Path(folder) if folder else _Path(".")
    for filename in embeddings.keys():
        img_path = folder_path / filename
        if not img_path.exists():
            continue

        width, height = 0, 0
        if cached_dimensions and filename in cached_dimensions:
            try:
                width, height = int(cached_dimensions[filename][0]), int(cached_dimensions[filename][1])
            except Exception:
                width, height = 0, 0
        if (width == 0 or height == 0) and Image is not None:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception:
                width, height = 0, 0

        img_info = {
            "id": int(image_id),
            "file_name": str(filename),
            "width": int(width),
            "height": int(height),
            "license": 1,
            "date_captured": "",
            "hidden": bool(hidden_flags.get(filename, False)),
        }
        data["images"].append(ensure_json_serializable(img_info))

        if filename in image_categories:
            cat = image_categories[filename]
            ann = {
                "id": int(ann_id),
                "image_id": int(image_id),
                "category_id": int(cat['class_id']),
                "category_name": str(cat['class_name']),
                "bbox": [0, 0, int(width), int(height)],
                "area": int(width * height),
                "iscrowd": 0,
                "segmentation": []
            }
            data["annotations"].append(ensure_json_serializable(ann))
            ann_id += 1

        image_id += 1

    return ensure_json_serializable(data)


def save_coco_json(npz_file_path: Path, coco_data: dict) -> str:
    json_path = get_classifications_json_path(npz_file_path)
    if not json_path:
        raise ValueError("Invalid npz path for JSON save")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    return str(json_path)


def load_coco_json(json_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        import json as _json
        return _json.load(f)