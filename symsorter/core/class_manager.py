from pathlib import Path
import yaml


class ClassManager:
    def __init__(self, class_file=None):
        self.classes = []                 # list[str] (UI order)
        self.class_keystrokes = {}        # idx -> str
        self.class_descriptions = {}      # idx -> str
        self.class_ids = {}               # idx -> persistent int id
        self.id_to_index = {}             # persistent id -> idx
        self.class_file_path = class_file
        self.last_used_class_idx = None

    def load(self, class_file_path) -> bool:
        if isinstance(class_file_path, str):
            class_file_path = Path(class_file_path)
        if not class_file_path or not class_file_path.exists():
            print(f"Class file not found: {class_file_path}")
            return False

        self.classes = []
        self.class_keystrokes = {}
        self.class_descriptions = {}
        self.class_ids = {}
        self.id_to_index = {}

        suffix = class_file_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            with open(class_file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            class_list = data.get("classes", [])
            for entry in class_list:
                if isinstance(entry, dict) and "name" in entry:
                    idx = len(self.classes)
                    name = (entry.get("name") or "").strip()
                    if not name:
                        continue
                    self.classes.append(name)
                    # persistent ID from YAML, fallback to index
                    cid = entry.get("class_id", idx)
                    try:
                        cid = int(cid)
                    except Exception:
                        cid = idx
                    self.class_ids[idx] = cid
                    self.id_to_index[cid] = idx

                    key = str(entry.get("keystroke", "") or "").strip()
                    if key:
                        self.class_keystrokes[idx] = key
                    desc = str(entry.get("description", "") or "").strip()
                    if desc:
                        self.class_descriptions[idx] = desc
                elif isinstance(entry, str):
                    idx = len(self.classes)
                    name = entry.strip()
                    if not name:
                        continue
                    self.classes.append(name)
                    self.class_ids[idx] = idx
                    self.id_to_index[idx] = idx
        else:
            # Simple text formats
            with open(class_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if ":" in line:
                        name, key = line.split(":", 1)
                        name, key = name.strip(), key.strip()
                        if not name:
                            continue
                        idx = len(self.classes)
                        self.classes.append(name)
                        self.class_ids[idx] = idx
                        self.id_to_index[idx] = idx
                        if key:
                            self.class_keystrokes[idx] = key
                    else:
                        idx = len(self.classes)
                        self.classes.append(line)
                        self.class_ids[idx] = idx
                        self.id_to_index[idx] = idx

        self.class_file_path = class_file_path
        print(f"Loaded {len(self.classes)} classes from {class_file_path}")
        return True

    def get_persistent_id(self, class_idx: int) -> int:
        return int(self.class_ids.get(class_idx, class_idx))