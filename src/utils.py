
import os, yaml, json
from dataclasses import dataclass
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
