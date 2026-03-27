from __future__ import annotations
import yaml
from pathlib import Path


def load_settings(path: str | Path = "configs/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
