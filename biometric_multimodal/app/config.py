from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class AppConfig:
    raw: Dict[str, Any]

    @property
    def data_root(self) -> Path:
        return Path(self.raw["paths"]["data_root"])

    @property
    def artifact_root(self) -> Path:
        return Path(self.raw["paths"]["artifact_root"])

    @property
    def models_root(self) -> Path:
        return Path(self.raw["paths"]["models_root"])

    @property
    def gallery_root(self) -> Path:
        return Path(self.raw["paths"]["gallery_root"])

    @property
    def reports_root(self) -> Path:
        return Path(self.raw["paths"]["reports_root"])


def load_config(path: str = "configs/settings.yaml") -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        return AppConfig(yaml.safe_load(f))
