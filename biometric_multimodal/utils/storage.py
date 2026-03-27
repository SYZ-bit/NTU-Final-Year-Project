from __future__ import annotations
import pickle
from pathlib import Path


def save_pickle(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)
