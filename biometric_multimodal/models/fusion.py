from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils.storage import save_joblib, load_joblib


@dataclass
class WeightedFusion:
    weights: Dict[str, float]

    def score(self, modality_scores: Dict[str, float]) -> float:
        total = 0.0
        for k, w in self.weights.items():
            total += w * float(modality_scores.get(k, 0.0))
        return float(total)


class LogisticFusion:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        save_joblib(self.model, path)

    @classmethod
    def load(cls, path: str) -> "LogisticFusion":
        obj = cls()
        obj.model = load_joblib(path)
        return obj
