from __future__ import annotations
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression


class ScoreFusion:
    def __init__(self, face_weight: float = 0.40, fingerprint_weight: float = 0.35, palm_weight: float = 0.25):
        self.face_weight = face_weight
        self.fingerprint_weight = fingerprint_weight
        self.palm_weight = palm_weight
        self.lr = None

    def weighted_sum(self, face_score: float, fingerprint_score: float, palm_score: float) -> float:
        return float(
            self.face_weight * face_score +
            self.fingerprint_weight * fingerprint_score +
            self.palm_weight * palm_score
        )

    def train_lr(self, X: np.ndarray, y: np.ndarray):
        self.lr = LogisticRegression(max_iter=1000)
        self.lr.fit(X, y)
        return self

    def predict_lr_score(self, face_score: float, fingerprint_score: float, palm_score: float, weighted_score: float) -> float:
        if self.lr is None:
            raise RuntimeError("Fusion LR model has not been trained.")
        x = np.array([[face_score, fingerprint_score, palm_score, weighted_score]], dtype=np.float32)
        return float(self.lr.predict_proba(x)[0, 1])

    def save(self, path: str):
        dump(self.lr, path)

    def load(self, path: str):
        self.lr = load(path)
        return self
