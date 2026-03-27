from __future__ import annotations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0, 0])


def euclidean_to_similarity(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a - b)
    return float(1.0 / (1.0 + d))


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    score = (score - min_val) / (max_val - min_val + 1e-8)
    return float(np.clip(score, 0.0, 1.0))


def find_best_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thresholds[idx])
