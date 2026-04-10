from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.dataset import list_images_by_subject, make_verification_pairs
from utils.fingerprint_module import FingerprintMatcher
from utils.metrics import compute_binary_metrics, far_frr
from utils.storage import save_json


def _select_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidate_thresholds = np.linspace(0.05, 0.80, 16)

    best_thr = 0.30
    best_obj = -1.0

    for thr in candidate_thresholds:
        y_pred = (scores >= thr).astype(int)
        metrics = compute_binary_metrics(y_true, y_pred)
        err = far_frr(y_true, scores, thr)

        objective = metrics["accuracy"] - 0.5 * err["FAR"] - 0.25 * err["FRR"]
        if objective > best_obj:
            best_obj = objective
            best_thr = float(thr)

    return best_thr


def train_fingerprint_system(
    fp_root: str = "data/fingerprint_single/train",
    max_pairs_per_subject: int = 6,
    max_negatives: int = 1200,
) -> Dict:
    matcher = FingerprintMatcher(max_keypoints=500)
    mapping = list_images_by_subject(fp_root)

    if len(mapping) == 0:
        raise ValueError("Fingerprint dataset is empty.")

    pairs = make_verification_pairs(
        mapping,
        max_pairs_per_subject=max_pairs_per_subject,
        max_negatives=max_negatives,
    )

    scores, y_true = [], []
    for a, b, label in tqdm(pairs, desc="Fingerprint verification"):
        try:
            s = matcher.pair_score(a, b)
        except Exception:
            s = 0.0
        scores.append(s)
        y_true.append(label)

    scores = np.asarray(scores, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int32)

    idx = np.arange(len(scores))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=42,
        stratify=y_true,
    )

    train_scores = scores[train_idx]
    train_y = y_true[train_idx]
    test_scores = scores[test_idx]
    test_y = y_true[test_idx]

    threshold = _select_best_threshold(train_y, train_scores)

    y_pred = (test_scores >= threshold).astype(int)
    metrics = compute_binary_metrics(test_y, y_pred)
    metrics.update(far_frr(test_y, test_scores, threshold))

    result = {
        "threshold": float(threshold),
        "metrics": metrics,
        "n_pairs_total": int(len(scores)),
        "n_pairs_train": int(len(train_scores)),
        "n_pairs_test": int(len(test_scores)),
        "dataset_root": fp_root,
    }
    save_json(result, "artifacts/reports/fingerprint_metrics.json")
    return result


if __name__ == "__main__":
    print(train_fingerprint_system())