from __future__ import annotations
from typing import Dict
from tqdm import tqdm
from models.dataset import list_images_by_subject, make_verification_pairs
from utils.face_module import FaceRecognizer
from utils.metrics import compute_binary_metrics, far_frr
from utils.storage import save_json


def train_face_system(face_root: str = "data/face/train", threshold: float = 0.40) -> Dict:
    recognizer = FaceRecognizer(model="hog", num_jitters=1)
    mapping = list_images_by_subject(face_root)
    pairs = make_verification_pairs(mapping, max_pairs_per_subject=6, max_negatives=600)

    scores, y_true = [], []
    for a, b, label in tqdm(pairs, desc="Face verification"):
        try:
            s = recognizer.pair_score(a, b)
        except Exception:
            s = 0.0
        scores.append(s)
        y_true.append(label)

    y_pred = [1 if s >= threshold else 0 for s in scores]
    metrics = compute_binary_metrics(y_true, y_pred)
    metrics.update(far_frr(y_true, scores, threshold))
    result = {"threshold": threshold, "metrics": metrics, "n_pairs": len(pairs)}
    save_json(result, "artifacts/reports/face_metrics.json")
    return result


if __name__ == "__main__":
    print(train_face_system())
