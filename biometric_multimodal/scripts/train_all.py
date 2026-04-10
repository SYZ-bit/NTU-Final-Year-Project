from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from models.dataset import intersect_subjects, list_images_by_subject, make_verification_pairs
from utils.face_module import FaceRecognizer
from utils.fingerprint_module import FingerprintMatcher
from utils.palm_module import PalmCNN, PalmFeatureExtractor
from models.train_face import train_face_system
from models.train_fingerprint import train_fingerprint_system
from models.train_palm import train_palm_cnn, evaluate_palm_system
from utils.metrics import compute_binary_metrics, far_frr, plot_metric_bars
from utils.storage import save_json


def _select_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Select the best threshold by balancing FAR and FRR,
    with accuracy used as a tie-breaker.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    candidate_thresholds = np.linspace(0.05, 0.95, 19)

    best_threshold = 0.5
    best_metrics: Dict[str, float] | None = None
    best_objective = -1.0
    best_accuracy = -1.0

    for thr in candidate_thresholds:
        y_pred = (scores >= thr).astype(int)
        cls_metrics = compute_binary_metrics(y_true, y_pred)
        err_metrics = far_frr(y_true, scores, thr)

        objective = 1.0 - 0.5 * (err_metrics["FAR"] + err_metrics["FRR"])
        acc = cls_metrics["accuracy"]

        if objective > best_objective or (abs(objective - best_objective) < 1e-12 and acc > best_accuracy):
            best_objective = objective
            best_accuracy = acc
            best_threshold = float(thr)
            best_metrics = {**cls_metrics, **err_metrics}

    assert best_metrics is not None
    return best_threshold, best_metrics


def build_fusion_training_set(limit_subjects: int = 100):
    fused = intersect_subjects(
        "data/face/train",
        "data/fingerprint/train",
        "data/palm/train",
        limit=limit_subjects,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    face = FaceRecognizer(model="hog", num_jitters=1)
    fp = FingerprintMatcher(max_keypoints=500)

    palm_root = Path("data/palm/train")
    num_palm_classes = len([p for p in palm_root.iterdir() if p.is_dir()]) if palm_root.exists() else 1
    num_palm_classes = max(num_palm_classes, 1)

    palm_cnn = PalmCNN(num_classes=num_palm_classes).to(device)

    model_path = Path("artifacts/models/palm_cnn.pt")
    if model_path.exists():
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict):
            state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
        palm_cnn.load_state_dict(state, strict=False)

    palm_cnn.eval()
    palm = PalmFeatureExtractor(palm_cnn, device=device)

    sids = sorted(fused)
    X, y = [], []

    # Positive pairs
    for sid in sids:
        f_imgs = fused[sid]["face"]
        fp_imgs = fused[sid]["fingerprint"]
        p_imgs = fused[sid]["palm"]

        if len(f_imgs) < 2 or len(fp_imgs) < 2 or len(p_imgs) < 2:
            continue

        row = [
            face.pair_score(f_imgs[0], f_imgs[1]),
            fp.pair_score(fp_imgs[0], fp_imgs[1]),
            palm.pair_score(p_imgs[0], p_imgs[1]),
        ]
        X.append(row)
        y.append(1)

    # Negative pairs
    for i in range(min(len(sids) - 1, len(X))):
        a = sids[i]
        b = sids[-(i + 1)]

        if a == b:
            continue

        row = [
            face.pair_score(fused[a]["face"][0], fused[b]["face"][0]),
            fp.pair_score(fused[a]["fingerprint"][0], fused[b]["fingerprint"][0]),
            palm.pair_score(fused[a]["palm"][0], fused[b]["palm"][0]),
        ]
        X.append(row)
        y.append(0)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def train_all():
    # Train/evaluate unimodal systems
    face_result = train_face_system()
    fingerprint_result = train_fingerprint_system()
    train_palm_cnn()
    palm_result = evaluate_palm_system()

    # Build fusion set
    X, y = build_fusion_training_set(limit_subjects=80)
    if len(X) < 10:
        raise ValueError("Fusion dataset is too small to evaluate reliably.")

    # Held-out split for honest evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    # Weighted fusion only
    weights = np.asarray([0.34, 0.33, 0.33], dtype=np.float32)
    weighted_scores_train = X_train @ weights
    weighted_scores_test = X_test @ weights

    best_weighted_threshold, _ = _select_best_threshold(y_train, weighted_scores_train)
    weighted_pred_test = (weighted_scores_test >= best_weighted_threshold).astype(int)

    systems = {
        "Face": face_result["metrics"],
        "Fingerprint": fingerprint_result["metrics"],
        "Palm": palm_result["metrics"],
        "Fusion(weighted)": {
            **compute_binary_metrics(y_test, weighted_pred_test),
            **far_frr(y_test, weighted_scores_test, best_weighted_threshold),
        },
    }

    extra = {
        "fusion_weighted_threshold": float(best_weighted_threshold),
        "fusion_train_size": int(len(X_train)),
        "fusion_test_size": int(len(X_test)),
        "fusion_weights": {
            "face": 0.34,
            "fingerprint": 0.33,
            "palm": 0.33,
        },
    }

    save_json(systems, "artifacts/reports/all_metrics.json")
    save_json(extra, "artifacts/reports/thresholds_and_splits.json")
    plot_metric_bars(systems, "artifacts/reports/comparison.png")

    print("Fusion weighted threshold:", best_weighted_threshold)
    print(systems)


if __name__ == "__main__":
    train_all()