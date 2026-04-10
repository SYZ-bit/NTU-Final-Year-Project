from __future__ import annotations

from pathlib import Path
from typing import Dict
import numpy as np
import torch

from utils.face_module import FaceRecognizer
from utils.fingerprint_module import FingerprintMatcher
from utils.palm_module import PalmCNN, PalmFeatureExtractor
from models.fusion import WeightedFusion, LogisticFusion
from utils.storage import load_json


class MultiModalInference:
    def __init__(self):
        self.face = FaceRecognizer(model="hog", num_jitters=1)
        self.fingerprint = FingerprintMatcher(max_keypoints=500)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        meta = {"classes": 1}
        if Path("artifacts/models/palm_cnn_meta.json").exists():
            meta = load_json("artifacts/models/palm_cnn_meta.json")
        self.palm_cnn = PalmCNN(num_classes=max(int(meta.get("classes", 1)), 1)).to(self.device)
        if Path("artifacts/models/palm_cnn.pt").exists():
            state = torch.load("artifacts/models/palm_cnn.pt", map_location=self.device)
            state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
            self.palm_cnn.load_state_dict(state, strict=False)
        self.palm_cnn.eval()
        self.palm = PalmFeatureExtractor(self.palm_cnn, device=self.device)

        self.weighted = WeightedFusion({"face": 0.34, "fingerprint": 0.33, "palm": 0.33})
        self.logistic = LogisticFusion.load("artifacts/models/fusion_lr.joblib") if Path("artifacts/models/fusion_lr.joblib").exists() else None

    def verify_face_pair(self, face_a: str, face_b: str) -> float:
        return self.face.pair_score(face_a, face_b)

    def verify_fingerprint_pair(self, fp_a: str, fp_b: str) -> float:
        return self.fingerprint.pair_score(fp_a, fp_b)

    def verify_palm_pair(self, palm_a: str, palm_b: str) -> float:
        return self.palm.pair_score(palm_a, palm_b)

    def verify_pair(self, face_a: str, face_b: str, fp_a: str, fp_b: str, palm_a: str, palm_b: str) -> Dict[str, float]:
        scores = {
            "face": self.verify_face_pair(face_a, face_b),
            "fingerprint": self.verify_fingerprint_pair(fp_a, fp_b),
            "palm": self.verify_palm_pair(palm_a, palm_b),
        }
        weighted_score = self.weighted.score(scores)
        prob = None
        if self.logistic is not None:
            arr = np.asarray([[scores["face"], scores["fingerprint"], scores["palm"]]], dtype=np.float32)
            prob = float(self.logistic.predict_proba(arr)[0])
        return {
            **scores,
            "weighted_fusion": weighted_score,
            "logistic_fusion": prob if prob is not None else weighted_score,
        }
