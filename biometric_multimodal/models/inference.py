from __future__ import annotations
from pathlib import Path
import numpy as np
from app.config import load_settings
from utils.face_module import FaceRecognizer
from utils.fingerprint_module import FingerprintRecognizer
from utils.palm_module import PalmRecognizer
from utils.storage import load_pickle
from utils.metrics import cosine_score
from models.fusion import ScoreFusion


class MultiModalAuthenticator:
    def __init__(self, settings: dict):
        self.settings = settings
        self.model_root = Path(settings["paths"]["model_root"])
        self.gallery_path = Path(settings["paths"]["gallery_path"])
        self.fusion_path = Path(settings["paths"]["fusion_model_path"])
        device = "cuda" if False else "cpu"

        self.face = FaceRecognizer(device=device, use_insightface=True)
        self.fingerprint = FingerprintRecognizer(device=device)
        self.palm = PalmRecognizer(device=device)

        face_w = self.model_root / "face_resnet50_embed.pt"
        fp_w = self.model_root / "fingerprint_resnet18_embed.pt"
        palm_w = self.model_root / "palm_resnet18_embed.pt"
        if face_w.exists() and self.face.model is not None:
            self.face.load_weights(face_w)
        if fp_w.exists():
            self.fingerprint.load_weights(fp_w)
        if palm_w.exists():
            self.palm.load_weights(palm_w)

        self.gallery = load_pickle(self.gallery_path) if self.gallery_path.exists() else {}
        self.fusion = ScoreFusion(
            face_weight=settings["fusion"]["weighted_score"]["face"],
            fingerprint_weight=settings["fusion"]["weighted_score"]["fingerprint"],
            palm_weight=settings["fusion"]["weighted_score"]["palm"],
        )
        if self.fusion_path.exists():
            self.fusion.load(self.fusion_path)

    def verify(self, subject_id: str, face_path: Path | None, fingerprint_path: Path | None, palm_path: Path | None) -> dict:
        if subject_id not in self.gallery:
            raise KeyError(f"Subject {subject_id} not found in gallery")
        template = self.gallery[subject_id]
        scores = {}

        face_score = 0.0
        if face_path is not None:
            probe_face = self.face.extract(face_path)
            gallery_face = template["face_embedding"]
            face_score = (cosine_score(probe_face, gallery_face) + 1.0) / 2.0
            scores["face"] = float(face_score)

        fingerprint_score = 0.0
        if fingerprint_path is not None:
            fp_res = self.fingerprint.compare(fingerprint_path, template["fingerprint_path"])
            fingerprint_score = fp_res["combined_score"]
            scores["fingerprint"] = float(fingerprint_score)
            scores["fingerprint_details"] = fp_res

        palm_score = 0.0
        if palm_path is not None:
            palm_res = self.palm.compare(palm_path, template["palm_path"])
            palm_score = palm_res["combined_score"]
            scores["palm"] = float(palm_score)
            scores["palm_details"] = palm_res

        weighted_score = self.fusion.weighted_sum(face_score, fingerprint_score, palm_score)
        fused_score = weighted_score
        if self.fusion.lr is not None:
            fused_score = self.fusion.predict_lr_score(face_score, fingerprint_score, palm_score, weighted_score)

        threshold = float(self.settings["thresholds"]["fused"])
        decision = "accept" if fused_score >= threshold else "reject"
        return {
            "subject_id": subject_id,
            "modality_scores": scores,
            "weighted_score": float(weighted_score),
            "fused_score": float(fused_score),
            "decision": decision,
            "threshold": threshold,
        }
