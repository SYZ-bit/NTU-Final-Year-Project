from __future__ import annotations

from typing import Dict
from models.inference import MultiModalInference


class VerificationService:
    def __init__(self):
        self.engine = MultiModalInference()
        self.thresholds = {
            "face": 0.52,
            "fingerprint": 0.42,
            "palm": 0.60,
            "fusion": 0.50,
        }

    def verify_face(self, enrollment_path: str, probe_path: str, claimed_subject_id: str) -> Dict:
        score = self.engine.verify_face_pair(enrollment_path, probe_path)
        decision = "accept" if score >= self.thresholds["face"] else "reject"
        return {
            "subject_id": claimed_subject_id,
            "modality": "face",
            "score": score,
            "threshold": self.thresholds["face"],
            "decision": decision,
        }

    def verify_fingerprint(self, enrollment_path: str, probe_path: str, claimed_subject_id: str) -> Dict:
        score = self.engine.verify_fingerprint_pair(enrollment_path, probe_path)
        decision = "accept" if score >= self.thresholds["fingerprint"] else "reject"
        return {
            "subject_id": claimed_subject_id,
            "modality": "fingerprint",
            "score": score,
            "threshold": self.thresholds["fingerprint"],
            "decision": decision,
        }

    def verify_palm(self, enrollment_path: str, probe_path: str, claimed_subject_id: str) -> Dict:
        score = self.engine.verify_palm_pair(enrollment_path, probe_path)
        decision = "accept" if score >= self.thresholds["palm"] else "reject"
        return {
            "subject_id": claimed_subject_id,
            "modality": "palm",
            "score": score,
            "threshold": self.thresholds["palm"],
            "decision": decision,
        }

    def verify(self, enrollment: Dict[str, str], probe: Dict[str, str], claimed_subject_id: str) -> Dict:
        scores = self.engine.verify_pair(
            enrollment["face"], probe["face"],
            enrollment["fingerprint"], probe["fingerprint"],
            enrollment["palm"], probe["palm"],
        )
        decision = "accept" if scores["logistic_fusion"] >= self.thresholds["fusion"] else "reject"
        return {
            "subject_id": claimed_subject_id,
            "modality_scores": {
                "face": scores["face"],
                "fingerprint": scores["fingerprint"],
                "palm": scores["palm"],
            },
            "weighted_fusion_score": scores["weighted_fusion"],
            "logistic_fusion_probability": scores["logistic_fusion"],
            "decision": decision,
        }
