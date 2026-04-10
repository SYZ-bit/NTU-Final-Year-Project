from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class VerifyRequest:
    subject_id: str
    face_path: str
    fingerprint_path: str
    palm_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerifyResponse:
    subject_id: str
    modality_scores: Dict[str, float]
    weighted_fusion_score: float
    logistic_fusion_probability: float
    decision: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
