from dataclasses import dataclass


@dataclass
class VerifyRequest:
    subject_id: str
    face_path: str | None = None
    fingerprint_path: str | None = None
    palm_path: str | None = None


@dataclass
class VerifyResponse:
    subject_id: str
    modality_scores: dict
    fused_score: float
    decision: str
    threshold: float
