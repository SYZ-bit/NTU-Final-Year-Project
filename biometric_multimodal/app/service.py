from __future__ import annotations
from pathlib import Path
from app.config import load_settings
from models.inference import MultiModalAuthenticator


class AuthService:
    def __init__(self, settings_path: str = "configs/settings.yaml"):
        self.settings = load_settings(settings_path)
        self.authenticator = MultiModalAuthenticator(self.settings)

    def verify(self, subject_id: str, face_path: str | None, fingerprint_path: str | None, palm_path: str | None) -> dict:
        result = self.authenticator.verify(
            subject_id=subject_id,
            face_path=Path(face_path) if face_path else None,
            fingerprint_path=Path(fingerprint_path) if fingerprint_path else None,
            palm_path=Path(palm_path) if palm_path else None,
        )
        return result
