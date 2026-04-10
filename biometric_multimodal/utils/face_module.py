from __future__ import annotations

from typing import Optional
import numpy as np
from utils.image_ops import read_bgr, to_rgb

try:
    import face_recognition  # type: ignore[reportMissingImports]
except Exception:
    face_recognition = None


class FaceRecognizer:
    def __init__(self, model: str = "hog", num_jitters: int = 1):
        self.model = model
        self.num_jitters = num_jitters

    def available(self) -> bool:
        return face_recognition is not None

    def encode(self, path: str) -> Optional[np.ndarray]:
        if face_recognition is None:
            raise ImportError("face_recognition is not installed. Install face-recognition and dlib.")

        img_bgr = read_bgr(path)
        img = to_rgb(img_bgr)

        boxes = face_recognition.face_locations(img, model=self.model)
        if not boxes:
            return None

        enc = face_recognition.face_encodings(
            img,
            known_face_locations=boxes,
            num_jitters=self.num_jitters,
        )
        return np.asarray(enc[0], dtype=np.float32) if enc else None

    @staticmethod
    def face_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 1.0
        return float(np.linalg.norm(a - b))

    def pair_score(self, path_a: str, path_b: str) -> float:
        ea = self.encode(path_a)
        eb = self.encode(path_b)
        dist = self.face_distance(ea, eb)

        # Convert distance to similarity-like score for reporting
        # smaller distance -> higher score
        return float(max(0.0, 1.0 - dist))