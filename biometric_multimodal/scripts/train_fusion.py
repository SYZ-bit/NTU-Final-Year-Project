from __future__ import annotations
from pathlib import Path
import random
import numpy as np
from app.config import load_settings
from utils.face_module import FaceRecognizer
from utils.fingerprint_module import FingerprintRecognizer
from utils.palm_module import PalmRecognizer
from utils.metrics import cosine_score
from models.fusion import ScoreFusion


def first_two_images(folder: Path):
    imgs = [p for p in sorted(folder.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if len(imgs) < 2:
        raise ValueError(f"Need at least two images in {folder}")
    return imgs[0], imgs[1]


def main():
    settings = load_settings("configs/settings.yaml")
    data_root = Path("data")
    subjects = sorted([p.name for p in (data_root / "face/train").iterdir() if p.is_dir()])

    face = FaceRecognizer(device="cpu", use_insightface=False)
    fp = FingerprintRecognizer(device="cpu")
    palm = PalmRecognizer(device="cpu")
    fusion = ScoreFusion(
        face_weight=settings["fusion"]["weighted_score"]["face"],
        fingerprint_weight=settings["fusion"]["weighted_score"]["fingerprint"],
        palm_weight=settings["fusion"]["weighted_score"]["palm"],
    )

    X, y = [], []

    # Genuine pairs
    for sid in subjects:
        face_a, face_b = first_two_images(data_root / "face/train" / sid)
        fp_a, fp_b = first_two_images(data_root / "fingerprint/train" / sid)
        palm_a, palm_b = first_two_images(data_root / "palm/train" / sid)

        face_score = (cosine_score(face.extract(face_a), face.extract(face_b)) + 1.0) / 2.0
        fp_score = fp.compare(fp_a, fp_b)["combined_score"]
        palm_score = palm.compare(palm_a, palm_b)["combined_score"]
        weighted = fusion.weighted_sum(face_score, fp_score, palm_score)
        X.append([face_score, fp_score, palm_score, weighted])
        y.append(1)

    # Impostor pairs
    for _ in range(len(subjects)):
        s1, s2 = random.sample(subjects, 2)
        face_a = first_two_images(data_root / "face/train" / s1)[0]
        face_b = first_two_images(data_root / "face/train" / s2)[0]
        fp_a = first_two_images(data_root / "fingerprint/train" / s1)[0]
        fp_b = first_two_images(data_root / "fingerprint/train" / s2)[0]
        palm_a = first_two_images(data_root / "palm/train" / s1)[0]
        palm_b = first_two_images(data_root / "palm/train" / s2)[0]

        face_score = (cosine_score(face.extract(face_a), face.extract(face_b)) + 1.0) / 2.0
        fp_score = fp.compare(fp_a, fp_b)["combined_score"]
        palm_score = palm.compare(palm_a, palm_b)["combined_score"]
        weighted = fusion.weighted_sum(face_score, fp_score, palm_score)
        X.append([face_score, fp_score, palm_score, weighted])
        y.append(0)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    fusion.train_lr(X, y)
    save_path = Path(settings["paths"]["fusion_model_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fusion.save(str(save_path))
    print(f"Saved fusion model to {save_path}")


if __name__ == "__main__":
    main()
