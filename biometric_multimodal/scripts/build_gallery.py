from __future__ import annotations
from pathlib import Path
from utils.face_module import FaceRecognizer
from utils.storage import save_pickle


def first_image(folder: Path) -> Path:
    for p in sorted(folder.glob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            return p
    raise FileNotFoundError(f"No image in {folder}")


def main():
    data_root = Path("data")
    save_path = Path("saved_models/gallery.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    face_model = FaceRecognizer(device="cpu", use_insightface=False)
    gallery = {}

    subjects = sorted([p.name for p in (data_root / "face/train").iterdir() if p.is_dir()])
    for sid in subjects:
        face_path = first_image(data_root / "face/train" / sid)
        fingerprint_path = first_image(data_root / "fingerprint/train" / sid)
        palm_path = first_image(data_root / "palm/train" / sid)
        face_embedding = face_model.extract(face_path)
        gallery[sid] = {
            "face_embedding": face_embedding,
            "fingerprint_path": str(fingerprint_path),
            "palm_path": str(palm_path),
        }
        print(f"Built gallery template for {sid}")

    save_pickle(gallery, save_path)
    print(f"Saved gallery to {save_path}")


if __name__ == "__main__":
    main()
