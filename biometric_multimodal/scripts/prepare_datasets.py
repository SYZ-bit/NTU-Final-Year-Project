from __future__ import annotations

from pathlib import Path
import shutil
from typing import Dict, List, Tuple

import kagglehub

from utils.storage import save_json

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _find_best_lfw_root(root: Path) -> Path:
    candidates = [
        p for p in root.rglob("*")
        if p.is_dir() and p.name.lower() in {"lfw", "lfw-deepfunneled"}
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not find LFW folder under: {root}")

    def score(folder: Path) -> int:
        total = 0
        for child in folder.iterdir():
            if child.is_dir() and any(child.glob("*.jpg")):
                total += 1
        return total

    return max(candidates, key=score)


def _collect_face_subjects() -> List[Tuple[str, List[Path]]]:
    path = Path(kagglehub.dataset_download("jessicali9530/lfw-dataset"))
    print(f"LFW downloaded to: {path}")
    lfw_root = _find_best_lfw_root(path)
    print(f"Using LFW folder: {lfw_root}")

    subjects: List[Tuple[str, List[Path]]] = []
    for person_dir in sorted([p for p in lfw_root.iterdir() if p.is_dir()]):
        imgs = sorted([p for p in person_dir.glob("*.jpg")])
        if len(imgs) >= 2:
            subjects.append((person_dir.name, imgs))
    return subjects


def _socofing_person_id(stem: str) -> str:
    if "__" not in stem:
        return stem
    return stem.split("__")[0]


def _socofing_finger_id(stem: str) -> str:
    """
    Same-finger identity for fingerprint-only verification.

    Examples:
    100__M_Left_index_finger
    100__M_Left_index_finger_CR
    100__M_Left_index_finger_Obl
    100__M_Left_index_finger_Zcut

    -> 100__M_Left_index_finger
    """
    if stem.endswith("_CR") or stem.endswith("_Obl") or stem.endswith("_Zcut"):
        return stem.rsplit("_", 1)[0]
    return stem


def _collect_fingerprint_subjects(include_altered: bool = True) -> List[Tuple[str, List[Path]]]:
    """
    Person-level fingerprint grouping for multimodal fusion.
    Each person folder contains all 10 real fingers, optionally plus altered images.
    """
    path = Path(kagglehub.dataset_download("ruizgara/socofing"))
    print(f"SOCOFing downloaded to: {path}")

    real_root = next((p for p in path.rglob("Real") if p.is_dir()), None)
    if real_root is None:
        raise FileNotFoundError("Could not find SOCOFing/Real folder")

    grouped: Dict[str, List[Path]] = {}

    # Group all REAL images by person ID
    real_imgs = sorted(real_root.glob("*.BMP")) + sorted(real_root.glob("*.bmp"))
    for img in real_imgs:
        person_id = _socofing_person_id(img.stem)
        grouped.setdefault(person_id, []).append(img)

    # Optionally add altered images to the same person's folder
    if include_altered:
        altered_dirs = []
        for name in ["Altered-Easy", "Altered-Medium", "Altered-Hard"]:
            altered_dirs.extend([p for p in path.rglob(name) if p.is_dir()])

        for altered_dir in altered_dirs:
            altered_imgs = sorted(altered_dir.glob("*.BMP")) + sorted(altered_dir.glob("*.bmp"))
            for img in altered_imgs:
                person_id = _socofing_person_id(img.stem)
                if person_id in grouped:
                    grouped[person_id].append(img)

    def sort_key(pid: str):
        try:
            return int(pid)
        except ValueError:
            return pid

    subjects = [
        (person_id, sorted(imgs))
        for person_id, imgs in sorted(grouped.items(), key=lambda x: sort_key(x[0]))
        if len(imgs) >= 10
    ]
    return subjects


def prepare_fingerprint_single_dataset() -> None:
    """
    Same-finger dataset for fingerprint-only verification.
    Each subject folder contains one real finger + its altered versions.
    """
    path = Path(kagglehub.dataset_download("ruizgara/socofing"))
    print(f"SOCOFing downloaded to: {path}")

    real_root = next((p for p in path.rglob("Real") if p.is_dir()), None)
    if real_root is None:
        raise FileNotFoundError("Could not find SOCOFing/Real folder")

    altered_dirs = []
    for name in ["Altered-Easy", "Altered-Medium", "Altered-Hard"]:
        altered_dirs.extend([p for p in path.rglob(name) if p.is_dir()])

    grouped: Dict[str, List[Path]] = {}

    for img in sorted(real_root.glob("*.BMP")) + sorted(real_root.glob("*.bmp")):
        finger_id = _socofing_finger_id(img.stem)
        grouped.setdefault(finger_id, []).append(img)

    for altered_dir in altered_dirs:
        for img in sorted(altered_dir.glob("*.BMP")) + sorted(altered_dir.glob("*.bmp")):
            finger_id = _socofing_finger_id(img.stem)
            if finger_id in grouped:
                grouped[finger_id].append(img)

    out_root = Path("data/fingerprint_single/train")
    reset_dir(out_root)

    mapping = {}
    count_subjects = 0
    count_images = 0

    for idx, (finger_id, imgs) in enumerate(sorted(grouped.items())):
        if len(imgs) < 2:
            continue

        pseudo_id = f"fp_{idx:05d}"
        for img in sorted(imgs):
            safe_copy(img, out_root / pseudo_id / img.name)
            count_images += 1

        mapping[pseudo_id] = finger_id
        count_subjects += 1

    save_json(mapping, "artifacts/reports/fingerprint_single_map.json")

    print(f"Prepared fingerprint-only same-finger data at {out_root}")
    print(f"Fingerprint single subjects copied: {count_subjects}")
    print(f"Fingerprint single images copied: {count_images}")


def _collect_palm_subjects() -> List[Tuple[str, List[Path]]]:
    path = Path(kagglehub.dataset_download("saqibshoaibdz/plam-dataset"))
    print(f"Palm dataset downloaded to: {path}")
    train_dir = next((p for p in path.rglob("_train_data") if p.is_dir()), None)
    if train_dir is None:
        raise FileNotFoundError("Could not find _train_data in palm dataset.")

    images = sorted([p for p in train_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    subjects: List[Tuple[str, List[Path]]] = []
    for idx in range(0, len(images), 3):
        chunk = images[idx: idx + 3]
        if len(chunk) == 3:
            source_id = f"{idx // 3:05d}"
            subjects.append((source_id, chunk))
    return subjects


def prepare_pseudo_aligned_datasets(limit_subjects: int | None = None) -> None:
    face_subjects = _collect_face_subjects()
    fp_subjects = _collect_fingerprint_subjects(include_altered=True)
    palm_subjects = _collect_palm_subjects()

    n = min(len(face_subjects), len(fp_subjects), len(palm_subjects))
    if limit_subjects is not None:
        n = min(n, limit_subjects)
    if n == 0:
        raise ValueError("No pseudo-aligned subjects could be prepared.")

    out_face = Path("data/face/train")
    out_fp = Path("data/fingerprint/train")
    out_palm = Path("data/palm/train")
    reset_dir(out_face)
    reset_dir(out_fp)
    reset_dir(out_palm)

    mapping = {}
    face_image_count = 0
    fp_image_count = 0
    palm_image_count = 0

    for idx in range(n):
        pseudo_id = f"s{idx:05d}"
        face_source, face_imgs = face_subjects[idx]
        fp_source, fp_imgs = fp_subjects[idx]
        palm_source, palm_imgs = palm_subjects[idx]

        for img in face_imgs:
            safe_copy(img, out_face / pseudo_id / img.name)
            face_image_count += 1

        for img in fp_imgs:
            safe_copy(img, out_fp / pseudo_id / img.name)
            fp_image_count += 1

        for img in palm_imgs:
            safe_copy(img, out_palm / pseudo_id / img.name)
            palm_image_count += 1

        mapping[pseudo_id] = {
            "face_source_id": face_source,
            "fingerprint_source_id": fp_source,
            "palm_source_id": palm_source,
        }

    save_json(mapping, "artifacts/reports/pseudo_identity_map.json")

    print(f"Prepared pseudo-aligned face data at {out_face}")
    print(f"Prepared pseudo-aligned fingerprint data at {out_fp}")
    print(f"Prepared pseudo-aligned palm data at {out_palm}")
    print(f"Pseudo subjects copied: {n}")
    print(f"Face images copied: {face_image_count}")
    print(f"Fingerprint images copied: {fp_image_count}")
    print(f"Palm images copied: {palm_image_count}")
    if n > 0:
        first_sid = f"s{0:05d}"
        print(f"Example mapping: {first_sid} -> {mapping[first_sid]}")


if __name__ == "__main__":
    prepare_pseudo_aligned_datasets(limit_subjects=120)
    prepare_fingerprint_single_dataset()