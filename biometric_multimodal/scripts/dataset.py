from pathlib import Path
import shutil
import random
import csv
import re

try:
    import kagglehub
except ImportError as e:
    raise ImportError(
        "kagglehub is not installed. Run: python -m pip install kagglehub"
    ) from e


# ============================================================
# CONFIG
# ============================================================

random.seed(42)

OUT_ROOT = Path(r"D:\SYZ\Github\Personal Projects\NTU Final Year Project\biometric_multimodal\data")

FACE_DATASET = "hearfool/vggface2"
PALM_DATASET = "saqibshoaibdz/plam-dataset"
FINGERPRINT_DATASET = "ruizgara/socofing"

MAX_SUBJECTS = 100
MAX_FACE_IMAGES = 5
MAX_PALM_IMAGES = 3
TRAIN_RATIO = 0.8

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================
# DOWNLOAD
# ============================================================

def download_datasets():
    face_download = Path(kagglehub.dataset_download(FACE_DATASET))
    palm_download = Path(kagglehub.dataset_download(PALM_DATASET))
    fp_download = Path(kagglehub.dataset_download(FINGERPRINT_DATASET))

    print(f"Face downloaded to: {face_download}")
    print(f"Palm downloaded to: {palm_download}")
    print(f"Fingerprint downloaded to: {fp_download}")

    return face_download, palm_download, fp_download


# ============================================================
# HELPERS
# ============================================================

def ensure_output_dirs(root: Path):
    for modality in ["face", "fingerprint", "palm"]:
        for split in ["train", "val"]:
            (root / modality / split).mkdir(parents=True, exist_ok=True)


def clear_existing_subject_dirs(root: Path):
    if not root.exists():
        return

    for modality in ["face", "fingerprint", "palm"]:
        for split in ["train", "val"]:
            split_dir = root / modality / split
            if split_dir.exists():
                for child in split_dir.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)

    mapping_csv = root / "synthetic_identity_mapping.csv"
    if mapping_csv.exists():
        mapping_csv.unlink()


def print_tree(root: Path, max_depth=2, prefix=""):
    if max_depth < 0 or not root.exists():
        return

    items = sorted(root.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    for item in items[:30]:
        print(prefix + item.name + ("/" if item.is_dir() else ""))
        if item.is_dir():
            print_tree(item, max_depth=max_depth - 1, prefix=prefix + "  ")


def list_images_in_dir(folder: Path):
    if not folder.exists() or not folder.is_dir():
        return []

    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda x: x.name.lower()
    )


def recursive_images(folder: Path):
    if not folder.exists():
        return []

    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda x: str(x).lower()
    )


def split_images(images, train_ratio=0.8):
    if len(images) < 2:
        return [], []

    split_idx = max(1, int(len(images) * train_ratio))
    if split_idx >= len(images):
        split_idx = len(images) - 1

    return images[:split_idx], images[split_idx:]


def copy_images(images, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, dst_dir / img.name)


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.stem.lower())
    return [int(p) if p.isdigit() else p for p in parts]


# ============================================================
# FACE: VGGFACE2
# ============================================================

def get_face_identity_map(face_download: Path):
    candidate_roots = [
        face_download / "train",
        face_download / "vggface2" / "train",
        face_download,
    ]

    for root in candidate_roots:
        if root.exists() and root.is_dir():
            subdirs = [p for p in root.iterdir() if p.is_dir()]
            face_map = {}
            for d in subdirs:
                imgs = list_images_in_dir(d)
                if len(imgs) >= 2:
                    face_map[d.name] = imgs[:MAX_FACE_IMAGES]
            if face_map:
                return face_map

    face_map = {}
    for d in face_download.rglob("*"):
        if d.is_dir():
            imgs = list_images_in_dir(d)
            if len(imgs) >= 2:
                face_map[d.name] = imgs[:MAX_FACE_IMAGES]

    return face_map


# ============================================================
# PALM: use _train_data only
# every 3 JPGs = one palm pseudo-identity
# ============================================================

def get_palm_identity_map(palm_download: Path):
    candidate_roots = [
        palm_download / "_train_data",
        palm_download / "train_data",
        palm_download / "Train_data",
        palm_download,
    ]

    train_root = None
    for root in candidate_roots:
        if root.exists() and root.is_dir():
            jpgs = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"]
            if jpgs:
                train_root = root
                break

    if train_root is None:
        raise FileNotFoundError(
            f"Could not find _train_data with jpg files inside: {palm_download}"
        )

    palm_images = sorted(
        [p for p in train_root.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"],
        key=natural_key
    )

    groups = {}
    for i in range(0, len(palm_images), 3):
        chunk = palm_images[i:i + 3]
        if len(chunk) == 3:
            identity = f"palm_{(i // 3) + 1:04d}"
            groups[identity] = chunk[:MAX_PALM_IMAGES]

    print(f"Palm train folder used: {train_root}")
    print(f"Total palm jpg files found: {len(palm_images)}")
    print(f"Total palm pseudo-identities created: {len(groups)}")

    preview_keys = list(groups.keys())[:5]
    for key in preview_keys:
        print(f"{key}: {[p.name for p in groups[key]]}")

    return groups


# ============================================================
# FINGERPRINT: SOCOFing
# train = Real
# val = Altered/Altered-Easy
# compare same finger position to same finger position
# ============================================================

def normalize_finger_position(name: str) -> str:
    stem = Path(name).stem.lower()

    patterns = [
        ("left_thumb", r"left[_\s-]*thumb"),
        ("left_index", r"left[_\s-]*index"),
        ("left_middle", r"left[_\s-]*middle"),
        ("left_ring", r"left[_\s-]*ring"),
        ("left_little", r"left[_\s-]*(little|pinky)"),
        ("right_thumb", r"right[_\s-]*thumb"),
        ("right_index", r"right[_\s-]*index"),
        ("right_middle", r"right[_\s-]*middle"),
        ("right_ring", r"right[_\s-]*ring"),
        ("right_little", r"right[_\s-]*(little|pinky)"),
    ]

    for label, pattern in patterns:
        if re.search(pattern, stem):
            return label

    return stem


def extract_person_id(filename: str):
    stem = Path(filename).stem

    m = re.match(r"(\d+)__", stem)
    if m:
        return m.group(1)

    m = re.match(r"(\d+)", stem)
    if m:
        return m.group(1)

    return None


def build_fingerprint_map_from_folder(folder: Path):
    images = recursive_images(folder)
    raw_groups = {}

    for img in images:
        person_id = extract_person_id(img.name)
        if person_id is None:
            continue

        finger_pos = normalize_finger_position(img.name)

        raw_groups.setdefault(person_id, {})
        if finger_pos not in raw_groups[person_id]:
            raw_groups[person_id][finger_pos] = img

    return raw_groups


def get_fingerprint_identity_map(fp_download: Path):
    real_candidates = [
        fp_download / "Real",
        fp_download / "SOCOFing" / "Real",
    ]

    altered_candidates = [
        fp_download / "Altered" / "Altered-Easy",
        fp_download / "SOCOFing" / "Altered" / "Altered-Easy",
        fp_download / "Altered-Easy",
        fp_download / "SOCOFing" / "Altered-Easy",
    ]

    real_root = None
    altered_root = None

    for root in real_candidates:
        if root.exists():
            real_root = root
            break

    for root in altered_candidates:
        if root.exists():
            altered_root = root
            break

    if real_root is None:
        raise FileNotFoundError("Could not find SOCOFing Real folder")

    if altered_root is None:
        raise FileNotFoundError("Could not find SOCOFing Altered-Easy folder")

    print(f"Fingerprint REAL folder used: {real_root}")
    print(f"Fingerprint Altered-Easy folder used: {altered_root}")

    real_map = build_fingerprint_map_from_folder(real_root)
    altered_map = build_fingerprint_map_from_folder(altered_root)

    fp_map = {}

    for person_id, real_fingers in real_map.items():
        altered_fingers = altered_map.get(person_id, {})
        shared_positions = sorted(set(real_fingers.keys()) & set(altered_fingers.keys()))

        if len(shared_positions) >= 4:
            fp_map[person_id] = {
                "train": {pos: real_fingers[pos] for pos in shared_positions},
                "val": {pos: altered_fingers[pos] for pos in shared_positions},
            }

    print(f"Total fingerprint person identities found: {len(fp_map)}")
    preview_keys = list(sorted(fp_map.keys()))[:5]
    for key in preview_keys:
        print(f"{key}: shared fingers = {sorted(fp_map[key]['train'].keys())}")

    return fp_map


def copy_fingerprint_person_set(fp_entry: dict, sid: str, out_root: Path):
    train_dst = out_root / "fingerprint" / "train" / sid
    val_dst = out_root / "fingerprint" / "val" / sid

    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for finger_pos, src_path in sorted(fp_entry["train"].items()):
        ext = src_path.suffix.lower()
        dst_name = f"{finger_pos}{ext}"
        shutil.copy2(src_path, train_dst / dst_name)

        manifest_rows.append({
            "split": "train",
            "finger_position": finger_pos,
            "source_file": src_path.name,
            "stored_file": dst_name,
        })

    for finger_pos, src_path in sorted(fp_entry["val"].items()):
        ext = src_path.suffix.lower()
        dst_name = f"{finger_pos}{ext}"
        shutil.copy2(src_path, val_dst / dst_name)

        manifest_rows.append({
            "split": "val",
            "finger_position": finger_pos,
            "source_file": src_path.name,
            "stored_file": dst_name,
        })

    manifest_path = train_dst / "finger_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "finger_position", "source_file", "stored_file"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return len(fp_entry["train"]), len(fp_entry["val"])


# ============================================================
# BUILD SYNTHETIC DATASET
# ============================================================

def build_synthetic_dataset(face_map, palm_map, fp_map):
    ensure_output_dirs(OUT_ROOT)
    clear_existing_subject_dirs(OUT_ROOT)

    print(f"\nUsable face identities: {len(face_map)}")
    print(f"Usable palm identities: {len(palm_map)}")
    print(f"Usable fingerprint person identities: {len(fp_map)}")

    num_subjects = min(len(face_map), len(palm_map), len(fp_map), MAX_SUBJECTS)
    if num_subjects == 0:
        raise RuntimeError("No usable identities found. Check grouping logic and dataset layout.")

    face_keys = sorted(face_map.keys())
    palm_keys = sorted(palm_map.keys())
    fp_keys = sorted(fp_map.keys())

    random.shuffle(face_keys)
    random.shuffle(palm_keys)
    random.shuffle(fp_keys)

    face_keys = face_keys[:num_subjects]
    palm_keys = palm_keys[:num_subjects]
    fp_keys = fp_keys[:num_subjects]

    mapping_rows = []
    actual_count = 0

    for i in range(num_subjects):
        sid = f"s{i + 1:03d}"

        face_id = face_keys[i]
        palm_id = palm_keys[i]
        fp_id = fp_keys[i]

        face_imgs = face_map[face_id][:MAX_FACE_IMAGES]
        palm_imgs = palm_map[palm_id][:MAX_PALM_IMAGES]
        fp_entry = fp_map[fp_id]

        if len(face_imgs) < 2 or len(palm_imgs) < 3:
            continue

        if len(fp_entry["train"]) < 4 or len(fp_entry["val"]) < 4:
            continue

        face_train, face_val = split_images(face_imgs, TRAIN_RATIO)
        palm_train, palm_val = split_images(palm_imgs, TRAIN_RATIO)

        if not all([face_train, face_val, palm_train, palm_val]):
            continue

        copy_images(face_train, OUT_ROOT / "face" / "train" / sid)
        copy_images(face_val, OUT_ROOT / "face" / "val" / sid)

        copy_images(palm_train, OUT_ROOT / "palm" / "train" / sid)
        copy_images(palm_val, OUT_ROOT / "palm" / "val" / sid)

        fp_train_count, fp_val_count = copy_fingerprint_person_set(fp_entry, sid, OUT_ROOT)

        mapping_rows.append({
            "synthetic_subject_id": sid,
            "face_source_identity": face_id,
            "palm_source_identity": palm_id,
            "fingerprint_source_person_id": fp_id,
            "face_total_images": len(face_imgs),
            "palm_total_images": len(palm_imgs),
            "fingerprint_train_fingers": fp_train_count,
            "fingerprint_val_fingers": fp_val_count,
        })

        actual_count += 1
        print(
            f"Created {sid} | "
            f"face={face_id} ({len(face_imgs)}) | "
            f"palm={palm_id} ({len(palm_imgs)}) | "
            f"fingerprint_person={fp_id} (train={fp_train_count}, val={fp_val_count})"
        )

    mapping_path = OUT_ROOT / "synthetic_identity_mapping.csv"
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "synthetic_subject_id",
                "face_source_identity",
                "palm_source_identity",
                "fingerprint_source_person_id",
                "face_total_images",
                "palm_total_images",
                "fingerprint_train_fingers",
                "fingerprint_val_fingers",
            ],
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    print("\nDone.")
    print(f"Synthetic subjects created: {actual_count}")
    print(f"Dataset written to: {OUT_ROOT}")
    print(f"Mapping saved to: {mapping_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    face_download, palm_download, fp_download = download_datasets()

    print("\n=== FACE TREE ===")
    print_tree(face_download, max_depth=2)

    print("\n=== PALM TREE ===")
    print_tree(palm_download, max_depth=2)

    print("\n=== FINGERPRINT TREE ===")
    print_tree(fp_download, max_depth=3)

    face_map = get_face_identity_map(face_download)
    palm_map = get_palm_identity_map(palm_download)
    fp_map = get_fingerprint_identity_map(fp_download)

    build_synthetic_dataset(face_map, palm_map, fp_map)


if __name__ == "__main__":
    main()