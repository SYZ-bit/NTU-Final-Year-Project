from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import random

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images_by_subject(root: str | Path) -> Dict[str, List[str]]:
    root = Path(root)
    mapping: Dict[str, List[str]] = defaultdict(list)
    if not root.exists():
        return {}
    for subject_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for img in sorted(subject_dir.rglob("*")):
            if img.suffix.lower() in IMG_EXTS:
                mapping[subject_dir.name].append(str(img))
    return dict(mapping)


def make_verification_pairs(mapping: Dict[str, List[str]], max_pairs_per_subject: int = 10,
                            max_negatives: int = 1000, seed: int = 42) -> List[Tuple[str, str, int]]:
    rng = random.Random(seed)
    subjects = [k for k, v in mapping.items() if len(v) >= 2]
    pairs = []

    for sid in subjects:
        imgs = mapping[sid]
        local_pairs = 0
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pairs.append((imgs[i], imgs[j], 1))
                local_pairs += 1
                if local_pairs >= max_pairs_per_subject:
                    break
            if local_pairs >= max_pairs_per_subject:
                break

    subject_ids = list(mapping.keys())
    negatives = 0
    while negatives < max_negatives and len(subject_ids) >= 2:
        a, b = rng.sample(subject_ids, 2)
        if not mapping[a] or not mapping[b]:
            continue
        pairs.append((rng.choice(mapping[a]), rng.choice(mapping[b]), 0))
        negatives += 1

    rng.shuffle(pairs)
    return pairs


def intersect_subjects(*roots: str | Path, limit: int | None = None) -> Dict[str, Dict[str, List[str]]]:
    mappings = [list_images_by_subject(r) for r in roots]
    subject_sets = [set(m.keys()) for m in mappings]
    common = sorted(set.intersection(*subject_sets)) if subject_sets else []

    if common:
        if limit is not None:
            common = common[:limit]
        return {
            sid: {
                "face": mappings[0][sid],
                "fingerprint": mappings[1][sid],
                "palm": mappings[2][sid],
            }
            for sid in common
        }

    # fallback to sorted-order pseudo alignment if true overlap is absent
    subject_lists = [sorted(m.keys()) for m in mappings]
    min_count = min((len(s) for s in subject_lists), default=0)
    if limit is not None:
        min_count = min(min_count, limit)
    fused: Dict[str, Dict[str, List[str]]] = {}
    for idx in range(min_count):
        sid = f"s{idx:05d}"
        fused[sid] = {
            "face": mappings[0][subject_lists[0][idx]],
            "fingerprint": mappings[1][subject_lists[1][idx]],
            "palm": mappings[2][subject_lists[2][idx]],
        }
    return fused
