from __future__ import annotations
from models.dataset import list_images_by_subject
from utils.storage import save_json, ensure_dir


def build_single_sample_gallery():
    gallery = {}
    for modality, root in [("face", "data/face/train"), ("fingerprint", "data/fingerprint/train"), ("palm", "data/palm/train")]:
        mapping = list_images_by_subject(root)
        gallery[modality] = {sid: imgs[0] for sid, imgs in mapping.items() if imgs}

    ensure_dir("artifacts/gallery")
    save_json(gallery, "artifacts/gallery/gallery.json")
    return gallery


if __name__ == "__main__":
    print(build_single_sample_gallery())
