from __future__ import annotations
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from utils.image_ops import read_image, enhance_fingerprint, binarize_and_skeletonize, resize_keep
from utils.metrics import cosine_score


class FingerprintCNN(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feats = base.fc.in_features
        base.fc = nn.Linear(in_feats, embedding_dim)
        self.backbone = base

    def forward(self, x):
        x = self.backbone(x)
        return nn.functional.normalize(x, p=2, dim=1)


class FingerprintRecognizer:
    def __init__(self, device: str = "cpu", embedding_dim: int = 256):
        self.device = torch.device(device)
        self.model = FingerprintCNN(embedding_dim=embedding_dim).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])

    def load_weights(self, path: str | Path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def extract_minutiae(self, image_path: str | Path) -> list[tuple[int, int, str]]:
        gray = read_image(image_path, grayscale=True)
        gray = enhance_fingerprint(gray)
        skel = binarize_and_skeletonize(gray)

        minutiae = []
        rows, cols = skel.shape
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if skel[y, x] == 1:
                    n = [
                        skel[y-1, x-1], skel[y-1, x], skel[y-1, x+1],
                        skel[y, x+1], skel[y+1, x+1], skel[y+1, x],
                        skel[y+1, x-1], skel[y, x-1], skel[y-1, x-1]
                    ]
                    crossing_number = sum(abs(int(n[i]) - int(n[i+1])) for i in range(8)) / 2
                    if crossing_number == 1:
                        minutiae.append((x, y, "ending"))
                    elif crossing_number == 3:
                        minutiae.append((x, y, "bifurcation"))
        return minutiae

    def minutiae_match_score(self, m1: list[tuple[int, int, str]], m2: list[tuple[int, int, str]], tolerance: int = 15) -> float:
        if not m1 or not m2:
            return 0.0
        matches = 0
        used = set()
        for x1, y1, t1 in m1:
            for idx, (x2, y2, t2) in enumerate(m2):
                if idx in used:
                    continue
                if t1 == t2 and abs(x1 - x2) <= tolerance and abs(y1 - y2) <= tolerance:
                    matches += 1
                    used.add(idx)
                    break
        return matches / max(len(m1), len(m2), 1)

    def extract_cnn_embedding(self, image_path: str | Path) -> np.ndarray:
        gray = read_image(image_path, grayscale=True)
        gray = enhance_fingerprint(gray)
        gray = resize_keep(gray, 224)
        gray_3 = np.stack([gray, gray, gray], axis=-1)
        x = self.transform(gray_3).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(x).cpu().numpy()[0]
        return emb.astype(np.float32)

    def compare(self, probe_path: str | Path, gallery_path: str | Path, cnn_weight: float = 0.5, minutiae_weight: float = 0.5):
        probe_minutiae = self.extract_minutiae(probe_path)
        gallery_minutiae = self.extract_minutiae(gallery_path)
        minutiae_score = self.minutiae_match_score(probe_minutiae, gallery_minutiae)

        probe_emb = self.extract_cnn_embedding(probe_path)
        gallery_emb = self.extract_cnn_embedding(gallery_path)
        cnn_score = cosine_score(probe_emb, gallery_emb)
        cnn_score = (cnn_score + 1.0) / 2.0

        fused = (cnn_weight * cnn_score) + (minutiae_weight * minutiae_score)
        return {
            "minutiae_score": float(minutiae_score),
            "cnn_score": float(cnn_score),
            "combined_score": float(fused),
            "probe_minutiae_count": len(probe_minutiae),
            "gallery_minutiae_count": len(gallery_minutiae),
        }
