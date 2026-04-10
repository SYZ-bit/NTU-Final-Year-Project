from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn

from utils.image_ops import read_bgr, clahe_gray, normalize_gray


def extract_palm_roi(path: str, out_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Extract a more stable palm ROI from a hand image by:
    1. grayscale conversion
    2. Otsu threshold segmentation
    3. largest contour extraction
    4. bounding-box crop with margin
    5. resize + contrast normalization
    """
    img = read_bgr(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fix inverted foreground/background if needed
    if np.mean(mask) > 127:
        mask = 255 - mask

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        roi = gray
    else:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        pad = int(0.08 * max(w, h))
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, gray.shape[1])
        y1 = min(y + h + pad, gray.shape[0])

        roi = gray[y0:y1, x0:x1]

    roi = cv2.resize(roi, out_size, interpolation=cv2.INTER_AREA)
    roi = clahe_gray(roi)
    roi = normalize_gray(roi)
    return roi.astype(np.uint8)


class PalmCNN(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        x = self.features(x).flatten(1)
        emb = self.embedding(x)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        if return_embedding:
            return emb
        return self.classifier(emb)


class PalmFeatureExtractor:
    def __init__(self, cnn: PalmCNN | None = None, device: str = "cpu", max_keypoints: int = 500):
        self.device = device
        self.cnn = cnn.to(device) if cnn is not None else None
        self.max_keypoints = max_keypoints

        if hasattr(cv2, "SIFT_create"):
            self.detector = cv2.SIFT_create(nfeatures=max_keypoints)
            self.norm = cv2.NORM_L2
        else:
            self.detector = cv2.ORB_create(nfeatures=max_keypoints)
            self.norm = cv2.NORM_HAMMING

        self.matcher = cv2.BFMatcher(self.norm, crossCheck=False)

    def sift_descriptors(self, path: str):
        gray = extract_palm_roi(path, out_size=(224, 224))
        kps, des = self.detector.detectAndCompute(gray, None)
        return kps, des

    def sift_score(self, path_a: str, path_b: str, ratio: float = 0.70) -> float:
        _, des1 = self.sift_descriptors(path_a)
        _, des2 = self.sift_descriptors(path_b)

        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return 0.0

        matches = self.matcher.knnMatch(des1, des2, k=2)

        good = 0
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < ratio * n.distance:
                good += 1

        denom = max(min(len(des1), len(des2)), 1)
        return float(good / denom)

    def cnn_embedding(self, path: str) -> np.ndarray:
        if self.cnn is None:
            raise ValueError("CNN model not provided.")

        self.cnn.eval()
        gray = extract_palm_roi(path, out_size=(224, 224))

        x = torch.from_numpy(gray).float() / 255.0
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.cnn(x, return_embedding=True).cpu().numpy()[0]

        return emb.astype(np.float32)

    @staticmethod
    def l2_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def pair_score(self, path_a: str, path_b: str) -> float:
        sift_sim = self.sift_score(path_a, path_b, ratio=0.70)

        if self.cnn is not None:
            ca = self.cnn_embedding(path_a)
            cb = self.cnn_embedding(path_b)
            cnn_sim = self.l2_similarity(ca, cb)

            # Make SIFT the dominant branch, CNN supportive
            return float(0.8 * sift_sim + 0.2 * cnn_sim)

        return float(sift_sim)