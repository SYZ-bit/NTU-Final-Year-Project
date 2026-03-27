from __future__ import annotations
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from utils.image_ops import read_image, preprocess_palm
from utils.metrics import cosine_score


class PalmCNN(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feats = base.fc.in_features
        base.fc = nn.Linear(in_feats, embedding_dim)
        self.backbone = base

    def forward(self, x):
        x = self.backbone(x)
        return nn.functional.normalize(x, p=2, dim=1)


class PalmRecognizer:
    def __init__(self, device: str = "cpu", embedding_dim: int = 256):
        self.device = torch.device(device)
        self.model = PalmCNN(embedding_dim=embedding_dim).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_weights(self, path: str | Path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def gabor_features(self, image_path: str | Path) -> np.ndarray:
        img = read_image(image_path, grayscale=True)
        img = preprocess_palm(img, 224)
        feats = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
            feats.extend([filtered.mean(), filtered.std()])
        feats = np.array(feats, dtype=np.float32)
        feats = feats / (np.linalg.norm(feats) + 1e-8)
        return feats

    def cnn_embedding(self, image_path: str | Path) -> np.ndarray:
        img = read_image(image_path, grayscale=False)
        img = preprocess_palm(img, 224)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(x).cpu().numpy()[0]
        return emb.astype(np.float32)

    def compare(self, probe_path: str | Path, gallery_path: str | Path, cnn_weight: float = 0.7, gabor_weight: float = 0.3):
        p_cnn = self.cnn_embedding(probe_path)
        g_cnn = self.cnn_embedding(gallery_path)
        cnn_score = (cosine_score(p_cnn, g_cnn) + 1.0) / 2.0

        p_g = self.gabor_features(probe_path)
        g_g = self.gabor_features(gallery_path)
        gabor_score = (cosine_score(p_g, g_g) + 1.0) / 2.0

        fused = (cnn_weight * cnn_score) + (gabor_weight * gabor_score)
        return {
            "cnn_score": float(cnn_score),
            "gabor_score": float(gabor_score),
            "combined_score": float(fused),
        }
