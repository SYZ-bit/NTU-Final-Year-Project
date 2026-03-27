from __future__ import annotations
import warnings
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from utils.image_ops import read_image, align_face_simple, bgr_to_rgb

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_OK = True
except Exception:
    INSIGHTFACE_OK = False


class ResNetEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_feats = base.fc.in_features
        base.fc = nn.Linear(in_feats, embedding_dim)
        self.model = base

    def forward(self, x):
        x = self.model(x)
        return nn.functional.normalize(x, p=2, dim=1)


class FaceRecognizer:
    def __init__(self, device: str = "cpu", embedding_dim: int = 256, use_insightface: bool = True):
        self.device = torch.device(device)
        self.use_insightface = use_insightface and INSIGHTFACE_OK
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.use_insightface:
            self.app = FaceAnalysis(name="buffalo_l")
            providers = ["CPUExecutionProvider"]
            self.app.prepare(ctx_id=0 if "cuda" in device else -1, providers=providers)
            self.model = None
        else:
            warnings.warn("InsightFace unavailable; falling back to ResNet50 embedding net.")
            self.app = None
            self.model = ResNetEmbeddingNet(embedding_dim=embedding_dim).to(self.device).eval()

    def load_weights(self, path: str | Path):
        if self.model is not None:
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()

    def extract(self, image_path: str | Path) -> np.ndarray:
        img = read_image(image_path, grayscale=False)
        img = align_face_simple(img)

        if self.use_insightface:
            rgb = bgr_to_rgb(img)
            faces = self.app.get(rgb)
            if not faces:
                raise ValueError(f"No face detected in {image_path}")
            emb = faces[0].embedding.astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            return emb

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(x).cpu().numpy()[0]
        return emb.astype(np.float32)
