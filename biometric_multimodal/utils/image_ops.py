from __future__ import annotations

from pathlib import Path
from typing import Tuple
import cv2
import numpy as np


def read_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def read_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_keep(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def normalize_gray(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    out = np.empty_like(img)
    cv2.normalize(img, out, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8)


def center_crop(img: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    y0 = max((h - ch) // 2, 0)
    x0 = max((w - cw) // 2, 0)
    return img[y0:y0 + ch, x0:x0 + cw]


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def clahe_gray(img_gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)