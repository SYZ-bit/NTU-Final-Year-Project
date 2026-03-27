from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from skimage.morphology import skeletonize


def read_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_keep(img: np.ndarray, size: int = 224) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32) / 255.0
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def align_face_simple(img: np.ndarray, size: int = 224) -> np.ndarray:
    # Placeholder for MTCNN / SCRFD alignment.
    return resize_keep(img, size)


def enhance_fingerprint(img_gray: np.ndarray) -> np.ndarray:
    img_gray = cv2.equalizeHist(img_gray)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    return img_gray


def binarize_and_skeletonize(img_gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = 255 - binary
    skel = skeletonize((binary > 0).astype(np.uint8)).astype(np.uint8)
    return skel


def crop_center_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y = (h - side) // 2
    x = (w - side) // 2
    return img[y:y+side, x:x+side]


def preprocess_palm(img: np.ndarray, size: int = 224) -> np.ndarray:
    img = crop_center_square(img)
    img = resize_keep(img, size)
    return img
