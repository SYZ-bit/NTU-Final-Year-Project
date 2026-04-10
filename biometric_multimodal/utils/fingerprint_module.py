from __future__ import annotations
import cv2
import numpy as np
from utils.image_ops import read_gray, normalize_gray, clahe_gray


class FingerprintMatcher:
    """
    Practical OpenCV-based matcher inspired by the OpenCV fingerprint matching pipeline:
    contrast enhancement -> ridge emphasis -> keypoints/descriptors -> ratio test.
    """

    def __init__(self, max_keypoints: int = 500):
        self.max_keypoints = max_keypoints
        if hasattr(cv2, "SIFT_create"):
            self.detector = cv2.SIFT_create(nfeatures=max_keypoints)
            self.norm = cv2.NORM_L2
        else:
            self.detector = cv2.ORB_create(nfeatures=max_keypoints)
            self.norm = cv2.NORM_HAMMING
        self.matcher = cv2.BFMatcher(self.norm, crossCheck=False)

    def preprocess(self, path: str) -> np.ndarray:
        img = read_gray(path)
        img = normalize_gray(img)
        img = clahe_gray(img)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]], dtype=np.float32)
        img = cv2.filter2D(img, -1, kernel)
        return img

    def describe(self, path: str):
        img = self.preprocess(path)
        kps, des = self.detector.detectAndCompute(img, None)
        return kps, des

    def pair_score(self, path_a: str, path_b: str) -> float:
        _, des1 = self.describe(path_a)
        _, des2 = self.describe(path_b)
        if des1 is None or des2 is None:
            return 0.0
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
        denom = max(min(len(des1), len(des2)), 1)
        return float(len(good) / denom)
