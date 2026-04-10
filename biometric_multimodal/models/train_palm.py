from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from models.dataset import list_images_by_subject, make_verification_pairs
from utils.palm_module import PalmCNN, PalmFeatureExtractor, extract_palm_roi
from utils.metrics import compute_binary_metrics, far_frr
from utils.storage import save_json, ensure_dir


class PalmClassificationDataset(Dataset):
    def __init__(self, root: str):
        mapping = list_images_by_subject(root)
        self.samples: List[Tuple[str, int]] = []
        self.subject_to_label = {sid: i for i, sid in enumerate(sorted(mapping))}

        for sid, imgs in mapping.items():
            for p in imgs:
                self.samples.append((p, self.subject_to_label[sid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        gray = extract_palm_roi(path, out_size=(224, 224))
        x = torch.from_numpy(gray).float() / 255.0
        x = x.unsqueeze(0)  # [1, H, W]
        return x, y


def train_palm_cnn(
    train_root: str = "data/palm/train",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3
):
    ds = PalmClassificationDataset(train_root)
    if len(ds) == 0:
        raise ValueError("Palm training dataset is empty.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = max(y for _, y in ds.samples) + 1

    val_size = max(1, int(0.2 * len(ds)))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PalmCNN(num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"Palm CNN epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()

            running += float(loss.item())

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())

        val_acc = correct / max(total, 1)
        print(f"Epoch {epoch+1}: loss={running / max(len(train_loader),1):.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    ensure_dir("artifacts/models")
    if best_state is not None:
        torch.save(best_state, "artifacts/models/palm_cnn.pt")
    else:
        torch.save(model.state_dict(), "artifacts/models/palm_cnn.pt")

    save_json(
        {"classes": n_classes, "best_val_acc": float(best_val_acc)},
        "artifacts/models/palm_cnn_meta.json"
    )
    return model


def _select_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidate_thresholds = np.linspace(0.05, 0.80, 16)

    best_thr = 0.25
    best_obj = -1.0

    for thr in candidate_thresholds:
        y_pred = (scores >= thr).astype(int)
        metrics = compute_binary_metrics(y_true, y_pred)
        err = far_frr(y_true, scores, thr)

        objective = metrics["accuracy"] - 0.5 * err["FAR"] - 0.25 * err["FRR"]
        if objective > best_obj:
            best_obj = objective
            best_thr = float(thr)

    return best_thr


def evaluate_palm_system(palm_root: str = "data/palm/train") -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mapping = list_images_by_subject(palm_root)
    if len(mapping) == 0:
        raise ValueError("Palm dataset is empty.")

    n_classes = len(mapping)
    model = PalmCNN(num_classes=max(n_classes, 1)).to(device)

    state_path = Path("artifacts/models/palm_cnn.pt")
    if state_path.exists():
        state = torch.load(state_path, map_location=device)
        if isinstance(state, dict):
            state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
        model.load_state_dict(state, strict=False)

    extractor = PalmFeatureExtractor(model, device=device)

    pairs = make_verification_pairs(mapping, max_pairs_per_subject=3, max_negatives=400)
    scores, y_true = [], []

    for a, b, label in tqdm(pairs, desc="Palm verification"):
        try:
            s = extractor.pair_score(a, b)
        except Exception:
            s = 0.0
        scores.append(s)
        y_true.append(label)

    scores = np.asarray(scores, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int32)

    train_idx, test_idx = train_test_split(
        np.arange(len(scores)),
        test_size=0.30,
        random_state=42,
        stratify=y_true,
    )

    train_scores = scores[train_idx]
    train_y = y_true[train_idx]
    test_scores = scores[test_idx]
    test_y = y_true[test_idx]

    threshold = _select_best_threshold(train_y, train_scores)

    y_pred = (test_scores >= threshold).astype(int)
    metrics = compute_binary_metrics(test_y, y_pred)
    metrics.update(far_frr(test_y, test_scores, threshold))

    result = {
        "threshold": float(threshold),
        "metrics": metrics,
        "n_pairs_total": int(len(scores)),
        "n_pairs_train": int(len(train_scores)),
        "n_pairs_test": int(len(test_scores)),
    }
    save_json(result, "artifacts/reports/palm_metrics.json")
    return result


if __name__ == "__main__":
    train_palm_cnn()
    print(evaluate_palm_system())