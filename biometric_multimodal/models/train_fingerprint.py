from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.dataset import SubjectFolderDataset
from utils.fingerprint_module import FingerprintCNN


def train_fingerprint_model(data_root: str, save_path: str, image_size: int = 224, batch_size: int = 16, epochs: int = 10, lr: float = 3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SubjectFolderDataset(data_root, image_size=image_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = FingerprintCNN(embedding_dim=256).to(device)
    clf = nn.Linear(256, len(ds.class_to_idx)).to(device)
    params = list(model.parameters()) + list(clf.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train(); clf.train()
    for epoch in range(epochs):
        running = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            if x.shape[1] == 3:
                x = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            emb = model(x)
            logits = clf(emb)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"[Fingerprint] epoch={epoch+1}/{epochs} loss={running/max(len(dl),1):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved fingerprint model to {save_path}")
