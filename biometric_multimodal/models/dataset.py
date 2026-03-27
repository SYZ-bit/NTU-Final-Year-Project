from __future__ import annotations
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SubjectFolderDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 224):
        self.root = Path(root)
        self.samples = []
        self.class_to_idx = {}
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for cls_name in classes:
            for img_path in (self.root / cls_name).glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
