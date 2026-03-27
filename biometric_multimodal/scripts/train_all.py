from __future__ import annotations
from pathlib import Path
from app.config import load_settings
from models.train_face import train_face_model
from models.train_fingerprint import train_fingerprint_model
from models.train_palm import train_palm_model


def main():
    settings = load_settings("configs/settings.yaml")
    t = settings["training"]
    save_dir = Path(settings["paths"]["model_root"])
    save_dir.mkdir(parents=True, exist_ok=True)

    train_face_model(
        data_root="data/face/train",
        save_path=str(save_dir / "face_resnet50_embed.pt"),
        image_size=t["image_size"],
        batch_size=t["batch_size"],
        epochs=t["epochs"],
        lr=t["lr"],
    )
    train_fingerprint_model(
        data_root="data/fingerprint/train",
        save_path=str(save_dir / "fingerprint_resnet18_embed.pt"),
        image_size=t["image_size"],
        batch_size=t["batch_size"],
        epochs=t["epochs"],
        lr=t["lr"],
    )
    train_palm_model(
        data_root="data/palm/train",
        save_path=str(save_dir / "palm_resnet18_embed.pt"),
        image_size=t["image_size"],
        batch_size=t["batch_size"],
        epochs=t["epochs"],
        lr=t["lr"],
    )


if __name__ == "__main__":
    main()
