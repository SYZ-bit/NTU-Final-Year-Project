# Multimodal Biometrics Authentication System

This project is an end-to-end Python template for a multimodal biometric authentication system with:

- **Face recognition**: ResNet embedding pipeline
- **Fingerprint recognition**: minutiae extraction + CNN embedding
- **Palm recognition**: CNN embedding + Gabor texture features
- **Fusion**: weighted score fusion + logistic-regression decision fusion
- **Frontend**: PyQt6 desktop UI
- **Backend**: Flask REST API
- **Image processing**: OpenCV

It is designed to be practical and extensible rather than a benchmark-ready production deployment.

## 1. Recommended public datasets

Use separate datasets for each modality.

- **Fingerprint**: FVC2004 / FVC2002 style datasets are widely used for fingerprint verification research. The FVC2004 site describes the competition and datasets; the full FVC datasets are distributed with the *Handbook of Fingerprint Recognition*. citeturn577236search0turn577236search4
- **Face**: LFW is a classic public face dataset for unconstrained face recognition evaluation. citeturn577236search2
- **Palm**: The IIT Delhi touchless palmprint database and the PolyU-IITD contactless palmprint database are public academic datasets; the PolyU-IITD terms page states it is for academic use only. citeturn577236search1turn577236search5turn577236search23
- **Face model stack**: InsightFace provides open-source ArcFace-style face recognition tooling and models. citeturn577236search3turn577236search6

## 2. Project structure

```text
biometric_multimodal/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   └── service.py
├── backend/
│   └── flask_api.py
├── frontend/
│   └── pyqt_app.py
├── models/
│   ├── dataset.py
│   ├── train_face.py
│   ├── train_fingerprint.py
│   ├── train_palm.py
│   ├── fusion.py
│   └── inference.py
├── utils/
│   ├── image_ops.py
│   ├── face_module.py
│   ├── fingerprint_module.py
│   ├── palm_module.py
│   ├── metrics.py
│   └── storage.py
├── configs/
│   └── settings.yaml
├── scripts/
│   ├── build_gallery.py
│   └── train_all.py
├── requirements.txt
└── README.md
```

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

## 4. Dataset layout expected by the code

Each modality is stored separately. The same subject id should be reused across modalities where possible.

```text
data/
├── face/
│   ├── train/
│   │   ├── s001/xxx1.jpg
│   │   ├── s001/xxx2.jpg
│   │   └── s002/...
│   └── val/
├── fingerprint/
│   ├── train/
│   │   ├── s001/fp1.png
│   │   ├── s001/fp2.png
│   │   └── s002/...
│   └── val/
└── palm/
    ├── train/
    │   ├── s001/p1.jpg
    │   ├── s001/p2.jpg
    │   └── s002/...
    └── val/
```

## 5. Workflow

1. Train or load the face, fingerprint, and palm embedding models.
2. Build a **gallery** from enrollment samples.
3. For each verification request:
   - preprocess each biometric image
   - extract features/scores from each modality
   - compare against gallery template
   - fuse scores using weighted sum and logistic regression
   - output accept/reject + confidence
4. Use the Flask backend as the model-serving layer.
5. Use the PyQt6 frontend to select images and call the backend.

## 6. Run

### Train models

```bash
python scripts/train_all.py
```

### Build gallery

```bash
python scripts/build_gallery.py
```

### Start backend

```bash
python backend/flask_api.py
```

### Start frontend

```bash
python frontend/pyqt_app.py
```

## 7. Notes

- This code uses a practical hybrid approach: classical features + deep embeddings.
- The fingerprint minutiae extractor is a lightweight academic implementation, not a vendor-grade matcher.
- For face embeddings, the code supports either **InsightFace/ArcFace** or a fallback ResNet-based embedding model.
- You will still need to download datasets and, if desired, pretrained checkpoints separately.
