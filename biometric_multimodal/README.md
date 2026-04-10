# Multimodal Biometrics Authentication System

This project gives you a complete Python starter implementation for:

- face recognition using `face_recognition`
- fingerprint recognition using OpenCV keypoint matching
- palm recognition using Gabor texture + CNN embeddings
- multimodal fusion using weighted fusion and logistic-regression score fusion
- a Flask backend
- a PyQt6 desktop frontend
- metric reporting with accuracy, precision, recall, F1, FAR, and FRR

## Important evaluation note

The three public datasets you requested are not identity-aligned across modalities. That means the same real person in the face dataset is not the same real person in the fingerprint and palm datasets. Because of that, the fusion experiment in `scripts/train_all.py` uses pseudo-aligned subject ids created by sorted-order matching across datasets. This is acceptable for a system prototype and software demonstration, but it is not a real-world biometric fusion benchmark.

## Installation

```bash
python -m venv .venv
.venv\Scriptsctivate
pip install -r requirements.txt
```

## Prepare datasets

```bash
python scripts/prepare_datasets.py
```

## Train everything and generate comparison chart

```bash
python scripts/train_all.py
```

Artifacts are written to:

```text
artifacts/
├── gallery/
├── models/
└── reports/
```

Main outputs:
- `artifacts/reports/face_metrics.json`
- `artifacts/reports/fingerprint_metrics.json`
- `artifacts/reports/palm_metrics.json`
- `artifacts/reports/all_metrics.json`
- `artifacts/reports/comparison.png`

## Start backend and frontend

```bash
python backend/flask_api.py
python frontend/pyqt_app.py
```
