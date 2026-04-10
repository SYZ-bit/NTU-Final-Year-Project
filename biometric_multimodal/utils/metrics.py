from __future__ import annotations
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_binary_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def far_frr(y_true, scores, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / max((fp + tn), 1)
    frr = fn / max((fn + tp), 1)
    return {"FAR": float(far), "FRR": float(frr)}


def plot_metric_bars(metric_table: Dict[str, Dict[str, float]], save_path: str | Path) -> None:
    systems = list(metric_table.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(systems))
    width = 0.18

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        values = [metric_table[s].get(metric, 0.0) for s in systems]
        plt.bar(x + i * width, values, width=width, label=metric.title())

    plt.xticks(x + 1.5 * width, systems, rotation=15)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("Comparison of biometric systems")
    plt.legend()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
