"""Regenerate report summaries using measured result files only.

Per-epoch curves and qualitative detections belong to the notebooks that
produce the underlying training logs and predictions. This helper deliberately
does not synthesize missing histories, confidence scores, or confusion data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

FIGURES = Path("report/figures")


def generate_yolo_summary():
    frame = pd.read_csv("results/detection_results.csv")
    metrics = ["Precision", "Recall", "mAP@50", "mAP@50-95"]
    x = np.arange(len(frame))
    width = 0.19
    plt.figure(figsize=(8.0, 4.8))
    for index, metric in enumerate(metrics):
        plt.bar(x + (index - 1.5) * width, frame[metric], width, label=metric)
    plt.xticks(x, frame["Run"].str.replace("_", " "), rotation=8)
    plt.ylim(0, 0.8)
    plt.ylabel("Score")
    plt.title("YOLOv8 test metrics by augmentation run")
    plt.legend(ncol=2)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES / "yolo_loss_curves.png", dpi=300)
    plt.close()


def generate_sentiment_summary():
    frame = pd.read_csv("results/sentiment_results.csv")
    neural = frame[frame["Model"] != "Majority Class Baseline"].copy()
    metrics = ["Val Macro-F1", "Test Accuracy", "Test Macro-F1"]
    x = np.arange(len(neural))
    width = 0.25
    plt.figure(figsize=(8.0, 4.8))
    for index, metric in enumerate(metrics):
        plt.bar(x + (index - 1) * width, neural[metric], width, label=metric)
    plt.xticks(x, neural["Model"])
    plt.ylim(0, 0.75)
    plt.ylabel("Score")
    plt.title("Sentiment model validation and test metrics")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES / "sentiment_training_curves.png", dpi=300)
    plt.close()


def generate_fusion_roc():
    frame = pd.read_csv("results/market_test_predictions.csv")
    models = {
        "Price-only": "price_probability",
        "News day t": "news_day_t_probability",
        "News lagged": "news_lagged_probability",
        "Chart-only": "chart_probability",
        "Fusion": "fusion_probability",
    }
    plt.figure(figsize=(7.2, 5.3))
    for label, column in models.items():
        fpr, tpr, _ = roc_curve(frame["target"], frame[column])
        plt.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Next-day direction ROC curves (2019-2020)")
    plt.grid(alpha=0.2)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES / "fusion_comparison_roc.png", dpi=300)
    plt.close()


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    generate_yolo_summary()
    generate_sentiment_summary()
    generate_fusion_roc()
    print("Generated measured YOLO, sentiment, and fusion summaries.")
    print("Run the notebooks to regenerate training histories and qualitative model outputs.")


if __name__ == "__main__":
    main()
