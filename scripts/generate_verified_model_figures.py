"""Generate qualitative and confusion figures from saved model checkpoints."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from ultralytics import YOLO

from rebuild_part2_evaluation import (
    BiLSTMModel,
    CustomTransformerEncoder,
    build_vocabulary,
    clean_financial_text,
    encode,
)

SEED = 42
FIGURES = Path("report/figures")
YOLO_NAMES = [
    "Doji", "Hammer", "Shooting Star", "Marubozu",
    "Bullish Engulfing", "Bearish Engulfing", "Morning Star", "Evening Star",
]


def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])


def iou(left, right):
    top_left = np.maximum(left[:2], right[:2])
    bottom_right = np.minimum(left[2:], right[2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0))
    left_area = np.prod(np.maximum(left[2:] - left[:2], 0))
    right_area = np.prod(np.maximum(right[2:] - right[:2], 0))
    return intersection / max(left_area + right_area - intersection, 1e-12)


def load_ground_truth(index):
    rows = []
    with open(f"results/demo_label_{index}.txt", encoding="utf-8") as stream:
        for line in stream:
            values = line.split()
            if values:
                rows.append((int(values[0]), xywh_to_xyxy(np.array(values[1:], dtype=float))))
    return rows


def generate_yolo_figures():
    image_paths = [str(Path(f"results/demo_chart_{index}.png")) for index in range(10)]
    results = YOLO("checkpoints/yolo_best.pt").predict(
        image_paths, device="cpu", conf=0.25, verbose=False
    )
    rendered = [result.plot()[..., ::-1] for result in results]

    figure, axes = plt.subplots(2, 5, figsize=(18, 7.2))
    for index, axis in enumerate(axes.ravel()):
        axis.imshow(rendered[index])
        axis.set_title(f"Chart {index + 1}: {len(results[index].boxes)} detections")
        axis.axis("off")
    figure.suptitle("YOLOv8 checkpoint detections with measured confidence scores", fontsize=15)
    plt.tight_layout()
    plt.savefig(FIGURES / "yolo_test_detections.png", dpi=220, bbox_inches="tight")
    plt.close()

    matrix = np.zeros((9, 9), dtype=int)
    error_counts = []
    for index, result in enumerate(results):
        ground_truth = load_ground_truth(index)
        predictions = [
            (int(cls), box)
            for cls, box in zip(
                result.boxes.cls.cpu().numpy(),
                result.boxes.xyxyn.cpu().numpy(),
            )
        ]
        candidates = sorted(
            (
                (iou(gt_box, pred_box), gt_index, pred_index)
                for gt_index, (_, gt_box) in enumerate(ground_truth)
                for pred_index, (_, pred_box) in enumerate(predictions)
            ),
            reverse=True,
        )
        matched_gt, matched_pred = set(), set()
        errors = 0
        for overlap, gt_index, pred_index in candidates:
            if overlap < 0.5 or gt_index in matched_gt or pred_index in matched_pred:
                continue
            true_class = ground_truth[gt_index][0]
            predicted_class = predictions[pred_index][0]
            matrix[true_class, predicted_class] += 1
            errors += true_class != predicted_class
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
        for gt_index, (true_class, _) in enumerate(ground_truth):
            if gt_index not in matched_gt:
                matrix[true_class, 8] += 1
                errors += 1
        for pred_index, (predicted_class, _) in enumerate(predictions):
            if pred_index not in matched_pred:
                matrix[8, predicted_class] += 1
                errors += 1
        error_counts.append(errors)

    labels = YOLO_NAMES + ["Background"]
    plt.figure(figsize=(8.6, 7.2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("YOLO checkpoint confusion counts on reproducible chart artifacts")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES / "yolo_confusion_matrix.png", dpi=260)
    plt.close()

    failure_indices = np.argsort(error_counts)[::-1][:5]
    figure, axes = plt.subplots(1, 5, figsize=(18, 4.0))
    for axis, index in zip(axes, failure_indices):
        axis.imshow(rendered[index])
        axis.set_title(f"Chart {index + 1}\nmatching errors: {error_counts[index]}")
        axis.axis("off")
    figure.suptitle("Highest-error checkpoint examples under IoU 0.50 matching", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES / "yolo_failure_cases.png", dpi=220, bbox_inches="tight")
    plt.close()


def generate_sentiment_confusion():
    vocab = build_vocabulary()
    frame = pd.read_csv("Data/dataset.csv")
    frame["cleaned_text"] = frame["text"].apply(clean_financial_text)
    frame = frame[frame["cleaned_text"].str.len() > 3].copy()
    conflicts = frame.groupby("cleaned_text")["sentiment"].nunique()
    frame = frame[~frame["cleaned_text"].isin(conflicts[conflicts > 1].index)]
    frame = frame.drop_duplicates("cleaned_text")
    labels = frame["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2}).to_numpy()
    features = np.array([encode(text, vocab) for text in frame["cleaned_text"]])
    _, temporary_x, _, temporary_y = train_test_split(
        features, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    _, test_x, _, test_y = train_test_split(
        temporary_x, temporary_y, test_size=0.5, random_state=SEED, stratify=temporary_y
    )
    state = torch.load("checkpoints/sentiment_best.pt", map_location="cpu", weights_only=True)
    model = (
        CustomTransformerEncoder(len(vocab))
        if any(key.startswith("transformer_encoder.") for key in state)
        else BiLSTMModel(len(vocab))
    )
    model.load_state_dict(state)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(torch.tensor(test_x, dtype=torch.long)), batch_size=256):
            predictions.extend(model(batch).argmax(1).numpy())
    matrix = confusion_matrix(test_y, predictions, normalize="true")
    plt.figure(figsize=(6.2, 5.2))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="Greens",
        xticklabels=["Negative", "Neutral", "Positive"],
        yticklabels=["Negative", "Neutral", "Positive"],
    )
    plt.xlabel("Predicted sentiment")
    plt.ylabel("True sentiment")
    plt.title("Transformer normalized confusion matrix (test split)")
    plt.tight_layout()
    plt.savefig(FIGURES / "sentiment_confusion_matrix.png", dpi=260)
    plt.close()


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    generate_yolo_figures()
    generate_sentiment_confusion()
    print("Generated verified YOLO and sentiment checkpoint figures.")


if __name__ == "__main__":
    main()
