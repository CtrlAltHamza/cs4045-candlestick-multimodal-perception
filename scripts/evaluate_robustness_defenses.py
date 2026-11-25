"""Select a YOLO robustness defense on corrupted validation data.

The supplied checkpoint is never modified. Candidate preprocessing/inference
pipelines are compared only on the 2018 validation charts after the same
480-pixel/JPEG-quality-40 corruption used by the required stress test. The
validation winner is then evaluated once on the corrupted 2019-2020 test set.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "yolo_dataset"
RESULTS = ROOT / "results"
CLASS_NAMES = (
    "Doji",
    "Hammer",
    "Shooting Star",
    "Marubozu",
    "Bullish Engulfing",
    "Bearish Engulfing",
    "Morning Star",
    "Evening Star",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    preprocessing: str
    inference_size: int


CANDIDATES = (
    Candidate("jpeg_q40_infer_480", "jpeg", 480),
    Candidate("jpeg_q40_infer_640", "jpeg", 640),
    Candidate("lanczos_restore_640", "lanczos", 640),
    Candidate("clahe_unsharp_restore_640", "clahe_unsharp", 640),
)


def restore_image(image, method: str):
    """Apply a deterministic defense after the mandatory JPEG corruption."""
    if method == "jpeg":
        return image
    if method == "lanczos":
        return cv2.resize(image, (640, 640), interpolation=cv2.INTER_LANCZOS4)
    if method == "clahe_unsharp":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = cv2.merge((clahe.apply(lightness), channel_a, channel_b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        enlarged = cv2.resize(enhanced, (640, 640), interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(enlarged, (0, 0), sigmaX=0.8)
        return cv2.addWeighted(enlarged, 1.35, blur, -0.35, 0)
    raise ValueError(f"Unknown preprocessing method: {method}")


def build_variant(split: str, candidate: Candidate) -> str:
    source_images = sorted((DATASET / "images" / split).glob("*.jpg"))
    if not source_images:
        raise FileNotFoundError(f"No source images found for split: {split}")

    variant = f"robust_{split}_{candidate.preprocessing}"
    image_dir = DATASET / "images" / variant
    label_dir = DATASET / "labels" / variant
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    expected_suffix = ".jpg" if candidate.preprocessing == "jpeg" else ".png"
    if len(list(image_dir.glob(f"*{expected_suffix}"))) == len(source_images):
        return variant

    for existing in image_dir.glob("*"):
        if existing.is_file():
            existing.unlink()
    for existing in label_dir.glob("*"):
        if existing.is_file():
            existing.unlink()

    for index, source in enumerate(source_images, start=1):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {source}")
        degraded = cv2.resize(image, (480, 480), interpolation=cv2.INTER_AREA)

        # Encode and decode once so every candidate begins with identical JPEG loss.
        ok, encoded = cv2.imencode(".jpg", degraded, [cv2.IMWRITE_JPEG_QUALITY, 40])
        if not ok:
            raise RuntimeError(f"JPEG encoding failed for {source}")
        degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        defended = restore_image(degraded, candidate.preprocessing)

        destination = image_dir / f"{source.stem}{expected_suffix}"
        if expected_suffix == ".jpg":
            # Preserve exactly one JPEG-quality-40 encode, matching the rubric
            # stress test rather than accidentally applying compression twice.
            destination.write_bytes(encoded.tobytes())
        else:
            cv2.imwrite(str(destination), defended, [cv2.IMWRITE_PNG_COMPRESSION, 2])

        source_label = DATASET / "labels" / split / f"{source.stem}.txt"
        if not source_label.is_file():
            raise FileNotFoundError(source_label)
        shutil.copy2(source_label, label_dir / source_label.name)
        if index % 250 == 0 or index == len(source_images):
            print(f"[{split}/{candidate.preprocessing}] {index}/{len(source_images)} images")
    return variant


def write_yaml(variant: str) -> Path:
    yaml_path = DATASET / f"{variant}.yaml"
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"path: {DATASET.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"test: images/{variant}\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def evaluate(model: YOLO, split: str, candidate: Candidate):
    variant = build_variant(split, candidate)
    metrics = model.val(
        data=str(write_yaml(variant)),
        split="test",
        imgsz=candidate.inference_size,
        batch=16,
        device=0,
        plots=False,
        project=str(ROOT / "runs" / "robustness_defense"),
        name=f"{split}_{candidate.name}",
    )
    return {
        "Candidate": candidate.name,
        "Preprocessing": candidate.preprocessing,
        "Inference Size": candidate.inference_size,
        "mAP@50": float(metrics.box.map50),
        "mAP@50-95": float(metrics.box.map),
        "Inference (ms/image)": float(metrics.speed.get("inference", float("nan"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/yolo_best.pt")
    args = parser.parse_args()

    checkpoint = (ROOT / args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = YOLO(str(checkpoint))

    validation_rows = [evaluate(model, "val", candidate) for candidate in CANDIDATES]
    validation = pd.DataFrame(validation_rows).sort_values(
        ["mAP@50", "mAP@50-95"], ascending=False
    ).reset_index(drop=True)
    validation["Selected on Validation"] = False
    validation.loc[0, "Selected on Validation"] = True
    validation.to_csv(RESULTS / "robustness_defense_validation.csv", index=False)
    print("\nValidation defense comparison:")
    print(validation.to_string(index=False))

    selected_name = str(validation.loc[0, "Candidate"])
    selected = next(candidate for candidate in CANDIDATES if candidate.name == selected_name)
    test_result = evaluate(model, "test", selected)

    previous = pd.read_csv(RESULTS / "detection_robustness.csv").iloc[0]
    original_map50 = float(previous["Original mAP@50"])
    unmitigated_map50 = float(previous["Robust mAP@50"])
    improved_map50 = float(test_result["mAP@50"])
    summary = pd.DataFrame([{
        "Selection Split": "2018 validation only",
        "Selected Defense": selected.name,
        "Original Test mAP@50": original_map50,
        "Unmitigated Robust Test mAP@50": unmitigated_map50,
        "Mitigated Robust Test mAP@50": improved_map50,
        "Absolute Improvement": improved_map50 - unmitigated_map50,
        "Remaining Drop vs Original": original_map50 - improved_map50,
        "Mitigated mAP@50-95": test_result["mAP@50-95"],
        "Inference (ms/image)": test_result["Inference (ms/image)"],
    }])
    summary.to_csv(RESULTS / "detection_robustness_improved.csv", index=False)
    print("\nHeld-out robust test result (evaluated once after validation selection):")
    print(summary.to_string(index=False))

    bundle = RESULTS / "robustness_defense_results.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("robustness_defense_validation.csv", "detection_robustness_improved.csv"):
            archive.write(RESULTS / name, f"results/{name}")
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Robustness defense result bundle failed CRC validation")
    print(f"Robustness defense bundle: {bundle}")


if __name__ == "__main__":
    main()
