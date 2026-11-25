"""Finish theme/recent YOLO evaluation from already-rendered Kaggle charts.

This recovery runner deliberately does not import notebook cells, use custom
Ultralytics split names, regenerate charts, or rerun the full test/robustness
evaluations. It evaluates absolute image lists through the standard ``test``
key and bundles all Part 1 evidence produced by the earlier run.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASET = ROOT / "yolo_dataset"
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
BUNDLE_FILES = (
    "detection_per_class.csv",
    "detection_theme_results.csv",
    "detection_robustness.csv",
    "detection_efficiency.csv",
    "test_metadata.csv",
    "recent_metadata.csv",
)


def evaluate_image_list(model: YOLO, label: str, image_paths: list[Path]) -> float:
    if not image_paths:
        raise ValueError(f"No images found for {label}")
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} contains {len(missing)} missing images")

    list_path = RESULTS / f"{label}_images.txt"
    list_path.write_text(
        "\n".join(path.resolve().as_posix() for path in image_paths) + "\n",
        encoding="utf-8",
    )
    yaml_path = DATASET / f"{label}_absolute_dataset.yaml"
    names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"path: {DATASET.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"test: {list_path.resolve().as_posix()}\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )
    metrics = model.val(data=str(yaml_path), split="test", imgsz=640, device=0)
    return float(metrics.box.map50)


def main() -> None:
    checkpoint = ROOT / "checkpoints" / "yolo_best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    test_metadata = pd.read_csv(RESULTS / "test_metadata.csv")
    if "theme" not in test_metadata.columns:
        raise ValueError("test_metadata.csv has no theme column; use the regenerated Kaggle metadata")

    model = YOLO(str(checkpoint))
    rows: list[dict[str, float | str]] = []
    for theme in ("light", "dark", "blue-white"):
        paths = [ROOT / value for value in test_metadata.loc[test_metadata["theme"] == theme, "image_path"]]
        rows.append({"Evaluation": theme, "mAP@50": evaluate_image_list(model, theme, paths)})

    recent_paths = sorted((DATASET / "images" / "recent").glob("*.jpg"))
    rows.append({"Evaluation": "recent", "mAP@50": evaluate_image_list(model, "recent", recent_paths)})

    theme_results = pd.DataFrame(rows)
    theme_results.to_csv(RESULTS / "detection_theme_results.csv", index=False)
    print(theme_results.to_string(index=False))

    missing_outputs = [name for name in BUNDLE_FILES if not (RESULTS / name).is_file()]
    if missing_outputs:
        raise FileNotFoundError(f"Missing prior evaluation outputs: {missing_outputs}")

    bundle_path = RESULTS / "part1_checkpoint_evaluation.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in BUNDLE_FILES:
            archive.write(RESULTS / name, f"results/{name}")
    with zipfile.ZipFile(bundle_path) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(BUNDLE_FILES):
            raise RuntimeError("Part 1 evaluation bundle validation failed")
    print(f"Part 1 evaluation bundle: {bundle_path}")


if __name__ == "__main__":
    main()
