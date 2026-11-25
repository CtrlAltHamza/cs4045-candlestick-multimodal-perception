"""Run the missing Part 1 YOLO evaluations without retraining any model.

The script reconstructs the deterministic chart splits from the Part 1 notebook,
loads ``checkpoints/yolo_best.pt``, and exports class-wise, theme-wise, recent,
robustness, and efficiency evidence required by the course rubric. It never
writes a checkpoint.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import matplotlib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "part1_vision.ipynb"
OUTPUT_NAMES = (
    "detection_per_class.csv",
    "detection_theme_results.csv",
    "detection_robustness.csv",
    "detection_efficiency.csv",
    "test_metadata.csv",
    "recent_metadata.csv",
)


def main() -> None:
    os.chdir(ROOT)
    matplotlib.use("Agg")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__part1_evaluation__"}

    # Environment/data, pattern rules, chart renderer, and deterministic splits.
    for index in (1, 3, 10, 11, 14):
        source = "".join(notebook["cells"][index]["source"])
        exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)

    from ultralytics import YOLO

    namespace["YOLO"] = YOLO
    namespace["df_yolo_comp"] = pd.read_csv(ROOT / "results" / "detection_results.csv")
    namespace["best_run"] = "Run_C_Chart_Aug"
    namespace["dataset_yaml_path"] = str((ROOT / "yolo_dataset" / "dataset.yaml").resolve())

    evaluation_source = "".join(notebook["cells"][18]["source"])
    exec(compile(evaluation_source, f"{NOTEBOOK.name}:cell-18", "exec"), namespace)

    bundle_path = ROOT / "results" / "part1_checkpoint_evaluation.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in OUTPUT_NAMES:
            path = ROOT / "results" / name
            if not path.exists():
                raise FileNotFoundError(f"Expected evaluation output was not created: {path}")
            archive.write(path, f"results/{name}")
    print(f"Part 1 evaluation bundle: {bundle_path}")


if __name__ == "__main__":
    main()
