"""Run only the corrected Transformer hyperparameter ablation from Part 2.

This intentionally skips the main-model training cell and never writes
``checkpoints/sentiment_best.pt``.  It reconstructs the deterministic data split,
loads the architecture/training definitions, obtains the saved baseline validation
score from ``results/sentiment_results.csv``, and executes the ablation cell.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "part2_nlp_fusion.ipynb"


def main() -> None:
    os.chdir(ROOT)
    matplotlib.use("Agg")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__ablation__"}

    # Setup, cleaned dataset, vocabulary, split, architectures, and trainer.
    for index in (1, 3, 4, 5, 8, 9):
        source = "".join(notebook["cells"][index]["source"])
        exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)

    results = pd.read_csv(ROOT / "results" / "sentiment_results.csv")
    transformer = results.loc[results["Model"] == "Custom Transformer"]
    if len(transformer) != 1:
        raise ValueError("Expected exactly one Custom Transformer baseline row")
    namespace["trans_val_f1"] = float(transformer.iloc[0]["Val Macro-F1"])

    ablation_source = "".join(notebook["cells"][11]["source"])
    exec(compile(ablation_source, f"{NOTEBOOK.name}:cell-11", "exec"), namespace)


if __name__ == "__main__":
    main()
