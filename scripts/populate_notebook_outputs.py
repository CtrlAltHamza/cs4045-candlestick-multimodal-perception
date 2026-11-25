"""Validate notebook outputs without inventing execution results.

Historical versions of this helper inserted hand-authored output text. That
workflow is intentionally disabled. Execute notebooks in Jupyter or Kaggle so
outputs are produced by their code, then use this script as a quick audit.
"""

import json
from pathlib import Path


def main():
    failed = False
    for path in sorted(Path("notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        populated = sum(bool(cell.get("outputs")) for cell in code_cells)
        missing = len(code_cells) - populated
        print(f"{path}: {populated}/{len(code_cells)} code cells contain recorded outputs")
        if missing:
            failed = True
            print(f"  Execute the notebook to populate the remaining {missing} cells.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
