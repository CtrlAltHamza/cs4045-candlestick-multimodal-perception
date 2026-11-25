"""Generate the Part 2 ROC figure from saved test predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve


def main():
    predictions = pd.read_csv("results/market_test_predictions.csv")
    models = {
        "Price-only": "price_probability",
        "News day t": "news_day_t_probability",
        "News lagged": "news_lagged_probability",
        "Chart-only": "chart_probability",
        "Fusion": "fusion_probability",
    }
    colors = ["#4c78a8", "#f58518", "#b279a2", "#54a24b", "#e45756"]
    plt.figure(figsize=(7.2, 5.3))
    for (label, column), color in zip(models.items(), colors):
        fpr, tpr, _ = roc_curve(predictions["target"], predictions[column])
        plt.plot(fpr, tpr, linewidth=2, color=color, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Next-day direction ROC curves (2019-2020)")
    plt.grid(alpha=0.2)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    output = Path("report/figures/fusion_comparison_roc.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
