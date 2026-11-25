"""Rebuild the leakage-free multimodal fusion checkpoint from saved features.

Run Part 2 through daily sentiment aggregation first. PCA and the classifier
are fitted on 2010-2017 only, and classifier regularization is selected on 2018.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def closed_period_mask(frame, start, end):
    return frame["date"].between(start, end) & frame["next_date"].between(start, end)


def main():
    required = [
        Path("results/market_basket_daily.csv"),
        Path("results/daily_sentiment.csv"),
        Path("results/chart_embeddings.npy"),
        Path("results/embedding_index.csv"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Part 1/2 outputs: {missing}")

    basket = pd.read_csv(required[0], parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    basket["ret"] = basket["close"].pct_change()
    for lag in range(1, 6):
        basket[f"ret_lag_{lag}"] = basket["ret"].shift(lag - 1).fillna(0.0)
    basket["next_close"] = basket["close"].shift(-1)
    basket["next_date"] = basket["date"].shift(-1)
    basket["target"] = (basket["next_close"] > basket["close"]).astype(int)
    basket = basket.dropna(subset=["next_close", "next_date"])

    daily_news = pd.read_csv(required[1], parse_dates=["Date"])
    basket = basket.merge(daily_news, left_on="date", right_on="Date", how="left")
    basket["has_news"] = basket["news_count"].notna().astype(int)
    basket[["prob_neg", "prob_pos", "news_count"]] = basket[
        ["prob_neg", "prob_pos", "news_count"]
    ].fillna(0.0)
    basket["prob_neu"] = basket["prob_neu"].fillna(1.0)

    embeddings = np.load(required[2])
    index = pd.read_csv(required[3], parse_dates=["date"])
    if len(embeddings) != len(index):
        raise ValueError("Embedding array and date index have different lengths")
    raw_chart_cols = [f"chart_embed_{i}" for i in range(embeddings.shape[1])]
    index[raw_chart_cols] = embeddings
    basket = basket.merge(index, on="date", how="left")

    train_mask = closed_period_mask(basket, "2010-01-01", "2017-12-31")
    val_mask = closed_period_mask(basket, "2018-01-01", "2018-12-31")
    pca = PCA(n_components=8, random_state=SEED)
    pca.fit(basket.loc[train_mask, raw_chart_cols].fillna(0.0).values)
    chart_cols = [f"chart_pca_{i}" for i in range(8)]
    basket[chart_cols] = pca.transform(basket[raw_chart_cols].fillna(0.0).values)

    price_cols = [f"ret_lag_{i}" for i in range(1, 6)]
    news_cols = ["prob_neg", "prob_neu", "prob_pos", "has_news", "news_count"]
    feature_cols = price_cols + news_cols + chart_cols
    X_train = basket.loc[train_mask, feature_cols].values
    y_train = basket.loc[train_mask, "target"].values
    X_val = basket.loc[val_mask, feature_cols].values
    y_val = basket.loc[val_mask, "target"].values

    best_c, best_auc = None, -np.inf
    for c_value in (0.01, 0.1, 1.0, 10.0):
        candidate = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=c_value, max_iter=2000, random_state=SEED)),
        ])
        candidate.fit(X_train, y_train)
        auc = roc_auc_score(y_val, candidate.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_c, best_auc = c_value, auc

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=best_c, max_iter=2000, random_state=SEED)),
    ])
    pipeline.fit(X_train, y_train)
    state = {
        "pipeline": pipeline,
        "pca": pca,
        "raw_chart_feature_names": raw_chart_cols,
        "feature_names": feature_cols,
        "training_period": ["2010-01-01", "2017-12-31"],
        "validation_selected_C": best_c,
        "validation_roc_auc": best_auc,
        "seed": SEED,
    }
    output = Path("checkpoints/fusion_best.pt")
    output.parent.mkdir(exist_ok=True)
    joblib.dump(state, output)
    restored = joblib.load(output)
    if not np.allclose(pipeline.predict_proba(X_val), restored["pipeline"].predict_proba(X_val)):
        raise RuntimeError("Reloaded checkpoint does not reproduce validation predictions")
    print(f"Saved and verified {output} with C={best_c} and validation ROC-AUC={best_auc:.4f}")


if __name__ == "__main__":
    main()
