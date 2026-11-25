"""Recompute Part 2 daily sentiment, leakage-free fusion metrics, and checkpoint.

This uses the saved best sentiment checkpoint. It does not retrain neural models.
"""

import html
import re
from collections import Counter
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
MAX_SEQ_LEN = 64


def clean_financial_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9\s.,!?:;$\-%]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True,
            batch_first=True, dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(self.embedding(x))
        return self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))


class CustomTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, max_len=64, embed_dim=128, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim // 2, 3),
        )

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        padding = x == 0
        encoded = self.transformer_encoder(
            self.dropout(self.token_embedding(x) + self.pos_embedding(positions)),
            src_key_padding_mask=padding,
        )
        mask = (~padding).unsqueeze(-1).float()
        pooled = (encoded * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
        return self.fc(pooled)


def build_vocabulary():
    frame = pd.read_csv("Data/dataset.csv")
    frame["cleaned_text"] = frame["text"].apply(clean_financial_text)
    frame = frame[frame["cleaned_text"].str.len() > 3].copy()
    conflicts = frame.groupby("cleaned_text")["sentiment"].nunique()
    frame = frame[~frame["cleaned_text"].isin(conflicts[conflicts > 1].index)]
    frame = frame.drop_duplicates("cleaned_text")
    counts = Counter()
    for text in frame["cleaned_text"]:
        counts.update(re.findall(r"\b\w+\b", text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in counts.items():
        if count >= 2:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab):
    tokens = re.findall(r"\b\w+\b", clean_financial_text(text))
    values = [vocab.get(token, 1) for token in tokens[:MAX_SEQ_LEN]]
    return values + [0] * (MAX_SEQ_LEN - len(values))


def infer_daily_sentiment():
    vocab = build_vocabulary()
    state = torch.load("checkpoints/sentiment_best.pt", map_location="cpu", weights_only=True)
    if any(key.startswith("transformer_encoder.") for key in state):
        model = CustomTransformerEncoder(len(vocab))
    else:
        model = BiLSTMModel(len(vocab))
    model.load_state_dict(state)
    model.eval()

    headlines = pd.read_csv("Data/sp500_headlines_2008_2024.csv")
    headlines["Date"] = pd.to_datetime(headlines["Date"])
    headlines["cleaned_title"] = headlines["Title"].apply(clean_financial_text)
    encoded = torch.tensor([encode(text, vocab) for text in headlines["cleaned_title"]], dtype=torch.long)
    probabilities = []
    started = perf_counter()
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(encoded), batch_size=256):
            probabilities.append(torch.softmax(model(batch), dim=1).numpy())
    probabilities = np.vstack(probabilities)
    headlines[["prob_neg", "prob_neu", "prob_pos"]] = probabilities
    daily = headlines.groupby("Date").agg(
        prob_neg=("prob_neg", "mean"), prob_neu=("prob_neu", "mean"),
        prob_pos=("prob_pos", "mean"), news_count=("Title", "count"),
    ).reset_index()
    daily.to_csv("results/daily_sentiment.csv", index=False)
    elapsed = perf_counter() - started
    print(f"Inferred {len(headlines):,} headlines in {elapsed:.1f}s using {type(model).__name__}")
    return daily, elapsed / len(headlines) * 1000


def closed_period_mask(frame, start, end):
    return frame["date"].between(start, end) & frame["next_date"].between(start, end)


def bootstrap_ci(y_true, predictions, probabilities, iterations=1000):
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(iterations):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) == 2:
            values.append([
                accuracy_score(y_true[idx], predictions[idx]),
                f1_score(y_true[idx], predictions[idx]),
                roc_auc_score(y_true[idx], probabilities[idx]),
            ])
    return np.percentile(values, [2.5, 97.5], axis=0).T


def paired_accuracy_ci(y_true, left, right, iterations=1000):
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(iterations):
        idx = rng.integers(0, len(y_true), len(y_true))
        values.append(
            accuracy_score(y_true[idx], left[idx]) - accuracy_score(y_true[idx], right[idx])
        )
    return np.percentile(values, [2.5, 97.5])


def rebuild_market_results(daily, sentiment_ms):
    basket = pd.read_csv("results/market_basket_daily.csv", parse_dates=["date"]).sort_values("date")
    basket["ret"] = basket["close"].pct_change()
    for lag in range(1, 6):
        basket[f"ret_lag_{lag}"] = basket["ret"].shift(lag - 1).fillna(0.0)
    basket["next_close"] = basket["close"].shift(-1)
    basket["next_date"] = basket["date"].shift(-1)
    basket["target"] = (basket["next_close"] > basket["close"]).astype(int)
    basket = basket.dropna(subset=["next_close", "next_date"])
    basket = basket.merge(daily, left_on="date", right_on="Date", how="left")
    basket["has_news"] = basket["news_count"].notna().astype(int)
    basket[["prob_neg", "prob_pos", "news_count"]] = basket[
        ["prob_neg", "prob_pos", "news_count"]
    ].fillna(0.0)
    basket["prob_neu"] = basket["prob_neu"].fillna(1.0)

    lagged = daily.copy()
    lagged["Date"] += pd.Timedelta(days=1)
    lagged = lagged.rename(columns={name: f"lag_{name}" for name in [
        "prob_neg", "prob_neu", "prob_pos", "news_count"
    ]})
    basket = basket.merge(lagged, left_on="date", right_on="Date", how="left")
    basket[["lag_prob_neg", "lag_prob_pos", "lag_news_count"]] = basket[
        ["lag_prob_neg", "lag_prob_pos", "lag_news_count"]
    ].fillna(0.0)
    basket["lag_prob_neu"] = basket["lag_prob_neu"].fillna(1.0)

    embeddings = np.load("results/chart_embeddings.npy")
    index = pd.read_csv("results/embedding_index.csv", parse_dates=["date"])
    raw_cols = [f"chart_embed_{i}" for i in range(embeddings.shape[1])]
    index = pd.concat([index, pd.DataFrame(embeddings, columns=raw_cols)], axis=1)
    basket = basket.merge(index, on="date", how="left")
    masks = {
        "train": closed_period_mask(basket, "2010-01-01", "2017-12-31"),
        "val": closed_period_mask(basket, "2018-01-01", "2018-12-31"),
        "test": closed_period_mask(basket, "2019-01-01", "2020-12-31"),
        "recent": closed_period_mask(basket, "2023-01-01", "2024-03-04"),
    }
    pca = PCA(n_components=8, random_state=SEED)
    pca.fit(basket.loc[masks["train"], raw_cols].fillna(0.0))
    chart_cols = [f"chart_pca_{i}" for i in range(8)]
    basket[chart_cols] = pca.transform(basket[raw_cols].fillna(0.0))

    feature_sets = {
        "Price-Only Baseline": [f"ret_lag_{i}" for i in range(1, 6)],
        "News-Only Model (Day t)": ["prob_neg", "prob_neu", "prob_pos", "has_news", "news_count"],
        "News-Only Lagged (Day t-1)": ["lag_prob_neg", "lag_prob_neu", "lag_prob_pos"],
        "Chart-Only Model (PCA 8)": chart_cols,
    }
    feature_sets["Multimodal Fusion (Price+News+Chart)"] = (
        feature_sets["Price-Only Baseline"]
        + feature_sets["News-Only Model (Day t)"] + chart_cols
    )
    y = {name: basket.loc[mask, "target"].to_numpy() for name, mask in masks.items()}
    fitted, rows = {}, []
    prediction_frame = pd.DataFrame({
        "date": basket.loc[masks["test"], "date"].to_numpy(),
        "target": y["test"],
    })
    for name, columns in feature_sets.items():
        X = {key: basket.loc[mask, columns].fillna(0.0).to_numpy() for key, mask in masks.items()}
        best_c, best_val_auc = None, -np.inf
        for c_value in (0.01, 0.1, 1.0, 10.0):
            candidate = Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=c_value, max_iter=2000, random_state=SEED)),
            ]).fit(X["train"], y["train"])
            score = roc_auc_score(y["val"], candidate.predict_proba(X["val"])[:, 1])
            if score > best_val_auc:
                best_c, best_val_auc = c_value, score
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=best_c, max_iter=2000, random_state=SEED)),
        ])
        started = perf_counter()
        model.fit(X["train"], y["train"])
        training_seconds = perf_counter() - started
        started = perf_counter()
        test_predictions = model.predict(X["test"])
        test_probabilities = model.predict_proba(X["test"])[:, 1]
        inference_ms = (perf_counter() - started) * 1000 / len(X["test"])
        recent_predictions = model.predict(X["recent"])
        recent_probabilities = model.predict_proba(X["recent"])[:, 1]
        ci = bootstrap_ci(y["test"], test_predictions, test_probabilities)
        fitted[name] = (model, test_predictions, test_probabilities)
        output_prefix = {
            "Price-Only Baseline": "price",
            "News-Only Model (Day t)": "news_day_t",
            "News-Only Lagged (Day t-1)": "news_lagged",
            "Chart-Only Model (PCA 8)": "chart",
            "Multimodal Fusion (Price+News+Chart)": "fusion",
        }[name]
        prediction_frame[f"{output_prefix}_prediction"] = test_predictions
        prediction_frame[f"{output_prefix}_probability"] = test_probabilities
        rows.append({
            "Model": name, "Validation ROC-AUC": best_val_auc, "Selected C": best_c,
            "Test Acc": accuracy_score(y["test"], test_predictions),
            "Test Acc 95% CI": f"[{ci[0,0]:.4f}, {ci[0,1]:.4f}]",
            "Test F1": f1_score(y["test"], test_predictions),
            "Test F1 95% CI": f"[{ci[1,0]:.4f}, {ci[1,1]:.4f}]",
            "Test ROC-AUC": roc_auc_score(y["test"], test_probabilities),
            "Test ROC-AUC 95% CI": f"[{ci[2,0]:.4f}, {ci[2,1]:.4f}]",
            "Recent Acc": accuracy_score(y["recent"], recent_predictions),
            "Recent F1": f1_score(y["recent"], recent_predictions),
            "Recent ROC-AUC": roc_auc_score(y["recent"], recent_probabilities),
            "Training Time (s)": training_seconds,
            "Inference (ms/day)": inference_ms,
        })

    majority = int(np.bincount(y["train"]).argmax())
    majority_predictions = np.full(len(y["test"]), majority)
    prediction_frame["majority_prediction"] = majority_predictions
    prediction_frame.to_csv("results/market_test_predictions.csv", index=False)
    price_predictions = fitted["Price-Only Baseline"][1]
    for row in rows:
        predictions = fitted[row["Model"]][1]
        row["Accuracy Delta vs Price 95% CI"] = str(
            np.round(paired_accuracy_ci(y["test"], predictions, price_predictions), 4).tolist()
        )
        row["Accuracy Delta vs Majority 95% CI"] = str(
            np.round(paired_accuracy_ci(y["test"], predictions, majority_predictions), 4).tolist()
        )
    rows.insert(0, {
        "Model": "Majority Class Reference",
        "Test Acc": accuracy_score(y["test"], majority_predictions),
        "Test F1": f1_score(y["test"], majority_predictions),
        "Test ROC-AUC": np.nan,
        "Recent Acc": accuracy_score(y["recent"], np.full(len(y["recent"]), majority)),
    })
    result = pd.DataFrame(rows)
    result.to_csv("results/market_results.csv", index=False)

    news_model = fitted["News-Only Model (Day t)"][0]
    news_columns = feature_sets["News-Only Model (Day t)"]
    test_news = basket.loc[masks["test"], news_columns].fillna(0.0).to_numpy()
    rng = np.random.default_rng(SEED)
    shuffled_auc = [
        roc_auc_score(y["test"], news_model.predict_proba(test_news[rng.permutation(len(test_news))])[:, 1])
        for _ in range(100)
    ]
    pd.DataFrame([{
        "Original News ROC-AUC": fitted["News-Only Model (Day t)"][2].shape
        and roc_auc_score(y["test"], fitted["News-Only Model (Day t)"][2]),
        "Shuffled ROC-AUC Mean": np.mean(shuffled_auc),
        "Shuffled ROC-AUC SD": np.std(shuffled_auc),
        "Iterations": 100,
    }]).to_csv("results/headline_shuffle_results.csv", index=False)

    fusion_name = "Multimodal Fusion (Price+News+Chart)"
    joblib.dump({
        "pipeline": fitted[fusion_name][0], "pca": pca,
        "raw_chart_feature_names": raw_cols,
        "feature_names": feature_sets[fusion_name],
        "training_period": ["2010-01-01", "2017-12-31"],
        "validation_selected_C": result.loc[result["Model"] == fusion_name, "Selected C"].iloc[0],
        "sentiment_inference_ms_per_headline": sentiment_ms, "seed": SEED,
    }, "checkpoints/fusion_best.pt")
    print(result.to_string(index=False))


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    daily_sentiment, sentiment_latency = infer_daily_sentiment()
    rebuild_market_results(daily_sentiment, sentiment_latency)
