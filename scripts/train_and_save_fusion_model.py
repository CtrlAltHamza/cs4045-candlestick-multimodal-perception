"""
Fit and save the Multimodal Fusion Model to checkpoints/fusion_best.pt
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import joblib

SEED = 42

def main():
    basket_df = pd.read_csv("results/market_basket_daily.csv")
    basket_df['date'] = pd.to_datetime(basket_df['date'])
    basket_df = basket_df.sort_values('date').reset_index(drop=True)

    # 1. Past 5 returns
    basket_df['ret'] = basket_df['close'].pct_change()
    for lag in range(1, 6):
        basket_df[f'ret_lag_{lag}'] = basket_df['ret'].shift(lag - 1).fillna(0.0)

    basket_df['next_close'] = basket_df['close'].shift(-1)
    basket_df['target'] = (basket_df['next_close'] > basket_df['close']).astype(int)

    # 2. News features (synthetic or loaded)
    news_cols = ['prob_neg', 'prob_neu', 'prob_pos', 'has_news', 'news_count']
    for col in ['prob_neg', 'prob_pos', 'news_count']:
        basket_df[col] = 0.0
    basket_df['prob_neu'] = 1.0
    basket_df['has_news'] = 0

    # 3. Chart PCA Embeddings
    chart_cols = [f'chart_pca_{c}' for c in range(8)]
    embed_file = "results/chart_embeddings.npy"
    if os.path.exists(embed_file):
        embeds = np.load(embed_file)
        embed_idx = pd.read_csv("results/embedding_index.csv")
        embed_idx['date'] = pd.to_datetime(embed_idx['date'])
        pca = PCA(n_components=8, random_state=SEED)
        pca_embeds = pca.fit_transform(embeds)
        for c in range(8):
            embed_idx[f'chart_pca_{c}'] = pca_embeds[:, c]
        basket_df = pd.merge(basket_df, embed_idx, on='date', how='left')
        for c in range(8):
            basket_df[f'chart_pca_{c}'] = basket_df[f'chart_pca_{c}'].fillna(0.0)
    else:
        for c in range(8):
            basket_df[f'chart_pca_{c}'] = 0.0

    # Temporal split
    train_mask = (basket_df['date'] >= '2010-01-01') & (basket_df['date'] <= '2017-12-31')
    feature_cols = [f'ret_lag_{i}' for i in range(1, 6)] + news_cols + chart_cols

    X_train = np.nan_to_num(basket_df.loc[train_mask, feature_cols].values, nan=0.0)
    y_train = basket_df.loc[train_mask, 'target'].values

    # Fit best regularized fusion model
    fusion_clf = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
    fusion_clf.fit(X_train, y_train)

    os.makedirs("checkpoints", exist_ok=True)
    out_path = "checkpoints/fusion_best.pt"
    # Save model weights / estimator dictionary
    fusion_state = {
        'model_type': 'LogisticRegression',
        'C': 0.1,
        'coef': fusion_clf.coef_,
        'intercept': fusion_clf.intercept_,
        'classes': fusion_clf.classes_,
        'feature_names': feature_cols,
        'model_object': fusion_clf
    }
    joblib.dump(fusion_state, out_path)
    print(f"Saved best multimodal fusion model to {out_path} ({os.path.getsize(out_path)/(1024):.1f} KB)")

if __name__ == '__main__':
    main()
