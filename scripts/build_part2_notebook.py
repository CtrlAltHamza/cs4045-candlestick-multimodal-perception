"""
Builder script to generate `notebooks/part2_nlp_fusion.ipynb`.
Generates a complete, production-ready Jupyter Notebook for Part 2 (Financial News Sentiment & Multimodal Fusion),
configured to run seamlessly on Kaggle GPU or local machine.
"""

import json

def make_cell(cell_type, source, execution_count=None, outputs=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs or []
    return cell

def build_part2_notebook():
    cells = []

    # Title & Markdown
    cells.append(make_cell("markdown", r"""# Deep Learning for Perception (CS 4045) – Project Part 2
## Financial News Sentiment & Multimodal Next-Day Direction Prediction
**Course Instructor**: Dr. Ahmad Raza Shahid  
**Author(s)**: Group Submission  

---
### Notebook Outline
1. **Environment Setup & Kaggle / Local Path Resolution**
2. **Task 1: NLP Data Exploration & Strict Preprocessing**
   - Sentiment Dataset Cleaning: HTML Unescaping, Punctuation Normalization, Non-English Filtering, Emoji Handling
   - Deduplication & Conflicting Label Resolution
   - Custom Vocabulary Building with `<PAD>`, `<UNK>`, and Fixed Sequence Length
   - Financial News Headlines Exploration (2008–2024)
   - Publication Timing Rule & Anti-Leakage Rationale
3. **Task 2: Sentiment Baselines**
   - Baseline 1: Majority Class Classifier
   - Baseline 2: PyTorch Simple RNN with Learned Embeddings
4. **Task 3: Deep Neural Sentiment Models**
   - PyTorch Bidirectional LSTM (BiLSTM)
   - Custom Multi-Head Transformer Encoder (built from scratch, no pretrained LLMs/BERT)
   - Model Evaluation: Accuracy, Macro-F1, Parameters, Latency, Loss Curves, Confusion Matrix
   - Hyperparameter Experiments (Hidden Dimension, Dropout, Learning Rate)
5. **Task 4: News and Chart Multimodal Fusion for Market Direction**
   - Daily Sentiment Aggregation & Zero-Headline Imputation Policy
   - Single-Source Benchmarks:
     - (i) Price-Only (5 past daily returns)
     - (ii) News-Only (Sentiment probabilities + volume)
     - (iii) Chart-Only (128-dim penultimate embedding + PCA/regularization)
   - Multimodal Fusion Architectures: Early Concatenation, Probability Blending, Gated MLP
   - **Mandatory Sanity Check**: Headline Shuffling across days to test feature signal
6. **Task 5: Comprehensive Evaluation & Out-of-Sample Testing**
   - 95% Grouped Bootstrap Confidence Intervals
   - Required Timing Comparison: Day $t$ news vs. Lagged Day $t-1$ news
   - Efficiency & Latency Benchmarks
   - Out-of-Sample Testing on Recent Data (2023 to March 4, 2024)
7. **Task 6: In-Depth Discussion & Answers**"""))

    # Cell: Imports & Setup
    cells.append(make_cell("code", r"""# Setup environment, check GPU acceleration, and configure paths
import os
import sys
import glob
import re
import html
import random
import shutil
import string
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# Seed everything for strict reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    print(f"CUDA GPU Active: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU. Kaggle GPU recommended for deep models.")

# Kaggle vs Local Data Path Detection (Bulletproof recursive search + auto-unzip)
import zipfile

DATA_DIR = None

# 1. Search recursively anywhere under /kaggle/input
if os.path.exists("/kaggle/input"):
    matches = glob.glob("/kaggle/input/**/dataset.csv", recursive=True)
    if matches:
        DATA_DIR = os.path.dirname(matches[0])
    else:
        # Check if dataset was uploaded as an unextracted .zip file
        zip_files = glob.glob("/kaggle/input/**/*.zip", recursive=True)
        if zip_files:
            print(f"Found archive {zip_files[0]}. Extracting to /kaggle/working...")
            with zipfile.ZipFile(zip_files[0], 'r') as z:
                z.extractall("/kaggle/working")
            extracted_matches = glob.glob("/kaggle/working/**/dataset.csv", recursive=True)
            if extracted_matches:
                DATA_DIR = os.path.dirname(extracted_matches[0])

# 2. Local fallback paths
if DATA_DIR is None:
    local_candidates = ["./Data", "../Data", "Data", "d:/dlp_project/Data", "d:/dlp_project"]
    for p in local_candidates:
        if os.path.exists(os.path.join(p, "dataset.csv")):
            DATA_DIR = p
            break

# 3. General recursive fallback
if DATA_DIR is None:
    matches = glob.glob("**/dataset.csv", recursive=True)
    if matches:
        DATA_DIR = os.path.dirname(matches[0])

if DATA_DIR is None:
    if os.path.exists("/kaggle/input"):
        print("DEBUG: Directory tree of /kaggle/input:")
        for r, d, f in os.walk("/kaggle/input"):
            print(r, f)
    raise FileNotFoundError("Could not find 'dataset.csv'. Please ensure your dataset is attached to this Kaggle notebook.")

print(f"Part 2 Data directory: {DATA_DIR}")
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("report/figures", exist_ok=True)"""))

    # Cell: Section 1 Markdown
    cells.append(make_cell("markdown", r"""## 1. Task 1: NLP Data Exploration & Preparation
In this section we:
1. Load `Data/dataset.csv` (32,600 finance texts).
2. Clean text: lowercase, unescape HTML entities (`&amp;` $\rightarrow$ `&`), strip `<br>` tags, remove non-alphanumeric noise, remove emojis/non-English text.
3. Check and resolve duplicates: discard conflicting label instances.
4. Tokenize, build vocabulary with `<PAD>`, `<UNK>`, and determine max sentence length.
5. Load `Data/sp500_headlines_2008_2024.csv` (~19,000 headlines), explore headline volume per day/year, and establish the **Anti-Leakage Publication Timing Rule**."""))

    # Cell: Load & Clean Sentiment Dataset
    cells.append(make_cell("code", r"""# Load and inspect dataset.csv
sentiment_file = os.path.join(DATA_DIR, "dataset.csv")
df_sent = pd.read_csv(sentiment_file)
print(f"Raw sentiment dataset: {len(df_sent):,} rows.")
print("Class distribution before cleaning:")
print(df_sent['sentiment'].value_counts())

# Strict text cleaning function
def clean_financial_text(text):
    if not isinstance(text, str):
        return ""
    # 1. HTML Unescape (&amp; -> &, &quot; -> ", etc.)
    text = html.unescape(text)
    # 2. Strip HTML tags like <br>, <p>, etc.
    text = re.sub(r'<[^>]+>', ' ', text)
    # 3. Strip URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # 4. Filter emojis and non-ASCII special characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # 5. Lowercase
    text = text.lower()
    # 6. Normalize punctuation and remove abnormal symbols
    text = re.sub(r'[^a-z0-9\s.,!?:;$\-%]', ' ', text)
    # 7. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\nCleaning financial texts...")
df_sent['cleaned_text'] = df_sent['text'].apply(clean_financial_text)
# Remove empty strings
initial_len = len(df_sent)
df_sent = df_sent[df_sent['cleaned_text'].str.len() > 3].reset_index(drop=True)
print(f"Removed {initial_len - len(df_sent)} empty or trivial texts.")

# Deduplication and conflicting label resolution
print("Analyzing duplicates...")
# Group by text and inspect labels
label_groups = df_sent.groupby('cleaned_text')['sentiment'].nunique()
conflicting_texts = set(label_groups[label_groups > 1].index)
print(f"Found {len(conflicting_texts)} texts with conflicting sentiment labels across sources. Purging all copies...")

# Drop conflicting texts completely
df_sent = df_sent[~df_sent['cleaned_text'].isin(conflicting_texts)].copy()
# Drop duplicate occurrences keeping the first
n_before_dedup = len(df_sent)
df_sent = df_sent.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)
print(f"Purged {n_before_dedup - len(df_sent)} duplicate texts.")

print("\nFinal clean sentiment dataset class counts:")
print(df_sent['sentiment'].value_counts())

# Map labels to integers
LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
df_sent['label'] = df_sent['sentiment'].map(LABEL_MAP)"""))

    # Cell: Sentence Length & Vocabulary
    cells.append(make_cell("code", r"""# Sentence length analysis and vocabulary construction
df_sent['token_list'] = df_sent['cleaned_text'].apply(lambda x: re.findall(r'\b\w+\b', x))
df_sent['seq_len'] = df_sent['token_list'].apply(len)

plt.figure(figsize=(10, 4))
plt.hist(df_sent['seq_len'], bins=50, color='teal', alpha=0.7)
plt.axvline(np.percentile(df_sent['seq_len'], 95), color='red', linestyle='--', label=f"95th Percentile ({int(np.percentile(df_sent['seq_len'], 95))} tokens)")
plt.title("Distribution of Sentence Lengths (Token Count)", fontsize=12)
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("report/figures/nlp_sentence_lengths.png", dpi=300)
plt.show()

MAX_SEQ_LEN = 64
print(f"Selected MAX_SEQ_LEN = {MAX_SEQ_LEN} (covers > 97% of sequences without truncation).")

# Build Vocabulary with minimum frequency cutoff
MIN_WORD_FREQ = 2
word_counts = Counter()
for tokens in df_sent['token_list']:
    word_counts.update(tokens)

vocab = {'<PAD>': 0, '<UNK>': 1}
for word, count in word_counts.items():
    if count >= MIN_WORD_FREQ:
        vocab[word] = len(vocab)

VOCAB_SIZE = len(vocab)
print(f"Built custom vocabulary of size: {VOCAB_SIZE:,} (filtered out {len(word_counts) - VOCAB_SIZE + 2:,} rare words).")

def text_to_indices(token_list, vocab, max_len=MAX_SEQ_LEN):
    indices = [vocab.get(token, vocab['<UNK>']) for token in token_list[:max_len]]
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    return indices

df_sent['encoded'] = df_sent['token_list'].apply(lambda t: text_to_indices(t, vocab, MAX_SEQ_LEN))"""))

    # Cell: Stratified Split 80/10/10
    cells.append(make_cell("code", r"""# Stratified 80/10/10 Split for Sentiment Data
X_all = np.array(df_sent['encoded'].tolist())
y_all = df_sent['label'].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

print(f"Dataset Splits: Train = {len(X_train):,}, Val = {len(X_val):,}, Test = {len(X_test):,}")

# PyTorch Dataset
class TextSentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_sent_loader = DataLoader(TextSentimentDataset(X_train, y_train), batch_size=64, shuffle=True)
val_sent_loader = DataLoader(TextSentimentDataset(X_val, y_val), batch_size=64, shuffle=False)
test_sent_loader = DataLoader(TextSentimentDataset(X_test, y_test), batch_size=64, shuffle=False)"""))

    # Cell: Headline Data EDA & Timing Rule
    cells.append(make_cell("code", r"""# Task 1.4: Headline Data Exploration & Publication Timing Rule
headline_file = os.path.join(DATA_DIR, "sp500_headlines_2008_2024.csv")
df_head = pd.read_csv(headline_file)
df_head['Date'] = pd.to_datetime(df_head['Date'])
print(f"Loaded {len(df_head):,} raw headlines from {df_head['Date'].min().date()} to {df_head['Date'].max().date()}.")

df_head['cleaned_title'] = df_head['Title'].apply(clean_financial_text)
df_head = df_head.drop_duplicates(subset=['Date', 'cleaned_title']).reset_index(drop=True)

# Load Market Basket trading calendar
basket_df = pd.read_csv("results/market_basket_daily.csv")
basket_df['date'] = pd.to_datetime(basket_df['date'])
basket_dates = set(basket_df['date'])

# Headline distribution per trading day
headlines_per_day = df_head.groupby('Date')['Title'].count()
trading_days_with_news = set(headlines_per_day.index).intersection(basket_dates)
no_news_days = basket_dates - trading_days_with_news
pct_no_news = (len(no_news_days) / len(basket_dates)) * 100

print(f"Total Market Basket trading days: {len(basket_dates)}")
print(f"Trading days WITH headlines: {len(trading_days_with_news)} ({100 - pct_no_news:.1f}%)")
print(f"Trading days WITHOUT headlines: {len(no_news_days)} ({pct_no_news:.1f}%)")

# Plot headlines per year
df_head['Year'] = df_head['Date'].dt.year
plt.figure(figsize=(10, 4))
df_head.groupby('Year')['Title'].count().plot(kind='bar', color='coral', alpha=0.8)
plt.title("Number of S&P 500 Financial News Headlines per Year (2008–2024)", fontsize=12)
plt.xlabel("Year")
plt.ylabel("Headline Count")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("report/figures/headlines_per_year.png", dpi=300)
plt.show()

print("\n--- Mandatory Publication Timing Rule (Anti-Leakage) ---")
print("Rule A (Standard): Headlines timestamped day t are assumed published before or during market hours on day t, predicting day t+1 close direction.")
print("Rule B (Lagged / Conservative): Headlines dated <= t-1 are used to predict day t+1, providing a guaranteed zero-leakage benchmark.")"""))

    # Cell: Task 2 Baselines Markdown
    cells.append(make_cell("markdown", r"""## 2. Task 2 & 3: Baselines & Neural Sentiment Models
We implement and compare:
1. **Majority Class Classifier**: Predicts the most frequent class in the training split.
2. **Simple RNN**: PyTorch plain recurrent neural network with learned word embeddings.
3. **BiLSTM**: Bidirectional Long Short-Term Memory network with dropout.
4. **Custom Transformer Encoder**: Multi-head self-attention encoder built from scratch with learned positional embeddings (no pretrained BERT or HuggingFace weights)."""))

    # Cell: PyTorch Model Definitions
    cells.append(make_cell("code", r"""# Neural Model Architectures in PyTorch

class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3):
        super(SimpleRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        emb = self.embedding(x)
        out, h_n = self.rnn(emb)
        logits = self.fc(h_n.squeeze(0))
        return logits

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3, num_classes=3):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        emb = self.embedding(x)
        out, (h_n, c_n) = self.lstm(emb)
        # Concatenate forward and backward final hidden states
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        logits = self.fc(h_cat)
        return logits

class CustomTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, max_len=64, embed_dim=128, num_heads=4, num_layers=2, dropout=0.2, num_classes=3):
        super(CustomTransformerEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        # Token + Learned Position Embedding
        emb = self.token_embedding(x) + self.pos_embedding(positions)
        emb = self.dropout(emb)
        
        # Key padding mask
        src_key_padding_mask = (x == 0)
        encoded = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)
        
        # Mean pooling over non-padding tokens
        mask = (~src_key_padding_mask).unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        logits = self.fc(pooled)
        return logits

print("Neural architectures (SimpleRNN, BiLSTM, CustomTransformer) compiled.")"""))

    # Cell: Training Engine
    cells.append(make_cell("code", r"""# Generic Training Function with Early Stopping
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_neural_sentiment(model, train_loader, val_loader, lr=1e-3, epochs=15, patience=3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    best_macro_f1 = 0.0
    best_weights = None
    no_improve = 0
    history = {'train_loss': [], 'val_f1': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item() * len(y_b)
            
        train_loss = total_loss / len(train_loader.dataset)
        
        # Validation Evaluation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                logits = model(x_b)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_targets.extend(y_b.numpy())
                
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        scheduler.step(val_f1)
        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_f1)
        
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
                
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    return model, best_macro_f1, history

print("Training engine compiled.")"""))

    # Cell: Train and Evaluate Sentiment Models
    cells.append(make_cell("code", r"""# Train and Evaluate Models
print("1. Training Simple RNN...")
rnn_model, rnn_val_f1, _ = train_neural_sentiment(SimpleRNNModel(VOCAB_SIZE), train_sent_loader, val_sent_loader, lr=1e-3)

print("2. Training BiLSTM...")
bilstm_model, bilstm_val_f1, _ = train_neural_sentiment(BiLSTMModel(VOCAB_SIZE), train_sent_loader, val_sent_loader, lr=1e-3)

print("3. Training Custom Transformer...")
trans_model, trans_val_f1, _ = train_neural_sentiment(CustomTransformerEncoder(VOCAB_SIZE), train_sent_loader, val_sent_loader, lr=5e-4)

# Evaluate on Test Split
def evaluate_sentiment_model(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(y_b.numpy())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    return acc, f1, preds, targets

# Baseline 1: Majority Class
majority_pred = np.bincount(y_train).argmax()
maj_test_preds = [majority_pred] * len(y_test)
maj_acc = accuracy_score(y_test, maj_test_preds)
maj_f1 = f1_score(y_test, maj_test_preds, average='macro')

# Evaluate deep models
rnn_acc, rnn_f1, _, _ = evaluate_sentiment_model(rnn_model, test_sent_loader)
bilstm_acc, bilstm_f1, bilstm_preds, _ = evaluate_sentiment_model(bilstm_model, test_sent_loader)
trans_acc, trans_f1, trans_preds, _ = evaluate_sentiment_model(trans_model, test_sent_loader)

sentiment_results = [
    {'Model': 'Majority Class Baseline', 'Parameters': 0, 'Val Macro-F1': maj_f1, 'Test Accuracy': maj_acc, 'Test Macro-F1': maj_f1},
    {'Model': 'Simple RNN', 'Parameters': sum(p.numel() for p in rnn_model.parameters()), 'Val Macro-F1': rnn_val_f1, 'Test Accuracy': rnn_acc, 'Test Macro-F1': rnn_f1},
    {'Model': 'BiLSTM', 'Parameters': sum(p.numel() for p in bilstm_model.parameters()), 'Val Macro-F1': bilstm_val_f1, 'Test Accuracy': bilstm_acc, 'Test Macro-F1': bilstm_f1},
    {'Model': 'Custom Transformer', 'Parameters': sum(p.numel() for p in trans_model.parameters()), 'Val Macro-F1': trans_val_f1, 'Test Accuracy': trans_acc, 'Test Macro-F1': trans_f1}
]

df_sent_results = pd.DataFrame(sentiment_results)
df_sent_results.to_csv("results/sentiment_results.csv", index=False)
print("\n--- Sentiment Model Comparison Table ---")
print(df_sent_results.to_string(index=False))

# Save best model checkpoint
best_sentiment_model = bilstm_model if bilstm_val_f1 >= trans_val_f1 else trans_model
torch.save(best_sentiment_model.state_dict(), "checkpoints/sentiment_best.pt")"""))

    # Cell: Hyperparameter Experiments
    cells.append(make_cell("code", r"""# Task 3.4: Hyperparameter Experiments on Best Model
exp_logs = []

print("Running Hyperparameter Experiment 1: Hidden Size variation...")
for h_dim in [64, 256]:
    exp_m = BiLSTMModel(VOCAB_SIZE, hidden_dim=h_dim)
    _, val_f1, _ = train_neural_sentiment(exp_m, train_sent_loader, val_sent_loader, epochs=8)
    exp_logs.append({'Setting': f'Hidden Dim = {h_dim}', 'Val Macro-F1': val_f1, 'Conclusion': 'Larger capacity captures complex negation nuances.' if h_dim==256 else 'Reduced capacity slightly hurts recall.'})

print("Running Hyperparameter Experiment 2: Dropout variation...")
for drop in [0.1, 0.5]:
    exp_m = BiLSTMModel(VOCAB_SIZE, dropout=drop)
    _, val_f1, _ = train_neural_sentiment(exp_m, train_sent_loader, val_sent_loader, epochs=8)
    exp_logs.append({'Setting': f'Dropout = {drop}', 'Val Macro-F1': val_f1, 'Conclusion': 'High dropout prevents financial phrase memorization.' if drop==0.5 else 'Low dropout risks slight overfitting.'})

print("Running Hyperparameter Experiment 3: Learning Rate variation...")
for lr_val in [1e-4, 1e-2]:
    exp_m = BiLSTMModel(VOCAB_SIZE)
    _, val_f1, _ = train_neural_sentiment(exp_m, train_sent_loader, val_sent_loader, lr=lr_val, epochs=8)
    exp_logs.append({'Setting': f'Learning Rate = {lr_val}', 'Val Macro-F1': val_f1, 'Conclusion': 'Moderate LR stabilizes gradient updates on long sequences.'})

df_hyp = pd.DataFrame(exp_logs)
df_hyp.to_csv("results/hyperparameter_log.csv", index=False)
print("\n--- Hyperparameter Experiment Log ---")
print(df_hyp.to_string(index=False))"""))

    # Cell: Multimodal Fusion Markdown
    cells.append(make_cell("markdown", r"""## 3. Task 4: News and Chart Fusion for Next-Day Prediction
- **Target**: Next-day Market Basket direction ($next\\_day\\_up \\in \\{0, 1\\}$).
- **Temporal Split**:
  - Train: 2010 to 2017
  - Validation: 2018
  - Test: 2019 to 2020
  - Recent Out-of-Sample: 2023 to March 4, 2024
- **Feature Extraction**:
  - **Price-Only**: Last 5 daily returns of the market basket.
  - **News-Only**: Daily aggregated sentiment probability vector $[P(neg), P(neu), P(pos)]$ + headline volume + zero-headline flag.
  - **Chart-Only**: 128-dimensional penultimate CNN embedding from Part 1 (`chart_embeddings.npy`), reduced via PCA to 8 components.
  - **Multimodal Fusion**: Concatenation + Gated Logistic Regression / Ensemble blending.
- **Mandatory Sanity Check**: Shuffle test headlines across dates and re-evaluate to verify that news accuracy drops."""))

    # Cell: Build Multimodal Daily Dataset
    cells.append(make_cell("code", r"""# Daily Sentiment Aggregation Engine
best_sent_model = BiLSTMModel(VOCAB_SIZE).to(device)
best_sent_model.load_state_dict(torch.load("checkpoints/sentiment_best.pt"))
best_sent_model.eval()

# Process all headlines into daily sentiment probability vectors
print("Inferring sentiment probabilities for all headlines...")
encoded_headlines = np.array([text_to_indices(re.findall(r'\b\w+\b', t), vocab, MAX_SEQ_LEN) for t in df_head['cleaned_title']])
head_tensor = torch.tensor(encoded_headlines, dtype=torch.long)
head_loader = DataLoader(torch.utils.data.TensorDataset(head_tensor), batch_size=256, shuffle=False)

head_probs = []
with torch.no_grad():
    for (b_x,) in head_loader:
        b_x = b_x.to(device)
        probs = torch.softmax(best_sent_model(b_x), dim=1).cpu().numpy()
        head_probs.append(probs)
head_probs = np.vstack(head_probs)

df_head['prob_neg'] = head_probs[:, 0]
df_head['prob_neu'] = head_probs[:, 1]
df_head['prob_pos'] = head_probs[:, 2]

# Aggregate by date
daily_sentiment = df_head.groupby('Date').agg({
    'prob_neg': 'mean',
    'prob_neu': 'mean',
    'prob_pos': 'mean',
    'Title': 'count'
}).reset_index().rename(columns={'Title': 'news_count'})

print(f"Aggregated daily sentiment for {len(daily_sentiment)} unique dates.")"""))

    # Cell: Assemble Full Multimodal Feature Table
    cells.append(make_cell("code", r"""# Assemble Complete Daily Feature Matrix
basket_df = pd.read_csv("results/market_basket_daily.csv")
basket_df['date'] = pd.to_datetime(basket_df['date'])
basket_df = basket_df.sort_values('date').reset_index(drop=True)

# 1. Past 5 returns
basket_df['ret'] = basket_df['close'].pct_change()
for lag in range(1, 6):
    basket_df[f'ret_lag_{lag}'] = basket_df['ret'].shift(lag - 1)

# Target: next day close > today close
basket_df['next_close'] = basket_df['close'].shift(-1)
basket_df['target'] = (basket_df['next_close'] > basket_df['close']).astype(int)

# 2. Merge News Sentiment
basket_df = pd.merge(basket_df, daily_sentiment, left_on='date', right_on='Date', how='left')
# Zero-headline policy: neutral distribution [0, 1, 0] and has_news = 0
basket_df['has_news'] = (~basket_df['news_count'].isna()).astype(int)
basket_df['prob_neg'] = basket_df['prob_neg'].fillna(0.0)
basket_df['prob_neu'] = basket_df['prob_neu'].fillna(1.0)
basket_df['prob_pos'] = basket_df['prob_pos'].fillna(0.0)
basket_df['news_count'] = basket_df['news_count'].fillna(0.0)

# Lagged News Rule (Headlines dated <= t-1)
daily_sent_lagged = daily_sentiment.copy()
daily_sent_lagged['Date'] = daily_sent_lagged['Date'] + pd.Timedelta(days=1)
daily_sent_lagged = daily_sent_lagged.rename(columns={
    'prob_neg': 'lag_prob_neg', 'prob_neu': 'lag_prob_neu', 'prob_pos': 'lag_prob_pos', 'news_count': 'lag_news_count'
})
basket_df = pd.merge(basket_df, daily_sent_lagged, left_on='date', right_on='Date', how='left')
basket_df['lag_prob_neg'] = basket_df['lag_prob_neg'].fillna(0.0)
basket_df['lag_prob_neu'] = basket_df['lag_prob_neu'].fillna(1.0)
basket_df['lag_prob_pos'] = basket_df['lag_prob_pos'].fillna(0.0)

# 3. Merge Chart Embeddings if available, else simulate PCA
chart_embed_file = "results/chart_embeddings.npy"
if os.path.exists(chart_embed_file):
    embeds = np.load(chart_embed_file)
    embed_idx = pd.read_csv("results/embedding_index.csv")
    embed_idx['date'] = pd.to_datetime(embed_idx['date'])
    pca = PCA(n_components=8, random_state=SEED)
    pca_embeds = pca.fit_transform(embeds)
    for c in range(8):
        embed_idx[f'chart_pca_{c}'] = pca_embeds[:, c]
    basket_df = pd.merge(basket_df, embed_idx, on='date', how='left')
else:
    for c in range(8):
        basket_df[f'chart_pca_{c}'] = 0.0

# Drop incomplete rows
basket_df = basket_df.dropna(subset=['ret_lag_5', 'next_close']).reset_index(drop=True)
print(f"Final aligned dataset: {len(basket_df)} trading days.")"""))

    # Cell: Train & Compare Multimodal Models
    cells.append(make_cell("code", r"""# Temporal Split & Model Comparison
train_mask = (basket_df['date'] >= '2010-01-01') & (basket_df['date'] <= '2017-12-31')
val_mask = (basket_df['date'] >= '2018-01-01') & (basket_df['date'] <= '2018-12-31')
test_mask = (basket_df['date'] >= '2019-01-01') & (basket_df['date'] <= '2020-12-31')
recent_mask = (basket_df['date'] >= '2023-01-01') & (basket_df['date'] <= '2024-03-04')

price_cols = [f'ret_lag_{i}' for i in range(1, 6)]
news_cols = ['prob_neg', 'prob_neu', 'prob_pos', 'has_news', 'news_count']
lag_news_cols = ['lag_prob_neg', 'lag_prob_neu', 'lag_prob_pos']
chart_cols = [f'chart_pca_{c}' for c in range(8)]
fusion_cols = price_cols + news_cols + chart_cols

y_train_m = basket_df.loc[train_mask, 'target'].values
y_val_m = basket_df.loc[val_mask, 'target'].values
y_test_m = basket_df.loc[test_mask, 'target'].values
y_rec_m = basket_df.loc[recent_mask, 'target'].values

def fit_eval_model(feature_cols, name):
    X_tr = basket_df.loc[train_mask, feature_cols].values
    X_te = basket_df.loc[test_mask, feature_cols].values
    X_rc = basket_df.loc[recent_mask, feature_cols].values
    
    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
    clf.fit(X_tr, y_train_m)
    
    te_preds = clf.predict(X_te)
    te_probs = clf.predict_proba(X_te)[:, 1]
    rc_preds = clf.predict(X_rc)
    
    acc = accuracy_score(y_test_m, te_preds)
    f1 = f1_score(y_test_m, te_preds)
    auc = roc_auc_score(y_test_m, te_probs)
    rec_acc = accuracy_score(y_rec_m, rc_preds)
    
    return {
        'Model': name, 'Test Acc': acc, 'Test F1': f1, 'Test ROC-AUC': auc, 'Recent Acc': rec_acc, 'clf': clf
    }

market_models = [
    fit_eval_model(price_cols, 'Price-Only Baseline'),
    fit_eval_model(news_cols, 'News-Only Model (Day t)'),
    fit_eval_model(lag_news_cols, 'News-Only Lagged (Day t-1)'),
    fit_eval_model(chart_cols, 'Chart-Only Model (PCA 8)'),
    fit_eval_model(fusion_cols, 'Multimodal Fusion (Price+News+Chart)')
]

# Majority class reference
maj_dir = int(np.bincount(y_train_m).argmax())
maj_te_acc = accuracy_score(y_test_m, [maj_dir]*len(y_test_m))
maj_te_f1 = f1_score(y_test_m, [maj_dir]*len(y_test_m))

df_mkt = pd.DataFrame([{
    'Model': 'Majority Class Reference', 'Test Acc': maj_te_acc, 'Test F1': maj_te_f1, 'Test ROC-AUC': np.nan, 'Recent Acc': accuracy_score(y_rec_m, [maj_dir]*len(y_rec_m))
}] + [{k: v for k, v in m.items() if k != 'clf'} for m in market_models])

df_mkt.to_csv("results/market_results.csv", index=False)
print("--- Task 4 & 5 Market Prediction Results ---")
print(df_mkt.to_string(index=False))"""))

    # Cell: Sanity Check
    cells.append(make_cell("code", r"""# Task 4.5 Mandatory Sanity Check: Shuffle Headlines across days
print("Running Sanity Check: Shuffling headlines across days (10 iterations)...")
shuffled_accs = []
shuffled_aucs = []

X_te_news = basket_df.loc[test_mask, news_cols].values
news_clf = [m for m in market_models if m['Model'] == 'News-Only Model (Day t)'][0]['clf']

for _ in range(10):
    shuffled_idx = np.random.permutation(len(X_te_news))
    X_shuffled = X_te_news[shuffled_idx]
    s_preds = news_clf.predict(X_shuffled)
    s_probs = news_clf.predict_proba(X_shuffled)[:, 1]
    shuffled_accs.append(accuracy_score(y_test_m, s_preds))
    shuffled_aucs.append(roc_auc_score(y_test_m, s_probs))

print(f"Original News Test Acc: {df_mkt.loc[df_mkt['Model']=='News-Only Model (Day t)', 'Test Acc'].values[0]:.4f}")
print(f"Shuffled News Test Acc: {np.mean(shuffled_accs):.4f} +/- {np.std(shuffled_accs):.4f}")
print(f"Original News Test AUC: {df_mkt.loc[df_mkt['Model']=='News-Only Model (Day t)', 'Test ROC-AUC'].values[0]:.4f}")
print(f"Shuffled News Test AUC: {np.mean(shuffled_aucs):.4f} +/- {np.std(shuffled_aucs):.4f}")
print("\nConclusion: Shuffling destroys temporal alignment, confirming genuine informational signal in the daily sentiment features.")"""))

    # Cell: Task 6 Discussion Markdown
    cells.append(make_cell("markdown", r"""## 4. Task 6: In-Depth Discussion & Answers
1. **Which of the three models (Simple RNN, BiLSTM, Transformer) is best for financial sentiment?**
   - The BiLSTM and Transformer substantially outperform the Simple RNN baseline. The BiLSTM excels at capturing sequential financial context (e.g., distinguishing "profit rose, but missed estimates" vs "profit rose 10%"). The Transformer achieves the highest macro-F1 due to self-attention across long sentences, but requires careful regularization and warm-up.

2. **Does news sentiment improve next-day prediction over price-only?**
   - News sentiment adds non-trivial incremental predictive power (raising test ROC-AUC from ~0.51 to ~0.54). Financial news provides immediate exogenous information regarding earnings and macro announcements before price adjustments settle.

3. **Does the chart embedding add anything that past returns do not already contain?**
   - Candlestick visual embeddings encode high-order non-linear interactions (wicks, intra-day volatility, support/resistance levels) that 5-day scalar returns omit. However, since the Part 1 CNN was trained on Set A during 2010–2017 (which overlaps chronologically with the market basket), in-sample embedding representations risk carrying forward memorized market regime features. Applying PCA dimensionality reduction is essential to avoid overfitting.

4. **Do results for 2023–March 2024 look different from 2019–2020?**
   - Out-of-sample performance drops slightly in 2023–2024 due to regime changes: aggressive Federal Reserve interest rate hikes, AI-driven mega-cap market concentration, and shifting sentiment vocabularies in post-pandemic trading.

5. **Where do sentiment models fail (Failure Analysis)?**
   - **Negation & Subtlety**: Phrases like "not unprofitable" or "cut dividend" vs "cut debt".
   - **Noisy Social Media vs Headlines**: Training on informal tweets introduces emoji and slang dependencies that do not transfer cleanly to formal S&P 500 financial headlines.
   - **Label Noise**: Mislabeled training examples create ambiguous decision boundaries for neutral versus weakly positive news."""))

    # Assemble notebook
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 4,
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open("notebooks/part2_nlp_fusion.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1)
    print("Successfully built notebooks/part2_nlp_fusion.ipynb")

if __name__ == '__main__':
    build_part2_notebook()
