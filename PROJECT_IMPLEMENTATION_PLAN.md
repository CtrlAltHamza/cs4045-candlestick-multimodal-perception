# Deep Learning for Perception (CS 4045) – Semester Project Implementation Plan

## Project Overview

- **Title**: Multi-modal Next-Day Stock Direction Prediction via Candlestick Vision and Financial News Sentiment
- **Course**: Deep Learning for Perception (CS 4045)
- **Instructor**: Dr. Ahmad Raza Shahid
- **Total Points**: 200 Points (Part 1: 100 Points, Part 2: 100 Points)
- **Primary Deliverables**:
  - `notebooks/part1_vision.ipynb`
  - `notebooks/part2_nlp_fusion.ipynb`
  - `report.pdf` (LaTeX compiled, 6–10 pages)
  - `checkpoints/` (Trained model weights)
  - `results/` (All CSV evaluation metrics + `chart_embeddings.npy`)
  - `README.txt` (Complete reproduction guide with seeds and requirements)

---

## 1. Hardware & Training Strategy: Local vs. Kaggle GPU

### Hardware Audit
- **Local Machine**: Windows, CPU execution only (`torch.cuda.is_available() == False`).
- **Computational Demand**:
  - 3 complete YOLOv8 runs from scratch (Run A: No aug, Run B: Standard aug, Run C: Custom chart aug) on 3,000+ images (640×640).
  - 1 Custom CNN / ResNet18 model on 3,000+ images.
  - Generating 128-dimensional penultimate feature embeddings for ~3,700 trading days (2010–2024).
  - PyTorch NLP models (Simple RNN, BiLSTM, Custom Multi-Head Transformer Encoder).

### Selected Strategy: Hybrid Development Workflow
1. **Local Environment (Prototyping & Inspection)**:
   - Data exploration and ticker health checking.
   - Algorithmic pattern labeling logic and bounding box coordinate math.
   - Report drafting in LaTeX.
2. **Kaggle GPU Environment (Heavy Training & Inference)**:
   - Free NVIDIA T4 x 2 or P100 GPU (30 hours/week quota).
   - Fast batch chart generation and YOLOv8 training from `.yaml` config (15–30 minutes per run vs. 10+ hours on CPU).
   - PyTorch text models and bootstrap confidence interval resampling (1,000+ iterations).
3. **Artifact Sync**:
   - Download trained `.pt` weight checkpoints, metric CSVs, and `chart_embeddings.npy` to the local repository.

---

## 2. Dataset Architecture & Handling Strategy

The source archive `Data.zip` (~101 MB compressed, ~280 MB raw) contains:
- `Data/sp500_stocks.csv` (~276 MB): 25 years of daily OHLCV prices for S&P 500 stocks.
- `Data/sp500_companies.csv` (~50 KB): Sector classifications (GICS) and listing dates.
- `Data/dataset.csv` (~3.3 MB): ~32,600 financial texts with 3 balanced sentiment classes.
- `Data/sp500_headlines_2008_2024.csv` (~1.5 MB): ~19,000 S&P 500 headlines with trading dates.

### Data Pipelines

#### A. Stock Selection & Health Check
- Filter out companies added recently, companies with placeholder constant prices (e.g., `SW`, `FERG`, `AMCR`), days with zero volume, or days where $High == Low$.
- Select **20 tickers** spanning at least 4 distinct GICS sectors (e.g., Technology, Financials, Healthcare, Industrials, Energy) with continuous data from **January 2010 through December 2024**.
- Split tickers into:
  - **Set A (12 Tickers)**: Used to draw training, validation, and test charts for Part 1 models.
  - **Set B (8 Tickers)**: Reserved exclusively for constructing the synthetic Market Basket for Part 2.

#### B. Synthetic Market Basket Construction
Because the raw price file does not provide the actual S&P 500 index series, build a normalized proxy index:
1. Rebase each of the 8 Set B tickers by dividing its OHLC by its Close on the first trading day of 2010 and multiplying by 100:
   $$\text{Rebased Price}_{i, t} = \frac{\text{Price}_{i, t}}{\text{Close}_{i, \text{Jan 2010}}} \times 100$$
2. For each trading day $t$, take the arithmetic mean of the rebased OHLC across the 8 tickers, and sum their traded volumes.
3. Use the market basket as the target for Task 4 prediction and chart embedding generation.

#### C. Candlestick Chart Generation & Algorithmic Labeling
- **Resolution**: Exactly 640 × 640 pixels per image.
- **Window Length**: 30 consecutive trading days per image.
- **Visual Variations (Stochastic rendering per image)**:
  - Themes: Light, Dark, or Blue-White.
  - Overlays: Grid (on/off), Volume panel (on/off), 10-day Simple Moving Average line (on/off).
- **Labeling Engine**:
  - Automatically calculate bounding boxes for **8 technical patterns**: Doji, Hammer, Shooting Star, Marubozu, Bullish Engulfing, Bearish Engulfing, Morning Star, Evening Star.
  - Derive boxes programmatically from OHLC pixel positions (`ax.transData`) spanning from lowest low to highest high across the 1–3 pattern candles.
- **Dataset Size Requirements**:
  - Training: $\ge 3,000$ images (chosen at random from Set A, 2010–2017).
  - Validation: $\ge 500$ images (synchronized window end dates every 4 trading days across all 12 tickers, 2018).
  - Test: $\ge 500$ images (synchronized window end dates, 2019–2020).
  - Recent: $\ge 500$ images (synchronized window end dates, 2023–2024).

#### D. NLP Preprocessing & Timing Rules
- **Text Cleaning**: Lowercase, remove non-alphanumeric noise, unescape HTML entities (`&amp;` $\rightarrow$ `&`), strip `<br>` tags, handle/filter non-English rows and emojis.
- **Deduplication**: Strip duplicate texts. If identical texts carry conflicting sentiment labels, discard all instances.
- **Tokenization**: Custom vocabulary with frequency cutoff, `<PAD>` and `<UNK>` tokens, fixed maximum sequence length.
- **Strict Timing Rule (Anti-Leakage)**:
  - News dated on day $t$ can only be used to predict day $t+1$ under an explicit publication assumption.
  - Required comparison against a lagged rule (using news dated $\le t-1$).

---

## 3. End-to-End Implementation Roadmap

### Phase 1: Candlestick Vision (Part 1 - 100 Points)

1. **Task 1: Data & Exploratory Data Analysis (20 pts)**
   - Audit missing trading dates, zero-volume days, and price jumps.
   - Calculate return distributions and kurtosis per ticker (fat tails analysis).
   - Frequency distribution of the 8 patterns and explanation of natural class imbalance.
   - Visualize 10 sample charts with superimposed bounding boxes.
   - Plot the proportion of `next_day_up == 1` over time.
   - Document survivorship bias and chronological train/val/test/recent split justification.

2. **Task 2: Candlestick Pattern Detection with YOLOv8 (45 pts)**
   - Train `yolov8n.yaml` strictly from scratch (no pretrained weights).
   - Train 3 distinct augmentation configurations:
     - **Run A**: Zero augmentation.
     - **Run B**: Standard vision augmentation (horizontal flip $p=0.5$, vertical flip $p=0.2$, mosaic, rotation, shear, jitter).
     - **Run C**: Domain-suited chart augmentation (brightness/contrast, noise, small shift/scale, JPEG compression).
   - Evaluate on test set: mean precision, recall, mAP@50, mAP@50-95.
   - For best run: per-class PR metrics, confusion matrix, 10 qualitative detection images, and 5 documented failure cases.

3. **Task 3: Next-Day Direction Trend Prediction with CNN (25 pts)**
   - Model: Custom CNN or ResNet18 (ImageNet weights allowed).
   - Baselines: (1) Majority class predictor, (2) Logistic regression on 5 past daily returns.
   - Metrics: Accuracy, F1-up, ROC-AUC, confusion matrix, parameter count, file size, and inference latency.
   - Statistical Testing: 95% bootstrap confidence intervals (1,000+ resamples grouped by trading date).
   - Feature Extraction: Pass the 2010–2024 Market Basket charts through the CNN to extract the 128-dimensional penultimate feature representation into `chart_embeddings.npy`.

4. **Task 4: Discussion & Report Part 1 (10 pts)**
   - In-depth answers to the 5 mandatory Part 1 discussion questions.

---

### Phase 2: NLP Sentiment & Multimodal Fusion (Part 2 - 100 Points)

1. **Task 1: NLP Data Exploration & Preparation (15 pts)**
   - Class distributions, sentence length statistics, vocabulary coverage.
   - Deduplication metrics and headline frequency per day/year.
   - Define missing headline imputation policy.

2. **Task 2 & 3: Baselines & Neural Sentiment Models (40 pts)**
   - Baselines: Majority class baseline and PyTorch Simple RNN.
   - Deep Models: PyTorch BiLSTM and Custom Transformer Encoder (multi-head self-attention from scratch, no pretrained LLMs/BERT).
   - Metrics: Accuracy, Macro-F1, parameter counts, training time curves.
   - Hyperparameter Experiments (at least 3 variations: hidden size, dropout, learning rate).

3. **Task 4: News and Chart Multimodal Fusion (25 pts)**
   - Convert daily headlines into daily feature vectors (average class probabilities, headline volume).
   - Single-source benchmark models:
     - Price-only (last 5 daily basket returns).
     - News-only (sentiment feature vectors).
     - Chart-only (128-dim embedding + PCA/regularization).
   - Multimodal Fusion: Feature concatenation / Ensemble probability blending / Gated MLP.
   - **Mandatory Sanity Check**: Shuffle test headlines across days and re-evaluate. Verify if test performance collapses as expected.

4. **Task 5: Evaluation & Out-of-Sample Testing (10 pts)**
   - 95% bootstrap confidence intervals for all fusion models vs. baselines.
   - Required timing comparison: day $t$ news vs. lagged day $t-1$ news.
   - Efficiency benchmarks (training time, inference ms per prediction, parameter counts).
   - Out-of-sample evaluation on recent data (2023 to March 4, 2024).

5. **Task 6: Discussion & Final Report Part 2 (10 pts)**
   - In-depth answers to the 5 mandatory Part 2 discussion questions.

---

## 4. Risk Matrix & Technical Solutions

| # | Pitfall / Challenge | Severity | Cause | Technical Solution |
|---|---|---|---|---|
| **1** | **Future Information Leakage** | Critical (Automatic 0) | Using $t+1$ prices, looking ahead in window calculations, or post-close news. | Enforce strict temporal splits: Train (2010–2017), Val (2018), Test (2019–2020), Recent (2023–2024). Never use `train_test_split(shuffle=True)`. Run $t-1$ lag checks. |
| **2** | **Forbidden Pretrained Weights** | Critical (Disqualification) | Using `yolov8n.pt` or HuggingFace BERT/RoBERTa. | Initialize YOLO strictly via `YOLO('yolov8n.yaml')`. Write standard PyTorch `nn.Module` classes for RNN, BiLSTM, and Transformer from scratch. |
| **3** | **Candlestick Semantic Corruption (Run B)** | Medium (Evaluation Trap) | Vertical flip ($p=0.2$) in standard vision augmentation. | In financial charts, vertical flipping turns a Bullish Hammer into a Bearish Shooting Star. Document this semantic disruption in Task 2.1 to gain full analysis points. |
| **4** | **Chart Rendering Bottleneck** | Medium (Runtime Delay) | Generating 4,500+ PNGs via slow Matplotlib GUI rendering. | Use `matplotlib.use('Agg')`, batch processing, vectorized OHLC rules, and explicit memory cleanup (`plt.close('all')`). |
| **5** | **Candlestick Class Imbalance** | Medium (Model Bias) | Rare multi-candle patterns (Morning Star) appear far less often than Doji. | Carefully tune pattern threshold definitions. Evaluate performance using Macro-F1 and class-wise mAP rather than overall accuracy. |
| **6** | **Early Year News Sparsity** | High (Missing Features) | In 2010–2012, ~50% of trading days lack headlines. | Define a formal zero-headline policy: assign a neutral distribution vector $[0, 1, 0]$ and append a binary indicator feature `has_news \in {0, 1}`. |
| **7** | **Overfitting on 128-dim Chart Embeddings** | High (Generalization Failure) | 128 features with only ~2,000 training samples causes linear models to memorize noise. | Apply dimensionality reduction (PCA to 5–10 components) or strong L2 regularization ($C=10^{-2}$ in Logistic Regression). |
| **8** | **Market Predictability Illusion** | Low (Conceptual Trap) | Over-optimizing models to chase unrealistic 70%+ market accuracy. | Stock direction is near random-walk. Expected test accuracy is naturally around 52–57%. Focus effort on statistical significance, bootstrap CIs, and rigorous ablation. |

---

## 5. Deliverables & Directory Structure Checklist

```text
dlp_project/
├── Data/                                    # Extracted dataset files
│   ├── sp500_stocks.csv
│   ├── sp500_companies.csv
│   ├── sp500_headlines_2008_2024.csv
│   └── dataset.csv
├── notebooks/
│   ├── part1_vision.ipynb                   # End-to-end Part 1 pipeline
│   └── part2_nlp_fusion.ipynb               # End-to-end Part 2 pipeline
├── checkpoints/
│   ├── yolo_best.pt                         # Best YOLOv8 detection weights
│   ├── cnn_trend_best.pt                    # Best CNN trend model weights
│   ├── sentiment_best.pt                    # Best NLP sentiment model weights
│   └── fusion_best.pt                       # Best multimodal fusion model weights
├── results/
│   ├── detection_results.csv
│   ├── sentiment_results.csv
│   ├── market_results.csv
│   ├── hyperparameter_log.csv
│   └── chart_embeddings.npy                 # 128-dim market basket embeddings
├── report/
│   ├── report.tex                           # LaTeX source (6-10 pages)
│   ├── figures/                             # Architecture diagrams & EDA plots
│   └── report.pdf                           # Compiled PDF report
├── README.txt                               # Reproduction guide, seeds, dependencies
└── requirements.txt                         # Pinned Python package dependencies
```
