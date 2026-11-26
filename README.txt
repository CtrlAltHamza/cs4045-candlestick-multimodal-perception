================================================================================
DEEP LEARNING FOR PERCEPTION (CS 4045) – SEMESTER PROJECT
Multi-modal Next-Day Stock Direction Prediction via Candlestick Vision
and Financial News Sentiment
================================================================================

Course Instructor: Dr. Ahmad Raza Shahid
Semester: Fall 2026
Total Points: 200 (Part 1: 100 Points, Part 2: 100 Points)

--------------------------------------------------------------------------------
1. GROUP MEMBERS & CONTRIBUTION MATRIX
--------------------------------------------------------------------------------
Group Members:
- Hamza Khurram (Roll No: i210735)
- Raja Abdullah Tahir (Roll No: i210708)

Contribution Statement:
Both group members actively collaborated and contributed across all project tasks in full compliance with course regulations (Page 10):
- Hamza Khurram (i210735):
  * Part 1: S&P 500 constituent health audit, ticker selection (Sets A and B), stochastic 640x640 candlestick rendering engine with ax.transData coordinate mapping, YOLOv8 scratch model configuration and training across Runs A, B, and C, augmentation semantic corruption analysis.
  * Part 2: Financial news headlines parsing and timing rules (Day t vs lagged Day t-1), Custom Multi-Head Transformer Encoder architecture implementation and attention mechanism, multimodal feature matrix assembly, and headline-shuffling sanity validation.
  * System reporting: System architecture diagrams (TikZ Diagrams 1 & 4), LaTeX paper drafting, and experimental evaluation tables.
- Raja Abdullah Tahir (i210708):
  * Part 1: Algorithmic bounding box math for the 8 candlestick patterns, daily return distributions and kurtosis fat-tails analysis, ResNet-18 trend classification architecture, 95% date-grouped bootstrap confidence interval calculation (1,000 resamples), and 128-dim penultimate market basket feature extraction.
  * Part 2: Financial sentiment text preprocessing (HTML decoding, duplicate and emoji purging), custom vocabulary tokenization (<PAD>, <UNK>, L_max=64), Simple RNN baseline and BiLSTM implementation, hyperparameter ablation study on the winning sentiment architecture, and single-source benchmark modeling.
  * System reporting: Discussion questions 1 to 5 for Part 1 & Part 2, TikZ Diagrams 2 & 3, failure case root-cause diagnostics, and final submission packaging.

--------------------------------------------------------------------------------
2. ENVIRONMENT & HARDWARE SETUP
--------------------------------------------------------------------------------
A. Kaggle GPU Environment (Recommended for heavy training):
   - Platform: Kaggle Notebooks
   - Accelerator: GPU P100 or 2x NVIDIA T4 (Free 30 hrs/week)
   - Environment: Python 3.10, PyTorch 2.1+, CUDA 12.1+

B. Local Environment:
   - Python: 3.10+ (the supplied requirements also resolve on Python 3.14)
   - OS: Windows / Linux / macOS
   - Recommended isolated setup on Windows:
       py -m venv .venv
       .venv\Scripts\python -m pip install -r requirements.txt
   - Dependencies on Linux/macOS: python -m pip install -r requirements.txt

--------------------------------------------------------------------------------
3. DATASET SETUP & STRUCTURE
--------------------------------------------------------------------------------
The project expects the extracted Google Classroom / Kaggle dataset files inside a 'Data/' directory:
   Data/
   ├── sp500_stocks.csv               (Daily OHLCV prices 2000-2024, ~289 MB)
   ├── sp500_companies.csv            (Company metadata, GICS sectors, ~50 KB)
   ├── sp500_headlines_2008_2024.csv  (Financial news headlines, ~1.5 MB)
   └── dataset.csv                    (Finance News Sentiments, ~3.4 MB)

On Kaggle:
   - Option A: Upload 'Data.zip' as a private Kaggle dataset named 'dlp-project-data'.
   - Option B: Link the public Kaggle datasets:
     * 'sp500-stocks-25-years-of-data'
     * 'finance-news-sentiments' (antobenedetti)
     * 's-and-p-500-with-financial-news-headlines-20082024' (dyutidasmahaptra)
   Both notebooks dynamically auto-detect paths at runtime across /kaggle/input and local ./Data.

--------------------------------------------------------------------------------
4. REPRODUCTION GUIDE: STEP-BY-STEP
--------------------------------------------------------------------------------
All experiments use a fixed master random seed: SEED = 42

Step 1: Part 1 Vision Pipeline (notebooks/part1_vision.ipynb)
   - Run all cells sequentially from top to bottom.
   - Tasks executed:
     1. Ticker selection & health audit (Set A: 12 tickers, Set B: 8 tickers).
     2. Daily return distributions and excess kurtosis calculation.
     3. Synthetic Market Basket construction from rebased Set B tickers.
     4. Candlestick chart generation (640x640) with stochastic variations (light, dark, blue-white themes, grid, SMA, volume).
     5. Programmatic bounding box calculation for 8 candlestick patterns via ax.transData.
     6. Dataset rendering: Train (>=3,000 images, 2010-2017), Val (>=500, 2018), Test (>=500, 2019-2020), Recent (>=500, 2023-2024).
     7. YOLOv8n training strictly from scratch (yolov8n.yaml) for Run A (no aug), Run B (standard vision aug), Run C (domain chart aug).
     8. Test split evaluation: precision, recall, mAP@50, mAP@50-95, and 480x480 robustness check.
     9. Next-Day Trend CNN (ResNet18) training, majority baseline, logistic regression on 5-day returns, and 95% grouped bootstrap CI.
     10. Penultimate 128-dim embedding extraction for 2010-2024 Market Basket -> chart_embeddings.npy.

Step 2: Part 2 NLP & Multimodal Fusion (notebooks/part2_nlp_fusion.ipynb)
   - Run all cells sequentially from top to bottom.
   - Tasks executed:
     1. Text cleaning (HTML unescaping, lowercasing, ASCII filter, non-alphanumeric strip).
     2. Deduplication and conflicting sentiment label removal.
     3. Custom vocabulary construction (<PAD>, <UNK>, max sequence length 64).
     4. Headline publication timing rule and zero-headline imputation policy.
     5. Training Simple RNN baseline, BiLSTM, and Custom Transformer Encoder from scratch.
     6. Hyperparameter ablation study (hidden size, dropout, learning rate).
     7. Multimodal feature assembly. PCA is fitted on 2010-2017 chart embeddings only and frozen for validation, test, and recent periods.
     8. Multimodal fusion vs. single-source models on 2019-2020 test split and 2023-March 2024 recent split.
     9. Mandatory sanity check: 100 headline-date shuffles, reported without assuming degradation.
   - Both day t and its target day t+1 must lie inside the same temporal split.
   - Model regularization is selected on 2018. Test years are evaluated only after selection.
   - To recompute Part 2 from the saved sentiment checkpoint without retraining it:
       python scripts/rebuild_part2_evaluation.py
   - To rerun only the six corrected Transformer ablations (including measured
     training time) without retraining or overwriting sentiment_best.pt:
       python scripts/run_sentiment_ablation.py
     A Kaggle GPU is recommended. This runner intentionally skips the main-model
     training cell and writes only results/hyperparameter_log.csv plus the
     existing sentence-length figure.
   - To regenerate chart splits and export the Part 1 per-class, theme, recent,
     robustness, and inference evidence without retraining or overwriting
     yolo_best.pt:
       python scripts/run_part1_checkpoint_evaluation.py
     The downloadable outputs are bundled as
     results/part1_checkpoint_evaluation.zip.

Step 3: How the submitted numbers were produced (final run) and how to reproduce them
   - Hardware/software of the final run: NVIDIA GeForce GTX 1650 (4 GB), Python 3.12,
     PyTorch 2.14.1 + CUDA 12.6, Ultralytics 8.4.218. Part 1 ran top to bottom in about
     11 minutes and Part 2 in about 10 minutes. YOLO Run C itself was trained on a Kaggle GPU.
   - Put the four data files in ./Data at the repository root and start Jupyter from the
     repository root (all paths in the notebooks are relative to it).
   - Part 1 switches (environment variables, all off by default):
       RETRAIN_YOLO=1        train Runs A, B and C from yolov8n.yaml (hours on a small GPU).
                             Default: evaluate checkpoints/yolo_best.pt (Run C, the run selected
                             on validation). Only the Run C weights were kept, so the Run A and
                             Run B rows of detection_results.csv come from the recorded Kaggle
                             evaluation in results/detection_results_kaggle_runs.csv.
       RETRAIN_TREND=1       retrain the ResNet-18 trend CNN. Default: evaluate
                             checkpoints/cnn_trend_best.pt.
       REBUILD_EMBEDDINGS=1  re-extract results/chart_embeddings.npy. Default: keep the saved
                             embeddings (used by Part 2) and verify 20 random dates against the
                             checkpoint (mean cosine similarity about 0.996).
     Example (Windows PowerShell):  $env:RETRAIN_TREND="1"; jupyter lab
   - Chart generation is seeded (SEED = 42). The windows, tickers, dates and labels of every
     split are fixed; the random style of each chart depends on the order of random draws.
   - The YOLO Run C training settings, training time (22.9 min, 20 epochs) and per-epoch loss
     history are read from the checkpoint and saved to results/yolo_run_c_training_history.csv.
   - Part 2 retrains the RNN, BiLSTM and Transformer when run, so small run-to-run differences
     from GPU non-determinism are expected. All numbers in the report are the outputs of the
     executed notebooks. scripts/rebuild_part2_evaluation.py aggregates daily sentiment slightly
     differently (ROC-AUC within 0.006) and is only used to draw report/figures/fusion_comparison_roc.png.
   - Inference times (ms per image) vary by a few ms between runs and machines.
   - Report: python scripts/create_overleaf_zip.py, then upload overleaf_upload.zip to Overleaf
     (New project -> Existing project (.zip)) and recompile main.tex.
   - Submission archive: python scripts/build_submission_zip.py

--------------------------------------------------------------------------------
5. OUTPUT ARTIFACTS INCLUDED IN SUBMISSION
--------------------------------------------------------------------------------
checkpoints/
├── yolo_best.pt              (Best YOLOv8 candlestick detector weights)
├── cnn_trend_best.pt         (Best ResNet18 trend predictor weights)
├── sentiment_best.pt         (Best BiLSTM/Transformer sentiment weights)
└── fusion_best.pt            (Best multimodal fusion weights)

results/
├── detection_results.csv     (YOLOv8 Run A/B/C metrics)
├── detection_per_class.csv   (Run C per-class metrics and test-box counts)
├── detection_theme_results.csv (Theme-specific and recent-set mAP@50)
├── detection_robustness.csv  (480px/JPEG stress-test degradation)
├── robustness_defense_validation.csv (Validation-only defense selection)
├── detection_robustness_improved.csv (Held-out mitigated robustness result)
├── detection_efficiency.csv  (Run C inference latency and checkpoint size)
├── sentiment_results.csv     (Simple RNN, BiLSTM, Transformer metrics)
├── market_results.csv        (Single-source vs. Multimodal fusion test & recent metrics)
├── daily_sentiment.csv       (Date-level sentiment probabilities from the saved Transformer)
├── headline_shuffle_results.csv (100-repeat headline-date sanity check)
├── market_test_predictions.csv  (Per-day held-out probabilities for reproducible ROC curves)
├── hyperparameter_log.csv    (Sentiment ablation log)
├── chart_embeddings.npy      (128-dim penultimate market basket embeddings)
└── embedding_index.csv       (Trading date index aligned with chart_embeddings.npy)

report/
├── report.tex                (Complete LaTeX paper source)
├── figures/                  (All architecture diagrams and EDA plots)
└── report.pdf                (Compiled 6-10 page submission PDF)

--------------------------------------------------------------------------------
6. ACADEMIC INTEGRITY & DISCLOSURE
--------------------------------------------------------------------------------
In compliance with CS 4045 academic integrity regulations (Page 11):
- Outside Datasets:
  * Kaggle "S&P 500 Stocks: 25 Years of Data" (sp500_stocks.csv, sp500_companies.csv)
  * Kaggle "Finance News Sentiments" by antobenedetti (dataset.csv)
  * Kaggle "S&P 500 with Financial News Headlines 2008 to 2024" by dyutidasmahaptra (sp500_headlines_2008_2024.csv)
- Libraries & Frameworks:
  * PyTorch (nn, optim, DataLoader) for BiLSTM, Custom Transformer Encoder, and Trend CNN
  * Ultralytics YOLOv8 (yolov8n.yaml built strictly from scratch, zero pretrained weights)
  * Scikit-Learn for Logistic Regression, PCA, evaluation metrics, and bootstrap resampling
  * Matplotlib / PIL for candlestick chart generation, coordinate projection, and figures
  * Pandas / NumPy for time-series rebasing, returns, kurtosis, and daily feature aggregation
- AI Usage Disclosure:
  * AI assistants (Antigravity) were utilized for code structuring, refactoring Matplotlib rendering loops for speed, drafting LaTeX figure formatting, and packaging reproduction scripts. All deep learning model configurations, bounding box mathematics, and experimental empirical evaluations were implemented, executed on Kaggle GPUs, and verified by group members Hamza Khurram (i210735) and Raja Abdullah Tahir (i210708).
  * Claude (Anthropic, via Claude Code) was used in the final stage to audit the repository against the project brief, re-run both notebooks top to bottom on a local NVIDIA GTX 1650, add evaluation cells (trend-CNN evaluation with date-grouped bootstrap, pattern counts from the label files, test-set YOLO detection and failure-case galleries), replace report numbers that did not match the executed notebooks, and draft revised report text. The final notebook runs used for every reported number were executed on that GTX 1650.
================================================================================
