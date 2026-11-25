"""
Helper script to package the final submission ZIP file: i210735_i210708_Project.zip.
Validates all required deliverables specified by Dr. Ahmad Raza Shahid's rubric before compressing.
"""

import os
import zipfile
import glob
import sys

REQUIRED_FILES = [
    "notebooks/part1_vision.ipynb",
    "notebooks/part2_nlp_fusion.ipynb",
    "checkpoints/yolo_best.pt",
    "checkpoints/cnn_trend_best.pt",
    "checkpoints/sentiment_best.pt",
    "checkpoints/fusion_best.pt",
    "results/detection_results.csv",
    "results/detection_per_class.csv",
    "results/detection_theme_results.csv",
    "results/detection_robustness.csv",
    "results/robustness_defense_validation.csv",
    "results/detection_robustness_improved.csv",
    "results/detection_efficiency.csv",
    "results/trend_results.csv",
    "results/trend_test_predictions.csv",
    "results/trend_recent_predictions.csv",
    "results/sentiment_results.csv",
    "results/market_results.csv",
    "results/daily_sentiment.csv",
    "results/headline_shuffle_results.csv",
    "results/market_test_predictions.csv",
    "results/hyperparameter_log.csv",
    "results/chart_embeddings.npy",
    "results/embedding_index.csv",
    "results/market_basket_daily.csv",
    "results/selected_tickers.json",
    "report/report.tex",
    "README.md",
    "README.txt",
    "requirements.txt"
]

def create_submission_zip(output_zip_name="i210735_i210708_Project.zip"):
    print("Checking required course deliverables against assignment manual...")
    missing = []
    for f in REQUIRED_FILES:
        if not os.path.exists(f):
            missing.append(f)
            
    if missing:
        print("ERROR: Missing the following required deliverables:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    # Check report.pdf
    pdf_path = "report/report.pdf"
    if not os.path.exists(pdf_path):
        if os.path.exists("report.pdf"):
            pdf_path = "report.pdf"
        else:
            print(f"NOTICE: Neither 'report/report.pdf' nor 'report.pdf' found yet.")
            print(f"        Compile 'report/report.tex' on Overleaf using 'report_overleaf.zip',")
            print(f"        place 'report.pdf' into 'report/', and rerun this script.")
            pdf_path = None
    else:
        print(f"Found compiled report: {pdf_path}")

    print(f"\nCompressing submission package to {output_zip_name}...")
    with zipfile.ZipFile(output_zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
        written = set()
        for f in REQUIRED_FILES:
            z.write(f, f)
            written.add(f.replace('\\', '/'))
            print(f"  Added: {f}")

        # Include supplementary metrics and reproducibility scripts when present.
        for pattern in ("results/*.csv", "scripts/*.py"):
            for extra in glob.glob(pattern):
                archive_name = extra.replace('\\', '/')
                if archive_name not in written:
                    z.write(extra, archive_name)
                    written.add(archive_name)
                    print(f"  Added: {archive_name}")
            
        # Add report figures
        for fig in glob.glob("report/figures/*.png"):
            archive_name = fig.replace('\\', '/')
            z.write(fig, archive_name)
            print(f"  Added: {archive_name}")
            
        if pdf_path and os.path.exists(pdf_path):
            z.write(pdf_path, "report.pdf")
            print(f"  Added: report.pdf")
            
    print(f"\nSUCCESS! Submission archive created at: {os.path.abspath(output_zip_name)}")
    print(f"Archive Size: {os.path.getsize(output_zip_name) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "i210735_i210708_Project.zip"
    create_submission_zip(name)
