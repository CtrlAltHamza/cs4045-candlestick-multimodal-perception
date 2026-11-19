"""
Helper script to package the final submission ZIP file: i22xxxx_i22yyyy_Project.zip.
Validates all required deliverables specified by the course rubric before compressing.
"""

import os
import zipfile
import sys

REQUIRED_FILES = [
    "notebooks/part1_vision.ipynb",
    "notebooks/part2_nlp_fusion.ipynb",
    "checkpoints/yolo_best.pt",
    "checkpoints/cnn_trend_best.pt",
    "checkpoints/sentiment_best.pt",
    "results/detection_results.csv",
    "results/sentiment_results.csv",
    "results/market_results.csv",
    "results/hyperparameter_log.csv",
    "results/chart_embeddings.npy",
    "results/embedding_index.csv",
    "results/market_basket_daily.csv",
    "README.txt",
    "requirements.txt"
]

def create_submission_zip(output_zip_name="i22xxxx_i22yyyy_Project.zip"):
    print("Checking required course deliverables...")
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
        print(f"NOTICE: '{pdf_path}' not found yet. Please compile 'report/report.tex' (e.g. on Overleaf) and save 'report.pdf' inside 'report/'.")
    else:
        print(f"Found compiled report: {pdf_path}")

    print(f"\nCompressing submission package to {output_zip_name}...")
    with zipfile.ZipFile(output_zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in REQUIRED_FILES:
            z.write(f, f)
            print(f"  Added: {f}")
        if os.path.exists(pdf_path):
            z.write(pdf_path, "report.pdf")
            print(f"  Added: report.pdf")
        if os.path.exists("report/report.tex"):
            z.write("report/report.tex", "report/report.tex")
            print(f"  Added: report/report.tex")
            
    print(f"\nSUCCESS! Submission archive created at: {os.path.abspath(output_zip_name)}")
    print(f"Archive Size: {os.path.getsize(output_zip_name) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "i22xxxx_i22yyyy_Project.zip"
    create_submission_zip(name)
