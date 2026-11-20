"""
Create a clean Overleaf upload bundle:
- Main file is 'main.tex' (what Overleaf expects as default entry point)
- All figures in 'figures/' subfolder
Instructions for Overleaf:
  1. Go to https://www.overleaf.com
  2. Click 'New Project' -> 'Upload Project'
  3. Upload this zip file (overleaf_upload.zip)
  4. Click 'Recompile'
  5. Download PDF
"""

import os
import zipfile
import glob

def make_overleaf_zip():
    zip_path = "overleaf_upload.zip"
    added = []
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        # Main .tex file named as main.tex (Overleaf default entry point)
        z.write("report/main.tex", "main.tex")
        added.append("main.tex")

        # All figures in figures/ subfolder
        figures = sorted(glob.glob("report/figures/*.png"))
        for fig in figures:
            arcname = "figures/" + os.path.basename(fig)
            z.write(fig, arcname)
            added.append(arcname)

    print(f"SUCCESS: Created {os.path.abspath(zip_path)} ({os.path.getsize(zip_path)/1024/1024:.1f} MB)")
    print(f"Contents ({len(added)} files):")
    for f in added:
        print(f"  {f}")
    print()
    print("--- OVERLEAF INSTRUCTIONS ---")
    print("1. Go to https://www.overleaf.com")
    print("2. Click 'New Project' -> 'Upload Project' (NOT 'Upload File')")
    print("3. Select overleaf_upload.zip and confirm")
    print("4. Click 'Recompile' (green button, top right)")
    print("5. Download PDF via the download button (downward arrow icon)")
    print("6. Save it as report/report.pdf in d:\\dlp_project\\")
    print("7. Run: python scripts/build_submission_zip.py")

if __name__ == "__main__":
    make_overleaf_zip()
