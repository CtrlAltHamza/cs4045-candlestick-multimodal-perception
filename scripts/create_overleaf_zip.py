"""
Create a clean zip file of the report and all figures for one-click upload to Overleaf.
"""

import os
import zipfile
import glob

def make_overleaf_zip():
    zip_path = "report_overleaf.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write("report/report.tex", "report.tex")
        figures = glob.glob("report/figures/*.png")
        for fig in figures:
            arcname = os.path.relpath(fig, "report")
            z.write(fig, arcname)
            print(f"Added to Overleaf bundle: {arcname}")
    print(f"\nSUCCESS! Created Overleaf upload bundle at: {os.path.abspath(zip_path)}")
    print("Instructions: On Overleaf (https://www.overleaf.com), click 'New Project' -> 'Upload Project' and upload this zip.")

if __name__ == "__main__":
    make_overleaf_zip()
