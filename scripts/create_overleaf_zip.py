"""
Create a lightweight Overleaf upload bundle (overleaf_upload.zip):
- main.tex at the root and all figures under figures/
- figures are downscaled to at most 1800 px wide and saved as 256-colour PNGs so the
  project compiles within the Overleaf free-plan time limit.
Upload with Overleaf: New Project -> Upload Project / Existing project (.zip), then Recompile.
"""
import glob, io, os, zipfile
from PIL import Image

MAX_W = 1800
out = "overleaf_upload.zip"
total_before = total_after = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write("report/main.tex", "main.tex")
    for f in sorted(glob.glob("report/figures/*.png")):
        im = Image.open(f).convert("RGB")
        if im.width > MAX_W:
            im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(buf, "PNG", optimize=True)
        total_before += os.path.getsize(f); total_after += buf.tell()
        z.writestr("figures/" + os.path.basename(f), buf.getvalue())
        print(f"{os.path.basename(f):35s} {os.path.getsize(f)/1024:7.0f} KB -> {buf.tell()/1024:6.0f} KB  ({im.width}x{im.height})")
print(f"figures: {total_before/1024/1024:.1f} MB -> {total_after/1024/1024:.1f} MB | zip {os.path.getsize(out)/1024/1024:.1f} MB")
