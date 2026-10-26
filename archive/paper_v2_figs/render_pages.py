"""Render PDF pages to PNG for visual checks: render_pages.py paper.pdf outdir [pages...] (uses pymupdf from a scratch pylib)."""
import sys, os
sys.path.insert(0, "/tmp/claude-2292/-homes-lbastian-latent-dynamics/ff433239-7ead-4777-86c4-f8c05d4a450b/scratchpad/pylib")
import fitz
d = fitz.open(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
pages = [int(x) for x in sys.argv[3:]] or range(1, len(d) + 1)
for p in pages:
    d[p - 1].get_pixmap(dpi=80).save(f"{out}/p{p:02d}.png")
