"""Print page count and first line(s) per page of a PDF (pypdf)."""
import sys
from pypdf import PdfReader
r = PdfReader(sys.argv[1])
print("pages", len(r.pages))
for i, p in enumerate(r.pages, 1):
    t = (p.extract_text() or "")
    key = [l for l in t.split("\n") if l.strip()][:2]
    hit = [m for m in ("Conclusion", "Impact Statement", "References") if m in t]
    print(i, key, hit)
