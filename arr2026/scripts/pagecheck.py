"""Measure where COUNTED CONTENT ends -- not where a heading appears.

ARR counts everything before Limitations against the 8-page limit. Two traps,
both hit in this project: testing which page the HEADING lands on reported
compliance while 486 chars of body sat on p9; and `\bLimitations\b` fails
because the review line number attaches to the word ("Limitations663").
"""
import re, sys
from pypdf import PdfReader

LIMIT = 8
r = PdfReader(sys.argv[1] if len(sys.argv) > 1 else "main.pdf")
lim_page = lead = None
for i, pg in enumerate(r.pages):
    t = pg.extract_text() or ""
    m = re.search(r"Limitations", t)       # no trailing \b: line numbers attach
    if m:
        lim_page, lead = i + 1, t[:m.start()]
        break
if lim_page is None:
    print("Limitations heading not found"); sys.exit(2)

# Anything left after dropping whitespace and line numbers is real body text.
residual = re.sub(r"[\s\d]+", "", lead)
end = lim_page if len(residual) >= 3 else lim_page - 1
ok = end <= LIMIT
print(f"{'OK' if ok else 'OVER'}: counted content ends on page {end} (limit {LIMIT}); "
      f"Limitations begins page {lim_page}; body text preceding it on that page: "
      f"{len(residual)} chars {residual[:60]!r}")
sys.exit(0 if ok else 1)
