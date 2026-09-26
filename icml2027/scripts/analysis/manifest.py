#!/usr/bin/env python
import argparse
import hashlib
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
a = ap.parse_args()
R = Path(a.results)
lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(R)}" for p in sorted(R.rglob("*"))
         if p.is_file() and p.suffix in {".json", ".jsonl", ".csv", ".yaml", ".md", ".pdf"} and p.name != "MANIFEST.sha256"]
(R / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
print(len(lines), "files")
