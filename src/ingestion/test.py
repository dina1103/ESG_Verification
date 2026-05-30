import json
from pathlib import Path

JSON_DIR = Path(r"C:\Users\dell\ESG_Verification\data\processed\json")

n_docs = tot = ext = ocr = skip = 0
for f in JSON_DIR.rglob("*.json"):
    d = json.loads(f.read_text(encoding="utf-8"))
    n_docs += 1
    tot  += d["total_pages"]
    ext  += d["pages_extracted"]
    ocr  += d["pages_ocr"]
    skip += d["pages_skipped"]

if n_docs == 0:
    print(f"No JSON found under {JSON_DIR} — check the path.")
else:
    print(f"Documents:       {n_docs}")
    print(f"Total pages:     {tot}")
    print(f"Pages extracted: {ext} ({ext/tot:.1%})")
    print(f"Pages via OCR:   {ocr}")
    print(f"Pages skipped:   {skip} ({skip/tot:.1%})")