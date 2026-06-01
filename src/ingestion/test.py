import json
from pathlib import Path

for f in sorted(Path("data/processed/json").rglob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    total = bad = 0
    for p in d["pages"]:
        t = p["text"]
        total += len(t)
        bad += sum(1 for c in t if ord(c) < 32 and c not in "\n\t\r")
    ratio = bad / max(total, 1)
    if ratio > 0.02:
        print(f"{ratio:6.1%}  {f.name}")