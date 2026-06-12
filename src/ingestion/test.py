import json
import shutil

JSONL  = r"data\processed\llm_claim_extraction_result.jsonl"
BACKUP = r"data\processed\llm_claim_extraction_result_preflatten.jsonl"

CLEAN_FIELDS = ["target_year", "baseline_year", "quantified_value"]


def main():
    shutil.copyfile(JSONL, BACKUP)
    print(f"backed up -> {BACKUP}")

    recs = []
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    folded = 0
    for r in recs:
        if r.get("parse_error"):
            continue
        for c in (r.get("parsed_claims") or []):
            for base in CLEAN_FIELDS:
                ck = base + "_clean"
                if ck in c:
                    c[base] = c.pop(ck)   # move cleaned value into the original field, drop _clean
                    folded += 1

    with open(JSONL, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"folded {folded:,} _clean fields back into their original fields")
    print(f"overwrote {JSONL}")
    print(f"original (with _clean fields) preserved at {BACKUP}")

    # verify: no _clean fields left, originals now scalar
    claims = [c for r in recs if not r.get("parse_error") for c in (r.get("parsed_claims") or [])]
    leftover = sum(1 for c in claims for k in c if k.endswith("_clean"))
    lists = sum(1 for c in claims if any(isinstance(c.get(b), list) for b in CLEAN_FIELDS))
    print(f"\nverification: _clean fields remaining: {leftover} | list-valued originals remaining: {lists}")


if __name__ == "__main__":
    main()