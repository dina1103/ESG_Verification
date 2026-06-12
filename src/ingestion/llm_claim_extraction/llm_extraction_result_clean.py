import json
import shutil

JSONL  = r"data\processed\llm_claim_extraction_result.jsonl"
BACKUP = r"data\processed\llm_claim_extraction_result_clean.jsonl"

NULLISH = {None, "", "none", "n/a", "na", "null", "unknown", "not specified"}
YEAR_MIN, YEAR_MAX = 1990, 2050


def is_null(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in NULLISH)


def clean_year(v):
    # list -> latest plausible year (most ambitious target horizon); scalar -> validated; bad -> None
    if is_null(v):
        return None
    if isinstance(v, list):
        ys = [clean_year(x) for x in v]
        ys = [y for y in ys if y is not None]
        return max(ys) if ys else None
    try:
        y = int(str(v).strip()[:4])
    except (ValueError, TypeError):
        return None
    return y if YEAR_MIN <= y <= YEAR_MAX else None


def clean_value(v):
    # list -> first numeric element; scalar -> float; non-numeric -> None
    if is_null(v):
        return None
    if isinstance(v, list):
        for x in v:
            r = clean_value(x)
            if r is not None:
                return r
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace(",", "").replace("%", "").replace("$", "").replace("€", "").replace("£", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def main():
    # back up the original before touching it
    shutil.copyfile(JSONL, BACKUP)
    print(f"backed up original -> {BACKUP}")

    recs = []
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    n = 0
    year_lists = value_lists = bad_years = 0

    for r in recs:
        if r.get("parse_error"):
            continue
        for c in (r.get("parsed_claims") or []):
            n += 1
            for yf in ["baseline_year", "target_year"]:
                raw = c.get(yf)
                if isinstance(raw, list):
                    year_lists += 1
                cleaned = clean_year(raw)
                if (not is_null(raw)) and cleaned is None and not isinstance(raw, list):
                    bad_years += 1
                c[yf + "_clean"] = cleaned

            raw_val = c.get("quantified_value")
            if isinstance(raw_val, list):
                value_lists += 1
            c["quantified_value_clean"] = clean_value(raw_val)

    # overwrite the canonical file in place
    with open(JSONL, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 55)
    print("LIST/SCALAR CLEANUP (in place)")
    print("=" * 55)
    print(f"claims processed:         {n:,}")
    print(f"year lists collapsed:     {year_lists}")
    print(f"value lists collapsed:    {value_lists}")
    print(f"bad scalar years -> null: {bad_years}")
    print(f"\noverwrote {JSONL}")
    print(f"original preserved at {BACKUP}")


if __name__ == "__main__":
    main()