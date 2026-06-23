import json
import math
from pathlib import Path
from collections import defaultdict
import pandas as pd

INPUT_JSONL    = r"data/processed/llm_claim_extraction_result.jsonl"
SDG_PARQUET    = r"data/processed/segments_esg_sdg.parquet"
VT_XLSX        = r"src/ingestion/external_benchmarks/sdg16_verification/Violation_Tracker_Global_Penalty_Records.xlsx"
OUTPUT_SUMMARY = r"data/processed/sdg16_penalty_company_year.json"

# SDG-16 governance/integrity offense categories. Environmental belongs to SDG-13;
# safety/labour are outside governance scope. This is the load-bearing scope choice.
GOVERNANCE_CATEGORIES = {
    "anti-competitive practices",
    "anti-money-laundering deficiencies",
    "accounting fraud or deficiencies",
    "investor protection violation",
    "government contracting fraud",
    "banking violation",
    "insurance violation",
    "unfair commercial practices",
    "consumer protection violation",
    "privacy violation",
}

# corpus company -> Violation Tracker 'Current Parent'. A parent that appears in the
# VT file with no governance penalties is CONFIRMED CLEAN (score 1.0); a company
# whose parent is absent from the file (None below, or a name not in the file) has
# NO INFORMATION and abstains (score None). Subaru and Ferrari carry a zero-penalty
# marker row in the file -> confirmed clean. Aston Martin is absent -> no information.
VT_PARENT_MAP = {
    "Bayerische Motoren Werke AG":              "BMW",
    "Volkswagen AG":                            "Volkswagen",
    "Ford Motor Company":                       "Ford Motor",
    "General Motors Company":                   "General Motors",
    "Nissan Motor Co., Ltd.":                   "Nissan",
    "Toyota Motor Corporation":                 "Toyota",
    "Tesla, Inc.":                              "Tesla Inc.",
    "Subaru Corporation":                       "Subaru Corporation",
    "Ferrari N.V.":                             "Ferrari N.V.",
    "Aston Martin Lagonda Global Holdings PLC": None,
}

# Dollar-severity scoring (replaces the count-based 1 - step*n):
#   score = max(0, 1 - log10(1 + total_usd/UNIT) / log10(1 + CAP/UNIT))
# higher = cleaner. Log-scaled so MAGNITUDE drives the score, not the count: a $75k
# fine barely moves it, a $700M record floors it. UNIT is the "noise floor" below
# which penalties are treated as negligible; CAP is the total at which the score
# reaches ~0. Both are absolute anchors (not corpus-relative), so the score does not
# shift if a company is added or removed.
UNIT_USD = 1_000_000      # $1M   - severity noise floor (fines below ~this barely register)
CAP_USD  = 1_000_000_000  # $1B   - severity saturates here (score ~0)
FREQ_STEP = 0.05          # frequency: credit lost per distinct penalty (20 -> 0)

# A penalty counts only if on the public record by the report year
# (penalty_year <= report_year). None = cumulative to date; int = rolling look-back.
VIOLATION_LOOKBACK_YEARS = None

_DENOM = math.log10(1 + CAP_USD / UNIT_USD)


def severity_score(total_usd):
    # HOW MUCH: 0 -> 1.0 (clean), CAP -> ~0.0, log-scaled in between
    if total_usd <= 0:
        return 1.0
    return round(max(0.0, 1.0 - math.log10(1 + total_usd / UNIT_USD) / _DENOM), 4)


def frequency_score(n_penalties):
    # HOW OFTEN: 1.0 at zero penalties, linear decay per distinct penalty
    return round(max(0.0, 1.0 - FREQ_STEP * n_penalties), 4)


def load_claims_companyyears(jsonl_path, parquet_path):
    # the SDG-16 company-year grid = company-years that have SDG-16 governance claims,
    # so the penalty score aligns 1:1 with the WBA / enforcement layers
    df = pd.read_parquet(parquet_path)
    df = df[df["sdg_label"] == "sdg16"]
    sdg16_blocks = set(df["source_document"].astype(str) + "__" + df["paragraph_id"].astype(str))
    cells = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec["block_id"] in sdg16_blocks:
                cells.add((rec["company_name"], rec["year"]))
    return sorted(cells)


def load_violations(xlsx_path):
    # returns (by_parent, covered_parents). covered_parents = every parent present in
    # the file (incl. zero-penalty marker rows) = the set that was actually searched.
    df = pd.read_excel(xlsx_path)
    covered_parents = set(df["Current Parent"].dropna().astype(str).str.strip())
    df = df[df["Offense Category"].isin(GOVERNANCE_CATEGORIES)].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    by_parent = defaultdict(lambda: defaultdict(list))
    for _, r in df.iterrows():
        if pd.isna(r["Year"]):
            continue
        by_parent[str(r["Current Parent"]).strip()][int(r["Year"])].append({
            "category": r["Offense Category"],
            "penalty_usd": float(r["Penalty Amount (USD)"]) if pd.notna(r["Penalty Amount (USD)"]) else None,
        })
    return by_parent, covered_parents


def penalties_on_record(parent_record, report_year):
    hits = []
    for vyear, recs in parent_record.items():
        if vyear > report_year:
            continue
        if VIOLATION_LOOKBACK_YEARS is not None and vyear < report_year - VIOLATION_LOOKBACK_YEARS:
            continue
        hits.extend(recs)
    return hits


def main():
    print(f"Loading SDG-16 company-year grid from claims/segments...")
    cells = load_claims_companyyears(INPUT_JSONL, SDG_PARQUET)
    print(f"  {len(cells)} SDG-16 company-years")

    print(f"\nLoading Violation Tracker records from {VT_XLSX}...")
    by_parent, covered_parents = load_violations(VT_XLSX)
    print(f"  parents searched (in file): {sorted(covered_parents)}")
    print(f"  parents with governance penalties: {sorted(by_parent)}")
    print(f"  severity scale: UNIT=${UNIT_USD:,}  CAP=${CAP_USD:,}")

    summary = {}
    for company, year in cells:
        parent = VT_PARENT_MAP.get(company)

        # no information: parent unmapped, or mapped name not present in the file
        if parent is None or parent not in covered_parents:
            summary[f"{company}__{year}"] = {
                "company_name": company, "year": year, "vt_parent": parent,
                "n_governance_penalties": 0, "total_penalty_usd": 0.0,
                "penalty_score": None, "score_note": "no_information_not_in_violation_tracker",
            }
            continue

        # searched and present in the file -> score by total penalty dollars on record
        hits = penalties_on_record(by_parent.get(parent, {}), year)
        n = len(hits)
        total_usd = sum(h["penalty_usd"] for h in hits if h["penalty_usd"])
        sev = severity_score(total_usd)      # how much (dollars)
        freq = frequency_score(n)            # how often (count)
        score = min(sev, freq)               # worst dimension sets the conduct score
        note = "confirmed_no_governance_penalties" if parent not in by_parent else None
        summary[f"{company}__{year}"] = {
            "company_name": company, "year": year, "vt_parent": parent,
            "n_governance_penalties": n,
            "total_penalty_usd": round(total_usd, 2),
            "severity_score": sev,
            "frequency_score": freq,
            "penalty_score": score,  # min(severity, frequency); higher = better conduct
            "binding_dimension": "severity" if sev <= freq else "frequency",
            "score_note": note,
        }

    Path(OUTPUT_SUMMARY).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- print each company's penalty score to the terminal ----------------
    print(f"\n{'='*70}\nPENALTY SCORE BY COMPANY = min(severity, frequency)  (higher = better conduct)\n{'='*70}")
    by_company = defaultdict(list)
    for v in summary.values():
        by_company[v["company_name"]].append(v)
    for company in sorted(by_company):
        rows = sorted(by_company[company], key=lambda r: r["year"])
        scores = [r["penalty_score"] for r in rows if r["penalty_score"] is not None]
        avg = f"{sum(scores) / len(scores):.3f}" if scores else "n/a"
        print(f"\n{company}  (avg {avg})")
        for r in rows:
            if r["penalty_score"] is None:
                print(f"   {r['year']}:   None   penalties={r['n_governance_penalties']:>2}   ${r['total_penalty_usd']:>14,.0f}  [{r['score_note']}]")
            else:
                bind = r["binding_dimension"]
                print(f"   {r['year']}:  {r['penalty_score']:.3f}   sev={r['severity_score']:.3f} freq={r['frequency_score']:.3f} (<-{bind})   penalties={r['n_governance_penalties']:>2}   ${r['total_penalty_usd']:>14,.0f}")

    print(f"\n{'='*70}\nCOMPLETE\n{'='*70}")
    print(f"Per-company-year penalty score: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()