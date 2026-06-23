import json
from pathlib import Path
from collections import defaultdict

WBA_SUMMARY     = r"data\processed\sdg16_governance_company_year.json"  
PENALTY_SUMMARY = r"data\processed\sdg16_penalty_company_year.json"      
OUTPUT_CY       = r"data\processed\sdg16_company_year.json"
OUTPUT_COMPANY  = r"data\processed\sdg16_company.json"

WBA_WEIGHT = 0.25     

ONLY_ONE_SIGNAL = "insufficient_info"

MIN_YEARS_FOR_COMPANY_SCORE = 3  # below this, the company average is flagged thin


def combine(wba, pen_score):
    return round(WBA_WEIGHT * wba + (1 - WBA_WEIGHT) * pen_score, 4)


def main():
    wba = json.load(open(WBA_SUMMARY, encoding="utf-8"))
    pen = json.load(open(PENALTY_SUMMARY, encoding="utf-8"))
    keys = sorted(set(wba) | set(pen))

    per_cy = {}
    by_company = defaultdict(list)
    for k in keys:
        w = wba.get(k, {})
        p = pen.get(k, {})
        company = w.get("company_name") or p.get("company_name")
        year = w.get("year") or p.get("year")
        wba_s = w.get("sdg16_alignment_score")
        pen_s = p.get("penalty_score")

        if wba_s is not None and pen_s is not None:
            combined, note = combine(wba_s, pen_s), None
        elif wba_s is None and pen_s is None:
            combined, note = None, "no_signal"
        elif ONLY_ONE_SIGNAL == "use_available":
            combined = round(wba_s if wba_s is not None else pen_s, 4)
            note = "penalty_signal_unavailable_wba_only" if wba_s is not None else "wba_signal_unavailable_penalty_only"
        else:  # "insufficient_info": a combined score needs both lenses
            combined, note = None, "insufficient_info_single_signal"

        per_cy[k] = {
            "company_name": company, "year": year,
            "wba_alignment_score": wba_s,
            "penalty_score": pen_s,
            "n_governance_penalties": p.get("n_governance_penalties"),
            "sdg16_combined_score": combined,
            "score_note": note,
        }
        if combined is not None:
            by_company[company].append(combined)

    per_company = {}
    for company, scores in sorted(by_company.items()):
        per_company[company] = {
            "company_name": company,
            "n_scored_years": len(scores),
            "sdg16_company_score": round(sum(scores) / len(scores), 4),
            "thin_coverage": len(scores) < MIN_YEARS_FOR_COMPANY_SCORE,
        }

    Path(OUTPUT_CY).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CY, "w", encoding="utf-8") as f:
        json.dump(per_cy, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_COMPANY, "w", encoding="utf-8") as f:
        json.dump(per_company, f, indent=2, ensure_ascii=False)

    # ---- print each company's combined score to the terminal ---------------
    print(f"{'='*70}\nSDG-16 COMBINED SCORE BY COMPANY  ({WBA_WEIGHT:.2f}*WBA + {1-WBA_WEIGHT:.2f}*penalty)\n{'='*70}")
    for company in sorted(per_company, key=lambda c: -per_company[c]["sdg16_company_score"]):
        pc = per_company[company]
        flag = "  [thin]" if pc["thin_coverage"] else ""
        print(f"\n{company}  ->  {pc['sdg16_company_score']:.3f}  ({pc['n_scored_years']}y){flag}")
        rows = sorted((v for v in per_cy.values() if v["company_name"] == company), key=lambda r: r["year"])
        for r in rows:
            if r["sdg16_combined_score"] is None:
                print(f"   {r['year']}:   None   [{r['score_note']}]")
            else:
                print(f"   {r['year']}:  {r['sdg16_combined_score']:.3f}   wba={r['wba_alignment_score']}  pen={r['penalty_score']}")
    # companies with no scored cell at all (e.g. Aston Martin, Ferrari)
    unscored = sorted({v["company_name"] for v in per_cy.values()} - set(per_company))
    for company in unscored:
        print(f"\n{company}  ->  None  (no scored company-years)")

    print(f"\n{'='*70}\nCOMPLETE\n{'='*70}")
    print(f"Per-company-year combined: {OUTPUT_CY}")
    print(f"Per-company final:         {OUTPUT_COMPANY}")


if __name__ == "__main__":
    main()