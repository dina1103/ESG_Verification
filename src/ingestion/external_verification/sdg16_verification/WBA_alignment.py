import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

INPUT_JSONL    = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\llm_claim_extraction_result.jsonl"
SDG_PARQUET    = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\segmentation_esg_sdg"
WBA_CSV        = r"C:\Users\dina_\Desktop\esg_verification_draft\src\ingestion\external_benchmarks\sdg16_verification\indicators.csv"
COMPANY_MAP    = r"C:\Users\dina_\Desktop\esg_verification_draft\src\ingestion\external_benchmarks\sdg16_verification\wba_company_mapping.json"
OUTPUT_CLAIMS  = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\sdg16_governance_claim_level.jsonl"
OUTPUT_SUMMARY = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\sdg16_governance_company_year.json"

# WBA 'Acting Ethically' indicators (measurement_area_id == 'C')
TARGET_AREA_ID = "C"

# threshold on 0-1 indicator score scale
# 0.5 = WBA's midpoint between "Not met" and "Met"
INDICATOR_THRESHOLD = 0.5

# claim-to-indicator keyword mapping (derived from WBA indicator definitions)
INDICATOR_KEYWORDS = {
    15: [  # Personal data protection fundamentals
        "data protection", "data privacy", "personal data", "gdpr",
        "data security", "privacy policy",
    ],
    16: [  # Responsible tax fundamentals
        "tax transparency", "responsible tax", "tax strategy", "tax disclosure",
        "tax practices", "country-by-country tax", "tax policy",
    ],
    17: [  # Anti-bribery and anti-corruption fundamentals
        "anti-corruption", "anti corruption", "anti-bribery", "anti bribery",
        "corruption", "bribery", "kickback", "facilitation payment",
    ],
    18: [  # Responsible lobbying and political engagement fundamentals
        "lobbying", "lobby", "political engagement", "political donation",
        "political contribution", "trade association",
    ],
}

# Type B: lexical leadership/excellence markers (TerraChoice Sin of Vagueness)
STRONG_POSITIVE_KEYWORDS = [
    "zero", "no incidents", "leading", "world-class", "world class",
    "best-in-class", "best in class", "highest standards",
    "100% compliance", "fully compliant", "industry leader",
    "global leader", "exemplary", "comprehensive",
]


def is_assessable_governance_claim(claim):
    # a claim is assessable (a substantive governance assertion, not vague
    # aspiration) if ANY of the following holds:
    #  Type A - quantitative zero on an incident-type metric (e.g. "zero breaches")
    #  Type B - contains a strong-positive leadership marker
    #  Type C - claim_type is achievement or commitment (Step 6 classification)
    #  Type D - carries any quantified value
    #  Type E - cites a reporting framework / standard / code
    # only pure vague narrative ("we value integrity") is excluded.

    # Type A - quantitative zero on incident-type metric
    qv = claim.get("quantified_value")
    metric = (claim.get("metric") or "").lower()
    if qv == 0 and any(kw in metric for kw in ["incident", "violation", "breach", "case"]):
        return True

    # Type B - lexical leadership marker
    claim_text = (claim.get("claim_text") or "").lower()
    if any(kw in claim_text for kw in STRONG_POSITIVE_KEYWORDS):
        return True

    # Type C - achievement or commitment (Step 6 claim_type)
    if claim.get("claim_type") in ("achievement", "commitment"):
        return True

    # Type D - any quantified value
    if qv is not None:
        return True

    # Type E - cites a framework / standard
    fw = str(claim.get("framework_reference") or "").lower().strip()
    if fw and fw not in ("none", "n/a", ""):
        return True

    return False


def match_indicator(claim):
    # check claim text against each indicator's keyword set; return first match
    claim_text = (claim.get("claim_text") or "").lower()
    metric = (claim.get("metric") or "").lower()
    haystack = claim_text + " " + metric
    for indicator_id, keywords in INDICATOR_KEYWORDS.items():
        if any(kw in haystack for kw in keywords):
            return indicator_id
    return None


def load_claims(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for claim in (rec.get("parsed_claims") or []):
                row = {
                    "block_id": rec["block_id"],
                    "company_name": rec["company_name"],
                    "year": rec["year"],
                    **claim,
                }
                # Llama sometimes emits fields as lists / wrong types - coerce
                for fld in ("metric", "unit", "claim_text", "scope", "claim_type",
                            "framework_reference", "geography"):
                    v = row.get(fld)
                    if isinstance(v, list):
                        row[fld] = " ".join(str(x) for x in v)
                    elif v is not None and not isinstance(v, str):
                        row[fld] = str(v)
                # numeric fields -> float or None
                for fld in ("quantified_value", "baseline_value"):
                    v = row.get(fld)
                    if isinstance(v, list):
                        v = v[0] if v else None
                    if v is not None:
                        try:
                            row[fld] = float(v)
                        except (ValueError, TypeError):
                            row[fld] = None
                    else:
                        row[fld] = None
                # year fields -> int or None
                for fld in ("target_year", "baseline_year"):
                    v = row.get(fld)
                    if isinstance(v, list):
                        v = v[0] if v else None
                    if v is not None:
                        try:
                            row[fld] = int(float(v))
                        except (ValueError, TypeError):
                            row[fld] = None
                    else:
                        row[fld] = None
                rows.append(row)
    return rows


def load_sdg16_block_ids(parquet_path):
    # load SDG classification, aggregate to paragraph level
    # return set of block_ids (= source_document + "__" + paragraph_id) that contain any SDG-16 sentence
    df = pd.read_parquet(parquet_path)
    df_sdg16 = df[df["sdg_label"] == "sdg16"]
    # block_id reconstruction matches Step 6's format
    block_ids = set(
        df_sdg16["source_document"].astype(str) + "__" + df_sdg16["paragraph_id"].astype(str)
    )
    return block_ids


def load_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_wba_indicators(csv_path, mapping):
    # read WBA indicators CSV, filter to Acting Ethically area
    # return: company → indicator_id → score
    df = pd.read_csv(csv_path)
    df = df[df["measurement_area_id"] == TARGET_AREA_ID]

    result = {}
    for our_name, entry in mapping.items():
        wba_name = entry.get("wba_name")
        if not wba_name:
            result[our_name] = None
            continue

        rows = df[df["company_name"].astype(str).str.strip() == wba_name.strip()]
        if len(rows) == 0:
            result[our_name] = None
            continue

        indicators = {}
        for _, row in rows.iterrows():
            ind_id = int(row["indicator_id"])
            indicators[ind_id] = {
                "score": float(row["indicator_score"]),
                "name": str(row["indicator_name"]),
            }

        result[our_name] = {
            "wba_name": wba_name,
            "methodology_year": int(rows.iloc[0]["methodology_year"]),
            "indicators": indicators,
        }

    return result


def assess_governance_claim(claim, wba_data):
    # B1 logic: the gate is whether a governance claim matches a WBA Acting-Ethically
    # indicator. The earlier "strong positive" lexical/quantitative gate rejected
    # ~99% of claims and produced no discriminating score, so it is removed.
    # A matched claim is then verified against WBA's independent indicator score:
    # plausible if WBA rates the company well on that indicator, flagged if poorly.
    if not wba_data:
        return "no_wba_data", None, "company not in WBA Social Benchmark"

    # inclusive assessability gate - excludes only vague narrative claims
    if not is_assessable_governance_claim(claim):
        return "weak_claim", None, "vague/narrative claim - not a substantive governance assertion"

    indicator_id = match_indicator(claim)
    if indicator_id is None:
        return "no_indicator_match", None, "claim does not match any Acting Ethically indicator topic"

    indicators = wba_data.get("indicators", {})
    ind = indicators.get(indicator_id)
    if not ind:
        return "no_indicator_score", indicator_id, f"company has no score for indicator {indicator_id}"

    score = ind["score"]
    if score >= INDICATOR_THRESHOLD:
        return "plausible", indicator_id, f"governance claim matches WBA Indicator {indicator_id}, which WBA scores {score:.2f}/1.0 (>= {INDICATOR_THRESHOLD})"
    return "flagged", indicator_id, f"governance claim matches WBA Indicator {indicator_id}, but WBA scores it only {score:.2f}/1.0 (< {INDICATOR_THRESHOLD})"


def main():
    print(f"Loading claims from {INPUT_JSONL}...")
    all_claims = load_claims(INPUT_JSONL)
    print(f"  loaded {len(all_claims):,} claims")

    print(f"\nLoading SDG-16 block IDs from {SDG_PARQUET}...")
    sdg16_blocks = load_sdg16_block_ids(SDG_PARQUET)
    print(f"  {len(sdg16_blocks):,} SDG-16 paragraph IDs")

    # filter claims by upstream SDG-16 paragraph classification
    governance_claims = [c for c in all_claims if c["block_id"] in sdg16_blocks]
    print(f"  governance claims (SDG-16 paragraphs): {len(governance_claims):,}")

    print(f"\nLoading WBA mapping from {COMPANY_MAP}...")
    mapping = load_mapping(COMPANY_MAP)

    print(f"Loading WBA indicators from {WBA_CSV}...")
    wba_data = load_wba_indicators(WBA_CSV, mapping)
    for company, data in wba_data.items():
        if data:
            print(f"  {company} ({data['wba_name']}):")
            for ind_id in sorted(data["indicators"]):
                ind = data["indicators"][ind_id]
                print(f"    Indicator {ind_id}: {ind['score']:.2f}/1.0  ({ind['name']})")
        else:
            print(f"  {company}: no WBA data")

    groups = defaultdict(list)
    for c in governance_claims:
        groups[(c["company_name"], c["year"])].append(c)

    Path(OUTPUT_CLAIMS).parent.mkdir(parents=True, exist_ok=True)
    per_claim_results = []
    per_group_summary = {}

    for (company, year), claims in sorted(groups.items()):
        data = wba_data.get(company)

        verdicts = []
        for claim in claims:
            verdict, indicator_id, reason = assess_governance_claim(claim, data)
            verdicts.append(verdict)
            per_claim_results.append({
                "company_name": company,
                "year": year,
                "block_id": claim["block_id"],
                "claim_text": claim.get("claim_text"),
                "claim_type": claim.get("claim_type"),
                "metric": claim.get("metric"),
                "quantified_value": claim.get("quantified_value"),
                "is_assessable": is_assessable_governance_claim(claim),
                "matched_indicator": indicator_id,
                "verdict": verdict,
                "reason": reason,
            })

        n_total = len(claims)
        n_plausible = sum(1 for v in verdicts if v == "plausible")
        n_flagged = sum(1 for v in verdicts if v == "flagged")
        n_weak = sum(1 for v in verdicts if v == "weak_claim")
        n_no_match = sum(1 for v in verdicts if v == "no_indicator_match")
        n_no_score = sum(1 for v in verdicts if v == "no_indicator_score")
        n_no_wba = sum(1 for v in verdicts if v == "no_wba_data")

        denom = n_plausible + n_flagged
        if denom > 0:
            score = round(n_plausible / denom, 4)
            score_note = None
        elif n_no_wba == n_total:
            score, score_note = None, "no_wba_data_for_company"
        elif n_weak == n_total:
            score, score_note = None, "no_strong_positive_claims"
        elif n_no_match == n_total:
            score, score_note = None, "no_claims_matched_indicators"
        else:
            score, score_note = None, "no_assessable_claims"

        per_group_summary[f"{company}__{year}"] = {
            "company_name": company,
            "year": year,
            "wba_name": data["wba_name"] if data else None,
            "wba_methodology_year": data["methodology_year"] if data else None,
            "wba_indicator_scores": {f"ind_{i}": data["indicators"][i]["score"] for i in sorted(data["indicators"])} if data else None,
            "n_governance_claims": n_total,
            "n_plausible": n_plausible,
            "n_flagged": n_flagged,
            "n_weak_claim": n_weak,
            "n_no_indicator_match": n_no_match,
            "n_no_indicator_score": n_no_score,
            "n_no_wba_data": n_no_wba,
            "sdg16_alignment_score": score,
            "score_note": score_note,
        }

    with open(OUTPUT_CLAIMS, "w", encoding="utf-8") as f:
        for rec in per_claim_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(per_group_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Per-claim verdicts:    {OUTPUT_CLAIMS}")
    print(f"Per-company-year:      {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()