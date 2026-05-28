import json
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd

INPUT_JSONL    = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\llm_claim_extraction_result.jsonl"
SDG_PARQUET    = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\segmentation_esg_sdg"
SBTI_XLSX      = r"C:\Users\dina_\Desktop\esg_verification_draft\src\ingestion\external_benchmarks\sdg13_verification\targets-excel.xlsx"
COMPANY_MAP    = r"C:\Users\dina_\Desktop\esg_verification_draft\src\ingestion\external_benchmarks\sdg13_verification\sbti_company_mapping.json"
OUTPUT_CLAIMS  = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\sdg13_climate_claim_level.jsonl"
OUTPUT_SUMMARY = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\sdg13_climate_company_year.json"

SBTI_SHEET = "WebsiteData"

# tolerance bounds for matching to SBTi
PCT_TOLERANCE = 5
YEAR_TOLERANCE = 2

# intensity indicators on the claim side
INTENSITY_PATTERNS = [
    r"\bper\b",
    r"/",
    r"\bintensity\b",
    r"\bper vehicle\b",
    r"\bper kilometer\b",
    r"\bper km\b",
    r"\bper unit\b",
    r"\bper produced\b",
]


def is_intensity_claim(claim):
    metric_text = (claim.get("metric") or "").lower()
    unit_text = (claim.get("unit") or "").lower().strip()
    claim_text = (claim.get("claim_text") or "").lower()
    combined = metric_text + " " + claim_text + " " + unit_text
    for pat in INTENSITY_PATTERNS:
        if re.search(pat, combined):
            return True
    return False


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


def load_sdg13_block_ids(parquet_path):
    # paragraphs containing any sentence classified as SDG-13 by upstream classifier
    df = pd.read_parquet(parquet_path)
    df_sdg13 = df[df["sdg_label"] == "sdg13"]
    return set(
        df_sdg13["source_document"].astype(str) + "__" + df_sdg13["paragraph_id"].astype(str)
    )


def load_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sbti_data(xlsx_path, mapping):
    df = pd.read_excel(xlsx_path, sheet_name=SBTI_SHEET)

    result = {}
    for our_name, entry in mapping.items():
        sbti_name = entry.get("sbti_name")
        if not sbti_name:
            result[our_name] = {"target": None, "commitment_removed_year": None, "status": entry.get("status")}
            continue

        target_rows = df[
            (df["company_name"] == sbti_name)
            & (df["action"] == "Target")
            & (df["target"] == "Near-term")
            & (df["scope"].astype(str) == "1+2")
        ]

        target = None
        if len(target_rows) > 0:
            row = target_rows.iloc[0]
            target_pct = float(row["target_value"]) * 100
            date_pub = pd.to_datetime(row["date_published"])
            target = {
                "target_pct": target_pct,
                "base_year": int(row["base_year"]),
                "target_year": int(row["target_year"]),
                "classification": str(row["target_classification_short"]),
                "year_type": str(row["year_type"]),
                "is_intensity": entry.get("is_intensity", False),
                "intensity_unit": entry.get("intensity_unit"),
                "validation_year": int(date_pub.year),
                "sbti_name": sbti_name,
            }

        removed_rows = df[
            (df["company_name"] == sbti_name)
            & (df["action"] == "Commitment")
            & (df["status"] == "Removed")
        ]
        commitment_removed_year = None
        if len(removed_rows) > 0:
            row = removed_rows.iloc[0]
            commit_date = pd.to_datetime(row["date_published"])
            commitment_removed_year = entry.get("commitment_removed_year") or (commit_date.year + 2)

        result[our_name] = {
            "target": target,
            "commitment_removed_year": commitment_removed_year,
            "status": entry.get("status"),
        }

    return result


def check_intensity_match(claim_is_intensity, target_is_intensity):
    return claim_is_intensity == target_is_intensity


def assess_target_claim(claim, sbti_data):
    target = sbti_data.get("target")
    status = sbti_data.get("status")
    report_year = claim.get("year")

    if not target:
        removed_year = sbti_data.get("commitment_removed_year")
        if status == "Commitment removed" and removed_year and report_year >= removed_year:
            return "commitment_removed", f"company's SBTi commitment was removed in {removed_year}"
        return "no_sbti_target", "no SBTi target on file for this company"

    if report_year < target["validation_year"]:
        return "target_not_yet_validated", \
               f"report year {report_year} predates SBTi validation in {target['validation_year']}"

    claim_is_intensity = is_intensity_claim(claim)
    target_is_intensity = target.get("is_intensity", False)

    if not check_intensity_match(claim_is_intensity, target_is_intensity):
        reason = (f"intensity mismatch: claim={'intensity' if claim_is_intensity else 'absolute'}, "
                  f"target={'intensity' if target_is_intensity else 'absolute'}")
        return "different_metric", reason

    claim_pct = claim.get("quantified_value")
    claim_year = claim.get("target_year")

    if claim_pct is None or claim_year is None:
        return "no_quantification", "missing reduction % or target year"

    pct_diff = claim_pct - target["target_pct"]
    year_diff = abs(claim_year - target["target_year"])

    if year_diff > YEAR_TOLERANCE:
        return "different_metric", f"target_year diff = {year_diff} (>{YEAR_TOLERANCE})"

    if abs(pct_diff) <= PCT_TOLERANCE:
        return "matches_sbti", f"pct_diff = {round(pct_diff, 2)}, year_diff = {year_diff}"

    if pct_diff > 0:
        return "stronger_than_sbti", f"claim {claim_pct}% vs sbti {target['target_pct']}%"
    return "weaker_than_sbti", f"claim {claim_pct}% vs sbti {target['target_pct']}%"


def assess_achievement_claim(claim, sbti_data):
    target = sbti_data.get("target")
    status = sbti_data.get("status")
    report_year = claim.get("year")

    if not target:
        removed_year = sbti_data.get("commitment_removed_year")
        if status == "Commitment removed" and removed_year and report_year >= removed_year:
            return "commitment_removed", f"company's SBTi commitment was removed in {removed_year}"
        return "no_sbti_target", "no SBTi target on file for this company"

    if report_year < target["validation_year"]:
        return "target_not_yet_validated", \
               f"report year {report_year} predates SBTi validation in {target['validation_year']}"

    claim_is_intensity = is_intensity_claim(claim)
    target_is_intensity = target.get("is_intensity", False)

    if not check_intensity_match(claim_is_intensity, target_is_intensity):
        reason = (f"intensity mismatch: claim={'intensity' if claim_is_intensity else 'absolute'}, "
                  f"target={'intensity' if target_is_intensity else 'absolute'}")
        return "different_metric", reason

    claim_pct = claim.get("quantified_value")
    baseline_year = claim.get("baseline_year") or target["base_year"]

    if claim_pct is None or baseline_year is None:
        return "no_quantification", "missing reduction % or baseline year"

    years_elapsed = report_year - baseline_year
    target_duration = target["target_year"] - baseline_year

    if target_duration <= 0 or years_elapsed < 0:
        return "different_metric", "invalid year math"

    expected_pct = target["target_pct"] * years_elapsed / target_duration
    diff = claim_pct - expected_pct

    if abs(diff) <= PCT_TOLERANCE:
        return "on_track", f"claim {claim_pct}% vs expected {round(expected_pct, 1)}% at year {report_year}"
    if diff > 0:
        return "ahead", f"claim {claim_pct}% vs expected {round(expected_pct, 1)}% (ahead by {round(diff, 1)})"
    return "behind", f"claim {claim_pct}% vs expected {round(expected_pct, 1)}% (behind by {round(-diff, 1)})"


def main():
    print(f"Loading claims from {INPUT_JSONL}...")
    all_claims = load_claims(INPUT_JSONL)
    print(f"  loaded {len(all_claims):,} claims")

    print(f"\nLoading SDG-13 block IDs from {SDG_PARQUET}...")
    sdg13_blocks = load_sdg13_block_ids(SDG_PARQUET)
    print(f"  {len(sdg13_blocks):,} SDG-13 paragraph IDs")

    climate_claims = [c for c in all_claims if c["block_id"] in sdg13_blocks]
    print(f"  climate claims (SDG-13 paragraphs): {len(climate_claims):,}")

    print(f"\nLoading SBTi mapping from {COMPANY_MAP}...")
    mapping = load_mapping(COMPANY_MAP)

    print(f"Loading SBTi data from {SBTI_XLSX}...")
    sbti_data = load_sbti_data(SBTI_XLSX, mapping)
    for company, data in sbti_data.items():
        target = data.get("target")
        if target:
            tag = " (intensity)" if target.get("is_intensity") else " (absolute)"
            print(f"  {company}: {target['target_pct']}% by {target['target_year']} "
                  f"(base {target['base_year']}, {target['year_type']}, {target['classification']})"
                  f"{tag} - validated {target['validation_year']}")
        else:
            removed = data.get("commitment_removed_year")
            if removed:
                print(f"  {company}: no target, commitment removed by {removed}")
            else:
                print(f"  {company}: no SBTi data (status: {data.get('status')})")

    groups = defaultdict(list)
    for c in climate_claims:
        groups[(c["company_name"], c["year"])].append(c)

    Path(OUTPUT_CLAIMS).parent.mkdir(parents=True, exist_ok=True)
    per_claim_results = []
    per_group_summary = {}

    for (company, year), claims in sorted(groups.items()):
        data = sbti_data.get(company, {"target": None, "commitment_removed_year": None, "status": None})
        target = data.get("target")

        verdicts = []
        for claim in claims:
            ctype = claim.get("claim_type")
            if ctype == "target":
                verdict, reason = assess_target_claim(claim, data)
            elif ctype == "achievement":
                verdict, reason = assess_achievement_claim(claim, data)
            else:
                if claim.get("quantified_value") is None:
                    verdict, reason = "no_quantification", "qualitative claim"
                else:
                    verdict, reason = "different_metric", f"non-target/achievement claim_type: {ctype}"

            verdicts.append(verdict)
            per_claim_results.append({
                "company_name": company,
                "year": year,
                "block_id": claim["block_id"],
                "claim_text": claim.get("claim_text"),
                "claim_type": ctype,
                "metric": claim.get("metric"),
                "quantified_value": claim.get("quantified_value"),
                "unit": claim.get("unit"),
                "target_year": claim.get("target_year"),
                "baseline_year": claim.get("baseline_year"),
                "scope": claim.get("scope"),
                "is_intensity_claim": is_intensity_claim(claim),
                "verdict": verdict,
                "reason": reason,
            })

        n_climate = len(claims)
        n_matches = sum(1 for v in verdicts if v == "matches_sbti")
        n_weaker = sum(1 for v in verdicts if v == "weaker_than_sbti")
        n_stronger = sum(1 for v in verdicts if v == "stronger_than_sbti")
        n_on_track = sum(1 for v in verdicts if v == "on_track")
        n_ahead = sum(1 for v in verdicts if v == "ahead")
        n_behind = sum(1 for v in verdicts if v == "behind")
        n_different = sum(1 for v in verdicts if v == "different_metric")
        n_no_quant = sum(1 for v in verdicts if v == "no_quantification")
        n_no_sbti = sum(1 for v in verdicts if v == "no_sbti_target")
        n_not_yet = sum(1 for v in verdicts if v == "target_not_yet_validated")
        n_commit_removed = sum(1 for v in verdicts if v == "commitment_removed")

        n_aligned = n_matches + n_on_track + n_ahead + n_stronger
        n_unassessable = n_no_sbti + n_no_quant + n_different + n_not_yet + n_commit_removed
        denom = n_climate - n_unassessable

        if n_commit_removed > 0 and n_commit_removed == n_climate:
            score = 0.0
            score_note = "commitment_removed_penalty"
        elif denom > 0:
            score = round(n_aligned / denom, 4)
            score_note = None
        else:
            score = None
            if n_not_yet > 0 and n_not_yet == n_climate:
                score_note = "all_claims_predate_validation"
            elif n_no_sbti > 0 and n_no_sbti == n_climate:
                score_note = "no_sbti_target_for_company"
            else:
                score_note = "no_assessable_claims"

        per_group_summary[f"{company}__{year}"] = {
            "company_name": company,
            "year": year,
            "sbti_status": data.get("status"),
            "has_sbti_validation": target is not None,
            "sbti_validation_year": target.get("validation_year") if target else None,
            "sbti_classification": target.get("classification") if target else None,
            "sbti_target_pct": target.get("target_pct") if target else None,
            "sbti_target_year": target.get("target_year") if target else None,
            "sbti_baseline_year": target.get("base_year") if target else None,
            "sbti_is_intensity": target.get("is_intensity") if target else None,
            "commitment_removed_year": data.get("commitment_removed_year"),
            "n_climate_claims": n_climate,
            "n_matches_sbti": n_matches,
            "n_weaker_than_sbti": n_weaker,
            "n_stronger_than_sbti": n_stronger,
            "n_on_track": n_on_track,
            "n_ahead": n_ahead,
            "n_behind": n_behind,
            "n_different_metric": n_different,
            "n_no_quantification": n_no_quant,
            "n_no_sbti_target": n_no_sbti,
            "n_target_not_yet_validated": n_not_yet,
            "n_commitment_removed": n_commit_removed,
            "n_aligned": n_aligned,
            "n_assessable": denom,
            "sdg13_alignment_score": score,
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