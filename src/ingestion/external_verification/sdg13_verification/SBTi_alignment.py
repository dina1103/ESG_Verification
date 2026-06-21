import json
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd

INPUT_JSONL    = r"data\processed\llm_claim_extraction_result.jsonl"
SDG_PARQUET    = r"data\processed\segments_esg_sdg.parquet"
SBTI_XLSX      = r"src\ingestion\external_benchmarks\sdg13_verification\targets-excel.xlsx"
COMPANY_MAP    = r"src\ingestion\external_benchmarks\sdg13_verification\sbti_company_mapping.json"
OUTPUT_CLAIMS  = r"data\processed\sdg13_climate_claim_level.jsonl"
OUTPUT_SUMMARY = r"data\processed\sdg13_climate_company_year.json"

SBTI_SHEET = "WebsiteData"

# tolerance bounds for matching to SBTi
PCT_TOLERANCE = 2
YEAR_TOLERANCE = 1
BASE_YEAR_TOLERANCE = 1   # claim baseline must align with the SBTi baseline for trajectory math

# --- intensity detection -------------------------------------------------
_INTENSITY_DENOM = (
    r"(vehicles?|cars?|km|kilomet\w+|miles?|units?|produced|production|capita|"
    r"employees?|fte|passengers?|products?|kwh|mwh|gj|m2|sqm|sales|revenue|turnover)"
)
_INTENSITY_RE = [
    re.compile(r"\bintensity\b", re.I),
    re.compile(r"\bper\s+" + _INTENSITY_DENOM + r"\b", re.I),
    re.compile(r"/\s*" + _INTENSITY_DENOM + r"\b", re.I),
]


def is_intensity_claim(claim):
    fields = (str(claim.get("metric") or "") + " " + str(claim.get("unit") or "")).strip()
    return any(r.search(fields) for r in _INTENSITY_RE)


# --- commensurability gate -----------------------------------------------
_GHG_RE      = re.compile(r"(ghg|greenhouse|co2|co₂|carbon|emission)", re.I)
_NONGHG_RE   = re.compile(r"(energy|water|waste|electricity|renewable|\bbev\b|\bev\b|battery|share|recycl|landfill)", re.I)
_USEPHASE_RE = re.compile(r"(use[ -]?phase|per vehicle kilom|vehicles? sold|new vehicles?|tailpipe|well[- ]to[- ]wheel|tank[- ]to[- ]wheel|fleet|lifecycle|life cycle|sales network)", re.I)
_ABSUNIT_RE  = re.compile(r"(g\s*co|/\s*km|/\s*vehicle|t\s*co|tonne|tco2|kwh|mwh|gwh|\bgj\b|kg\b|liter|litre)", re.I)


_SCOPE3_RE = re.compile(r"\\b3\\b")

def route_scope_group(claim):
    # decide which SBTi scope a claim should be checked against
    text = str(claim.get("metric") or "") + " " + str(claim.get("claim_text") or "")
    scope = claim.get("scope") or "N/A"
    if "3" in scope and "1+2" not in scope and "1 & 2" not in scope:
        return "3"
    if _USEPHASE_RE.search(text):
        return "3"   # use-phase / sold-product / fleet emissions are scope 3 for an automaker
    return "1+2"


def pick_target(claim, candidates):
    # among scope-matched candidates, pick the benchmark this claim most plausibly
    # refers to: closest target_year, then closest reduction percentage
    qv = claim.get("quantified_value")
    cy = claim.get("target_year")
    def key(t):
        yd = abs((cy - t["target_year"])) if cy is not None else 0
        pd_ = abs(qv - t["target_pct"]) if qv is not None else 0
        return (yd, pd_)
    return sorted(candidates, key=key)[0] if candidates else None


def commensurability_reason(claim, target):
    # returns None if the claim is comparable, else a short reason for the mismatch
    text = str(claim.get("metric") or "") + " " + str(claim.get("claim_text") or "")
    if not _GHG_RE.search(text) or _NONGHG_RE.search(str(claim.get("metric") or "")):
        return "non-GHG metric (energy/water/waste/share)"
    if is_intensity_claim(claim) != target.get("is_intensity", False):
        return (f"intensity mismatch: claim={'intensity' if is_intensity_claim(claim) else 'absolute'}, "
                f"target={'intensity' if target.get('is_intensity') else 'absolute'}")
    qv = claim.get("quantified_value")
    if qv is None or not (0 < qv <= 100):
        return "value is not a reduction percentage in (0,100]"
    if _ABSUNIT_RE.search(str(claim.get("unit") or "")):
        return "value is an absolute level (e.g. g CO2/km), not a reduction %"
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


def check_mapping_coverage(climate_claims, mapping):
    # loud guard: any corpus company absent from the mapping would silently score no_sbti_target
    corpus_companies = sorted({c["company_name"] for c in climate_claims})
    unmapped = [c for c in corpus_companies if c not in mapping]
    print(f"\nMapping coverage: {len(corpus_companies) - len(unmapped)}/{len(corpus_companies)} corpus companies mapped")
    for c in corpus_companies:
        print(f"  [{'OK' if c in mapping else '!! UNMAPPED'}] {c}")
    if unmapped:
        print("\n  WARNING: these companies are NOT in sbti_company_mapping.json and will all")
        print("           score 'no_sbti_target'. Add them (use sbti_name=null if no SBTi record):")
        for c in unmapped:
            print(f"             - {c}")
    return unmapped


def load_sbti_data(xlsx_path, mapping):
    df = pd.read_excel(xlsx_path, sheet_name=SBTI_SHEET)

    result = {}
    for our_name, entry in mapping.items():
        sbti_name = entry.get("sbti_name")
        if not sbti_name:
            result[our_name] = {"target": None, "targets": [], "validation_year": None, "commitment_removed_year": None, "status": entry.get("status")}
            continue

        nt = df[
            (df["company_name"] == sbti_name)
            & (df["action"] == "Target")
            & (df["target"] == "Near-term")
        ]

        # collect every near-term target row, tagged by scope group, so each claim
        # can be routed to the benchmark that matches its scope (1+2 vs 3)
        targets = []
        for _, row in nt.iterrows():
            sc = str(row["scope"])
            if sc == "1+2":
                grp = "1+2"
            elif sc == "3":
                grp = "3"
            else:
                continue  # ignore 1, 2, 1+2+3 standalone rows for routing
            targets.append({
                "scope_group": grp,
                "target_pct": float(row["target_value"]) * 100,
                "base_year": int(row["base_year"]),
                "target_year": int(row["target_year"]),
                "classification": str(row["target_classification_short"]),
                "year_type": str(row["year_type"]),
                # scope 1+2 intensity flag comes from the mapping; scope-3 auto targets
                # in this corpus are intensity-based (per the SBTi 'type' column)
                "is_intensity": entry.get("is_intensity", False) if grp == "1+2" else (str(row["type"]) == "Intensity"),
                "validation_year": int(pd.to_datetime(row["date_published"]).year),
                "sbti_name": sbti_name,
            })

        target = None
        s12 = [t for t in targets if t["scope_group"] == "1+2"]
        if s12:
            target = dict(s12[0])
            target["intensity_unit"] = entry.get("intensity_unit")

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

        validation_year = min((t["validation_year"] for t in targets), default=None)
        result[our_name] = {
            "target": target,            # scope 1+2 primary, for summary fields
            "targets": targets,          # all near-term targets, for routing
            "validation_year": validation_year,
            "commitment_removed_year": commitment_removed_year,
            "status": entry.get("status"),
        }

    return result


def _no_target_verdict(sbti_data, report_year):
    # shared abstain logic when no validated target exists; reason reflects SBTi status
    status = sbti_data.get("status")
    removed_year = sbti_data.get("commitment_removed_year")
    if status == "Commitment removed" and removed_year and report_year >= removed_year:
        return "commitment_removed", f"company's SBTi commitment was removed in {removed_year}"
    if status == "Commitment removed":
        return "no_sbti_target", f"SBTi commitment removed (effective {removed_year}); report year {report_year} precedes removal"
    if status == "Committed":
        return "no_sbti_target", "company committed to SBTi but has no validated target to assess against"
    return "no_sbti_target", "no SBTi target on file for this company"


def assess_target_claim(claim, sbti_data):
    targets = sbti_data.get("targets") or []
    report_year = claim.get("year")
    val_year = sbti_data.get("validation_year")

    if not targets:
        return _no_target_verdict(sbti_data, report_year)

    if val_year is not None and report_year < val_year:
        return "target_not_yet_validated", \
               f"report year {report_year} predates SBTi validation in {val_year}"

    grp = route_scope_group(claim)
    candidates = [t for t in targets if t["scope_group"] == grp]
    if not candidates:
        return "not_commensurable", f"no SBTi scope-{grp} target for this company"
    target = pick_target(claim, candidates)

    reason = commensurability_reason(claim, target)
    if reason:
        return "not_commensurable", reason

    claim_pct = claim.get("quantified_value")
    claim_year = claim.get("target_year")
    if claim_year is None:
        return "no_quantification", "missing target year"

    if abs(claim_year - target["target_year"]) > YEAR_TOLERANCE:
        return "not_commensurable", f"target_year diff = {abs(claim_year - target['target_year'])} (>{YEAR_TOLERANCE})"

    pct_diff = claim_pct - target["target_pct"]
    tag = f"[scope {grp}]"
    if abs(pct_diff) <= PCT_TOLERANCE:
        return "matches_sbti", f"{tag} pct_diff = {round(pct_diff, 2)}"
    if pct_diff > 0:
        return "stronger_than_sbti", f"{tag} claim {claim_pct}% vs sbti {target['target_pct']}%"
    return "weaker_than_sbti", f"{tag} claim {claim_pct}% vs sbti {target['target_pct']}%"


def assess_achievement_claim(claim, sbti_data):
    targets = sbti_data.get("targets") or []
    report_year = claim.get("year")
    val_year = sbti_data.get("validation_year")

    if not targets:
        return _no_target_verdict(sbti_data, report_year)

    if val_year is not None and report_year < val_year:
        return "target_not_yet_validated", \
               f"report year {report_year} predates SBTi validation in {val_year}"

    grp = route_scope_group(claim)
    candidates = [t for t in targets if t["scope_group"] == grp]
    if not candidates:
        return "not_commensurable", f"no SBTi scope-{grp} target for this company"
    target = pick_target(claim, candidates)

    reason = commensurability_reason(claim, target)
    if reason:
        return "not_commensurable", reason

    claim_pct = claim.get("quantified_value")
    baseline_year = claim.get("baseline_year")

    # the linear-trajectory check is only meaningful for a cumulative reduction
    # measured from the SBTi baseline; otherwise the claim can't be placed on the path
    if baseline_year is None:
        return "not_commensurable", "achievement has no baseline year; cannot place on SBTi trajectory"
    if abs(baseline_year - target["base_year"]) > BASE_YEAR_TOLERANCE:
        return "not_commensurable", f"baseline {baseline_year} != SBTi baseline {target['base_year']}"
    # a reported "achievement" >= the full target reduction, years before the target
    # year, is almost always a restated target mislabeled as an achievement
    if claim_pct >= target["target_pct"]:
        return "not_commensurable", \
               f"claimed reduction {claim_pct}% >= full target {target['target_pct']}%; likely a restated target"

    years_elapsed = report_year - baseline_year
    target_duration = target["target_year"] - baseline_year
    if target_duration <= 0 or years_elapsed < 0:
        return "not_commensurable", "invalid year math"

    expected_pct = target["target_pct"] * years_elapsed / target_duration
    diff = claim_pct - expected_pct
    tag = f"[scope {grp}]"

    if abs(diff) <= PCT_TOLERANCE:
        return "on_track", f"{tag} claim {claim_pct}% vs expected {round(expected_pct, 1)}% at year {report_year}"
    if diff > 0:
        return "ahead", f"{tag} claim {claim_pct}% vs expected {round(expected_pct, 1)}% (ahead by {round(diff, 1)})"
    return "behind", f"{tag} claim {claim_pct}% vs expected {round(expected_pct, 1)}% (behind by {round(-diff, 1)})"


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

    check_mapping_coverage(climate_claims, mapping)

    print(f"\nLoading SBTi data from {SBTI_XLSX}...")
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
        n_not_comm = sum(1 for v in verdicts if v == "not_commensurable")
        n_no_quant = sum(1 for v in verdicts if v == "no_quantification")
        n_no_sbti = sum(1 for v in verdicts if v == "no_sbti_target")
        n_not_yet = sum(1 for v in verdicts if v == "target_not_yet_validated")
        n_commit_removed = sum(1 for v in verdicts if v == "commitment_removed")

        n_aligned = n_matches + n_on_track + n_ahead + n_stronger
        n_unassessable = (n_no_sbti + n_no_quant + n_different + n_not_yet
                          + n_commit_removed + n_not_comm)
        denom = n_climate - n_unassessable

        # ---- status-driven scoring ----------------------------------------
        # The score FLOOR is set by the company's SBTi status, not by counting how
        # many individual claims happen to carry a given verdict. Per-claim alignment
        # (aligned/denom) only applies to firms that actually hold a validated target.
        status = data.get("status")
        removed_year = data.get("commitment_removed_year")

        if status == "Commitment removed" and removed_year and year >= removed_year:
            # commitment withdrawn by this report year -> bad score
            score = 0.0
            score_note = "commitment_removed_penalty"
        elif status == "Commitment removed":
            # report year precedes removal: commitment still active, no validated
            # target yet -> insufficient info (do NOT back-date the penalty: that
            # would use future information the report could not have reflected)
            score = None
            score_note = "commitment_active_no_validated_target"
        elif status == "Committed":
            # promised a target but none validated -> nothing external to assess
            score = None
            score_note = "committed_no_validated_target"
        elif status == "No commitment":
            # no SBTi engagement at all -> insufficient info
            score = None
            score_note = "no_sbti_commitment"
        elif denom > 0:
            # validated target with >=1 commensurable claim this year
            score = round(n_aligned / denom, 4)
            score_note = None
        else:
            # validated target but no commensurable claim this year
            score = None
            if n_not_yet > 0 and n_not_yet == n_climate:
                score_note = "all_claims_predate_validation"
            else:
                score_note = "no_commensurable_claim_this_year"

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
            "n_not_commensurable": n_not_comm,
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