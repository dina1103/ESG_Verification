import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

INPUT_JSONL     = r"data\processed\llm_claim_extraction_result.jsonl"
OUTPUT_CLUSTERS = r"data\processed\peer_comparison_metric_clusters.json"
OUTPUT_CLAIMS   = r"data\processed\peer_comparison_claim_level.jsonl"
OUTPUT_SUMMARY  = r"data\processed\peer_comparison_company_year.json"

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
# clustering kept tight: loosening over-merges different denominators without
# widening peer groups (peer width is capped by disclosure overlap, not granularity)
CLUSTER_THRESHOLD = 0.4
# minimum companies for a valid peer comparison. 3 is the floor; comparisons are
# additionally WEIGHTED by peer-group width so a 3-company z counts less than a 10-company z.
MIN_COMPANIES_FOR_PEER = 3
# a company-year score must rest on at least this many cluster comparisons to be reported
MIN_CLUSTERS_FOR_SCORE = 3
# this script scores ONLY "level" metrics (current state values);
# the companion script handles the other type. levels and changes are never compared together.
VALUE_TYPE_THIS_SCRIPT = "level"

BAD_TOPIC_KEYWORDS = [
    "emission", "ghg", "co2", "carbon footprint",
    "water use", "water consumption", "water withdrawal",
    "waste", "hazardous", "landfill",
    "energy consumption", "energy use",
    "incident", "injury", "accident", "fatality",
    "spill", "release", "violation", "fine",
    "complaint", "grievance",
]
REDUCTION_INDICATORS = [
    "reduced", "reduction", "decrease", "decreased", "lowered", "lower",
    "cut", "avoided", "avoidance", "savings", "saved",
    "below baseline", "less than", "fewer than",
    "improvement", "improved",
]
INTENSITY_PATTERNS = [
    r"\bper\b", r"/", r"\bintensity\b", r"\bshare\b",
    r"\bpercentage\b", r"\bratio\b", r"\brate\b", r"\baverage\b",
]
INTENSITY_UNITS = {
    "%", "percent", "percentage", "per vehicle", "per car", "per unit",
    "per employee", "per fte", "per million hours", "per 100 employees",
    "per revenue", "per euro", "per dollar", "per kwh", "per gj", "rate",
}
UNIT_NORMALIZATION = {
    "mtco2e": ("tCO2e", 1_000_000), "ktco2e": ("tCO2e", 1_000), "tco2e": ("tCO2e", 1),
    "tonnes co2": ("tCO2e", 1), "tons co2": ("tCO2e", 1),
    "million tonnes co2": ("tCO2e", 1_000_000), "kg co2e": ("tCO2e", 0.001),
    "m3": ("m3", 1), "cubic meters": ("m3", 1), "litres": ("m3", 0.001),
    "liters": ("m3", 0.001), "ml": ("m3", 1e-6), "million m3": ("m3", 1_000_000),
    "tonnes": ("tonnes", 1), "tons": ("tonnes", 1), "kg": ("tonnes", 0.001),
    "kilograms": ("tonnes", 0.001), "mwh": ("MWh", 1), "gwh": ("MWh", 1000),
    "twh": ("MWh", 1_000_000), "kwh": ("MWh", 0.001), "gj": ("GJ", 1),
    "tj": ("GJ", 1000), "pj": ("GJ", 1_000_000), "%": ("%", 1),
    "percent": ("%", 1), "percentage": ("%", 1), "count": ("count", 1),
    "number": ("count", 1), "incidents": ("count", 1), "employees": ("employees", 1),
}


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
                row = {"block_id": rec["block_id"], "company_name": rec["company_name"],
                       "year": rec["year"], **claim}
                for fld in ("metric", "unit", "claim_text", "scope", "claim_type"):
                    v = row.get(fld)
                    if isinstance(v, list):
                        row[fld] = " ".join(str(x) for x in v)
                    elif v is not None and not isinstance(v, str):
                        row[fld] = str(v)
                qv = row.get("quantified_value")
                if isinstance(qv, list):
                    qv = qv[0] if qv else None
                if qv is not None:
                    try:
                        row["quantified_value"] = float(qv)
                    except (ValueError, TypeError):
                        row["quantified_value"] = None
                else:
                    row["quantified_value"] = None
                rows.append(row)
    return rows


def is_intensity_claim(claim):
    metric_text = (claim.get("metric") or "").lower()
    unit_text = (claim.get("unit") or "").lower().strip()
    claim_text = (claim.get("claim_text") or "").lower()
    if unit_text in INTENSITY_UNITS:
        return True
    combined = metric_text + " " + claim_text
    for pat in INTENSITY_PATTERNS:
        if re.search(pat, combined):
            return True
    if "/" in unit_text:
        return True
    return False


def normalize_unit(value, unit):
    if value is None:
        return None, None
    if unit is None or unit == "":
        return value, None
    unit_lower = unit.lower().strip()

    # exact-match table first (percentages, plain mass/energy/volume units)
    if unit_lower in UNIT_NORMALIZATION:
        canonical, multiplier = UNIT_NORMALIZATION[unit_lower]
        return value * multiplier, canonical

    # unit-FAMILY normalization: collapse the many string variants of the same
    # physical intensity onto one canonical unit+scale, so they form one peer group.
    # CO2 per vehicle -> tCO2/vehicle (kg variants scaled down by 1000)
    if "vehicle" in unit_lower and ("co2" in unit_lower or "carbon" in unit_lower):
        if unit_lower.startswith("kg") or "kg " in unit_lower or "kg of" in unit_lower:
            return value * 0.001, "tCO2/vehicle"
        return value, "tCO2/vehicle"
    # CO2 per km -> gCO2/km (these are already in grams in the corpus)
    if "km" in unit_lower and ("co2" in unit_lower or unit_lower in ("g/km", "g/km co2")):
        return value, "gCO2/km"
    # water per vehicle -> m3/vehicle
    if "m3" in unit_lower and "vehicle" in unit_lower:
        return value, "m3/vehicle"

    return value, unit_lower


# a claim is a CHANGE (delta/movement) vs a LEVEL (state). changes and levels are
# different quantities and must never be z-scored together (a reduction-% is not a
# level value). detected from the metric/claim text.
CHANGE_PATTERN = re.compile(r'(?i)\b(reduc\w*|decreas\w*|increas\w*|cut|lower\w*|declin\w*|'
                            r'change in|chang\w*|improv\w*|grew|growth|rose|fell|drop\w*|'
                            r'saving|saved|avoid\w*|reduction)\b')

def value_type(claim):
    text = (claim.get("metric") or "").lower() + " " + (claim.get("claim_text") or "").lower()
    return "change" if CHANGE_PATTERN.search(text) else "level"


def has_bad_topic(metric_name):
    metric_lower = (metric_name or "").lower()
    return any(kw in metric_lower for kw in BAD_TOPIC_KEYWORDS)


def is_measuring_reduction(claim):
    combined = (claim.get("metric") or "").lower() + " " + (claim.get("claim_text") or "").lower()
    return any(ind in combined for ind in REDUCTION_INDICATORS)


def is_bad_direction_value(claim):
    # flip the z-score sign when: bad topic AND measuring a LEVEL (not a reduction).
    # then, after flipping, positive z = better-than-peers for ALL claims, which
    # matches the integrity-score convention (higher = more integrity).
    if not has_bad_topic(claim.get("metric", "")):
        return False
    if is_measuring_reduction(claim):
        return False
    return True


def cluster_metrics(metric_names, model):
    if len(metric_names) < 2:
        return {m: 0 for m in metric_names}
    embeddings = model.encode(metric_names, show_progress_bar=False, batch_size=64)
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=CLUSTER_THRESHOLD,
        metric="cosine", linkage="average",
    )
    labels = clustering.fit_predict(embeddings)
    return dict(zip(metric_names, labels.tolist()))


def main():
    print(f"Loading claims from {INPUT_JSONL}...")
    all_claims = load_claims(INPUT_JSONL)
    print(f"  loaded {len(all_claims):,} claims")

    quant_claims = [
        c for c in all_claims
        if c.get("claim_type") == "achievement"
        and c.get("quantified_value") is not None
        and c.get("metric") and c.get("metric") != "N/A"
    ]
    print(f"  quantitative-achievement: {len(quant_claims):,}")

    intensity_claims = [c for c in quant_claims if is_intensity_claim(c)]
    print(f"  intensity-only: {len(intensity_claims):,} "
          f"(excluded {len(quant_claims) - len(intensity_claims):,} absolute, size-confounded)")
    if not intensity_claims:
        print("No intensity claims. Exiting.")
        return

    for c in intensity_claims:
        nv, nu = normalize_unit(c.get("quantified_value"), c.get("unit"))
        c["normalized_value"] = nv
        c["normalized_unit"] = nu
        c["flip_sign"] = is_bad_direction_value(c)
        c["value_type"] = value_type(c)   # level vs change - kept in separate peer groups

    # keep only the value_type this script handles
    before = len(intensity_claims)
    intensity_claims = [c for c in intensity_claims if c["value_type"] == VALUE_TYPE_THIS_SCRIPT]
    print(f"  filtered to {VALUE_TYPE_THIS_SCRIPT!r} claims: {len(intensity_claims):,} (dropped {before-len(intensity_claims):,} of other type)")

    unique_metrics = sorted(set(c["metric"] for c in intensity_claims))
    print(f"\nClustering {len(unique_metrics)} unique metric names (threshold {CLUSTER_THRESHOLD})...")
    model = SentenceTransformer(SBERT_MODEL)
    metric_to_cluster = cluster_metrics(unique_metrics, model)
    print(f"  {len(set(metric_to_cluster.values()))} clusters")
    for c in intensity_claims:
        c["cluster_id"] = metric_to_cluster[c["metric"]]

    cluster_to_metrics = defaultdict(list)
    for metric, cid in metric_to_cluster.items():
        cluster_to_metrics[cid].append(metric)
    cluster_catalog = {}
    for cid, metrics in cluster_to_metrics.items():
        claims_in_c = [c for c in intensity_claims if c["cluster_id"] == cid]
        companies = sorted(set(c["company_name"] for c in claims_in_c))
        units = sorted(set(c["normalized_unit"] for c in claims_in_c if c["normalized_unit"]))
        n_flipped = sum(1 for c in claims_in_c if c["flip_sign"])
        cluster_catalog[f"cluster_{cid}"] = {
            "metrics_in_cluster": sorted(metrics), "n_claims": len(claims_in_c),
            "n_companies": len(companies), "companies": companies, "units_seen": units,
            "has_bad_topic": any(has_bad_topic(m) for m in metrics),
            "n_claims_flipped_sign": n_flipped, "n_claims_unflipped": len(claims_in_c) - n_flipped,
        }
    Path(OUTPUT_CLUSTERS).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CLUSTERS, "w", encoding="utf-8") as f:
        json.dump(cluster_catalog, f, indent=2, ensure_ascii=False)

    groups = defaultdict(list)
    for c in intensity_claims:
        # value_type in the key: levels compared only to levels, changes only to changes
        groups[(c["cluster_id"], c["year"], c["normalized_unit"], c["flip_sign"], c["value_type"])].append(c)

    print(f"\nComputing peer z-scores (weighted by peer-group width)...")
    per_claim_results = []
    # store per company-year a list of (z, weight) so we can take a weighted mean
    per_company_year = defaultdict(lambda: {"weighted": [], "z_scores": {}})
    n_groups = n_skipped = 0

    for (cid, year, unit, flip_sign, vtype), group_claims in groups.items():
        per_company = defaultdict(list)
        for c in group_claims:
            nv = c["normalized_value"]
            if isinstance(nv, (int, float)) and not isinstance(nv, bool):
                per_company[c["company_name"]].append(nv)
        company_values = {comp: float(np.mean(vals)) for comp, vals in per_company.items() if vals}

        sign_subgroups = {
            "nonneg": {c: v for c, v in company_values.items() if v >= 0},
            "neg":    {c: v for c, v in company_values.items() if v < 0},
        }
        produced = False
        for sign_label, subgroup in sign_subgroups.items():
            n_peers = len(subgroup)
            if n_peers < MIN_COMPANIES_FOR_PEER:
                n_skipped += 1
                continue
            values = list(subgroup.values())
            peer_mean = float(np.mean(values))
            peer_std = float(np.std(values))
            if peer_std == 0:
                n_skipped += 1
                continue
            # weight: wider peer groups give more reliable z-scores. weight = n_peers - 2
            # so a 3-company comparison (the minimum) gets weight 1, a 10-company gets 8.
            weight = n_peers - (MIN_COMPANIES_FOR_PEER - 1)
            for company, val in subgroup.items():
                z = (val - peer_mean) / peer_std
                if flip_sign:
                    z = -z   # after flip, positive z = better than peers (higher integrity)
                sign_tag = "level" if flip_sign else "reduction_or_good"
                cluster_key = f"cluster_{cid}__{unit}__{vtype}__{sign_tag}__{sign_label}"
                kcy = f"{company}__{year}"
                per_company_year[kcy]["z_scores"][cluster_key] = round(z, 4)
                per_company_year[kcy]["weighted"].append((z, weight))
                per_company_year[kcy]["company_name"] = company
                per_company_year[kcy]["year"] = year
                per_claim_results.append({
                    "company_name": company, "year": year, "cluster_id": cid,
                    "normalized_unit": unit, "sign_subgroup": sign_label,
                    "company_value": val, "peer_mean": peer_mean, "peer_std": peer_std,
                    "n_peers": n_peers, "weight": weight, "z_score": round(z, 4),
                    "flip_sign_applied": flip_sign, "value_type": vtype,
                    "interpretation": "level of bad thing (lower=better, sign flipped)" if flip_sign
                                      else "reduction or good thing (higher=better)",
                })
            produced = True
        if produced:
            n_groups += 1

    print(f"  computed {n_groups} groups | skipped {n_skipped} (insufficient peers or zero variance)")

    final_summary = {}
    for key, data in per_company_year.items():
        wz = data["weighted"]
        if wz and len(wz) >= MIN_CLUSTERS_FOR_SCORE:
            zs = np.array([z for z, w in wz], dtype=float)
            ws = np.array([w for z, w in wz], dtype=float)
            peer_score = float(np.sum(zs * ws) / np.sum(ws))      # weighted mean z
            simple_mean = float(np.mean(zs))                       # unweighted, for reference
            note = None
        else:
            peer_score = simple_mean = None
            note = f"insufficient_evidence ({len(wz)} cluster comparisons)"
        final_summary[key] = {
            "company_name": data["company_name"], "year": data["year"],
            "n_clusters_with_data": len(wz),
            "z_scores_by_cluster": data["z_scores"],
            "peer_deviation_score": round(peer_score, 4) if peer_score is not None else None,
            "peer_deviation_score_unweighted": round(simple_mean, 4) if simple_mean is not None else None,
            "note": note,
        }

    with open(OUTPUT_CLAIMS, "w", encoding="utf-8") as f:
        for rec in per_claim_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    scored = sum(1 for v in final_summary.values() if v["peer_deviation_score"] is not None)
    print(f"\n{'='*70}\nCOMPLETE")
    print(f"company-years with a peer score: {scored}/{len(final_summary)}")
    print(f"per-claim z-scores: {len(per_claim_results)}")
    print(f"  {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()