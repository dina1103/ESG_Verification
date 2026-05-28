import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

INPUT_JSONL    = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\llm_claim_extraction_result.jsonl"
OUTPUT_CLUSTERS = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\peer_metric_clusters.json"
OUTPUT_CLAIMS   = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\peer_comparison_claim_level.jsonl"
OUTPUT_SUMMARY  = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\peer_comparison_company_year.json"

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
CLUSTER_THRESHOLD = 0.4
MIN_COMPANIES_FOR_PEER = 3

# topics where the underlying thing is "bad" - lower level = better
BAD_TOPIC_KEYWORDS = [
    "emission", "ghg", "co2", "carbon footprint",
    "water use", "water consumption", "water withdrawal",
    "waste", "hazardous", "landfill",
    "energy consumption", "energy use",
    "incident", "injury", "accident", "fatality",
    "spill", "release", "violation", "fine",
    "complaint", "grievance",
]

# language indicating the claim is measuring REDUCTION/AVOIDANCE of a bad thing
# in which case higher value = better (we don't flip the z-score)
REDUCTION_INDICATORS = [
    "reduced", "reduction", "decrease", "decreased", "lowered", "lower",
    "cut", "avoided", "avoidance", "savings", "saved",
    "below baseline", "less than", "fewer than",
    "improvement", "improved",
]

# intensity indicators - metric is normalized per unit of company activity
INTENSITY_PATTERNS = [
    r"\bper\b",
    r"/",
    r"\bintensity\b",
    r"\bshare\b",
    r"\bpercentage\b",
    r"\bratio\b",
    r"\brate\b",
    r"\baverage\b",
]

INTENSITY_UNITS = {
    "%", "percent", "percentage",
    "per vehicle", "per car", "per unit",
    "per employee", "per fte",
    "per million hours", "per 100 employees",
    "per revenue", "per euro", "per dollar",
    "per kwh", "per gj",
    "rate",
}

UNIT_NORMALIZATION = {
    "mtco2e": ("tCO2e", 1_000_000),
    "ktco2e": ("tCO2e", 1_000),
    "tco2e": ("tCO2e", 1),
    "tonnes co2": ("tCO2e", 1),
    "tons co2": ("tCO2e", 1),
    "million tonnes co2": ("tCO2e", 1_000_000),
    "kg co2e": ("tCO2e", 0.001),
    "m3": ("m3", 1),
    "cubic meters": ("m3", 1),
    "litres": ("m3", 0.001),
    "liters": ("m3", 0.001),
    "ml": ("m3", 1e-6),
    "million m3": ("m3", 1_000_000),
    "tonnes": ("tonnes", 1),
    "tons": ("tonnes", 1),
    "kg": ("tonnes", 0.001),
    "kilograms": ("tonnes", 0.001),
    "mwh": ("MWh", 1),
    "gwh": ("MWh", 1000),
    "twh": ("MWh", 1_000_000),
    "kwh": ("MWh", 0.001),
    "gj": ("GJ", 1),
    "tj": ("GJ", 1000),
    "pj": ("GJ", 1_000_000),
    "%": ("%", 1),
    "percent": ("%", 1),
    "percentage": ("%", 1),
    "count": ("count", 1),
    "number": ("count", 1),
    "incidents": ("count", 1),
    "employees": ("employees", 1),
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
                row = {
                    "block_id": rec["block_id"],
                    "company_name": rec["company_name"],
                    "year": rec["year"],
                    **claim,
                }
                # Llama sometimes emits text fields as lists - coerce to strings
                for fld in ("metric", "unit", "claim_text", "scope", "claim_type"):
                    v = row.get(fld)
                    if isinstance(v, list):
                        row[fld] = " ".join(str(x) for x in v)
                    elif v is not None and not isinstance(v, str):
                        row[fld] = str(v)
                # quantified_value must be numeric for the math - coerce or null it
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
    if unit_lower in UNIT_NORMALIZATION:
        canonical, multiplier = UNIT_NORMALIZATION[unit_lower]
        return value * multiplier, canonical
    return value, unit_lower


def has_bad_topic(metric_name):
    # is the underlying topic a "bad thing" (emissions, waste, injuries, etc.)
    metric_lower = (metric_name or "").lower()
    return any(kw in metric_lower for kw in BAD_TOPIC_KEYWORDS)


def is_measuring_reduction(claim):
    # does the claim measure a REDUCTION/AVOIDANCE rather than a level?
    # if so, higher value is better even for bad topics
    metric_text = (claim.get("metric") or "").lower()
    claim_text = (claim.get("claim_text") or "").lower()
    combined = metric_text + " " + claim_text
    return any(ind in combined for ind in REDUCTION_INDICATORS)


def is_bad_direction_value(claim):
    # final determination: should we flip the z-score sign for this claim?
    # we flip when: the topic is bad AND the value measures the level (not reduction)
    if not has_bad_topic(claim.get("metric", "")):
        return False  # good topic - high value = better, no flip
    if is_measuring_reduction(claim):
        return False  # bad topic but measuring reduction - high value = better, no flip
    return True  # bad topic AND measuring level - high value = worse, flip


def cluster_metrics(metric_names, model):
    if len(metric_names) < 2:
        return {m: 0 for m in metric_names}

    embeddings = model.encode(metric_names, show_progress_bar=False)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CLUSTER_THRESHOLD,
        metric="cosine",
        linkage="average",
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
    print(f"  after quantitative-achievement filter: {len(quant_claims):,}")

    intensity_claims = [c for c in quant_claims if is_intensity_claim(c)]
    n_excluded = len(quant_claims) - len(intensity_claims)
    print(f"  after intensity-only filter: {len(intensity_claims):,}")
    print(f"  excluded {n_excluded:,} absolute metric claims (size-confounded)")

    if not intensity_claims:
        print("\nNo intensity claims to compare. Exiting.")
        return

    # normalize units
    for c in intensity_claims:
        norm_val, norm_unit = normalize_unit(c.get("quantified_value"), c.get("unit"))
        c["normalized_value"] = norm_val
        c["normalized_unit"] = norm_unit
        # also tag each claim with its direction-sign decision (per-claim, not per-cluster)
        c["flip_sign"] = is_bad_direction_value(c)

    # cluster unique metric names
    unique_metrics = sorted(set(c["metric"] for c in intensity_claims))
    print(f"\nClustering {len(unique_metrics)} unique metric names...")
    print(f"Loading SBERT model: {SBERT_MODEL}")
    model = SentenceTransformer(SBERT_MODEL)
    metric_to_cluster = cluster_metrics(unique_metrics, model)
    n_clusters = len(set(metric_to_cluster.values()))
    print(f"  produced {n_clusters} clusters")

    for c in intensity_claims:
        c["cluster_id"] = metric_to_cluster[c["metric"]]

    # build cluster catalog
    cluster_to_metrics = defaultdict(list)
    for metric, cid in metric_to_cluster.items():
        cluster_to_metrics[cid].append(metric)

    cluster_catalog = {}
    for cid, metrics in cluster_to_metrics.items():
        claims_in_c = [c for c in intensity_claims if c["cluster_id"] == cid]
        companies = sorted(set(c["company_name"] for c in claims_in_c))
        units = sorted(set(c["normalized_unit"] for c in claims_in_c if c["normalized_unit"]))
        # show how many claims in this cluster were treated as flipped vs not
        n_flipped = sum(1 for c in claims_in_c if c["flip_sign"])
        cluster_catalog[f"cluster_{cid}"] = {
            "metrics_in_cluster": sorted(metrics),
            "n_claims": len(claims_in_c),
            "n_companies": len(companies),
            "companies": companies,
            "units_seen": units,
            "has_bad_topic": any(has_bad_topic(m) for m in metrics),
            "n_claims_flipped_sign": n_flipped,
            "n_claims_unflipped": len(claims_in_c) - n_flipped,
        }

    Path(OUTPUT_CLUSTERS).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CLUSTERS, "w", encoding="utf-8") as f:
        json.dump(cluster_catalog, f, indent=2, ensure_ascii=False)
    print(f"  cluster catalog saved")

    # group claims by (cluster, year, unit)
    # within a group we additionally split by flip_sign to avoid mixing
    # reduction-% with level-value within the same z-score computation
    cluster_year_unit_groups = defaultdict(list)
    for c in intensity_claims:
        key = (c["cluster_id"], c["year"], c["normalized_unit"], c["flip_sign"])
        cluster_year_unit_groups[key].append(c)

    print(f"\nComputing peer z-scores per (cluster, year, unit, flip_sign) group...")
    per_claim_results = []
    per_company_year = defaultdict(lambda: {"z_scores": {}})

    n_groups_computed = 0
    n_groups_skipped = 0
    for (cid, year, unit, flip_sign), group_claims in cluster_year_unit_groups.items():
        # aggregate to one value per company within this group
        per_company = defaultdict(list)
        for c in group_claims:
            nv = c["normalized_value"]
            # backstop: only keep genuinely numeric values
            if isinstance(nv, (int, float)) and not isinstance(nv, bool):
                per_company[c["company_name"]].append(nv)
        per_company = {k: v for k, v in per_company.items() if v}
        company_values = {comp: float(np.mean(vals)) for comp, vals in per_company.items()}

        sample_metric = group_claims[0]["metric"] if group_claims else "?"

        # SIGN-SPLIT: a metric cluster can mix non-comparable sub-types - e.g. a
        # reduction-% (negative) and a share-% (positive). z-scoring them together
        # is invalid. split the group by sign of the value and z-score each
        # sign-consistent subgroup independently (each needs >= MIN_COMPANIES_FOR_PEER).
        sign_subgroups = {
            "nonneg": {c: v for c, v in company_values.items() if v >= 0},
            "neg":    {c: v for c, v in company_values.items() if v < 0},
        }

        group_produced = False
        for sign_label, subgroup in sign_subgroups.items():
            if len(subgroup) < MIN_COMPANIES_FOR_PEER:
                n_groups_skipped += 1
                continue

            values = list(subgroup.values())
            peer_mean = float(np.mean(values))
            peer_std = float(np.std(values))
            if peer_std == 0:
                n_groups_skipped += 1
                continue

            for company, val in subgroup.items():
                z = (val - peer_mean) / peer_std
                if flip_sign:
                    z = -z

                sign_tag = "level" if flip_sign else "reduction_or_good"
                # cluster key now also records the sign subgroup, so a reduction-%
                # subgroup and a share-% subgroup stay distinct
                cluster_key = f"cluster_{cid}__{unit}__{sign_tag}__{sign_label}"

                key_cy = f"{company}__{year}"
                per_company_year[key_cy]["z_scores"][cluster_key] = round(z, 4)
                per_company_year[key_cy]["company_name"] = company
                per_company_year[key_cy]["year"] = year

                per_claim_results.append({
                    "company_name": company,
                    "year": year,
                    "cluster_id": cid,
                    "metric_sample": sample_metric,
                    "normalized_unit": unit,
                    "sign_subgroup": sign_label,
                    "company_value": val,
                    "peer_mean": peer_mean,
                    "peer_std": peer_std,
                    "n_peers": len(subgroup),
                    "z_score": z,
                    "flip_sign_applied": flip_sign,
                    "interpretation": "level of bad thing (lower=better)" if flip_sign
                                      else "reduction of bad thing or good thing (higher=better)",
                })
            group_produced = True

        if group_produced:
            n_groups_computed += 1

    print(f"  computed: {n_groups_computed} groups | skipped: {n_groups_skipped} (insufficient peers or zero variance)")

    # aggregate per company-year
    final_summary = {}
    for key, data in per_company_year.items():
        z_values = list(data["z_scores"].values())
        peer_deviation_score = round(float(np.mean(z_values)), 4) if z_values else None
        final_summary[key] = {
            "company_name": data["company_name"],
            "year": data["year"],
            "n_clusters_with_data": len(z_values),
            "z_scores_by_cluster": data["z_scores"],
            "peer_deviation_score": peer_deviation_score,
        }

    with open(OUTPUT_CLAIMS, "w", encoding="utf-8") as f:
        for rec in per_claim_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Per-claim z-scores:    {OUTPUT_CLAIMS}")
    print(f"Per-company-year:      {OUTPUT_SUMMARY}")
    print(f"Cluster catalog:       {OUTPUT_CLUSTERS}")


if __name__ == "__main__":
    main()