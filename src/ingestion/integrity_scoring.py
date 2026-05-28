import json
import math
from pathlib import Path

SDG13_JSON   = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\sdg13_climate_company_year.json"
SDG16_JSON   = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\sdg16_governance_company_year.json"
INTCON_JSON  = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\internal_consistency_company_year.json"
PEER_JSON    = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\peer_comparison_company_year.json"

OUTPUT_JSON  = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\integrity_score_company_year.json"
OUTPUT_CSV   = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\integrity_score_company_year.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm_key(company, year):
    # defensive: strip whitespace and trailing punctuation from company name
    c = str(company).strip().rstrip(".").strip()
    return f"{c}__{year}"


def sigmoid(z):
    # map a z-score (~ -3..+3) to 0-1; z=0 -> 0.5
    return 1.0 / (1.0 + math.exp(-z))


def reindex(raw):
    # rebuild a layer dict keyed by normalised (company, year)
    out = {}
    for v in raw.values():
        company = v.get("company_name")
        year = v.get("year")
        if company is None or year is None:
            continue
        out[norm_key(company, year)] = v
    return out


def main():
    print("Loading the four verification layers...")
    sdg13  = reindex(load_json(SDG13_JSON))
    sdg16  = reindex(load_json(SDG16_JSON))
    intcon = reindex(load_json(INTCON_JSON))
    peer   = reindex(load_json(PEER_JSON))
    print(f"  sdg13:  {len(sdg13)} company-years")
    print(f"  sdg16:  {len(sdg16)} company-years")
    print(f"  intcon: {len(intcon)} company-years")
    print(f"  peer:   {len(peer)} company-years")

    # union of every company-year seen in any layer
    all_keys = sorted(set(sdg13) | set(sdg16) | set(intcon) | set(peer))
    print(f"\nTotal distinct company-years: {len(all_keys)}")

    results = {}
    for key in all_keys:
        # pull each layer's score; absent key or null both -> None
        s13 = (sdg13.get(key)  or {}).get("sdg13_alignment_score")
        s16 = (sdg16.get(key)  or {}).get("sdg16_alignment_score")
        sic = (intcon.get(key) or {}).get("internal_consistency_score")
        zpeer = (peer.get(key) or {}).get("peer_deviation_score")

        # peer z-score -> 0-1 via sigmoid (only if present)
        speer = sigmoid(zpeer) if zpeer is not None else None

        # collect available layers
        components = {
            "sdg13_alignment": s13,
            "sdg16_alignment": s16,
            "internal_consistency": sic,
            "peer_comparison": speer,
        }
        available = {k: v for k, v in components.items() if v is not None}

        # equal-weighted mean of available layers
        if available:
            integrity = round(sum(available.values()) / len(available), 4)
        else:
            integrity = None

        # recover company / year from whichever layer has the record
        src = (sdg13.get(key) or sdg16.get(key)
               or intcon.get(key) or peer.get(key) or {})

        results[key] = {
            "company_name": src.get("company_name"),
            "year": src.get("year"),
            "sdg13_alignment": s13,
            "sdg16_alignment": s16,
            "internal_consistency": sic,
            "peer_comparison_raw_z": zpeer,
            "peer_comparison_sigmoid": round(speer, 4) if speer is not None else None,
            "n_layers_available": len(available),
            "layers_used": sorted(available.keys()),
            "integrity_score": integrity,
        }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # flat CSV for the Phase 5 modelling step
    cols = ["company_name", "year", "sdg13_alignment", "sdg16_alignment",
            "internal_consistency", "peer_comparison_sigmoid",
            "n_layers_available", "integrity_score"]
    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for key in all_keys:
            r = results[key]
            row = [r.get(c) for c in cols]
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")

    # console summary
    scored = [r for r in results.values() if r["integrity_score"] is not None]
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Company-years with an integrity score: {len(scored)}/{len(results)}")
    if scored:
        vals = [r["integrity_score"] for r in scored]
        print(f"  integrity score range: {min(vals):.3f} - {max(vals):.3f}")
        print(f"  mean: {sum(vals)/len(vals):.3f}")
    from collections import Counter
    dist = Counter(r["n_layers_available"] for r in results.values())
    print(f"  layers-available distribution: {dict(sorted(dist.items()))}")
    print(f"\nJSON: {OUTPUT_JSON}")
    print(f"CSV:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()