import json
import re
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_JSONL    = r"data\processed\llm_claim_extraction_result.jsonl"
OUTPUT_CLAIMS  = r"data\processed\internal_consistency_claim_level.jsonl"
OUTPUT_SUMMARY = r"data\processed\internal_consistency_company_year.json"

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.7

# minimum directional verdicts (aligned+contradicted) required to compute a score
# below this, the score rests on too few claims to be meaningful -> null
MIN_VERDICTS_FOR_SCORE = 2

# OPTION 1: minimum metric-field similarity to accept a qual->quant pairing
METRIC_MATCH_THRESHOLD = 0.75

# metric keyword -> direction that means "improvement"
# applied when narrative uses ambiguous words like "improved", "progress"
METRIC_IMPROVEMENT_DIRECTION = {
    "emission": "down",
    "ghg": "down",
    "co2": "down",
    "carbon": "down",
    "water": "down",
    "waste": "down",
    "energy": "down",       # energy consumption
    "incident": "down",
    "injury": "down",
    "accident": "down",
    "spill": "down",
    "violation": "down",
    "renewable": "up",
    "recycled": "up",
    "recycling": "up",
    "diversity": "up",
    "women": "up",
    "training": "up",
    "certification": "up",
    "efficiency": "up",
    "saving": "up",
}

# directional verbs/adjectives
DOWN_WORDS = ["reduced", "reducing", "decrease", "decreased", "lowered", "lower",
              "cut", "declined", "declining", "minimized", "minimizing", "fell",
              "dropped", "diminished"]
UP_WORDS   = ["increased", "increasing", "grew", "growing", "expanded", "expanding",
              "raised", "rose", "boosted", "doubled", "tripled"]
AMBIGUOUS_IMPROVEMENT_WORDS = ["improved", "improving", "progress", "better", "enhanced",
                                "advanced", "strengthened"]
STABLE_WORDS = ["maintained", "sustained", "kept stable", "remained"]

CONJUNCTION_PATTERN = r"\b(?:but|while|however|although|whereas|yet)\b"


def load_claims(path):
    # read jsonl, flatten to one record per claim
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
            claims = rec.get("parsed_claims") or []
            for claim in claims:
                # attach parent record context
                rows.append({
                    "block_id": rec["block_id"],
                    "company_name": rec["company_name"],
                    "year": rec["year"],
                    "claim_text": claim.get("claim_text", ""),
                    "claim_type": claim.get("claim_type", ""),
                    "metric": claim.get("metric", "N/A"),
                    "quantified_value": claim.get("quantified_value"),
                    "unit": claim.get("unit"),
                    "baseline_year": claim.get("baseline_year"),
                    "baseline_value": claim.get("baseline_value"),
                    "target_year": claim.get("target_year"),
                    "scope": claim.get("scope", "N/A"),
                    "geography": claim.get("geography", "N/A"),
                    "framework_reference": claim.get("framework_reference", "none"),
                })
    return rows



def has_real_metric(c):
    # a usable metric field - not N/A / None / empty
    m = c.get("metric")
    if m is None:
        return False
    m = str(m).strip().lower()
    return m not in ("", "n/a", "none", "na")

def is_qualitative(c):
    # narrative/commitment without a number
    return c["claim_type"] in ("narrative", "commitment") and c["quantified_value"] is None


def is_quantitative_evidence(c):
    # achievements with a real measured value (skip future targets)
    return c["claim_type"] == "achievement" and c["quantified_value"] is not None


def split_on_conjunctions(text):
    # split on contrastive conjunctions for multi-direction narratives
    parts = re.split(CONJUNCTION_PATTERN, text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def expected_direction_from_text(text, metric):
    # extract directional intent from narrative text
    text_lower = text.lower()
    metric_lower = (metric or "").lower()

    # check explicit direction words first
    if any(w in text_lower for w in DOWN_WORDS):
        return "down"
    if any(w in text_lower for w in UP_WORDS):
        return "up"
    if any(w in text_lower for w in STABLE_WORDS):
        return "stable"

    # ambiguous "improvement" words resolved via metric lookup
    if any(w in text_lower for w in AMBIGUOUS_IMPROVEMENT_WORDS):
        for keyword, direction in METRIC_IMPROVEMENT_DIRECTION.items():
            if keyword in metric_lower or keyword in text_lower:
                return direction
        return "unknown"

    return "unknown"


def actual_direction_from_quant(quant_claim):
    # compute actual direction from quantitative claim
    baseline = quant_claim.get("baseline_value")
    current = quant_claim.get("quantified_value")

    # if explicit baseline + current, compute directly
    if baseline is not None and current is not None:
        if current < baseline:
            return "down"
        elif current > baseline:
            return "up"
        else:
            return "stable"

    # otherwise infer from quantitative claim text (e.g., "reduced by 30%")
    text_lower = quant_claim.get("claim_text", "").lower()
    if any(w in text_lower for w in DOWN_WORDS):
        return "down"
    if any(w in text_lower for w in UP_WORDS):
        return "up"
    return "unknown"


def assess_clause(qual_clause, qual_metric, quant_claim, similarity):
    # full assessment of one qualitative clause vs its best quantitative match
    if similarity < SIMILARITY_THRESHOLD:
        return {
            "verdict": "unsupported",
            "expected_direction": None,
            "actual_direction": None,
            "reason": f"no quantitative match above threshold (best sim={similarity:.3f})",
        }

    expected = expected_direction_from_text(qual_clause, qual_metric)
    actual = actual_direction_from_quant(quant_claim)

    if expected == "unknown" or actual == "unknown":
        return {
            "verdict": "direction_unknown",
            "expected_direction": expected,
            "actual_direction": actual,
            "reason": "could not determine direction",
        }

    if expected == actual:
        verdict = "aligned"
    else:
        verdict = "contradicted"

    return {
        "verdict": verdict,
        "expected_direction": expected,
        "actual_direction": actual,
        "reason": f"expected={expected}, actual={actual}",
    }


def main():
    print(f"Loading claims from {INPUT_JSONL}...")
    all_claims = load_claims(INPUT_JSONL)
    print(f"  loaded {len(all_claims):,} claims total")

    # group by (company, year)
    groups = defaultdict(list)
    for c in all_claims:
        groups[(c["company_name"], c["year"])].append(c)
    print(f"  grouped into {len(groups)} company-year combinations")

    print(f"\nLoading SBERT model: {SBERT_MODEL}...")
    model = SentenceTransformer(SBERT_MODEL)

    Path(OUTPUT_CLAIMS).parent.mkdir(parents=True, exist_ok=True)

    per_claim_results = []
    per_group_summary = {}

    print(f"\nProcessing company-year groups...\n")
    t0 = time.time()

    for i, ((company, year), claims) in enumerate(sorted(groups.items()), start=1):
        qualitative = [c for c in claims if is_qualitative(c)]
        quantitative = [c for c in claims if is_quantitative_evidence(c)]

        # skip groups where consistency cannot be checked
        if not qualitative or not quantitative:
            per_group_summary[f"{company}__{year}"] = {
                "company_name": company,
                "year": year,
                "n_qualitative": len(qualitative),
                "n_quantitative": len(quantitative),
                "n_aligned": 0,
                "n_contradicted": 0,
                "n_unsupported": 0,
                "n_direction_unknown": 0,
                "internal_consistency_score": None,
                "note": "insufficient data" if not qualitative or not quantitative else None,
            }
            print(f"  [{i:3d}/{len(groups)}] {company} {year}: skipped (qual={len(qualitative)}, quant={len(quantitative)})")
            continue

        # embed both pools
        quant_texts = [c["claim_text"] for c in quantitative]
        quant_embeds = model.encode(quant_texts, show_progress_bar=False)

        n_aligned = 0
        n_contradicted = 0
        n_unsupported = 0
        n_direction_unknown = 0

        # OPTION 1: match qualitative -> quantitative on METRIC-FIELD similarity.
        # only qualitative claims WITH a real metric are assessed; the rest are
        # recorded as unsupported (no metric to match on).
        qual_with_metric = [q for q in qualitative if has_real_metric(q)]
        qual_no_metric   = [q for q in qualitative if not has_real_metric(q)]

        # every no-metric qualitative claim is unsupported by construction
        for q in qual_no_metric:
            n_unsupported += 1
            per_claim_results.append({
                "company_name": company, "year": year,
                "qualitative_block_id": q["block_id"],
                "qualitative_clause": q["claim_text"],
                "qualitative_metric": q.get("metric"),
                "best_match_block_id": None, "best_match_text": None,
                "best_match_metric": None, "similarity": None,
                "expected_direction": None, "actual_direction": None,
                "verdict": "unsupported",
                "reason": "qualitative claim has no usable metric field",
            })

        if qual_with_metric:
            qual_metric_texts = [str(q["metric"]) for q in qual_with_metric]
            quant_metric_texts = [str(c.get("metric")) for c in quantitative]
            qmet_embeds = model.encode(qual_metric_texts, show_progress_bar=False, batch_size=64)
            cmet_embeds = model.encode(quant_metric_texts, show_progress_bar=False, batch_size=64)
            metric_sim_matrix = cosine_similarity(qmet_embeds, cmet_embeds)
        else:
            metric_sim_matrix = None

        for row_idx, qual_claim in enumerate(qual_with_metric):
            clause = qual_claim["claim_text"]
            msims = metric_sim_matrix[row_idx]
            best_idx = int(msims.argmax())
            best_sim = float(msims[best_idx])
            best_quant = quantitative[best_idx]

            # require metric-field similarity above METRIC_MATCH_THRESHOLD
            if best_sim < METRIC_MATCH_THRESHOLD:
                n_unsupported += 1
                per_claim_results.append({
                    "company_name": company, "year": year,
                    "qualitative_block_id": qual_claim["block_id"],
                    "qualitative_clause": clause,
                    "qualitative_metric": qual_claim.get("metric"),
                    "best_match_block_id": best_quant["block_id"],
                    "best_match_text": best_quant["claim_text"],
                    "best_match_metric": best_quant.get("metric"),
                    "similarity": round(best_sim, 4),
                    "expected_direction": None, "actual_direction": None,
                    "verdict": "unsupported",
                    "reason": f"no quantitative claim with matching metric (best metric sim={best_sim:.3f})",
                })
                continue

            # already gated on metric match; pass 1.0 so assess_clause's own
            # similarity gate does not re-reject
            result = assess_clause(clause, qual_claim.get("metric"), best_quant, 1.0)

            if result["verdict"] == "aligned":
                n_aligned += 1
            elif result["verdict"] == "contradicted":
                n_contradicted += 1
            elif result["verdict"] == "unsupported":
                n_unsupported += 1
            else:
                n_direction_unknown += 1

            per_claim_results.append({
                "company_name": company,
                "year": year,
                "qualitative_block_id": qual_claim["block_id"],
                "qualitative_clause": clause,
                "qualitative_metric": qual_claim.get("metric"),
                "best_match_block_id": best_quant["block_id"],
                "best_match_text": best_quant["claim_text"],
                "best_match_metric": best_quant.get("metric"),
                "similarity": round(best_sim, 4),
                "expected_direction": result["expected_direction"],
                "actual_direction": result["actual_direction"],
                "verdict": result["verdict"],
                "reason": result["reason"],
            })

        # internal consistency score: aligned / (aligned + contradicted)
        # excludes unsupported and direction_unknown from denominator
        denom = n_aligned + n_contradicted

        # minimum-evidence threshold: a score from <MIN_VERDICTS claims is noise,
        # not a measurement - treat as insufficient evidence (null)
        if denom < MIN_VERDICTS_FOR_SCORE:
            score = None
            score_note = f"insufficient_evidence (only {denom} directional verdicts)"
        else:
            score = n_aligned / denom
            score_note = None

        per_group_summary[f"{company}__{year}"] = {
            "company_name": company,
            "year": year,
            "n_qualitative": len(qualitative),
            "n_quantitative": len(quantitative),
            "n_aligned": n_aligned,
            "n_contradicted": n_contradicted,
            "n_unsupported": n_unsupported,
            "n_direction_unknown": n_direction_unknown,
            "n_directional_verdicts": denom,
            "internal_consistency_score": round(score, 4) if score is not None else None,
            "note": score_note,
        }

        print(f"  [{i:3d}/{len(groups)}] {company} {year}: "
              f"qual={len(qualitative)} quant={len(quantitative)} | "
              f"aligned={n_aligned} contradicted={n_contradicted} "
              f"unsupported={n_unsupported} unknown={n_direction_unknown} | "
              f"score={score:.3f}" if score is not None else
              f"  [{i:3d}/{len(groups)}] {company} {year}: "
              f"qual={len(qualitative)} quant={len(quantitative)} | "
              f"aligned={n_aligned} contradicted={n_contradicted} "
              f"unsupported={n_unsupported} unknown={n_direction_unknown} | "
              f"score=N/A")

    # save per-claim results
    with open(OUTPUT_CLAIMS, "w", encoding="utf-8") as f:
        for rec in per_claim_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # save per-group summary
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(per_group_summary, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")
    print(f"Total time:            {elapsed:.0f}s")
    print(f"Per-claim records:     {len(per_claim_results):,}")
    print(f"Company-year groups:   {len(per_group_summary)}")
    print(f"Per-claim output:      {OUTPUT_CLAIMS}")
    print(f"Summary output:        {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()