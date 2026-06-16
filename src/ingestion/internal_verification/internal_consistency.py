import json
import re
import time
from pathlib import Path
from collections import defaultdict
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_JSONL    = r"data\processed\llm_claim_extraction_result.jsonl"
INPUT_TABLES   = r"data\processed\tables_clean.parquet"
OUTPUT_CLAIMS  = r"data\processed\internal_consistency_claim_level.jsonl"
OUTPUT_SUMMARY = r"data\processed\internal_consistency_company_year.json"

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
METRIC_MATCH_THRESHOLD = 0.75
MIN_VERDICTS_FOR_SCORE = 3

NULLISH = {"", "n/a", "none", "na", "null"}

FAMILIES = [
    ("ghg_emissions",      re.compile(r'(?i)\b(emission|co2|co\u2082|carbon|ghg|scope\s*[123]|decarboni)\b'), "down"),
    ("energy_consumption", re.compile(r'(?i)\b(energy consumption|energy use|electricity consumption|'
                                      r'power consumption|energy intensity|fuel consumption|specific energy)\b'), "down"),
    ("renewable_energy",   re.compile(r'(?i)\b(renewable|solar|wind|green tariff|clean energy|green electricity)\b'), "up"),
    ("water",              re.compile(r'(?i)\bwater\b'), "down"),
    ("waste",              re.compile(r'(?i)\b(waste|landfill)\b'), "down"),
    ("recycling",          re.compile(r'(?i)\b(recycl|circular|secondary material|reuse)\b'), "up"),
    ("diversity",          re.compile(r'(?i)\b(diversity|women|gender|female|inclusion)\b'), "up"),
    ("safety",             re.compile(r'(?i)\b(safety|injur|accident|incident|fatalit)\b'), "down"),
]

MEASURABLE = re.compile(r'(?i)\b(emission|co2|co\u2082|carbon|ghg|scope\s*[123]|energy|electricity|'
                        r'fuel|water|waste|landfill|recycl|renewable|consumption|intensity|'
                        r'spill|pollution|nox|sox|biodiversit|'
                        r'diversity|women|gender|female|safety|injur|accident|incident|fatalit|turnover)\b')
TARGET = re.compile(r'(?i)(\bby\s*20\d{2}|target|aim to|goal|will reduce|plan to|commit|'
                    r'save an additional|reduce .* by \d)')
NONDIRECTIONAL = re.compile(r'(?i)\b(measures|measured by|documents|defined as|refers to|'
                            r'is calculated|are calculated|represents|comprises|consists of|'
                            r'not directly comparable|not comparable|methodology|definition of)\b')
# ---- false-positive filters (raise contradiction precision) ----
RISK_FRAMING = re.compile(r'(?i)\b(could (significantly )?impact|categoris\w+ climate|'
                          r'climate-related risks|risks and opportunities|impact demand|'
                          r'demand for our|may affect|exposure to|could face|slower progress)\b')
WRONG_SUBJECT = re.compile(r'(?i)\b(unit sales|business activities|growth in both unit|'
                           r'continued growth|impact demand for|eu set|set new ambitious|'
                           r'set new|governmental regulation|negative impact on earnings)\b')
TEMP_FRAMING = re.compile(r'(?i)(keep the increase|global temperature|temperature .*1\.5|'
                          r'aims to keep|do our part|aligned with the (united nations|paris))')
TARGET_LANG = re.compile(r'(?i)\b(commit to|we commit|by 20[2-9]\d|net zero target|'
                         r'we anticipate|will reduce|target set|intend to)\b')
FRAMING = re.compile(r'(?i)\b(global leader|leader in|recognizes that|is one of the most|'
                     r'most pressing|pressing global|we recognize|advancements)\b')
ACTION = re.compile(r'(?i)\b(reduc\w+|cut|cutting|lower\w+|decreas\w+|minimi\w+|increas\w+|'
                    r'expand\w+|improv\w+|grow\w+|switch\w+|avoid\w+|achiev\w+)\b')

# evidence-side filters: exclude product comparisons and financial-only rows from being used as evidence
PRODUCT_COMP = re.compile(r'(?i)(compared to conventional|lifecycle emissions of|all-in lifecycle|'
                          r'less than a (conventional|gasoline)|vs\.? a (conventional|gasoline)|'
                          r'well-to-wheels|greet|model [3sy]\b|leaf emits|than its gasoline|'
                          r'conventional vehicles of the same)')
FINANCIAL_EV = re.compile(r'(?i)(earning power|earnings|profit margin|revenue|invest)')
EVID_MEASURABLE = re.compile(r'(?i)\b(emission|co2|ghg|carbon|scope|energy|electricity|water|waste|'
                             r'renewable|recycl|women|gender|diversity|injur|incident|safety)\b')
# ambiguous-intent qual words that are not a clear directional commitment on their own
AMBIG_INTENT = re.compile(r'(?i)\b(control|maintain|manage)\b')
CLEAR_INTENT = re.compile(r'(?i)\b(reduc|cut|lower|decreas|increas|expand|'
                          r'improv\w+ (the )?(share|percentage|rate|intensity))\b')

MACRO = re.compile(r'(?i)(\bthe world\b|world\u2019?s |\bglobal |\bworldwide\b|industry-wide|'
                   r'\bthe planet\b|\bglobally\b|across the (sector|industry))')

TBL_DOWN = re.compile(r'(?i)\b(decreas\w*|reduc\w*|lower\w*|fell|declin\w*|cut|fall)\b')
TBL_UP   = re.compile(r'(?i)\b(increas\w*|grew|rose|higher|growth)\b')
TBL_PCT  = re.compile(r'(\d+(?:\.\d+)?)\s*%')

DOWN_WORDS = ["reduced","reducing","reduce","decrease","decreased","lowered","lower","cut","cutting",
              "declined","declining","minimized","fell","dropped","diminished","lowering"]
UP_WORDS   = ["increased","increasing","increase","grew","growing","grow","expanded","expanding",
              "raised","rose","boosted","doubled","tripled","improving","improved"]


def present(v):
    return not (v is None or (isinstance(v, str) and v.strip().lower() in NULLISH))


def metric_family(text):
    for name, rx, improve in FAMILIES:
        if rx.search(text):
            return name, improve
    return None, None


def has_measurable_metric(c):
    return c.get("metric") and bool(MEASURABLE.search(str(c["metric"])))


def has_number(text):
    return bool(re.search(r'\d', text))


def stated_direction(text):
    t = text.lower()
    if any(w in t for w in DOWN_WORDS): return "down"
    if any(w in t for w in UP_WORDS):   return "up"
    return None


def measured_direction(c):
    if c.get("source") == "table":
        return c.get("direction")
    bv, qv = c.get("baseline_value"), c.get("quantified_value")
    if present(bv) and present(qv):
        try:
            b, q = float(bv), float(qv)
            return "down" if q < b else "up" if q > b else "stable"
        except (ValueError, TypeError):
            pass
    t = c["claim_text"].lower()
    if any(w in t for w in DOWN_WORDS): return "down"
    if any(w in t for w in UP_WORDS):   return "up"
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
            if rec.get("parse_error"):
                continue
            for claim in (rec.get("parsed_claims") or []):
                rows.append({
                    "block_id": rec.get("block_id"),
                    "company_name": rec.get("company_name"),
                    "year": rec.get("year"),
                    "claim_text": claim.get("claim_text", "") or "",
                    "claim_type": (claim.get("claim_type") or "").lower(),
                    "metric": claim.get("metric"),
                    "quantified_value": claim.get("quantified_value"),
                    "baseline_value": claim.get("baseline_value"),
                    "source": "claim",
                })
    return rows


def load_table_evidence(path):
    t = pd.read_parquet(path)
    ev = []
    for _, r in t.iterrows():
        txt = re.sub(r'\s+', ' ', str(r["text"])).strip()
        if len(txt) > 160 or not MEASURABLE.search(txt):
            continue
        if not TBL_PCT.search(txt) or TARGET.search(txt) or MACRO.search(txt):
            continue
        has_down = bool(TBL_DOWN.search(txt)); has_up = bool(TBL_UP.search(txt))
        if has_down == has_up:
            continue
        ev.append({
            "block_id": r["table_row_id"], "company_name": r["company_name"], "year": r["year"],
            "claim_text": txt, "metric": txt,
            "direction": "down" if has_down else "up", "source": "table",
        })
    return ev


def is_qualitative(c):
    if c["claim_type"] not in ("narrative", "commitment"):
        return False
    if present(c["quantified_value"]):
        return False
    if not has_measurable_metric(c):
        return False
    q = c["claim_text"]
    if stated_direction(q) is None:
        return False
    if NONDIRECTIONAL.search(q):
        return False
    # false-positive filters: framing, risk-hedging, wrong-subject, temperature, targets, identity
    if RISK_FRAMING.search(q) or WRONG_SUBJECT.search(q) or TEMP_FRAMING.search(q):
        return False
    if TARGET_LANG.search(q) or FRAMING.search(q):
        return False
    # require a first-person action verb on the metric (firm acting, not just topic mention)
    if not ACTION.search(q):
        return False
    # ambiguous "control/maintain/manage" without a clear directional verb is not directional
    if AMBIG_INTENT.search(q) and not CLEAR_INTENT.search(q):
        return False
    fam, _ = metric_family(q + " " + str(c["metric"]))
    return fam is not None


def is_quant_evidence(c):
    if c["claim_type"] != "achievement":
        return False
    if not has_measurable_metric(c):
        return False
    if not (present(c["quantified_value"]) or has_number(c["claim_text"])):
        return False
    if TARGET.search(c["claim_text"]):
        return False
    if MACRO.search(c["claim_text"]):
        return False
    # evidence-side filters: not a product comparison, not financial-only
    if PRODUCT_COMP.search(c["claim_text"]):
        return False
    if FINANCIAL_EV.search(c["claim_text"]) and not EVID_MEASURABLE.search(c["claim_text"]):
        return False
    if measured_direction(c) is None:
        return False
    fam, _ = metric_family(c["claim_text"] + " " + str(c["metric"]))
    return fam is not None


def dedupe(items):
    seen, out = set(), []
    for c in items:
        key = (c["claim_text"].strip().lower(), str(c["metric"]).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def assess(qc, ev):
    qf, q_improve = metric_family(qc["claim_text"] + " " + str(qc["metric"]))
    ef, e_improve = metric_family(ev["claim_text"] + " " + str(ev["metric"]))
    if qf is None or ef is None or qf != ef:
        return None
    stated = stated_direction(qc["claim_text"])
    measured = measured_direction(ev)
    if stated is None or measured is None:
        return None
    stated_is_improvement = (stated == q_improve)
    measured_is_improvement = (measured == e_improve)
    if stated_is_improvement == measured_is_improvement:
        return ("aligned", qf, stated, measured)
    return ("contradicted", qf, stated, measured)


def main():
    print(f"Loading claims from {INPUT_JSONL}...")
    claims = load_claims(INPUT_JSONL)
    print(f"  {len(claims):,} claims")

    print(f"Loading table evidence from {INPUT_TABLES}...")
    table_ev = load_table_evidence(INPUT_TABLES)
    print(f"  {len(table_ev):,} directional table-evidence rows")

    by_group = defaultdict(lambda: {"claims": [], "tables": []})
    for c in claims:
        by_group[(c["company_name"], c["year"])]["claims"].append(c)
    for e in table_ev:
        by_group[(e["company_name"], e["year"])]["tables"].append(e)
    print(f"  {len(by_group)} company-year groups")

    print(f"\nLoading SBERT: {SBERT_MODEL}...")
    model = SentenceTransformer(SBERT_MODEL)

    Path(OUTPUT_CLAIMS).parent.mkdir(parents=True, exist_ok=True)
    per_claim, summary = [], {}
    t0 = time.time()

    for i, ((company, year), g) in enumerate(sorted(by_group.items()), 1):
        quals = dedupe([c for c in g["claims"] if is_qualitative(c)])
        claim_ev = dedupe([c for c in g["claims"] if is_quant_evidence(c)])
        evidence = claim_ev + g["tables"]

        if not quals or not evidence:
            summary[f"{company}__{year}"] = {
                "company_name": company, "year": year,
                "n_qualitative": len(quals), "n_evidence": len(evidence),
                "n_evidence_claim": len(claim_ev), "n_evidence_table": len(g["tables"]),
                "n_aligned": 0, "n_contradicted": 0, "n_directional_verdicts": 0,
                "consistency_score": None, "contradiction_score": None,
                "n_contradictions": 0, "note": "insufficient data",
            }
            print(f"  [{i:3d}/{len(by_group)}] {company} {year}: skip (qual={len(quals)} ev={len(evidence)})")
            continue

        q_emb = model.encode([str(c["metric"]) for c in quals], show_progress_bar=False, batch_size=64)
        e_emb = model.encode([str(c["metric"]) for c in evidence], show_progress_bar=False, batch_size=64)
        sim = cosine_similarity(q_emb, e_emb)

        n_al = n_con = 0
        seen_pairs = set()
        for r, qc in enumerate(quals):
            j = int(sim[r].argmax())
            s = float(sim[r][j])
            if s < METRIC_MATCH_THRESHOLD:
                continue
            ev = evidence[j]
            # skip self/near-self matches (a claim matched to itself is meaningless)
            if qc["claim_text"].strip().lower()[:40] == ev["claim_text"].strip().lower()[:40]:
                continue
            pair_key = (qc["claim_text"].strip().lower(), ev["claim_text"].strip().lower())
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            verdict_info = assess(qc, ev)
            if verdict_info is None:
                continue
            verdict, fam, stated, measured = verdict_info
            if verdict == "aligned":
                n_al += 1
            else:
                n_con += 1
            per_claim.append({
                "company_name": company, "year": year, "metric_family": fam,
                "qual_block_id": qc["block_id"], "qual_claim": qc["claim_text"],
                "qual_metric": qc["metric"], "stated_direction": stated,
                "evidence_source": ev["source"], "evidence_block_id": ev["block_id"],
                "evidence_claim": ev["claim_text"], "evidence_metric": ev["metric"],
                "measured_direction": measured, "metric_similarity": round(s, 4),
                "verdict": verdict,
            })

        denom = n_al + n_con
        if denom < MIN_VERDICTS_FOR_SCORE:
            cons_score = contra_score = None
            note = f"insufficient_evidence ({denom} verdicts)"
        else:
            cons_score = round(n_al / denom, 4)
            contra_score = round(n_con / denom, 4)
            note = None

        summary[f"{company}__{year}"] = {
            "company_name": company, "year": year,
            "n_qualitative": len(quals), "n_evidence": len(evidence),
            "n_evidence_claim": len(claim_ev), "n_evidence_table": len(g["tables"]),
            "n_aligned": n_al, "n_contradicted": n_con, "n_directional_verdicts": denom,
            "consistency_score": cons_score, "contradiction_score": contra_score,
            "n_contradictions": n_con, "note": note,
        }
        cs = f"{cons_score:.3f}" if cons_score is not None else "N/A"
        print(f"  [{i:3d}/{len(by_group)}] {company} {year}: qual={len(quals)} "
              f"ev={len(evidence)}(c{len(claim_ev)}/t{len(g['tables'])}) | "
              f"aligned={n_al} contradicted={n_con} | consistency={cs}")

    with open(OUTPUT_CLAIMS, "w", encoding="utf-8") as f:
        for rec in per_claim:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    scored = sum(1 for v in summary.values() if v["consistency_score"] is not None)
    total_con = sum(v["n_contradictions"] for v in summary.values())
    print(f"\n{'='*70}\nCOMPLETE in {time.time()-t0:.0f}s")
    print(f"company-years scored: {scored}/{len(summary)} | contradictions: {total_con}")
    print(f"verdicts: {len(per_claim):,}")
    print(f"  {OUTPUT_CLAIMS}\n  {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()