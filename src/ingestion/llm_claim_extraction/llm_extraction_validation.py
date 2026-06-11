import json
import re
import time
import requests
from pathlib import Path

INPUT_FILE  = r"src\ingestion\ml_promise_dataset\English_test.json"
PROMPT_FILE = r"src\ingestion\llm_claim_extraction\llm_extraction_prompt.txt"
OUTPUT_FILE = r"data\processed\llm_extraction_validation_results.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

# ML-Promise "promise" aligns with forward-looking claim types
PROMISE_TYPES = {"target", "commitment"}


def call_llm(prompt_template, paragraph_text):
    full_prompt = prompt_template.replace("{paragraph}", paragraph_text.replace('"', "'"))
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 4096, "num_ctx": 8192},
        },
        timeout=450,
    )
    response.raise_for_status()
    return response.json()["response"]


def parse_llm_response(raw_text):
    if not raw_text:
        return None, "empty response"
    text = raw_text.strip()
    text = text.replace("```json", "").replace("```", "")
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    def try_load(s):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    parsed = try_load(text)
    if parsed is not None:
        return parsed, None
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    parsed = try_load(cleaned)
    if parsed is not None:
        return parsed, "repaired_control_chars"
    last = cleaned.rfind("}")
    if last != -1:
        for suffix in ("]}", "}]}", "}"):
            parsed = try_load(cleaned[:last + 1] + suffix)
            if parsed is not None:
                return parsed, "repaired_from_truncation"
        last_claim = cleaned.rfind("},")
        if last_claim != -1:
            parsed = try_load(cleaned[:last_claim + 1] + "]}")
            if parsed is not None:
                return parsed, "repaired_from_truncation"
    return None, "unrecoverable"


def confusion(records, pred_key):
    tp = tn = fp = fn = 0
    for r in records:
        g, p = r["gold_yes"], r[pred_key]
        if g and p: tp += 1
        elif g and not p: fn += 1
        elif not g and p: fp += 1
        else: tn += 1
    n = len(records)
    acc  = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": acc,
            "recall": rec, "specificity": spec, "precision": prec, "f1": f1}


def report(label, m, n):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    print(f"                  pred claim   pred no-claim")
    print(f"  gold promise     {m['tp']:4d} (TP)    {m['fn']:4d} (FN)")
    print(f"  gold no-promise  {m['fp']:4d} (FP)    {m['tn']:4d} (TN)")
    print(f"\nAccuracy:    {m['accuracy']:.3f}")
    print(f"Recall:      {m['recall']:.3f}   <-- of real promises, fraction caught")
    print(f"Specificity: {m['specificity']:.3f}")
    print(f"Precision:   {m['precision']:.3f}")
    print(f"F1:          {m['f1']:.3f}")


def main():
    prompt_template = open(PROMPT_FILE, encoding="utf-8").read()
    print(f"Prompt: {len(prompt_template)} chars")

    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: Ollama unreachable — {e}")
        return

    clean = json.load(open(INPUT_FILE, encoding="utf-8"))
    print(f"Validating on {len(clean)} clean test paragraphs\n")

    results = []
    parse_fail = 0
    run_start = time.time()

    for i, rec in enumerate(clean):
        gold_yes = (rec["promise_status"] == "Yes")
        try:
            raw = call_llm(prompt_template, rec["data"])
            parsed, repair = parse_llm_response(raw)
        except Exception as e:
            raw, parsed, repair = None, None, f"llm_error: {e}"

        if parsed is not None and isinstance(parsed, dict):
            claims = parsed.get("claims", [])
            err = repair
        else:
            claims, err = [], repair
            parse_fail += 1

        types = [str(c.get("claim_type", "")).lower() for c in claims]
        pred_yes_full = len(claims) > 0
        pred_yes_promise = any(t in PROMISE_TYPES for t in types)

        results.append({
            "url": rec["URL"], "page": rec["page_number"],
            "gold": rec["promise_status"], "gold_yes": gold_yes,
            "pred_yes": pred_yes_full,                # full taxonomy (any claim)
            "pred_yes_promise": pred_yes_promise,     # target/commitment only
            "n_claims": len(claims), "claim_types": types,
            "parse_error": err,
        })

        if (i + 1) % 25 == 0:
            eta = (time.time() - run_start) / (i + 1) * (len(clean) - i - 1)
            print(f"  [{i+1}/{len(clean)}] done, parse_fail={parse_fail}, eta={eta/60:.0f}min")

    n = len(clean)
    m_full = confusion(results, "pred_yes")
    m_prom = confusion(results, "pred_yes_promise")

    print(f"\nClean test paragraphs: {n}  (Yes={m_full['tp']+m_full['fn']}, No={m_full['tn']+m_full['fp']})")
    report("FULL TAXONOMY (any extracted claim = Yes)", m_full, n)
    report("PROMISE-RESTRICTED (target/commitment only = Yes)", m_prom, n)
    print(f"\nParse failures: {parse_fail}")
    print(f"Total time: {(time.time()-run_start)/60:.0f} min")

    json.dump({"summary_full": m_full, "summary_promise_restricted": m_prom,
               "per_record": results},
              open(OUTPUT_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()