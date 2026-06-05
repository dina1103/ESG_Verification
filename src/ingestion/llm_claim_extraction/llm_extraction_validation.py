import json
import re
import time
import requests
from pathlib import Path

INPUT_FILE  = r"C:\Users\dina_\Desktop\esg_verification\src\ingestion\ml_promise_dataset\English_test.json"
PROMPT_FILE = r"C:\Users\dina_\Desktop\esg_verification\src\ingestion\llm_claim_extraction\llm_extraction_prompt.txt"
OUTPUT_FILE = r"C:\Users\dina_\Desktop\esg_verification\data\processed\llm_extraction_validation_results.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"


def call_llm(prompt_template, paragraph_text):
    full_prompt = prompt_template.replace("{paragraph}", paragraph_text.replace('"', "'"))
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 2048, "num_ctx": 8192},
        },
        timeout=450,
    )
    response.raise_for_status()
    return response.json()["response"]


def parse_llm_response(raw_text):
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        original_error = str(e)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), None
        except json.JSONDecodeError:
            pass
    last = text.rfind("},")
    if last != -1:
        try:
            return json.loads(text[:last + 1] + "]}"), "repaired_from_truncation"
        except json.JSONDecodeError:
            pass
    return None, f"JSON decode error: {original_error}"


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
    tp = tn = fp = fn = 0
    parse_fail = 0
    run_start = time.time()

    for i, rec in enumerate(clean):
        gold_yes = (rec["promise_status"] == "Yes")
        t0 = time.time()
        try:
            raw = call_llm(prompt_template, rec["data"])
            parsed, repair = parse_llm_response(raw)
        except Exception as e:
            raw, parsed, repair = None, None, f"llm_error: {e}"
        elapsed = time.time() - t0

        if parsed is not None and isinstance(parsed, dict):
            claims = parsed.get("claims", [])
            pred_yes = len(claims) > 0
            err = repair
        else:
            claims, pred_yes, err = [], False, repair
            parse_fail += 1

        if gold_yes and pred_yes: tp += 1
        elif gold_yes and not pred_yes: fn += 1
        elif not gold_yes and pred_yes: fp += 1
        else: tn += 1

        results.append({
            "url": rec["URL"], "page": rec["page_number"],
            "gold": rec["promise_status"], "gold_yes": gold_yes,
            "pred_yes": pred_yes, "n_claims": len(claims),
            "parse_error": err, "elapsed": elapsed,
        })

        if (i + 1) % 25 == 0:
            eta = (time.time() - run_start) / (i + 1) * (len(clean) - i - 1)
            print(f"  [{i+1}/{len(clean)}] tp={tp} fn={fn} fp={fp} tn={tn} "
                  f"parse_fail={parse_fail} eta={eta/60:.0f}min")

    n = len(clean)
    acc  = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0          # of flagged, how many real
    rec  = tp / (tp + fn) if (tp + fn) else 0          # of real promises, how many caught
    spec = tn / (tn + fp) if (tn + fp) else 0          # of no-promise, how many correctly empty
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    print("\n" + "=" * 70)
    print("BINARY DETECTION vs ML-Promise promise_status")
    print("=" * 70)
    print(f"Clean test paragraphs: {n}  (Yes={tp+fn}, No={tn+fp})")
    print(f"\n                  pred claim   pred no-claim")
    print(f"  gold promise     {tp:4d} (TP)    {fn:4d} (FN)")
    print(f"  gold no-promise  {fp:4d} (FP)    {tn:4d} (TN)")
    print(f"\nAccuracy:    {acc:.3f}   (note: 67% Yes baseline — don't lead with this)")
    print(f"Recall:      {rec:.3f}   <-- KEY: of real promises, fraction caught")
    print(f"Specificity: {spec:.3f}   of no-promise paragraphs, fraction correctly left empty")
    print(f"Precision:   {prec:.3f}   of flagged paragraphs, fraction that were real promises")
    print(f"F1:          {f1:.3f}")
    print(f"\nParse failures (counted as no-claim): {parse_fail}")
    print(f"Total time: {(time.time()-run_start)/60:.0f} min")

    summary = {"n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
               "accuracy": acc, "recall": rec, "specificity": spec,
               "precision": prec, "f1": f1, "parse_failures": parse_fail}
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "per_record": results},
              open(OUTPUT_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()