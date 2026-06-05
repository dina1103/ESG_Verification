import json
from pathlib import Path

TEST_FILE  = r"C:\Users\dina_\Desktop\esg_verification\src\ingestion\ml_promise_dataset\English_test.json"
TRAIN_FILE = r"C:\Users\dina_\Desktop\esg_verification\src\ingestion\ml_promise_dataset\PromiseEval_Trainset_English.json"
PROMPT_FILE= r"C:\Users\dina_\Desktop\esg_verification\src\ingestion\llm_claim_extraction\llm_extraction_prompt.txt"
CLEAN_OUT  = r"C:\Users\dina_\Desktop\esg_verification\src\ingestion\ml_promise_dataset\English_test_clean.json"


def norm(s):
    return "".join(s.split()).lower()


def extract_fewshot_paragraphs(prompt_text):
    # the prompt embeds examples as: Paragraph: "<text>"
    # pull those quoted paragraphs out so we can check them against the test set
    import re
    return re.findall(r'Paragraph:\s*"(.*?)"\s*Output:', prompt_text, re.DOTALL)


def main():
    test = json.load(open(TEST_FILE, encoding="utf-8"))
    train = json.load(open(TRAIN_FILE, encoding="utf-8"))
    prompt = open(PROMPT_FILE, encoding="utf-8").read()

    fewshots = extract_fewshot_paragraphs(prompt)
    print(f"Few-shot paragraphs found in prompt: {len(fewshots)}")
    for fs in fewshots:
        print(f"  - {fs[:70]}...")
    print()

    train_norm = {norm(r["data"]) for r in train}
    fewshot_norm = {norm(fs) for fs in fewshots}

    clean, rm_train, rm_fs = [], [], []
    for i, r in enumerate(test):
        n = norm(r["data"])
        if n in fewshot_norm:
            rm_fs.append(i); continue
        if n in train_norm:
            rm_train.append(i); continue
        clean.append(r)

    print(f"Original test:           {len(test)}")
    print(f"Removed (few-shot match): {len(rm_fs)}  indices={rm_fs}")
    print(f"Removed (train overlap):  {len(rm_train)}  indices={rm_train}")
    print(f"Clean test set:           {len(clean)}")

    Path(CLEAN_OUT).parent.mkdir(parents=True, exist_ok=True)
    json.dump(clean, open(CLEAN_OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved to {CLEAN_OUT}")


if __name__ == "__main__":
    main()