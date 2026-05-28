import pandas as pd
from collections import Counter
from transformers import pipeline

OSDG_PATH = r"C:\Users\dina_\Desktop\esg_verification_draft\src\ingestion\sdg_testing_dataset\osdg-community-data-v2023-10-01.csv"

# sample size per SDG — balance coverage with CPU time
SAMPLE_PER_SDG = 100

# only test on sentences where all annotators agreed (highest-quality labels)
MIN_AGREEMENT = 1.0

# same config as sdg_classification.py
MIN_SCORE = 0.7
IN_SCOPE_SDGS = {"sdg12", "sdg13", "sdg16"}


def classify_pipeline(texts, clf):
    # run raw sdgBERT, apply threshold and in-scope filter
    results = clf(texts)
    final = []
    for r in results:
        label = r["label"].lower()
        score = r["score"]
        if label in IN_SCOPE_SDGS and score >= MIN_SCORE:
            final.append({"label": label, "score": score})
        else:
            final.append({"label": None, "score": score, "raw_label": label})
    return final


# load OSDG
print("Loading OSDG dataset...")
df = pd.read_csv(OSDG_PATH, sep="\t")
print(f"  Total: {len(df)} sentences\n")

# sample from each scope SDG
samples = {}
for sdg in [12, 13, 16]:
    pool = df[(df["sdg"] == sdg) & (df["agreement"] >= MIN_AGREEMENT)]
    n = min(SAMPLE_PER_SDG, len(pool))
    samples[f"sdg{sdg}"] = pool.sample(n, random_state=42)["text"].tolist()
    print(f"  SDG{sdg}: sampled {n} from {len(pool)} unanimous-agreement sentences")

# also sample confounders — SDGs sdgBERT might confuse with scope SDGs
for sdg in [7, 8, 9]:
    pool = df[(df["sdg"] == sdg) & (df["agreement"] >= MIN_AGREEMENT)]
    n = min(50, len(pool))
    samples[f"sdg{sdg}_confounder"] = pool.sample(n, random_state=42)["text"].tolist()
    print(f"  SDG{sdg} (confounder): sampled {n} from {len(pool)} unanimous-agreement sentences")

# load model
print("\nLoading sdgBERT...")
clf = pipeline("text-classification", model="sadickam/sdgBERT", truncation=True, max_length=512)
print("Loaded\n")

# evaluate
print("=" * 70)
print("EVALUATION: raw sdgBERT on OSDG held-out data")
print("=" * 70)

total_correct = 0
total_tested = 0

for label, texts in samples.items():
    expected = label.replace("_confounder", "")
    is_confounder = "_confounder" in label

    print(f"\n--- {label.upper()} ({len(texts)} sentences) ---")
    preds = classify_pipeline(texts, clf)

    correct = 0
    none_count = 0
    for pred in preds:
        final_label = pred["label"]

        if is_confounder:
            # for confounders: "correct" means pipeline didn't wrongly tag as scope SDG
            # (since our pipeline intentionally discards out-of-scope predictions, these should be None)
            if final_label is None:
                correct += 1
        else:
            if final_label == expected:
                correct += 1
        if final_label is None:
            none_count += 1

    if not is_confounder:
        total_correct += correct
        total_tested += len(texts)
        print(f"  Correct: {correct}/{len(texts)} ({correct/len(texts)*100:.1f}%)")
        print(f"  Labeled None: {none_count}/{len(texts)}")
    else:
        false_pos = sum(1 for p in preds if p["label"] in IN_SCOPE_SDGS)
        print(f"  Correctly not tagged to scope: {correct}/{len(texts)}")
        print(f"  False positives into scope SDGs: {false_pos}/{len(texts)}")
        label_counts = Counter(p["label"] for p in preds)
        print(f"  Label distribution: {dict(label_counts)}")

print()
print("=" * 70)
print("SUMMARY (scope SDGs only)")
print("=" * 70)
print(f"Overall accuracy: {total_correct}/{total_tested} ({total_correct/total_tested*100:.1f}%)")