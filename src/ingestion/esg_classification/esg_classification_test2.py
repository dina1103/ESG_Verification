from collections import Counter
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

MIN_SCORE = 0.5


def get_pillar_score(result, pillar_label):
    # binary models return {pillar, none}; flip the score when "none" is returned
    label = result["label"].lower()
    score = result["score"]
    return score if label == pillar_label.lower() else 1 - score


def classify_sentence(e_result, s_result, g_result):
    # raw per-pillar probabilities — argmax, 0.5 threshold, no keyword boost
    candidates = {
        "Environmental": get_pillar_score(e_result, "environmental"),
        "Social":        get_pillar_score(s_result, "social"),
        "Governance":    get_pillar_score(g_result, "governance"),
    }
    best_label = max(candidates, key=candidates.get)
    best_score = candidates[best_label]
    return best_label if best_score >= MIN_SCORE else "None"


print("Loading models...")
e_clf = pipeline("text-classification", model="ESGBERT/EnvironmentalBERT-environmental", truncation=True, max_length=512)
s_clf = pipeline("text-classification", model="ESGBERT/SocialBERT-social",               truncation=True, max_length=512)
g_clf = pipeline("text-classification", model="ESGBERT/GovernanceBERT-governance",       truncation=True, max_length=512)


def evaluate_on_dataset(dataset_name, label_column, expected_label):
    ds = load_dataset(dataset_name, split="train")
    texts = ds["text"]
    true_labels = ds[label_column]

    print(f"\nRunning pipeline on {dataset_name} ({len(texts)} sentences)...")

    batch_size = 32
    e_results, s_results, g_results = [], [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="  classifying"):
        batch = texts[i:i+batch_size]
        e_results.extend(e_clf(batch))
        s_results.extend(s_clf(batch))
        g_results.extend(g_clf(batch))

    predictions = [
        classify_sentence(e, s, g)
        for e, s, g in zip(e_results, s_results, g_results)
    ]

    correct = 0
    confusion = {"correct_positive": 0, "false_negative": 0, "correct_negative": 0, "false_positive": 0}
    for pred, true in zip(predictions, true_labels):
        if true == 1:
            if pred == expected_label:
                confusion["correct_positive"] += 1
                correct += 1
            else:
                confusion["false_negative"] += 1
        else:
            if pred != expected_label:
                confusion["correct_negative"] += 1
                correct += 1
            else:
                confusion["false_positive"] += 1

    accuracy = correct / len(texts) * 100
    print(f"  Accuracy: {accuracy:.1f}% ({correct}/{len(texts)})")
    print(f"  True positives (correct {expected_label}):   {confusion['correct_positive']}")
    print(f"  False negatives (missed {expected_label}):    {confusion['false_negative']}")
    print(f"  True negatives (correct not-{expected_label}): {confusion['correct_negative']}")
    print(f"  False positives (wrong {expected_label}):     {confusion['false_positive']}")

    neg_preds = [pred for pred, true in zip(predictions, true_labels) if true == 0]
    print(f"  Output distribution on negative sentences: {dict(Counter(neg_preds))}")

    return accuracy


env_acc = evaluate_on_dataset("ESGBERT/environmental_2k", "env", "Environmental")
soc_acc = evaluate_on_dataset("ESGBERT/social_2k",        "soc", "Social")
gov_acc = evaluate_on_dataset("ESGBERT/governance_2k",    "gov", "Governance")

print("\n=== SUMMARY (raw config — no boost, no backstop) ===")
print(f"Environmental: {env_acc:.1f}%  (ESGBERT paper reports ~93%)")
print(f"Social:        {soc_acc:.1f}%  (ESGBERT paper reports ~93%)")
print(f"Governance:    {gov_acc:.1f}%  (ESGBERT paper reports ~89%)")