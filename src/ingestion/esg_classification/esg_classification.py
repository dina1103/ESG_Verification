import pandas as pd
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm

INPUT_FILE  = "data/processed/segments.parquet"
OUTPUT_FILE = "data/processed/segments_esg.parquet"
CHECKPOINT_FILE = "data/processed/segments_esg_checkpoint.parquet"

# batch size — lower to 16 if you run out of memory
BATCH_SIZE = 32

# save progress every N batches so a crash doesn't lose the whole run
CHECKPOINT_EVERY = 100

# minimum confidence to assign a label, otherwise stays None
MIN_SCORE = 0.5

print("ESG classification running...")


def load_classifiers():
    # three independent binary ESGBERT models, each {pillar, none}
    e_clf = pipeline("text-classification", model="ESGBERT/EnvironmentalBERT-environmental", truncation=True, max_length=512)
    s_clf = pipeline("text-classification", model="ESGBERT/SocialBERT-social",               truncation=True, max_length=512)
    g_clf = pipeline("text-classification", model="ESGBERT/GovernanceBERT-governance",       truncation=True, max_length=512)
    return e_clf, s_clf, g_clf


def get_pillar_score(result, pillar_label):
    # binary models return {pillar, none}; flip the score when "none" is returned
    label = result["label"].lower()
    score = result["score"]
    return score if label == pillar_label.lower() else 1 - score


def classify_batch(texts, e_clf, s_clf, g_clf):
    e_results = e_clf(texts)
    s_results = s_clf(texts)
    g_results = g_clf(texts)

    labels, scores = [], []

    for e, s, g in zip(e_results, s_results, g_results):
        # raw per-pillar probabilities — each pillar competes on its own model score
        candidates = {
            "Environmental": get_pillar_score(e, "environmental"),
            "Social":        get_pillar_score(s, "social"),
            "Governance":    get_pillar_score(g, "governance"),
        }
        best_label = max(candidates, key=candidates.get)
        best_score = candidates[best_label]

        if best_score >= MIN_SCORE:
            labels.append(best_label)
            scores.append(round(best_score, 4))
        else:
            labels.append(None)
            scores.append(None)

    return labels, scores


def process():
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} segments")

    # resume from checkpoint if one exists
    start_batch = 0
    all_labels, all_scores = [], []
    if Path(CHECKPOINT_FILE).exists():
        ckpt = pd.read_parquet(CHECKPOINT_FILE)
        n_done = int(ckpt["_processed"].fillna(False).sum())
        if n_done > 0:
            all_labels = ckpt.loc[:n_done-1, "esg_label"].tolist()
            all_scores = ckpt.loc[:n_done-1, "esg_score"].tolist()
            start_batch = n_done // BATCH_SIZE
            print(f"Resuming from checkpoint: {n_done:,} segments already done")

    texts = df["text"].tolist()

    print("Loading ESGBERT models...")
    e_clf, s_clf, g_clf = load_classifiers()
    print("Models loaded\n")

    batch_starts = list(range(0, len(texts), BATCH_SIZE))
    for bi, i in enumerate(tqdm(batch_starts, desc="Classifying")):
        if bi < start_batch:
            continue
        batch = texts[i : i + BATCH_SIZE]
        labels, scores = classify_batch(batch, e_clf, s_clf, g_clf)
        all_labels.extend(labels)
        all_scores.extend(scores)

        # periodic checkpoint
        if (bi + 1) % CHECKPOINT_EVERY == 0:
            n = len(all_labels)
            ckpt = df.copy()
            ckpt["esg_label"] = all_labels + [None] * (len(df) - n)
            ckpt["esg_score"] = all_scores + [None] * (len(df) - n)
            ckpt["_processed"] = [True] * n + [False] * (len(df) - n)
            Path(CHECKPOINT_FILE).parent.mkdir(parents=True, exist_ok=True)
            ckpt.to_parquet(CHECKPOINT_FILE, index=False)

    df["esg_label"] = all_labels
    df["esg_score"] = all_scores

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    # clean up checkpoint on successful completion
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()

    print(f"\nTotal segments: {len(df):,}")
    print(f"Labelled:       {df['esg_label'].notna().sum():,}")
    print(f"Unlabelled:     {df['esg_label'].isna().sum():,}")
    print("\nLabel distribution:")
    print(df["esg_label"].value_counts(dropna=False))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process()