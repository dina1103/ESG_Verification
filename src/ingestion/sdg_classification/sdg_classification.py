import pandas as pd
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm

INPUT_FILE  = r"data\processed\segments_esg.parquet"
OUTPUT_FILE = r"data\processed\segments_esg_sdg.parquet"
CHECKPOINT_FILE = r"data\processed\segments_esg_sdg_checkpoint.parquet"

# batch size — lower to 16 if you run out of memory
BATCH_SIZE = 32

# save progress every N batches
CHECKPOINT_EVERY = 100

# default minimum confidence to assign an SDG
MIN_SCORE = 0.7

# stricter threshold for the noisy Governance-SDG12 bucket
G_SDG12_THRESHOLD = 0.85

# SDGs in thesis scope — out-of-scope predictions are discarded
IN_SCOPE_SDGS = {"sdg12", "sdg13", "sdg16"}

print("SDG classification running...")


def load_classifier():
    # sdgBERT — multi-class over SDGs 1-16, no "none" class; every input is forced
    # into some SDG, so the confidence threshold + scope filter reject out-of-domain content
    return pipeline(
        "text-classification",
        model="sadickam/sdgBERT",
        truncation=True,
        max_length=512,
    )


def normalize_sdg_label(raw_label):
    # sdgBERT returns "sdg12" etc. (confirmed); normalize defensively in case of format drift
    digits = "".join(ch for ch in raw_label if ch.isdigit())
    return f"sdg{digits}" if digits else raw_label.lower().replace(" ", "")


def get_threshold(esg_label, sdg_label):
    # pillar-aware: stricter cutoff for the known-noisy Governance-SDG12 bucket
    if esg_label == "Governance" and sdg_label == "sdg12":
        return G_SDG12_THRESHOLD
    return MIN_SCORE


def classify_batch(texts, esg_labels, clf):
    results = clf(texts)
    labels, scores = [], []
    for esg_label, r in zip(esg_labels, results):
        sdg = normalize_sdg_label(r["label"])
        score = r["score"]
        if sdg in IN_SCOPE_SDGS and score >= get_threshold(esg_label, sdg):
            labels.append(sdg)
            scores.append(round(score, 4))
        else:
            labels.append(None)
            scores.append(None)
    return labels, scores


def process():
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} segments\n")

    # sdgBERT has no "none" class; on non-ESG content it produces meaningless labels,
    # so SDG classification is restricted to ESG-labelled segments
    is_esg = df["esg_label"].notna()
    esg_indices = df.index[is_esg].tolist()
    esg_texts = df.loc[is_esg, "text"].tolist()
    esg_pillars = df.loc[is_esg, "esg_label"].tolist()
    print(f"Running SDG classification on {len(esg_texts):,} ESG-labelled segments\n")

    # resume from checkpoint if present
    start_batch = 0
    all_labels, all_scores = [], []
    if Path(CHECKPOINT_FILE).exists():
        ckpt = pd.read_parquet(CHECKPOINT_FILE)
        n_done = len(ckpt)
        if n_done > 0:
            all_labels = ckpt["sdg_label"].tolist()
            all_scores = ckpt["sdg_score"].tolist()
            start_batch = n_done // BATCH_SIZE
            print(f"Resuming from checkpoint: {n_done:,} segments already done")

    print("Loading sdgBERT...")
    clf = load_classifier()
    print("Model loaded\n")

    batch_starts = list(range(0, len(esg_texts), BATCH_SIZE))
    for bi, i in enumerate(tqdm(batch_starts, desc="Classifying")):
        if bi < start_batch:
            continue
        labels, scores = classify_batch(
            esg_texts[i : i + BATCH_SIZE],
            esg_pillars[i : i + BATCH_SIZE],
            clf,
        )
        all_labels.extend(labels)
        all_scores.extend(scores)

        if (bi + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame({"sdg_label": all_labels, "sdg_score": all_scores}).to_parquet(CHECKPOINT_FILE, index=False)

    # write back: None everywhere, then fill the ESG-labelled rows
    df["sdg_label"] = None
    df["sdg_score"] = None
    df.loc[esg_indices, "sdg_label"] = pd.Series(all_labels, index=esg_indices)
    df.loc[esg_indices, "sdg_score"] = pd.Series(all_scores, index=esg_indices)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()

    print(f"\nTotal segments:          {len(df):,}")
    print(f"ESG-labelled:            {is_esg.sum():,}")
    print(f"SDG-labelled (in scope): {df['sdg_label'].notna().sum():,}")
    print("\nIn-scope SDG distribution:")
    print(df["sdg_label"].value_counts())
    print("\nSDG distribution by ESG pillar:")
    print(pd.crosstab(df["esg_label"], df["sdg_label"].fillna("None")))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process()