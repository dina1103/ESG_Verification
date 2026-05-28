import pandas as pd
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm

INPUT_FILE  = "data/processed/segmentation_esg"
OUTPUT_FILE = "data/processed/segmentation_esg_sdg"

# batch size — lower to 16 if you run out of memory
BATCH_SIZE = 32

# default minimum confidence to assign an SDG
MIN_SCORE = 0.7

# stricter threshold for the noisy G-SDG12, diagnostic testing showed mean confidence 0.881 for G-SDG12 vs 0.918 for E-SDG12
G_SDG12_THRESHOLD = 0.85

# SDGs in thesis scope — out-of-scope predictions are discarded
IN_SCOPE_SDGS = {"sdg12", "sdg13", "sdg16"}

print("SDG classification running...")


def load_classifier():
    # sdgBERT — multi-class classifier over SDGs 1-16
    # no "none" class: every input is forced into one SDG
    # we rely on the confidence threshold and scope filter to reject out-of-domain content
    return pipeline(
        "text-classification",
        model="sadickam/sdgBERT",
        truncation=True,
        max_length=512,
    )


def get_threshold(esg_label, sdg_label):
    # pillar-aware threshold: stricter for known-noisy buckets
    if esg_label == "Governance" and sdg_label == "sdg12":
        return G_SDG12_THRESHOLD
    return MIN_SCORE


def classify_batch(texts, esg_labels, clf):
    results = clf(texts)
    labels = []
    scores = []
    for text, esg_label, r in zip(texts, esg_labels, results):
        label = r["label"].lower()
        score = r["score"]
        # keep only in-scope SDGs above the pillar-aware threshold
        if label in IN_SCOPE_SDGS and score >= get_threshold(esg_label, label):
            labels.append(label)
            scores.append(round(score, 4))
        else:
            labels.append(None)
            scores.append(None)
    return labels, scores


def process():
    # load ESG-classified segments
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} segments\n")

    # only run SDG classification on ESG-labeled sentences
    # sdgBERT has no "none" class; running on non-ESG content produces wrong labels
    is_esg = df["esg_label"].notna()
    esg_indices = df.index[is_esg].tolist()
    esg_texts = df.loc[is_esg, "text"].tolist()
    esg_pillar_labels = df.loc[is_esg, "esg_label"].tolist()
    print(f"Running SDG classification on {len(esg_texts):,} ESG-labeled sentences\n")

    # load model
    print("Loading sdgBERT...")
    clf = load_classifier()
    print("Model loaded\n")

    # classify in batches
    all_labels = []
    all_scores = []
    for i in tqdm(range(0, len(esg_texts), BATCH_SIZE), desc="Classifying"):
        batch_texts = esg_texts[i : i + BATCH_SIZE]
        batch_pillars = esg_pillar_labels[i : i + BATCH_SIZE]
        labels, scores = classify_batch(batch_texts, batch_pillars, clf)
        all_labels.extend(labels)
        all_scores.extend(scores)

    # add new columns — default None for non-ESG sentences
    df["sdg_label"] = None
    df["sdg_score"] = None

    # fill in the ESG-labeled rows
    for idx, label, score in zip(esg_indices, all_labels, all_scores):
        df.at[idx, "sdg_label"] = label
        df.at[idx, "sdg_score"] = score

    # save
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    # summary
    print(f"\nTotal segments:              {len(df):,}")
    print(f"ESG-labeled:                 {is_esg.sum():,}")
    print(f"SDG-labeled (in scope):      {df['sdg_label'].notna().sum():,}")
    print()
    print("In-scope SDG distribution:")
    print(df["sdg_label"].value_counts())
    print()
    print("SDG distribution by ESG pillar:")
    print(pd.crosstab(df["esg_label"], df["sdg_label"].fillna("None")))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process()