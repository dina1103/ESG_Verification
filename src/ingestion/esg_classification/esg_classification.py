import re
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

# a "none" segment is reconsidered for Governance only if keyword evidence is
# this strong; promotes missed-governance cases without touching confident E/S
GOV_BACKSTOP_MIN_MATCHES = 2
GOV_BACKSTOP_SCORE = 0.5

print(f"ESG classification running...")


# Governance keywords from Baier, Berninger & Kiesel (2020), Table 3
# DOI: 10.1111/fmii.12132
GOVERNANCE_KEYWORDS = [
    "align", "aligned", "aligning", "alignment", "aligns", "bylaw", "bylaws",
    "charter", "charters", "culture", "death", "duly", "parents", "independent",
    "compliance", "conduct", "conformity", "governance", "misconduct",
    "parachute", "parachutes", "perquisites", "plane", "planes", "poison",
    "retirement",
    "approval", "approvals", "approve", "approved", "approves", "approving",
    "assess", "assessed", "assesses", "assessing", "assessment", "assessments",
    "audit", "audited", "auditing", "auditor", "auditors", "audits", "control",
    "controls", "coso", "detect", "detected", "detecting", "detection",
    "evaluate", "evaluated", "evaluates", "evaluating", "evaluation",
    "evaluations", "examination", "examinations", "examine", "examined",
    "examines", "examining", "irs", "oversee", "overseeing", "oversees",
    "oversight", "review", "reviewed", "reviewing", "reviews", "rotation",
    "test", "tested", "testing", "tests", "treadway",
    "backgrounds", "independence", "leadership", "nomination", "nominations",
    "nominee", "nominees", "perspectives", "qualifications", "refreshment",
    "skill", "skills", "succession", "tenure", "vacancies", "vacancy",
    "appreciation", "award", "awarded", "awarding", "awards", "bonus",
    "bonuses", "cd", "compensate", "compensated", "compensates",
    "compensating", "compensation", "eip", "iso", "isos", "payout", "payouts",
    "pension", "prsu", "prsus", "recoupment", "remuneration", "reward",
    "rewarding", "rewards", "rsu", "rsus", "salaries", "salary", "severance",
    "vest", "vested", "vesting", "vests",
    "ballot", "ballots", "cast", "consent", "elect", "elected", "electing",
    "election", "elections", "elects", "nominate", "nominated", "plurality",
    "proponent", "proponents", "proposal", "proposals", "proxies", "quorum",
    "vote", "voted", "votes", "voting",
    "brother", "clicking", "conflict", "conflicts", "family", "grandchildren",
    "grandparent", "grandparents", "inform", "insider", "insiders", "inspector",
    "inspectors", "interlocks", "nephews", "nieces", "posting", "relatives",
    "siblings", "sister", "son", "spousal", "spouse", "spouses", "stepchildren",
    "stepparents", "transparency", "transparent", "visit", "visiting", "visits",
    "webpage", "website",
    "attract", "attracting", "attracts", "incentive", "incentives", "interview",
    "interviews", "motivate", "motivated", "motivates", "motivating",
    "motivation", "recruit", "recruiting", "recruitment", "retain", "retainer",
    "retainers", "retaining", "retention", "talent", "talented", "talents",
    "cobc", "ethic", "ethical", "ethically", "ethics", "honesty",
    "bribery", "corrupt", "corruption", "crimes", "embezzlement",
    "grassroots", "influence", "influences", "influencing", "lobbied",
    "lobbies", "lobby", "lobbying", "lobbyist", "lobbyists",
    "whistleblower",
    "announce", "announced", "announcement", "announcements", "announces",
    "announcing", "communicate", "communicated", "communicates",
    "communicating", "erm", "fairly", "integrity", "liaison", "presentation",
    "presentations", "sustainable",
    "asc", "disclose", "disclosed", "discloses", "disclosing", "disclosure",
    "disclosures", "fasb", "gaap", "objectivity", "press", "sarbanes",
    "engagement", "engagements", "feedback", "hotline", "investor", "invite",
    "invited", "mail", "mailed", "mailing", "mailings", "notice", "relations",
    "stakeholder", "stakeholders",
    "compact", "ungc",
]

GOVERNANCE_PATTERNS = [re.compile(rf"\b{re.escape(kw)}\b") for kw in GOVERNANCE_KEYWORDS]


def load_classifiers():
    e_clf = pipeline("text-classification", model="ESGBERT/EnvironmentalBERT-environmental", truncation=True, max_length=512)
    s_clf = pipeline("text-classification", model="ESGBERT/SocialBERT-social",              truncation=True, max_length=512)
    g_clf = pipeline("text-classification", model="ESGBERT/GovernanceBERT-governance",      truncation=True, max_length=512)
    return e_clf, s_clf, g_clf


def get_pillar_score(result, pillar_label):
    # binary models return {pillar, none}; flip the score when "none" is returned
    label = result["label"].lower()
    score = result["score"]
    return score if label == pillar_label.lower() else 1 - score


def count_gov_keywords(text_lower):
    return sum(1 for p in GOVERNANCE_PATTERNS if p.search(text_lower))


def classify_batch(texts, e_clf, s_clf, g_clf):
    e_results = e_clf(texts)
    s_results = s_clf(texts)
    g_results = g_clf(texts)

    labels, scores = [], []

    for text, e, s, g in zip(texts, e_results, s_results, g_results):
        # raw per-pillar probabilities — no keyword interference in the argmax
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
            # nothing cleared threshold -> recall backstop:
            # promote to Governance only if keyword evidence is strong enough.
            # compensates for GovernanceBERT's lower recall without overriding
            # confident Environmental/Social predictions (which already won above)
            if count_gov_keywords(text.lower()) >= GOV_BACKSTOP_MIN_MATCHES:
                labels.append("Governance")
                scores.append(GOV_BACKSTOP_SCORE)
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
        done = ckpt["esg_label"].notna() | ckpt["_processed"].fillna(False)
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
    print(df["esg_label"].value_counts())
    gov_backstop = (df["esg_score"] == GOV_BACKSTOP_SCORE).sum()
    print(f"\n(of which promoted by governance backstop: {gov_backstop:,})")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process()