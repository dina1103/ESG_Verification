import re
import pandas as pd
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm

INPUT_FILE  = "data/processed/segmentation"
OUTPUT_FILE = "data/processed/segmentation_esg"

# batch size — lower to 16 if you run out of memory
BATCH_SIZE = 32

# minimum confidence to assign a label, otherwise stays None
MIN_SCORE = 0.5

print("ESG classification running...")


# Governance keywords from Baier, Berninger & Kiesel (2020), Table 3
# DOI: 10.1111/fmii.12132
# Applied verbatim to compensate for GovernanceBERT's lower recall (~89% vs ~93% for E and S).
# Environmental and Social models receive no keyword boost — they already perform well on their own.

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

# compile patterns once at module level
GOVERNANCE_PATTERNS = [re.compile(rf"\b{re.escape(kw)}\b") for kw in GOVERNANCE_KEYWORDS]


def load_classifiers():
    # load all three ESGBERT models
    e_clf = pipeline("text-classification", model="ESGBERT/EnvironmentalBERT-environmental", truncation=True, max_length=512)
    s_clf = pipeline("text-classification", model="ESGBERT/SocialBERT-social",              truncation=True, max_length=512)
    g_clf = pipeline("text-classification", model="ESGBERT/GovernanceBERT-governance",      truncation=True, max_length=512)
    return e_clf, s_clf, g_clf


def get_pillar_score(result, pillar_label):
    # if model label matches the pillar, use score directly
    # if model returned "none", flip the score (binary softmax: P(pillar) + P(none) = 1)
    label = result["label"].lower()
    score = result["score"]
    if label == pillar_label.lower():
        return score
    else:
        return 1 - score


def governance_boost(text_lower):
    # count G keyword matches, bounded additive boost to compensate for model's lower recall
    matches = sum(1 for p in GOVERNANCE_PATTERNS if p.search(text_lower))
    return min(matches * 0.15, 0.4)


def classify_batch(texts, e_clf, s_clf, g_clf):
    e_results = e_clf(texts)
    s_results = s_clf(texts)
    g_results = g_clf(texts)

    labels = []
    scores = []

    for text, e, s, g in zip(texts, e_results, s_results, g_results):
        text_lower = text.lower()
        # G gets a keyword boost to compensate for GovernanceBERT's weaker recall
        candidates = {
            "Environmental": get_pillar_score(e, "environmental"),
            "Social":        get_pillar_score(s, "social"),
            "Governance":    min(get_pillar_score(g, "governance") + governance_boost(text_lower), 1.0),
        }
        best_label = max(candidates, key=candidates.get)
        best_score = candidates[best_label]

        # only assign label if confidence is above threshold
        if best_score >= MIN_SCORE:
            labels.append(best_label)
            scores.append(round(best_score, 4))
        else:
            labels.append(None)
            scores.append(None)

    return labels, scores


def process():
    # load segments
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} segments\n")

    texts = df["text"].tolist()
    all_labels = []
    all_scores = []

    # load models
    print("Loading ESGBERT models...")
    e_clf, s_clf, g_clf = load_classifiers()
    print("Models loaded\n")

    # classify in batches
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Classifying"):
        batch = texts[i : i + BATCH_SIZE]
        labels, scores = classify_batch(batch, e_clf, s_clf, g_clf)
        all_labels.extend(labels)
        all_scores.extend(scores)

    # fill in the placeholder columns
    df["esg_label"] = all_labels
    df["esg_score"] = all_scores

    # save
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    # summary
    print(f"\nTotal segments: {len(df):,}")
    print(f"Labelled:       {df['esg_label'].notna().sum():,}")
    print(f"Unlabelled:     {df['esg_label'].isna().sum():,}")
    print("\nLabel distribution:")
    print(df["esg_label"].value_counts())
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process()