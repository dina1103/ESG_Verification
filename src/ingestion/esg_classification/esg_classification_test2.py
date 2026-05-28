import re
from collections import Counter
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

MIN_SCORE = 0.5


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


def get_pillar_score(result, pillar_label):
    if result["label"].lower() == pillar_label.lower():
        return result["score"]
    return 1 - result["score"]


def governance_boost(text_lower):
    matches = sum(1 for p in GOVERNANCE_PATTERNS if p.search(text_lower))
    return min(matches * 0.15, 0.4)


def classify_sentence(text, e_result, s_result, g_result):
    # E and S: raw model score. G: model score + keyword boost.
    text_lower = text.lower()
    candidates = {
        "Environmental": get_pillar_score(e_result, "environmental"),
        "Social":        get_pillar_score(s_result, "social"),
        "Governance":    min(get_pillar_score(g_result, "governance") + governance_boost(text_lower), 1.0),
    }
    best_label = max(candidates, key=candidates.get)
    best_score = candidates[best_label]
    return best_label if best_score >= MIN_SCORE else "None"


# ---- load models once ----
print("Loading models...")
e_clf = pipeline("text-classification", model="ESGBERT/EnvironmentalBERT-environmental", truncation=True, max_length=512)
s_clf = pipeline("text-classification", model="ESGBERT/SocialBERT-social",               truncation=True, max_length=512)
g_clf = pipeline("text-classification", model="ESGBERT/GovernanceBERT-governance",       truncation=True, max_length=512)


def evaluate_on_dataset(dataset_name, label_column, expected_label):
    # load the 2k dataset — has "text" and a pillar-specific binary label column
    ds = load_dataset(dataset_name, split="train")
    texts = ds["text"]
    true_labels = ds[label_column]

    print(f"\nRunning pipeline on {dataset_name} ({len(texts)} sentences)...")

    # batch-process all three models
    batch_size = 32
    e_results, s_results, g_results = [], [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="  classifying"):
        batch = texts[i:i+batch_size]
        e_results.extend(e_clf(batch))
        s_results.extend(s_clf(batch))
        g_results.extend(g_clf(batch))

    predictions = [
        classify_sentence(t, e, s, g)
        for t, e, s, g in zip(texts, e_results, s_results, g_results)
    ]

    # evaluate: label=1 means pipeline should predict expected_label
    #           label=0 means pipeline should predict anything except expected_label
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
    print(f"  True positives (correct {expected_label}):  {confusion['correct_positive']}")
    print(f"  False negatives (missed {expected_label}):   {confusion['false_negative']}")
    print(f"  True negatives (correct not-{expected_label}): {confusion['correct_negative']}")
    print(f"  False positives (wrong {expected_label}):   {confusion['false_positive']}")

    neg_preds = [pred for pred, true in zip(predictions, true_labels) if true == 0]
    print(f"  Pipeline output distribution on negative sentences: {dict(Counter(neg_preds))}")

    return accuracy


# run all three
env_acc = evaluate_on_dataset("ESGBERT/environmental_2k", "env", "Environmental")
soc_acc = evaluate_on_dataset("ESGBERT/social_2k",        "soc", "Social")
gov_acc = evaluate_on_dataset("ESGBERT/governance_2k",    "gov", "Governance")

print("\n=== SUMMARY (G-only boost, no E/S boost) ===")
print(f"Environmental: {env_acc:.1f}%  (previous with E boost: 95.0%, expected ~93%)")
print(f"Social:        {soc_acc:.1f}%  (previous with S boost: 90.8%, expected ~93%)")
print(f"Governance:    {gov_acc:.1f}%  (previous with G boost: 87.8%, expected ~89%)")