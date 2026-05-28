import pandas as pd

df = pd.read_parquet(r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\segmentation_esg")

# label distribution
print("=== Label distribution ===")
print(df["esg_label"].value_counts(dropna=False))
print()

# score stats — check confidence distribution
print("=== Score stats ===")
print(df.groupby("esg_label")["esg_score"].describe().round(4))
print()

# score saturation at 1.0 — if too many scores are exactly 1.0, boost cap is saturating
print("=== Score saturation (scores at exactly 1.0) ===")
for label in ["Environmental", "Social", "Governance"]:
    labeled = df[df["esg_label"] == label]
    at_one = (labeled["esg_score"] == 1.0).sum()
    print(f"  {label:15s} {at_one:>6,} / {len(labeled):>6,} ({at_one/max(len(labeled),1)*100:.1f}%)")
print()

# sample 5 sentences from each label — check they make sense
for label in ["Environmental", "Social", "Governance"]:
    print(f"--- {label} samples ---")
    sub = df[df["esg_label"] == label]
    sample = sub["text"].sample(min(5, len(sub)), random_state=42).tolist()
    for s in sample:
        print(f"  - {s[:150]}")
    print()

# high confidence samples per pillar — should be very clearly ESG
for label in ["Environmental", "Social", "Governance"]:
    high = df[(df["esg_label"] == label) & (df["esg_score"] > 0.95)]
    print(f"--- High confidence {label} (score > 0.95, n={len(high):,}) ---")
    if len(high) > 0:
        for s in high["text"].sample(min(5, len(high)), random_state=42).tolist():
            print(f"  - {s[:150]}")
    print()

# borderline cases — score 0.5 to 0.6, these are rescued by the boost
print("=== Borderline labels (score 0.5 - 0.6) ===")
for label in ["Environmental", "Social", "Governance"]:
    bl = df[(df["esg_label"] == label) & (df["esg_score"] >= 0.5) & (df["esg_score"] < 0.6)]
    print(f"--- Borderline {label} (n={len(bl):,}) ---")
    if len(bl) > 0:
        for s in bl["text"].sample(min(5, len(bl)), random_state=42).tolist():
            print(f"  - {s[:150]}")
    print()

# unlabelled — should be non-ESG (finance, legal, product descriptions, boilerplate)
print("--- Unlabelled samples ---")
unlabelled = df[df["esg_label"].isna()]
for s in unlabelled["text"].sample(min(10, len(unlabelled)), random_state=42).tolist():
    print(f"  - {s[:150]}")
print()

# cross-contamination check — does label distribution match report type?
# sustainability reports should be heavy on E+S, annual reports heavier on G
print("=== Label distribution by report type ===")
ct = pd.crosstab(df["report_type"], df["esg_label"].fillna("None"), normalize="index").round(3)
print(ct)
print()

# governance over-match check — is G label dominated by long sentences with many generic words?
print("=== Governance label — sentence length distribution ===")
gov = df[df["esg_label"] == "Governance"]
print(f"  Mean length: {gov['text'].str.len().mean():.0f} chars")
print(f"  Median length: {gov['text'].str.len().median():.0f} chars")
print(f"  vs all labeled (E+S+G) median: {df[df['esg_label'].notna()]['text'].str.len().median():.0f} chars")
print()

# spot-check: sentences labeled Governance that mention core environmental terms
# if these exist in volume, the G boost is poaching from E
print("=== Possible G mis-labels (contain emissions/carbon/climate) ===")
suspicious_g = df[
    (df["esg_label"] == "Governance")
    & df["text"].str.contains(r"\b(emissions|carbon|climate|scope [123])\b", case=False, regex=True)
]
print(f"  Count: {len(suspicious_g):,}")
if len(suspicious_g) > 0:
    for s in suspicious_g["text"].sample(min(5, len(suspicious_g)), random_state=42).tolist():
        print(f"  - {s[:200]}")
print()

# spot-check: sentences labeled Environmental that are clearly governance boilerplate
print("=== Possible E mis-labels (contain board/committee/audit) ===")
suspicious_e = df[
    (df["esg_label"] == "Environmental")
    & df["text"].str.contains(r"\b(board of directors|audit committee|compensation committee)\b", case=False, regex=True)
]
print(f"  Count: {len(suspicious_e):,}")
if len(suspicious_e) > 0:
    for s in suspicious_e["text"].sample(min(5, len(suspicious_e)), random_state=42).tolist():
        print(f"  - {s[:200]}")