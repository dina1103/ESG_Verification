import pandas as pd

df = pd.read_parquet(r"C:\Users\dina_\Desktop\esg_verification\data\processed\segments_esg.parquet")

# label distribution
print("=== Label distribution ===")
print(df["esg_label"].value_counts(dropna=False))
print()

# score stats per pillar — confidence distribution
print("=== Score stats per label ===")
print(df.groupby("esg_label")["esg_score"].describe().round(4))
print()

# how many labels are low-confidence (just over the 0.5 threshold)?
print("=== Low-confidence labels (0.50 - 0.60) ===")
for label in ["Environmental", "Social", "Governance"]:
    sub = df[(df["esg_label"] == label) & (df["esg_score"] < 0.60)]
    tot = (df["esg_label"] == label).sum()
    print(f"  {label:15s} {len(sub):>6,} / {tot:>6,} ({len(sub)/max(tot,1)*100:.1f}%)")
print()

# random samples per label — do they read correctly?
for label in ["Environmental", "Social", "Governance"]:
    print(f"--- {label} samples ---")
    sub = df[df["esg_label"] == label]
    for s in sub["text"].sample(min(5, len(sub)), random_state=42).tolist():
        print(f"  - {s[:150]}")
    print()

# high-confidence samples — should be unambiguously on-pillar
for label in ["Environmental", "Social", "Governance"]:
    high = df[(df["esg_label"] == label) & (df["esg_score"] > 0.95)]
    print(f"--- High confidence {label} (>0.95, n={len(high):,}) ---")
    for s in high["text"].sample(min(5, len(high)), random_state=42).tolist():
        print(f"  - {s[:150]}")
    print()

# unlabelled — should be non-ESG (finance, legal, product, boilerplate)
print("--- Unlabelled samples ---")
unlabelled = df[df["esg_label"].isna()]
for s in unlabelled["text"].sample(min(10, len(unlabelled)), random_state=42).tolist():
    print(f"  - {s[:150]}")
print()

# label distribution by report type — sanity of the panel
print("=== Label distribution by report type ===")
print(pd.crosstab(df["report_type"], df["esg_label"].fillna("None"), normalize="index").round(3))
print()

# cross-contamination spot-checks
print("=== Possible Governance mis-labels (contain emissions/carbon/climate) ===")
sus_g = df[(df["esg_label"] == "Governance") &
           df["text"].str.contains(r"\b(?:emissions|carbon|climate|scope [123])\b", case=False, regex=True)]
print(f"  Count: {len(sus_g):,}")
for s in sus_g["text"].sample(min(5, len(sus_g)), random_state=42).tolist():
    print(f"  - {s[:200]}")
print()

print("=== Possible Environmental mis-labels (contain board/committee/audit) ===")
sus_e = df[(df["esg_label"] == "Environmental") &
           df["text"].str.contains(r"\b(?:board of directors|audit committee|compensation committee)\b", case=False, regex=True)]
print(f"  Count: {len(sus_e):,}")
for s in sus_e["text"].sample(min(5, len(sus_e)), random_state=42).tolist():
    print(f"  - {s[:200]}")