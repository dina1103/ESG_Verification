import pandas as pd

df = pd.read_parquet(r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\segmentation_esg_sdg")

# basic shape and column check
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print()

# label distribution
print("=== SDG label distribution ===")
print(df["sdg_label"].value_counts(dropna=False))
print()

# score stats per SDG
print("=== SDG score stats ===")
print(df.groupby("sdg_label")["sdg_score"].describe().round(4))
print()

# sample each SDG to verify quality
for sdg in ["sdg12", "sdg13", "sdg16"]:
    sample = df[df["sdg_label"] == sdg]
    print(f"--- {sdg} samples (n={len(sample):,}) ---")
    if len(sample) > 0:
        for s in sample["text"].sample(min(5, len(sample)), random_state=42).tolist():
            print(f"  - {s[:200]}")
    print()

# how many ESG-labelled sentences got no in-scope SDG tag
esg_labelled = df[df["esg_label"].notna()]
no_sdg = esg_labelled[esg_labelled["sdg_label"].isna()]
print(f"ESG labelled: {len(esg_labelled):,}")
print(f"ESG labelled but no in-scope SDG tag: {len(no_sdg):,} ({len(no_sdg)/len(esg_labelled)*100:.1f}%)")
print("\nSample of ESG sentences that got no SDG tag:")
for s in no_sdg["text"].sample(min(10, len(no_sdg)), random_state=42).tolist():
    print(f"  - {s[:200]}")
print()

# cross-tab of ESG label vs SDG label
# expected: E sentences -> mostly sdg13, G sentences -> mostly sdg16, S sentences -> spread
print("=== Cross-tab: ESG label x SDG label ===")
print(pd.crosstab(df["esg_label"], df["sdg_label"].fillna("None"), dropna=False))
print()

# normalized version — what fraction of each ESG pillar maps to each SDG
print("=== Cross-tab (row-normalized — % of each ESG pillar) ===")
print(pd.crosstab(df["esg_label"], df["sdg_label"].fillna("None"), normalize="index").round(3))
print()

# climate vocabulary in SDG12 — these are likely SDG13 misclassifications
print("=== Climate vocabulary in SDG12-labelled sentences ===")
print("(SDG12 sentences mentioning climate terms may be miscategorized SDG13 content)")
sdg12 = df[df["sdg_label"] == "sdg12"]
climate_in_sdg12 = sdg12[
    sdg12["text"].str.contains(
        r"\b(emissions|carbon|climate|scope [123]|net zero|paris|ghg|greenhouse)\b",
        case=False, regex=True, na=False
    )
]
print(f"SDG12 sentences mentioning climate terms: {len(climate_in_sdg12):,} of {len(sdg12):,} ({len(climate_in_sdg12)/len(sdg12)*100:.1f}%)")
print("\nSamples:")
for s in climate_in_sdg12["text"].sample(min(5, len(climate_in_sdg12)), random_state=42).tolist():
    print(f"  - {s[:200]}")
print()

# distribution by report type — sustainability reports should have more SDG13
print("=== SDG distribution by report type ===")
ct = pd.crosstab(df["report_type"], df["sdg_label"].fillna("None"), normalize="index").round(3)
print(ct)
print()

# distribution by company — sanity check for any anomalies
print("=== SDG counts by company ===")
print(pd.crosstab(df["company_name"], df["sdg_label"].fillna("None")))

# look at G sentences tagged SDG12 — is this a misclassification or legitimate overlap?
print("\n=== Governance sentences tagged SDG12 (n=2,683) ===")
g_sdg12 = df[(df["esg_label"] == "Governance") & (df["sdg_label"] == "sdg12")]
print(f"Count: {len(g_sdg12):,}")
print("\nRandom samples:")
for s in g_sdg12["text"].sample(min(15, len(g_sdg12)), random_state=42).tolist():
    print(f"  - {s[:250]}")
print()

# look at S sentences tagged SDG12 — same question
print("=== Social sentences tagged SDG12 (n=2,974) ===")
s_sdg12 = df[(df["esg_label"] == "Social") & (df["sdg_label"] == "sdg12")]
print(f"Count: {len(s_sdg12):,}")
print("\nRandom samples:")
for s in s_sdg12["text"].sample(min(15, len(s_sdg12)), random_state=42).tolist():
    print(f"  - {s[:250]}")
print()

# what's the SDG12 score distribution for G/S sentences vs E sentences?
# if G/S sentences are getting lower scores, sdgBERT is genuinely uncertain about them
print("=== SDG12 score distribution by ESG pillar ===")
sdg12 = df[df["sdg_label"] == "sdg12"]
print(sdg12.groupby("esg_label")["sdg_score"].describe().round(4))


print("\n=== Surviving G-SDG12 bucket ===")
g_sdg12 = df[(df["esg_label"] == "Governance") & (df["sdg_label"] == "sdg12")]
print(f"Count: {len(g_sdg12):,}")
print(f"Score range: {g_sdg12['sdg_score'].min():.4f} - {g_sdg12['sdg_score'].max():.4f}")
print(f"Mean score: {g_sdg12['sdg_score'].mean():.4f}")
print()
print("Random samples (should be more legitimate sustainability/supply chain governance, less financial boilerplate):")
for s in g_sdg12["text"].sample(min(15, len(g_sdg12)), random_state=42).tolist():
    print(f"  - {s[:250]}")