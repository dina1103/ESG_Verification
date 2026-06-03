import pandas as pd

df = pd.read_parquet(r"C:\Users\dell\ESG_Verification\data\processed\segments_esg_sdg.parquet")

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

# VERIFY the pillar restriction held: SDG12 must be Environmental-only,
# so these two counts must both be 0
g_sdg12 = len(df[(df["esg_label"] == "Governance") & (df["sdg_label"] == "sdg12")])
s_sdg12 = len(df[(df["esg_label"] == "Social") & (df["sdg_label"] == "sdg12")])
print("=== Pillar-restriction check (both must be 0) ===")
print(f"Governance-SDG12: {g_sdg12}")
print(f"Social-SDG12:     {s_sdg12}")
print(f"  -> {'PASS' if g_sdg12 == 0 and s_sdg12 == 0 else 'FAIL — restriction did not apply'}")
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
# expected now: SDG12 column is Environmental-only; E -> sdg12/sdg13, G -> sdg16, S -> sdg16
print("=== Cross-tab: ESG label x SDG label ===")
print(pd.crosstab(df["esg_label"], df["sdg_label"].fillna("None"), dropna=False))
print()

# normalized version — what fraction of each ESG pillar maps to each SDG
print("=== Cross-tab (row-normalized — % of each ESG pillar) ===")
print(pd.crosstab(df["esg_label"], df["sdg_label"].fillna("None"), normalize="index").round(3))
print()

# climate vocabulary in SDG12 — leakage of climate (SDG13) content into SDG12
print("=== Climate vocabulary in SDG12-labelled sentences ===")
print("(SDG12 sentences mentioning climate terms may be miscategorized SDG13 content)")
sdg12 = df[df["sdg_label"] == "sdg12"]
climate_in_sdg12 = sdg12[
    sdg12["text"].str.contains(
        r"\b(?:emissions|carbon|climate|scope [123]|net zero|paris|ghg|greenhouse)\b",
        case=False, regex=True, na=False
    )
]
pct = len(climate_in_sdg12) / max(len(sdg12), 1) * 100
print(f"SDG12 sentences mentioning climate terms: {len(climate_in_sdg12):,} of {len(sdg12):,} ({pct:.1f}%)")
print("\nSamples:")
for s in climate_in_sdg12["text"].sample(min(5, len(climate_in_sdg12)), random_state=42).tolist():
    print(f"  - {s[:200]}")
print()

# distribution by report type — sustainability reports should have more SDG content
print("=== SDG distribution by report type ===")
print(pd.crosstab(df["report_type"], df["sdg_label"].fillna("None"), normalize="index").round(3))
print()

# distribution by company — sanity check for any anomalies
print("=== SDG counts by company ===")
print(pd.crosstab(df["company_name"], df["sdg_label"].fillna("None")))
print()

# SDG12 score distribution — now Environmental-only, single bucket
print("=== Environmental-SDG12 score distribution ===")
print(sdg12["sdg_score"].describe().round(4))
print()

# inspect the surviving Environmental-SDG12 bucket — should be responsible
# production/consumption content (recycling, materials, procurement, resources)
print("=== Surviving SDG12 bucket (Environmental pillar only) ===")
print(f"Count: {len(sdg12):,}")
if len(sdg12) > 0:
    print("Random samples (expect recycling / circularity / procurement / resource use):")
    for s in sdg12["text"].sample(min(15, len(sdg12)), random_state=42).tolist():
        print(f"  - {s[:250]}")