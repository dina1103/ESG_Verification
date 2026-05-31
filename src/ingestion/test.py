import pandas as pd
df = pd.read_parquet("data/processed/segments.parquet")

# segment counts per document — spot the F_2020 / GM_2020 anomalies in context
print(df.groupby("source_document").size().sort_values().head(10))

# confirm no empty/garbage text survived, and length distribution looks sane
print(df["text"].str.len().describe())