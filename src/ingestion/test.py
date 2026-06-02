import pandas as pd
df = pd.read_parquet("data/processed/segments.parquet")
print("max segment length:", df["text"].str.len().max())
print("over 1500:", (df["text"].str.len() > 1500).sum())