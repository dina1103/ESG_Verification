import pandas as pd
df = pd.read_parquet(r"C:\Users\dina_\Desktop\esg_verification\data\processed\segments_esg_sdg.parquet")

# how many distinct paragraph_ids exist, and what do they look like?
print("distinct paragraph_id values:", df["paragraph_id"].nunique())
print("distinct source_documents:", df["source_document"].nunique())
print("sample paragraph_ids:", sorted(df["paragraph_id"].unique())[:5], "...", sorted(df["paragraph_id"].unique())[-5:])
print()

# for ONE document, do paragraph_ids map to contiguous sentence runs?
doc = df["source_document"].sample(1, random_state=1).iloc[0]
one = df[df["source_document"] == doc].sort_values("sentence_id")
print("document:", doc)
print(one.groupby("paragraph_id")["sentence_id"].agg(["count", "first", "last"]).head(15))