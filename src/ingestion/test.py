import pandas as pd
df = pd.read_parquet(r"C:\Users\dina_\Desktop\esg_verification\data\processed\segments_esg_sdg.parquet")

e = df[(df["esg_label"]=="Environmental") & (df["sdg_label"]=="sdg12")]
for s in e["text"].sample(10, random_state=1): print("-", s[:150])