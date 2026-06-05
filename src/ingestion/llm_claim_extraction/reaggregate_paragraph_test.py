import pandas as pd

PATH = r"C:\Users\dina_\Desktop\esg_verification\data\processed\llm_paragraphs.parquet"

df = pd.read_parquet(PATH)
print("rows:", len(df), "| cols:", len(df.columns))
print("columns:", df.columns.tolist())
print()

# 1. length distribution — the number that drives num_ctx
L = df["text"].str.len()
print("=== text length (chars) ===")
print(f"  median {L.median():.0f}  P90 {L.quantile(0.9):.0f}  P99 {L.quantile(0.99):.0f}  max {L.max()}")
# rough token estimate at ~4 chars/token
print(f"  est. max tokens (~chars/4): {L.max()/4:.0f}")
print(f"  paragraphs still over 6000 chars: {(L > 6000).sum()}")
print()

# 2. windowing summary
print("=== windowing ===")
print(f"  windowed paragraphs: {int(df['was_windowed'].sum())} / {len(df)}")
w = df[df["was_windowed"]]
if len(w):
    print(f"  windowed length: median {w['text'].str.len().median():.0f}  max {w['text'].str.len().max():.0f}")
    print(f"  these had median {w['n_sentences'].median():.0f} sentences, {w['n_esg_sentences'].median():.0f} ESG")
print()

# 3. integrity checks
print("=== integrity ===")
print(f"  null/empty text: {(df['text'].str.len() == 0).sum() + df['text'].isna().sum()}")
print(f"  duplicate block_ids: {df['block_id'].duplicated().sum()}")
print(f"  every paragraph has >=1 ESG sentence: {(df['n_esg_sentences'] >= 1).all()}")
print(f"  every paragraph has sentence_ids: {df['sentence_ids'].apply(lambda x: len(x) > 0).all()}")
print()

# 4. inspect a windowed paragraph — confirm [...] gaps and ESG content preserved
print("=== sample windowed paragraph (check [...] gaps + ESG content kept) ===")
if len(w):
    s = w.iloc[0]
    print(f"  {s['block_id']} | {s['n_sentences']} sents, {s['n_esg_sentences']} ESG | {len(s['text'])} chars")
    print("  ---")
    print("  " + s["text"][:1200])
print()

# 5. inspect a normal (non-windowed) paragraph for comparison
print("=== sample normal paragraph ===")
nw = df[~df["was_windowed"]]
s = nw.iloc[0]
print(f"  {s['block_id']} | {s['n_sentences']} sents, {s['n_esg_sentences']} ESG | {len(s['text'])} chars")
print("  ---")
print("  " + s["text"][:600])