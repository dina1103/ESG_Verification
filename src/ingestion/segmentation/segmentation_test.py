import pandas as pd
import glob
import json
print("imported")

df = pd.read_parquet(r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\segmentation")

# basic shape
print(df.shape)

# check metadata is filled correctly
print(df[["company_name", "year", "report_type"]].drop_duplicates())

# check a few sentences look clean
print(df["text"].iloc[0])
print(df["text"].iloc[100])
print(df["text"].iloc[1000])

# check segments per company
print(df.groupby("company_name").size())

# check segments per document
print(df.groupby("source_document").size())

# check for empty or very short text
print("Empty rows:", df["text"].isna().sum())
print("Short rows (<20 chars):", (df["text"].str.len() < 20).sum())

# print 10 random sentences
print(df["text"].sample(10).to_list())

# check sentences that mention emissions (should look like real ESG claims)
emissions = df[df["text"].str.contains("emission|carbon|CO2", case=False)]
print(emissions["text"].head(10).to_list())

# check for remaining noise — very repetitive or suspiciously short
short = df[df["text"].str.len() < 60]
print("Short (<60 chars):", len(short))
if len(short) > 0:
    print(short["text"].sample(min(10, len(short))).to_list())

# length distribution — check for TOC/index dumps at the long end
print("\nLength percentiles:")
print(df["text"].str.len().describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
print("Sentences over 1000 chars:", (df["text"].str.len() > 1000).sum())
print("Sentences over 2000 chars:", (df["text"].str.len() > 2000).sum())

# page coverage per document — look for suspicious gaps
per_doc = df.groupby("source_document")["page_number"].agg(["min", "max", "nunique"])
print("\nPage coverage per document (first 10):")
print(per_doc.head(10))

# section heading distribution — see if headings were preserved or mostly generic
print("\nTop section headings:")
print(df["section_heading"].value_counts().head(20))

# OCR spot-check in parquet
print(f"\nOCR sentences in parquet: {df['is_ocr'].sum()} ({df['is_ocr'].mean()*100:.1f}%)")
if df["is_ocr"].sum() > 0:
    print("Sample OCR sentences:")
    print(df[df["is_ocr"]]["text"].sample(min(5, df["is_ocr"].sum()), random_state=42).to_list())

# stratified look at long sentences — are they TOCs, tables, or real prose?
print("\nStratified sample of long sentences:")
for low, high in [(1000, 1500), (1500, 2000), (2000, 3000), (3000, 5000), (5000, 10000)]:
    bucket = df[(df["text"].str.len() >= low) & (df["text"].str.len() < high)]
    print(f"\n=== {low}-{high} chars: {len(bucket)} sentences ===")
    for t in bucket["text"].sample(min(3, len(bucket)), random_state=42):
        print("---")
        print(t[:400] + ("..." if len(t) > 400 else ""))

# long sentences grouped by document
long_sents = df[df["text"].str.len() > 2000]
if len(long_sents) > 0:
    print("\nLong sentences by document (top 10):")
    print(long_sents["source_document"].value_counts().head(10))

# verify is_ocr propagation by checking across all json files
print("\nOCR check across all JSON files:")
json_files = glob.glob(r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\json\**\*.json", recursive=True)
total_ocr = 0
docs_with_ocr = 0
for path in json_files:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    n_ocr = sum(1 for p in doc["pages"] if p["is_ocr"])
    if n_ocr > 0:
        docs_with_ocr += 1
        total_ocr += n_ocr
        print(f"  {path.split(chr(92))[-1]}: {n_ocr} OCR pages")
print(f"\n{docs_with_ocr}/{len(json_files)} documents had OCR pages, {total_ocr} OCR pages total")

# sanity check metadata propagation — one sentence per company
print("\nMetadata sanity check (one sentence per company):")
for company in df["company_name"].unique():
    row = df[df["company_name"] == company].iloc[0]
    print(f"\n{company}:")
    print(f"  {row['source_document']}")
    print(f"  page {row['page_number']}, sentence {row['sentence_id']}")
    print(f"  {row['text'][:200]}...")

    