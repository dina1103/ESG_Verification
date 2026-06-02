import pandas as pd
import glob
import json
from pathlib import Path
print("imported")

# single source of truth for paths so the parquet and JSON checks can't diverge
ROOT = Path(r"C:\Users\dina_\Desktop\esg_verification")
SEGMENTS = ROOT / "data" / "processed" / "segments.parquet"
JSON_DIR = ROOT / "data" / "processed" / "json"

df = pd.read_parquet(SEGMENTS)

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
print("Sentences over 1500 chars:", (df["text"].str.len() > 1500).sum())

# composite-key uniqueness — the classifier must write labels back on
# (source_document, sentence_id); sentence_id alone repeats across documents
print("\nKey checks:")
print("Duplicate (source_document, sentence_id) keys:",
      df.duplicated(subset=["source_document", "sentence_id"]).sum())
print("sentence_id alone unique:", df["sentence_id"].is_unique)

# provenance completeness — anything 'unknown' breaks the later financial join
print("\nProvenance gaps:")
print("company_id == unknown:", (df["company_id"] == "unknown").sum())
print("year == 0:", (df["year"] == 0).sum())
print("report_type == unknown:", (df["report_type"] == "unknown").sum())

# page coverage per document — look for suspicious gaps
per_doc = df.groupby("source_document")["page_number"].agg(["min", "max", "nunique"])
print("\nPage coverage per document (first 10):")
print(per_doc.head(10))

# OCR spot-check in parquet
print(f"\nOCR sentences in parquet: {df['is_ocr'].sum()} ({df['is_ocr'].mean()*100:.1f}%)")
if df["is_ocr"].sum() > 0:
    print("Sample OCR sentences:")
    print(df[df["is_ocr"]]["text"].sample(min(5, df["is_ocr"].sum()), random_state=42).to_list())

# stratified look at long sentences — are they TOCs, tables, or real prose?
print("\nStratified sample of long sentences:")
for low, high in [(800, 1000), (1000, 1200), (1200, 1500)]:
    bucket = df[(df["text"].str.len() >= low) & (df["text"].str.len() < high)]
    print(f"\n=== {low}-{high} chars: {len(bucket)} sentences ===")
    for t in bucket["text"].sample(min(3, len(bucket)), random_state=42):
        print("---")
        print(t[:400] + ("..." if len(t) > 400 else ""))

# verify is_ocr propagation by re-walking the JSON files (same ROOT)
print("\nOCR check across all JSON files:")
json_files = glob.glob(str(JSON_DIR / "**" / "*.json"), recursive=True)
total_ocr = 0
docs_with_ocr = 0
for path in json_files:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    n_ocr = sum(1 for p in doc["pages"] if p["is_ocr"])
    if n_ocr > 0:
        docs_with_ocr += 1
        total_ocr += n_ocr
        print(f"  {Path(path).name}: {n_ocr} OCR pages")
print(f"\n{docs_with_ocr}/{len(json_files)} documents had OCR pages, {total_ocr} OCR pages total")

# sanity check metadata propagation — one sentence per company
print("\nMetadata sanity check (one sentence per company):")
for company in df["company_name"].unique():
    row = df[df["company_name"] == company].iloc[0]
    print(f"\n{company}:")
    print(f"  {row['source_document']}")
    print(f"  page {row['page_number']}, sentence {row['sentence_id']}")
    print(f"  {row['text'][:200]}...")