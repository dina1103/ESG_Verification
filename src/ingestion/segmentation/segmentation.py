import os
import json
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

INPUT_DIR = "data/processed/json"
OUTPUT_FILE = "data/processed/segmentation"

# segments shorter than this are dropped
MIN_SEGMENT_CHARS = 60

print("Segmentation script running...")


# heading patterns common in sustainability reports
HEADING_PATTERNS = [
    re.compile(r"^[A-Z][A-Z\s\-&/]{4,60}$"),              # ALL CAPS
    re.compile(r"^\d+(\.\d+)*\s+[A-Z][A-Za-z\s]{3,60}$"), # 1.2 Numbered
    re.compile(r"^[A-Z][a-z].*:$"),                         # Title ending with colon
]


def looks_like_heading(line):
    line = line.strip()
    if len(line) < 4 or len(line) > 80:
        return False
    return any(p.match(line) for p in HEADING_PATTERNS)


def split_sentences(text):
    # protect known non-sentence periods before splitting
    text = re.sub(r"(\b(?:e\.g|i\.e|vs|Mr|Dr|Corp|Ltd|Inc|Fig|No|CO2|GHG))\.", r"\1<DOT>", text)
    text = re.sub(r"(\d+)\.(\d+)", r"\1<DOT>\2", text)

    # split on sentence-ending punctuation followed by capital letter
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    # restore protected dots and drop very short fragments
    sentences = [p.replace("<DOT>", ".").strip() for p in parts]
    return [s for s in sentences if len(s) > MIN_SEGMENT_CHARS]

def is_noise(text):
    # drop lines that are mostly digits (table rows, page references)
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    if digit_ratio > 0.3:
        return True
    # drop lines that look like page references or headers
    if re.match(r"^\d+\s+\w+", text) and len(text) < 60:
        return True
    # drop lines with too many bullet separators or GRI-style codes
    if re.match(r"^(GRI|SASB|SDG|ISO)\s+[\d\-,\s]+$", text):
        return True
    return False

def segment_page(page, doc_meta, para_counter, sent_counter):
    segments = []
    current_heading = "Preamble"
    current_paragraph = []

    def flush_paragraph():
        nonlocal para_counter, sent_counter, current_paragraph
        if not current_paragraph:
            return
        para_text = " ".join(current_paragraph).strip()
        if len(para_text) < MIN_SEGMENT_CHARS:
            current_paragraph = []
            return

        para_id = f"p_{para_counter:04d}"
        para_counter += 1

        for sent in split_sentences(para_text):
            if is_noise(sent):
                continue
            sent_id = f"s_{sent_counter:05d}"
            sent_counter += 1
            segments.append({
                # document-level metadata
                "company_id":       doc_meta["company_id"],
                "company_name":     doc_meta["company_name"],
                "year":             doc_meta["year"],
                "report_type":      doc_meta["report_type"],
                "framework":        doc_meta["framework"],
                "source_document":  doc_meta["filename"],
                # page-level metadata
                "page_number":      page["page_number"],
                "is_ocr":           page["is_ocr"],
                # segment-level metadata
                "section_heading":  current_heading,
                "paragraph_id":     para_id,
                "sentence_id":      sent_id,
                "text":             sent,
                # placeholders — filled in by the ESG/SDG classifier step
                "esg_label":        None,   # Environmental | Social | Governance | None
                "esg_score":        None,   # confidence score from classifier
                "sdg_tags":         None,   # e.g. ["SDG12", "SDG13"]
            })
        current_paragraph = []

    lines = page["text"].split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
        elif looks_like_heading(stripped):
            flush_paragraph()
            current_heading = stripped
        else:
            current_paragraph.append(stripped)

    # flush remaining paragraph at end of page
    flush_paragraph()

    return segments, para_counter, sent_counter


def segment_document(doc):
    segments = []
    para_counter = 0
    sent_counter = 0

    for page in doc["pages"]:
        page_segments, para_counter, sent_counter = segment_page(
            page, doc, para_counter, sent_counter
        )
        segments.extend(page_segments)

    return segments


def process():
    json_paths = list(Path(INPUT_DIR).rglob("*.json"))
    if not json_paths:
        print(f"No JSON files found in {INPUT_DIR}")
        return

    print(f"Found {len(json_paths)} document(s) to segment.\n")

    all_segments = []

    for json_path in tqdm(json_paths, desc="Segmenting"):
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        segments = segment_document(doc)
        all_segments.extend(segments)

        tqdm.write(f"Done: {json_path.name} — {len(segments):,} segments")

    # save all segments as a single parquet file ready for the classifier
    df = pd.DataFrame(all_segments)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\nTotal segments: {len(df):,}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Columns: {list(df.columns)}")


# For Testing:
"""
# print sample of removed sentences
removed_sample = []
for json_path in list(Path(INPUT_DIR).rglob("*.json"))[:3]:  # just first 3 docs
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    for page in doc["pages"]:
        for sent in split_sentences(page["text"]):
            if is_noise(sent):
                removed_sample.append(sent)

print("\nSample of removed sentences:")
for s in removed_sample[:20]:
    print(" -", s)
"""

if __name__ == "__main__":
    process()