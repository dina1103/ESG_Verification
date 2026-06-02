import os
import json
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

INPUT_DIR = "data/processed/json"
OUTPUT_FILE = "data/processed/segments.parquet"
TABLES_FILE = "data/processed/tables.parquet"

# segments shorter than this are dropped
MIN_SEGMENT_CHARS = 60

# segments longer than this are hard-wrapped so they fit ESGBERT's ~512-token window
MAX_SEGMENT_CHARS = 1500

print("Segmentation script running...")


def hard_wrap(sentence):
    if len(sentence) <= MAX_SEGMENT_CHARS:
        return [sentence]
    clauses = re.split(r"(?<=[;:])\s+|(?<=,)\s+", sentence)
    # if clause-splitting didn't help (no punctuation), fall back to word chunks
    if max(len(c) for c in clauses) > MAX_SEGMENT_CHARS:
        words = sentence.split()
        clauses, buf = [], ""
        for w in words:
            if len(buf) + len(w) + 1 > MAX_SEGMENT_CHARS and buf:
                clauses.append(buf); buf = w
            else:
                buf = f"{buf} {w}".strip()
        if buf:
            clauses.append(buf)
    chunks, buf = [], ""
    for c in clauses:
        if len(buf) + len(c) + 1 > MAX_SEGMENT_CHARS and buf:
            chunks.append(buf.strip()); buf = c
        else:
            buf = f"{buf} {c}".strip()
    if buf:
        chunks.append(buf.strip())
    return chunks


def split_sentences(text):
    # protect known non-sentence periods before splitting
    text = re.sub(r"(\b(?:e\.g|i\.e|vs|Mr|Dr|Corp|Ltd|Inc|Fig|No|CO2|GHG))\.", r"\1<DOT>", text)
    text = re.sub(r"(\d+)\.(\d+)", r"\1<DOT>\2", text)

    # split on sentence-ending punctuation OR bullet markers (lists lack terminal punctuation)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])|\s+•\s+", text)

    sentences = [p.replace("<DOT>", ".").strip() for p in parts]
    sentences = [s for s in sentences if len(s) > MIN_SEGMENT_CHARS]

    # hard-wrap any remaining over-long segment so the classifier never truncates
    wrapped = []
    for s in sentences:
        wrapped.extend(hard_wrap(s))
    return [s for s in wrapped if len(s) > MIN_SEGMENT_CHARS]


def is_noise(text):
    if not text:
        return True
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


def collect_tables(doc):
    # tables are preserved verbatim in a parallel channel, not segmented as prose
    rows = []
    t_counter = 0
    for page in doc["pages"]:
        for row_text in page.get("tables", []):
            row_text = row_text.strip()
            if not row_text:
                continue
            rows.append({
                "company_id":      doc["company_id"],
                "company_name":    doc["company_name"],
                "year":            doc["year"],
                "report_type":     doc["report_type"],
                "framework":       doc["framework"],
                "source_document": doc["filename"],
                "page_number":     page["page_number"],
                "is_ocr":          page["is_ocr"],
                "table_row_id":    f"t_{t_counter:05d}",
                "text":            row_text,
            })
            t_counter += 1
    return rows


def process():
    json_paths = list(Path(INPUT_DIR).rglob("*.json"))
    if not json_paths:
        print(f"No JSON files found in {INPUT_DIR}")
        return

    print(f"Found {len(json_paths)} document(s) to segment.\n")

    all_segments = []
    all_tables = []

    for json_path in tqdm(json_paths, desc="Segmenting"):
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        segments = segment_document(doc)
        all_segments.extend(segments)
        all_tables.extend(collect_tables(doc))

        tqdm.write(f"Done: {json_path.name} — {len(segments):,} segments")

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    # prose segments -> classifier
    df = pd.DataFrame(all_segments)
    df.to_parquet(OUTPUT_FILE, index=False)

    # table rows -> preserved verbatim for Step 8 metric verification
    df_tables = pd.DataFrame(all_tables)
    df_tables.to_parquet(TABLES_FILE, index=False)

    print(f"\nTotal segments: {len(df):,}")
    print(f"Total table rows: {len(df_tables):,}")
    print(f"Saved segments to: {OUTPUT_FILE}")
    print(f"Saved tables to:   {TABLES_FILE}")
    print(f"Segment columns: {list(df.columns)}")


if __name__ == "__main__":
    process()


# For Testing (run manually, not on import):

removed_sample = []
for json_path in list(Path(INPUT_DIR).rglob("*.json"))[:3]:
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    for page in doc["pages"]:
        for para in page["text"].split("\n"):
            for sent in split_sentences(para):
                if is_noise(sent):
                    removed_sample.append(sent)

print("\nSample of removed sentences:")
for s in removed_sample[:20]:
    print(" -", s)
