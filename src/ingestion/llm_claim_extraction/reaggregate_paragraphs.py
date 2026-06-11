import pandas as pd
from pathlib import Path

INPUT_PATH  = r"data\processed\segments_esg_sdg.parquet"
OUTPUT_PATH = r"data\processed\llm_paragraphs.parquet"

ESG_LABELS = {"Environmental", "Social", "Governance"}

# paragraphs longer than this get windowed down to ESG sentences + context
MAX_CHARS = 6000
# neighbour sentences kept on each side of each ESG sentence when windowing
WINDOW = 2


def build_text(group):
    # group: sentence rows for one paragraph, already in reading order.
    # returns (text, was_windowed). full concatenation unless over MAX_CHARS,
    # in which case keep only ESG sentences +/- WINDOW neighbours, with [...] gaps.
    full = " ".join(group["text"].astype(str).tolist())
    if len(full) <= MAX_CHARS:
        return full, False

    texts = group["text"].astype(str).tolist()
    is_esg = group["esg_label"].isin(ESG_LABELS).tolist()
    n = len(texts)

    keep = [False] * n
    for i in range(n):
        if is_esg[i]:
            for j in range(max(0, i - WINDOW), min(n, i + WINDOW + 1)):
                keep[j] = True

    parts = []
    prev_kept = False
    skipped_any = False
    for i in range(n):
        if keep[i]:
            if not prev_kept and skipped_any:
                parts.append("[...]")
            parts.append(texts[i])
            prev_kept = True
        else:
            prev_kept = False
            skipped_any = True
    return " ".join(parts), True


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  loaded {len(df):,} sentence rows")

    # reading order within each paragraph
    df = df.sort_values(["source_document", "paragraph_id", "sentence_id"]).reset_index(drop=True)

    print("\nAggregating into paragraphs...")
    grouped = df.groupby(["source_document", "paragraph_id"], sort=False)

    records = []
    for (src_doc, para_id), group in grouped:
        n_esg = int(group["esg_label"].isin(ESG_LABELS).sum())

        esg_counts = {k: int(v) for k, v in group["esg_label"].value_counts().to_dict().items()}
        sdg_labels = [s for s in group["sdg_label"].dropna().unique().tolist() if s and s != "None"]

        text, was_windowed = build_text(group)
        first = group.iloc[0]

        records.append({
            "block_id": f"{src_doc}__{para_id}",
            "source_document": src_doc,
            "paragraph_id": para_id,
            "company_name": first["company_name"],
            "year": int(first["year"]),
            "report_type": first["report_type"],
            "framework": first["framework"],
            "page_number_min": int(group["page_number"].min()),
            "page_number_max": int(group["page_number"].max()),
            "text": text,
            "was_windowed": was_windowed,
            "n_sentences": len(group),
            "n_esg_sentences": n_esg,
            "esg_label_distribution": str(esg_counts),
            "sdg_labels": sdg_labels,
            "sentence_ids": group["sentence_id"].tolist(),
        })

    paragraphs = pd.DataFrame(records)
    print(f"  built {len(paragraphs):,} paragraph rows (all)")

    before = len(paragraphs)
    paragraphs = paragraphs[paragraphs["n_esg_sentences"] > 0].reset_index(drop=True)
    print(f"  filtered to ESG-containing: {len(paragraphs):,} ({before - len(paragraphs):,} dropped)")

    n_windowed = int(paragraphs["was_windowed"].sum())
    print(f"  windowed (over {MAX_CHARS} chars): {n_windowed:,}")

    # diagnostics
    L = paragraphs["text"].str.len()
    print("\nDiagnostics:")
    print(f"  unique documents: {paragraphs['source_document'].nunique()}")
    print(f"  unique companies: {paragraphs['company_name'].nunique()}")
    print(f"  year range: {paragraphs['year'].min()}-{paragraphs['year'].max()}")
    print(f"  text length (chars): median {L.median():.0f}  P90 {L.quantile(0.9):.0f}  "
          f"P99 {L.quantile(0.99):.0f}  max {L.max()}")
    print(f"  sentences/paragraph: median {paragraphs['n_sentences'].median():.0f}  "
          f"max {paragraphs['n_sentences'].max()}")
    print(f"  ESG sentences/paragraph: median {paragraphs['n_esg_sentences'].median():.0f}  "
          f"max {paragraphs['n_esg_sentences'].max()}")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    paragraphs.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()