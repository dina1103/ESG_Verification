import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pymupdf
import pdfplumber
import pytesseract
from PIL import Image
import io
from tqdm import tqdm

INPUT_DIR = "data/reports"
OUTPUT_DIR = "data/processed"

# pages with fewer chars than this trigger OCR
OCR_FALLBACK_THRESHOLD = 80

# pages below this after OCR are skipped entirely
MIN_CHARS_FINAL = 30

pymupdf.TOOLS.mupdf_display_errors(False)

# folder name -> (company_name, ISIN). Keyed by the folder, since the new
# filenames don't contain the company name. Keys must match folder names exactly.
COMPANY_MAP = {
    "BMW":            ("Bayerische Motoren Werke AG",              "DE0005190003"),
    "Aston Martin":   ("Aston Martin Lagonda Global Holdings PLC", "GB00BFXZC448"),
    "Ferrari":        ("Ferrari N.V.",                             "NL0011585146"),
    "Ford":           ("Ford Motor Company",                       "US3453708600"),
    "General Motors": ("General Motors Company",                   "US37045V1008"),
    "Nissan":         ("Nissan Motor Co., Ltd.",                   "JP3672400003"),
    "Subaru":         ("Subaru Corporation",                       "JP3814800003"),
    "Tesla":          ("Tesla, Inc.",                              "US88160R1014"),
    "Toyota":         ("Toyota Motor Corporation",                 "JP3633400001"),
    "Volkswagen":     ("Volkswagen AG",                            "DE0007664005"),
}


def parse_filename(filename, company_folder):
    # company comes from the folder; year + report type are parsed from the filename
    stem = Path(filename).stem
    tokens = re.split(r"[_\-\s.]+", stem.lower())

    year = 0
    # pass 1: a standalone 4-digit year token (e.g. "2021")
    for tok in tokens:
        if re.fullmatch(r"(19|20)\d{2}", tok):
            year = int(tok)
            break
    # pass 2: 2-digit year, possibly glued to a type code (e.g. "sr20", "ir24")
    if year == 0:
        for tok in tokens:
            m = re.fullmatch(r"([a-z]{2,3})?(\d{2})", tok)
            if m:
                year = 2000 + int(m.group(2))
                if m.group(1):
                    tokens.append(m.group(1))  # expose "sr"/"ir"/"ar" for type detection
                break

    # report type by priority: integrated > sustainability > annual
    report_type = "unknown"
    if any(t in ("integrated", "ir") for t in tokens):
        report_type = "integrated_report"
    elif any(t in ("sustainability", "impact", "esg", "sr") for t in tokens):
        report_type = "sustainability_report"
    elif any(t in ("annual", "financial", "ar") for t in tokens):
        report_type = "annual_report"

    company_name, company_id = COMPANY_MAP.get(company_folder, ("unknown", "unknown"))

    return {
        "company_id":   company_id,
        "company_name": company_name,
        "year":         year,
        "report_type":  report_type,
        "framework":    "unknown",
    }


@dataclass
class PageRecord:
    page_number: int
    text: str
    tables: list[str]
    is_ocr: bool
    char_count: int


@dataclass
class DocumentRecord:
    filename: str
    source_path: str
    company_id: str
    company_name: str
    year: int
    report_type: str
    framework: str
    total_pages: int
    pages_extracted: int
    pages_ocr: int
    pages_skipped: int
    pages: list[PageRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def clean_text(text):
    # normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # remove download watermarks
    text = re.sub(r"Downloaded from\s+\S+\s*\|.*", "", text)
    # collapse all whitespace to single spaces first
    text = re.sub(r"\s+", " ", text)
    # normal all-caps words keep their spaces ("ANNUAL REPORT" stays two words)
    text = re.sub(r"\b(?:[A-Z] )+[A-Z]\b", lambda m: m.group().replace(" ", ""), text)
    return text.strip()

def _extract_tables_from_page(plumber_page):
    # extract tables from an already-open pdfplumber page
    rows = []
    try:
        for table in plumber_page.find_tables():
            extracted = table.extract()
            if extracted:
                for row in extracted:
                    cells = [c.strip() for c in row if c and c.strip()]
                    if cells:
                        rows.append(" | ".join(cells))
    except Exception:
        pass
    return rows


def _ocr_page(fitz_page):
    # rasterize at 300 DPI and run tesseract
    try:
        mat = pymupdf.Matrix(300 / 72, 300 / 72)
        pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang="eng").strip()
    except Exception:
        return ""


def extract_pdf(pdf_path, metadata=None, use_ocr=True, use_tables=True, max_pages=None):
    pdf_path = Path(pdf_path)
    meta = metadata or {}

    doc_record = DocumentRecord(
        filename=pdf_path.name,
        source_path=str(pdf_path.resolve()),
        company_id=meta.get("company_id", "unknown"),
        company_name=meta.get("company_name", "unknown"),
        year=meta.get("year", 0),
        report_type=meta.get("report_type", "unknown"),
        framework=meta.get("framework", "unknown"),
        total_pages=0,
        pages_extracted=0,
        pages_ocr=0,
        pages_skipped=0,
    )

    try:
        fitz_doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        print(f"Error reading {pdf_path.name}: {e}")
        return doc_record

    total = len(fitz_doc)
    doc_record.total_pages = total
    limit = min(total, max_pages) if max_pages else total

    # open pdfplumber once for the whole document
    plumber_doc = None
    if use_tables:
        try:
            plumber_doc = pdfplumber.open(str(pdf_path))
        except Exception:
            pass

    for i in range(limit):
        fitz_page = fitz_doc[i]
        page_num = i + 1

        # extract native text layer
        raw_text = fitz_page.get_text("text").strip()
        is_ocr = False

        # fall back to OCR if page is sparse or scanned
        if use_ocr and len(raw_text) < OCR_FALLBACK_THRESHOLD:
            ocr_text = _ocr_page(fitz_page)
            if len(ocr_text) > len(raw_text):
                raw_text = ocr_text
                is_ocr = True

        text = clean_text(raw_text)

        # extract tables from the already-open pdfplumber doc
        tables = []
        if plumber_doc and i < len(plumber_doc.pages):
            tables = _extract_tables_from_page(plumber_doc.pages[i])

        # skip empty or very small pages
        total_content = len(text) + sum(len(t) for t in tables)
        if total_content < MIN_CHARS_FINAL:
            doc_record.pages_skipped += 1
            continue

        doc_record.pages.append(PageRecord(
            page_number=page_num,
            text=text,
            tables=tables,
            is_ocr=is_ocr,
            char_count=len(text),
        ))
        doc_record.pages_extracted += 1
        if is_ocr:
            doc_record.pages_ocr += 1

    fitz_doc.close()
    if plumber_doc:
        plumber_doc.close()

    return doc_record


def write_txt(doc, output_path):
    # write human-readable txt with page and table markers
    lines = []
    for page in doc.pages:
        block = f"[PAGE {page.page_number}]\n{page.text}"
        if page.tables:
            table_block = "\n".join(f"[TABLE] {row}" for row in page.tables)
            block += "\n\n" + table_block
        lines.append(block)
    output_path.write_text("\n\n".join(lines), encoding="utf-8")


def write_json(doc, output_path):
    # write structured json with metadata for downstream steps
    output_path.write_text(
        json.dumps(doc.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def process(use_ocr=True, use_tables=True, max_pages=None, skip_existing=True):
    pdf_paths = list(Path(INPUT_DIR).rglob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {INPUT_DIR}")
        return

    print(f"Found {len(pdf_paths)} PDF(s) to process.\n")

    for pdf_path in tqdm(pdf_paths, desc="Extracting PDFs"):
        relative = pdf_path.relative_to(INPUT_DIR)
        company_folder = pdf_path.parent.name      # folder = company
        stem = relative.with_suffix("")

        # mirror folder structure for both output formats
        json_out = Path(OUTPUT_DIR) / "json" / stem.with_suffix(".json")
        txt_out  = Path(OUTPUT_DIR) / "text" / stem.with_suffix(".txt")

        json_out.parent.mkdir(parents=True, exist_ok=True)
        txt_out.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and json_out.exists():
            tqdm.write(f"⏭ Skipping (already done): {pdf_path.name}")
            continue

        # parse metadata from filename
        metadata = parse_filename(pdf_path.name, company_folder)

        doc = extract_pdf(
            pdf_path,
            metadata=metadata,
            use_ocr=use_ocr,
            use_tables=use_tables,
            max_pages=max_pages,
        )

        write_json(doc, json_out)
        write_txt(doc, txt_out)

        tqdm.write(
            f"Done: {pdf_path.name} — "
            f"{doc.pages_extracted}/{doc.total_pages} pages "
            f"({doc.pages_ocr} OCR, {doc.pages_skipped} skipped)"
        )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-tables", action="store_true")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--reprocess", action="store_true")
    args = parser.parse_args()

    process(
        use_ocr=not args.no_ocr,
        use_tables=not args.no_tables,
        max_pages=args.max_pages,
        skip_existing=not args.reprocess,
    )