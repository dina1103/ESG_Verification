import re
import pandas as pd
from pathlib import Path

INPUT  = r"data\processed\tables.parquet"
OUTPUT = r"data\processed\tables_clean.parquet"

# a row is KEPT if it mentions any ESG topic
ESG = re.compile(r'(?i)\b('
    r'emission|co2|co\u2082|carbon|ghg|greenhouse|scope\s*[123]|climate|'
    r'energy|electricity|fuel|renewable|solar|wind|kwh|mwh|gwh|'
    r'water|waste|landfill|recycl|circular|'
    r'biodiversit|pollution|spill|air quality|nox|sox|voc|'
    r'diversity|inclusion|women|gender|minorit|ethnic|'
    r'safety|injur|accident|incident|fatalit|health|wellbeing|'
    r'training|development|employee|workforce|labou?r|human rights|'
    r'turnover|retention|engagement|'
    r'supply chain|supplier|sourcing|conflict mineral|'
    r'community|philanthrop|donation|volunteer|'
    r'sustainab|esg|environment|social responsib'
    r')\b')

# dropped: finance / accounting / disclosure-index rows
NOISE = re.compile(r'(?i)('
    r'item\s*\d|note\s*\d|ifrs|gaap|ebitda|ebit\b|'
    r'earnings per share|\beps\b|dividend|coupon|maturit|'
    r'audit fee|director compensation|executive compensation|remuneration table|'
    r'fair value|amorti[sz]ation|depreciation|goodwill|deferred tax|'
    r'\bgri\b|\bsasb\b|tcfd\s*(?:index|content)|content index|'
    r'page\s*\d{1,3}'
    r')')

# dropped: repeated report-navigation boilerplate (CEO message / contents headers)
NAV = re.compile(r'(?i)(message (on |from )|contents .*(environment|social|governance)|'
                 r'letter from the chair|appendix .*content)')


def is_junk(txt):
    # only true OCR garbage — legitimate number-heavy metric rows are kept
    if not txt or len(txt.strip()) < 6:
        return True
    if "(cid:" in txt:
        return True
    alpha = sum(c.isalpha() for c in txt)
    if alpha < 3 and not re.search(r'\d', txt):
        return True
    return False


def clean_text(s):
    # collapse newlines/whitespace for matching (does not alter the stored text)
    return re.sub(r'\s+', ' ', str(s).replace('\n', ' ')).strip()


def main():
    print(f"Loading {INPUT}...")
    t = pd.read_parquet(INPUT)
    print(f"  {len(t):,} table rows")

    ct = t["text"].apply(clean_text)
    is_esg   = ct.apply(lambda s: bool(ESG.search(s)))
    is_noise = ct.apply(lambda s: bool(NOISE.search(s)))
    is_nav   = ct.apply(lambda s: bool(NAV.search(s)))
    is_jnk   = ct.apply(is_junk)

    keep = is_esg & ~is_noise & ~is_nav & ~is_jnk
    out = t[keep].copy()

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT, index=False)

    print(f"\n  mention ESG topic:           {int(is_esg.sum()):,}")
    print(f"  dropped finance/index noise: {int((is_esg & is_noise).sum()):,}")
    print(f"  dropped nav boilerplate:     {int((is_esg & ~is_noise & is_nav).sum()):,}")
    print(f"  dropped OCR junk:            {int((is_esg & ~is_noise & ~is_nav & is_jnk).sum()):,}")
    print(f"KEPT clean ESG rows:           {len(out):,} ({len(out)/len(t)*100:.1f}%)")
    print()
    print("kept rows per company:")
    print(out.groupby("company_name").size().sort_values(ascending=False).to_string())
    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()