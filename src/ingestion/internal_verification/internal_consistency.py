from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# INTERNAL CONSISTENCY V2
# Qualitative performance claims = LLM JSONL
# Quantitative evidence = raw rows/windows from tables.parquet
#
# V2 improvements over the first version:
# 1) Filters out future targets, policies, regulations, generic commitments.
# 2) Filters out GRI/index/page-reference table rows.
# 3) Uses table windows as evidence, not isolated rows like "70".
# 4) Requires ESG metric-category overlap between claim and table evidence.
# 5) Counts aligned/contradicted only when table evidence is meaningful.
# 6) Adds evidence_confidence and review statuses instead of forcing noisy contradictions.
# ============================================================

DEFAULT_CLAIMS_JSONL = Path(r"data\processed\llm_claim_extraction_result.jsonl")
DEFAULT_TABLES_PARQUET = Path(r"data\processed\tables.parquet")
DEFAULT_OUT_CLAIMS = Path(r"data\processed\internal_consistency_claim_level_v2.jsonl")
DEFAULT_OUT_SUMMARY = Path(r"data\processed\internal_consistency_company_year_v2.json")
DEFAULT_OUT_DEBUG = Path(r"data\processed\internal_consistency_top_matches_v2.csv")

TOP_K_DEBUG = 5
MIN_VERDICTS_FOR_SCORE = 3
MATCH_THRESHOLD = 0.12
HIGH_CONF_THRESHOLD = 0.22

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")
PERCENT_RE = re.compile(r"[-+]?\d{1,3}(?:\.\d+)?\s*%")
CONTRASTIVE_SPLIT_RE = re.compile(r"\b(?:but|while|however|although|whereas|yet)\b", re.I)

DOWN_WORDS = [
    "reduced", "reducing", "reduce", "reduction", "decrease", "decreased", "decreasing",
    "lowered", "lower", "cut", "declined", "declining", "fell", "dropped", "minimized",
    "minimised", "less", "lowered by", "reduction in"
]
UP_WORDS = [
    "increased", "increasing", "increase", "grew", "growing", "growth", "expanded", "expanding",
    "raised", "rose", "boosted", "higher", "improved by", "more", "rise"
]
STABLE_WORDS = ["maintained", "sustained", "remained stable", "kept stable", "unchanged", "flat"]
AMBIGUOUS_IMPROVEMENT_WORDS = ["improved", "improving", "progress", "better", "enhanced", "advanced", "strengthened"]

# Words that often create false internal-consistency checks.
# These are usually future-looking, regulatory, policy, or generic strategy statements.
NON_PERFORMANCE_PATTERNS = [
    r"\bwill\b", r"\bshall\b", r"\bplan(?:s|ned|ning)?\b", r"\baim(?:s|ed|ing)?\b",
    r"\btarget(?:s|ed|ing)?\b", r"\bgoal(?:s)?\b", r"\baspire(?:s|d)?\b",
    r"\bcommit(?:s|ted|ment)?\s+to\b", r"\bworking\s+to\b", r"\bstrive(?:s|d)?\s+to\b",
    r"\btoward\b", r"\bon track\b", r"\broadmap\b", r"\bstrategy\b", r"\bpolicy\b",
    r"\bregulation(?:s)?\b", r"\bstandard(?:s)?\b", r"\brequirement(?:s)?\b", r"\blaw(?:s)?\b",
    r"\bEPA\b", r"\bCARB\b", r"\bChina Stage\b", r"\bMinistry\b",
    r"\bscenario\b", r"\b1\.5\s*°?c\b", r"\b2\s*°?c\b",
    r"\bcertification\b", r"\bISO\s*\d+\b",
    r"\bpartnership\b", r"\bmember\s+of\b", r"\bjoined\b",
]

# Past/current performance phrases override some weak future wording if the clause clearly reports actual results.
PERFORMANCE_EVIDENCE_PATTERNS = [
    r"\bhas\s+(?:reduced|decreased|increased|improved|grown|expanded|fallen|dropped|declined)\b",
    r"\bhave\s+(?:reduced|decreased|increased|improved|grown|expanded|fallen|dropped|declined)\b",
    r"\bwas\s+(?:reduced|decreased|increased|improved|lowered|raised)\b",
    r"\bwere\s+(?:reduced|decreased|increased|improved|lowered|raised)\b",
    r"\b(?:reduced|decreased|increased|improved|grew|expanded|fell|dropped|declined|rose)\s+by\s+[-+]?\d",
    r"\b(?:from|compared\s+to|versus|vs\.)\s+(?:19|20)\d{2}\b",
    r"\b(?:in|during|for)\s+(?:19|20)\d{2}\b",
    r"\bachieved\b", r"\breached\b", r"\brecorded\b", r"\breported\b",
]

METRIC_IMPROVEMENT_DIRECTION = {
    "emission": "down", "emissions": "down", "ghg": "down", "co2": "down", "co₂": "down", "carbon": "down",
    "scope 1": "down", "scope 2": "down", "scope 3": "down", "decarbonization": "down",
    "energy consumption": "down", "fuel consumption": "down", "electricity consumption": "down", "gas consumption": "down",
    "water": "down", "waste": "down", "landfill": "down",
    "incident": "down", "injury": "down", "accident": "down", "fatality": "down", "spill": "down", "violation": "down",
    "renewable": "up", "recycled": "up", "recycling": "up", "reuse": "up", "diversion": "up",
    "diversity": "up", "women": "up", "female": "up", "training": "up", "efficiency": "up",
    "electric vehicle": "up", "ev": "up", "zero emission vehicle": "up", "zev": "up", "hybrid": "up",
}

CATEGORY_KEYWORDS = {
    "emissions": ["emission", "emissions", "ghg", "co2", "co₂", "carbon", "scope 1", "scope 2", "scope 3", "decarbonization", "decarbonisation", "climate"],
    "energy": ["energy", "electricity", "fuel", "gas", "natural gas", "renewable", "solar", "wind", "power"],
    "water": ["water", "wastewater", "withdrawal", "discharge"],
    "waste": ["waste", "recycled", "recycling", "landfill", "hazardous", "non-hazardous", "reuse", "diverted"],
    "safety": ["injury", "injuries", "accident", "incident", "fatality", "fatalities", "lost time", "ltifr", "trir", "safety"],
    "diversity": ["women", "female", "gender", "diversity", "minority", "employee", "employees", "workforce"],
    "training": ["training", "trained", "learning", "hours", "education"],
    "vehicles": ["electric vehicle", "electric vehicles", "ev", "zev", "zero emission", "hybrid", "bev", "phev", "fleet", "vehicle", "vehicles"],
    "supplier": ["supplier", "suppliers", "supply chain", "procurement"],
}

# Very broad categories are allowed only if paired with a more specific metric match.
BROAD_ONLY_CATEGORIES = {"supplier"}

ALL_METRIC_TERMS = sorted({term for terms in CATEGORY_KEYWORDS.values() for term in terms}, key=len, reverse=True)
METRIC_REGEX = re.compile("|".join(re.escape(t) for t in ALL_METRIC_TERMS), re.I)
DIRECTION_REGEX = re.compile("|".join(re.escape(t) for t in (DOWN_WORDS + UP_WORDS + STABLE_WORDS)), re.I)

TABLE_REFERENCE_PATTERNS = [
    r"^\s*GRI\s+\d", r"\bGRI\s*\d{3}\b", r"\bESRS\s*[A-Z0-9-]+\b", r"\bSASB\b",
    r"\bpage(?:s)?\s+\d+", r"\bannual report pages?\b", r"\bmanagement approach\b",
    r"^\s*\d{3}-\d+\s*\|", r"^\s*\d{3}\s*[-–]", r"\bdisclosure\b",
]


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return re.sub(r"\s+", " ", str(x).replace("\r", " ").replace("\n", " ")).strip()


def clean_multiline(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in str(x).replace("\r", "\n").split("\n")]
    return "\n".join([ln for ln in lines if ln])




def json_safe(obj: Any) -> Any:
    """Convert numpy/pandas values into normal Python JSON-safe values."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def normalize_company_name(name: Any) -> str:
    text = clean_text(name).lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    stop = {"ag", "se", "sa", "plc", "inc", "corp", "corporation", "company", "co", "ltd", "limited", "nv", "group", "holdings", "holding", "motor", "motors", "automobiles", "automobile"}
    return " ".join(t for t in text.split() if t not in stop)


def safe_int_year(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)) and not pd.isna(x):
        return int(x)
    m = YEAR_RE.search(clean_text(x))
    return int(m.group(0)) if m else None


def contains_any(text: str, words: Iterable[str]) -> bool:
    t = text.lower()
    return any(w.lower() in t for w in words)


def regex_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)


def split_clauses(text: str) -> List[str]:
    return [p.strip() for p in CONTRASTIVE_SPLIT_RE.split(clean_text(text)) if p.strip()]


def metric_categories(text: str) -> set[str]:
    t = clean_text(text).lower()
    cats = set()
    for cat, keys in CATEGORY_KEYWORDS.items():
        if any(k in t for k in keys):
            cats.add(cat)
    return cats


def meaningful_metric_overlap(claim_text: str, evidence_text: str) -> bool:
    claim_cats = metric_categories(claim_text)
    evidence_cats = metric_categories(evidence_text)
    shared = claim_cats & evidence_cats
    shared = {c for c in shared if c not in BROAD_ONLY_CATEGORIES}
    return bool(shared)


def expected_direction_from_text(text: str, metric: str = "") -> str:
    text_l = clean_text(text).lower()
    metric_l = clean_text(metric).lower()

    if contains_any(text_l, DOWN_WORDS):
        return "down"
    if contains_any(text_l, UP_WORDS):
        return "up"
    if contains_any(text_l, STABLE_WORDS):
        return "stable"

    if contains_any(text_l, AMBIGUOUS_IMPROVEMENT_WORDS):
        for key, direction in METRIC_IMPROVEMENT_DIRECTION.items():
            if key in metric_l:
                return direction
        return "unknown"

    return "unknown"


def actual_direction_from_values(current: float, previous: float) -> str:
    # 1% tolerance avoids calling tiny rounded differences contradictions.
    if previous == 0:
        if current == 0:
            return "stable"
        # Avoid strong conclusion from zero baseline unless the row is clearly a percentage/share metric.
        return "up" if current > 0 else "down"
    rel_change = (current - previous) / abs(previous)
    if abs(rel_change) <= 0.01:
        return "stable"
    return "up" if rel_change > 0 else "down"


def remove_noise_numbers(text: str) -> str:
    t = clean_text(text)
    t = re.sub(r"\bGRI\s*\d{3}\s*[-–]?\s*\d*\b", " ", t, flags=re.I)
    t = re.sub(r"\bESRS\s*[A-Z0-9-]+\b", " ", t, flags=re.I)
    t = re.sub(r"\bISO\s*\d+(?::\d+)?\b", " ", t, flags=re.I)
    t = re.sub(r"\bSASB\s*[A-Z0-9-]+\b", " ", t, flags=re.I)
    t = re.sub(r"\bPage(?:s)?\s+\d+(?:[-–]\d+)?\b", " ", t, flags=re.I)
    t = YEAR_RE.sub(" ", t)
    return t


def extract_numbers(text: str) -> List[float]:
    t = remove_noise_numbers(text)
    nums: List[float] = []
    for m in NUMBER_RE.finditer(t):
        raw = m.group(0).replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        nums.append(val)
    return nums


def alpha_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{3,}", clean_text(text)))


def is_reference_table_text(text: str) -> bool:
    t = clean_text(text)
    if regex_any(t, TABLE_REFERENCE_PATTERNS):
        # Keep if it is also obviously a data row with many values and metric words.
        if len(extract_numbers(t)) >= 3 and metric_categories(t):
            return False
        return True
    return False


def is_meaningful_value_text(text: str) -> bool:
    t = clean_text(text)
    if not t or is_reference_table_text(t):
        return False
    if len(extract_numbers(t)) == 0 and not PERCENT_RE.search(t):
        return False
    if alpha_word_count(t) < 2:
        return False
    if not metric_categories(t):
        return False
    return True


def explicit_direction_from_evidence(text: str) -> str:
    t = clean_text(text).lower()
    # Must be a data/performance-like phrase, not merely "target to reduce".
    if regex_any(t, NON_PERFORMANCE_PATTERNS) and not regex_any(t, PERFORMANCE_EVIDENCE_PATTERNS):
        return "unknown"
    has_number = bool(extract_numbers(t) or PERCENT_RE.search(t))
    if not has_number:
        return "unknown"
    if contains_any(t, DOWN_WORDS):
        return "down"
    if contains_any(t, UP_WORDS):
        return "up"
    if contains_any(t, STABLE_WORDS):
        return "stable"
    return "unknown"


def best_single_value_from_text(text: str) -> Optional[float]:
    if not is_meaningful_value_text(text):
        return None
    nums = extract_numbers(text)
    if not nums:
        return None
    # Ignore rows where the only number is clearly a tiny footnote/index and no percent/unit context exists.
    if len(nums) == 1 and nums[0] <= 5 and "%" not in text and not re.search(r"\b(t|tonnes?|mt|mwh|kwh|gwh|tj|m3|hours?|employees?)\b", text, re.I):
        return None
    return nums[-1]


def extract_year_value_pair_from_context(context: str, row_text: str, report_year: Optional[int], baseline_year: Optional[int]) -> Tuple[Optional[float], Optional[float], Optional[int], str]:
    if report_year is None or not is_meaningful_value_text(row_text):
        return None, None, None, "not_meaningful_row"

    possible_baselines: List[int] = []
    if baseline_year is not None and baseline_year < report_year:
        possible_baselines.append(int(baseline_year))
    for y in [report_year - 1, report_year - 2, report_year - 3]:
        if y not in possible_baselines:
            possible_baselines.append(int(y))

    # Use original line breaks when available.
    lines = [ln.strip() for ln in str(context).split("\n") if ln.strip()]
    if not lines:
        lines = [row_text]

    row_nums = extract_numbers(row_text)
    if len(row_nums) < 2:
        return None, None, None, "row_has_fewer_than_two_values"

    # Find any nearby header that contains current year + baseline year.
    for line in lines:
        years = [int(y) for y in YEAR_RE.findall(line)]
        if report_year not in years:
            continue
        for base in possible_baselines:
            if base not in years:
                continue
            n = len(years)
            if len(row_nums) >= n:
                values = row_nums[-n:]
                yv = dict(zip(years, values))
                if report_year in yv and base in yv:
                    return yv[report_year], yv[base], base, "same_table_year_columns"
    return None, None, None, "no_year_header_match"


def baseline_year_from_claim_text(text: str, report_year: Optional[int]) -> Optional[int]:
    if report_year is None:
        return None
    years = [int(y) for y in YEAR_RE.findall(clean_text(text))]
    prev = [y for y in years if y < report_year]
    return max(prev) if prev else None


def claim_has_measureable_keyword(claim_text: str, metric: str) -> bool:
    return bool(metric_categories(f"{metric} {claim_text}"))


def skip_reason_for_claim(clause: str, metric: str, claim_type: str) -> Optional[str]:
    text = clean_text(clause)
    joined = f"{metric} {text}"

    if not claim_has_measureable_keyword(text, metric):
        return "not_measurable_esg_metric"

    direction = expected_direction_from_text(text, metric)
    if direction == "unknown":
        return "no_clear_direction"

    lower = text.lower()

    # Claims about standards/regulations becoming stricter are not company performance.
    if regex_any(text, NON_PERFORMANCE_PATTERNS) and not regex_any(text, PERFORMANCE_EVIDENCE_PATTERNS):
        return "not_actual_performance_claim"

    # Skip generic commitments/targets even if the LLM marked them as commitments.
    if claim_type == "commitment" and not regex_any(text, PERFORMANCE_EVIDENCE_PATTERNS):
        return "commitment_or_target_not_current_performance"

    # Require actual performance language for non-numeric vague clauses.
    has_number = bool(extract_numbers(text) or PERCENT_RE.search(text))
    has_perf_phrase = regex_any(text, PERFORMANCE_EVIDENCE_PATTERNS)
    has_past_direction = any(w in lower for w in ["reduced", "decreased", "increased", "grew", "fell", "dropped", "declined", "rose", "maintained", "achieved", "reported", "recorded"])
    if not (has_number or has_perf_phrase or has_past_direction):
        return "directional_but_not_evidence_like"

    return None


# ============================================================
# Load claims
# ============================================================

def load_claim_clauses(path: Path, keep_skipped: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            company = rec.get("company_name")
            year = safe_int_year(rec.get("year"))
            claims = rec.get("parsed_claims") or []

            for claim_idx, claim in enumerate(claims):
                claim_text = clean_text(claim.get("claim_text"))
                claim_type = clean_text(claim.get("claim_type")).lower()
                metric = clean_text(claim.get("metric"))

                # We want qualitative/narrative side from JSONL. If the LLM extracted a numeric value,
                # it is not the qualitative side for this check.
                if claim.get("quantified_value") is not None:
                    continue

                if claim_type not in {"narrative", "commitment", "achievement", ""}:
                    continue

                for clause_idx, clause in enumerate(split_clauses(claim_text)):
                    expected = expected_direction_from_text(clause, metric)
                    skip_reason = skip_reason_for_claim(clause, metric, claim_type)
                    if skip_reason and not keep_skipped:
                        continue

                    rows.append({
                        "company_name": company,
                        "company_key": normalize_company_name(company),
                        "year": year,
                        "block_id": rec.get("block_id"),
                        "page_number_min": rec.get("page_number_min"),
                        "page_number_max": rec.get("page_number_max"),
                        "claim_idx": claim_idx,
                        "clause_idx": clause_idx,
                        "claim_type": claim_type,
                        "claim_text": claim_text,
                        "qualitative_clause": clause,
                        "metric": metric,
                        "scope": clean_text(claim.get("scope")),
                        "geography": clean_text(claim.get("geography")),
                        "expected_direction": expected,
                        "baseline_year_from_claim": baseline_year_from_claim_text(clause, year),
                        "skip_reason": skip_reason,
                    })
    return rows


def iter_assessable_claim_clauses(path: Path) -> Iterable[Dict[str, Any]]:
    """Memory-safe stream of only assessable qualitative performance claim clauses."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            company = rec.get("company_name")
            year = safe_int_year(rec.get("year"))
            claims = rec.get("parsed_claims") or []

            for claim_idx, claim in enumerate(claims):
                claim_text = clean_text(claim.get("claim_text"))
                claim_type = clean_text(claim.get("claim_type")).lower()
                metric = clean_text(claim.get("metric"))

                if claim.get("quantified_value") is not None:
                    continue
                if claim_type not in {"narrative", "commitment", "achievement", ""}:
                    continue

                for clause_idx, clause in enumerate(split_clauses(claim_text)):
                    expected = expected_direction_from_text(clause, metric)
                    skip_reason = skip_reason_for_claim(clause, metric, claim_type)
                    if skip_reason:
                        continue
                    yield {
                        "company_name": company,
                        "company_key": normalize_company_name(company),
                        "year": year,
                        "block_id": rec.get("block_id"),
                        "page_number_min": rec.get("page_number_min"),
                        "page_number_max": rec.get("page_number_max"),
                        "claim_idx": claim_idx,
                        "clause_idx": clause_idx,
                        "claim_type": claim_type,
                        "claim_text": claim_text[:1000],
                        "qualitative_clause": clause,
                        "metric": metric,
                        "scope": clean_text(claim.get("scope")),
                        "geography": clean_text(claim.get("geography")),
                        "expected_direction": expected,
                        "baseline_year_from_claim": baseline_year_from_claim_text(clause, year),
                        "skip_reason": None,
                    }


# ============================================================
# Load/build table evidence
# ============================================================

def row_number_from_id(row_id: Any) -> int:
    m = re.search(r"(\d+)", clean_text(row_id))
    return int(m.group(1)) if m else 0


def load_table_evidence(path: Path, window_before: int = 2, window_after: int = 2) -> pd.DataFrame:
    """Build filtered table-evidence windows quickly from raw extracted table rows."""
    df = pd.read_parquet(path).copy()
    required = {"company_name", "year", "table_row_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tables.parquet is missing required columns: {sorted(missing)}")

    for col in ["source_document", "page_number", "report_type", "framework"]:
        if col not in df.columns:
            df[col] = "" if col != "page_number" else -1

    df["flat_text"] = (
        df["text"].fillna("").astype(str)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["company_key"] = df["company_name"].apply(normalize_company_name)
    df["year"] = df["year"].apply(safe_int_year)
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df["row_num"] = df["table_row_id"].apply(row_number_from_id)

    df["has_number"] = df["flat_text"].str.contains(r"\d", regex=True, na=False)
    df = df[df["has_number"]].copy()
    df = df.sort_values(["company_name", "year", "source_document", "page_number", "row_num"]).reset_index(drop=True)

    records: List[Dict[str, Any]] = []
    group_cols = ["company_name", "year", "source_document", "page_number"]

    for _, g in df.groupby(group_cols, sort=False):
        rows = g.to_dict("records")
        texts = [clean_text(r.get("flat_text"))[:500] for r in rows]
        n = len(rows)
        for pos, row in enumerate(rows):
            row_text = texts[pos]
            if not row_text or is_reference_table_text(row_text):
                continue

            start = max(0, pos - window_before)
            end = min(n, pos + window_after + 1)
            context_flat = " ".join(texts[start:end])[:900]

            if len(context_flat) < 8:
                continue
            if not METRIC_REGEX.search(context_flat):
                continue
            if alpha_word_count(context_flat) < 4:
                continue

            cats = metric_categories(context_flat)
            if not cats:
                continue

            records.append({
                "company_name": row.get("company_name"),
                "company_key": row.get("company_key"),
                "year": int(row.get("year")),
                "source_document": row.get("source_document"),
                "page_number": row.get("page_number"),
                "table_row_id": row.get("table_row_id"),
                "row_num": row.get("row_num"),
                "row_text": row_text,
                "evidence_text": context_flat,
                "evidence_multiline": context_flat,
                "metric_categories": ",".join(sorted(cats)),
            })

    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError("No usable quantitative table evidence rows found after filtering.")

    out = out.drop_duplicates(subset=["company_name", "year", "source_document", "page_number", "table_row_id"]).reset_index(drop=True)
    out["match_text"] = (
        out["company_name"].astype(str) + " | " + out["year"].astype(str) + " | " +
        out["row_text"].astype(str) + " | " + out["evidence_text"].astype(str)
    )
    return out

# ============================================================
# Matching/assessment
# ============================================================

def make_claim_query(claim: Dict[str, Any]) -> str:
    return " | ".join(x for x in [
        clean_text(claim.get("metric")),
        clean_text(claim.get("scope")),
        clean_text(claim.get("geography")),
        clean_text(claim.get("qualitative_clause")),
    ] if x)


def top_matches(query: str, candidate_df: pd.DataFrame, candidate_matrix, vectorizer: TfidfVectorizer, k: int = TOP_K_DEBUG) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df.copy()
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, candidate_matrix)[0]
    out = candidate_df.copy()
    out["similarity"] = sims
    return out.sort_values("similarity", ascending=False).head(k)


def select_current_matches(claim: Dict[str, Any], company_df: pd.DataFrame, company_matrix, vectorizer: TfidfVectorizer) -> pd.DataFrame:
    year = claim.get("year")
    query = make_claim_query(claim)
    if year is not None:
        same_year_positions = company_df.index[company_df["year"] == int(year)].tolist()
        if same_year_positions:
            cand = company_df.loc[same_year_positions]
            mat = company_matrix[[company_df.index.get_loc(i) for i in same_year_positions]]
            return top_matches(query, cand, mat, vectorizer, TOP_K_DEBUG)
    return top_matches(query, company_df, company_matrix, vectorizer, TOP_K_DEBUG)


def select_baseline_match(claim: Dict[str, Any], current_row: pd.Series, company_df: pd.DataFrame, company_matrix, vectorizer: TfidfVectorizer) -> Optional[pd.Series]:
    claim_year = claim.get("year")
    if claim_year is None:
        return None
    baseline_year = claim.get("baseline_year_from_claim") or int(claim_year) - 1

    pool = company_df[company_df["year"] == int(baseline_year)]
    if pool.empty:
        earlier = company_df[company_df["year"] < int(claim_year)]
        if earlier.empty:
            return None
        closest = earlier["year"].max()
        pool = earlier[earlier["year"] == closest]

    # Require evidence category overlap with current evidence.
    current_cats = set(clean_text(current_row.get("metric_categories")).split(","))
    if current_cats:
        pool = pool[pool["metric_categories"].apply(lambda x: bool(current_cats & set(clean_text(x).split(","))))]
    if pool.empty:
        return None

    query = " | ".join([
        clean_text(claim.get("metric")),
        clean_text(claim.get("qualitative_clause")),
        clean_text(current_row.get("row_text")),
    ])
    positions = pool.index.tolist()
    mat = company_matrix[[company_df.index.get_loc(i) for i in positions]]
    top = top_matches(query, pool, mat, vectorizer, 1)
    if top.empty:
        return None
    return top.iloc[0]


def confidence_from_similarity(sim: float, has_metric_overlap: bool, evidence_method: str) -> str:
    if sim >= HIGH_CONF_THRESHOLD and has_metric_overlap:
        return "high"
    if sim >= MATCH_THRESHOLD and has_metric_overlap and evidence_method in {"explicit_direction_in_table_evidence", "same_table_year_columns"}:
        return "medium"
    if sim >= MATCH_THRESHOLD and has_metric_overlap:
        return "low"
    return "weak"


def assess_claim(claim: Dict[str, Any], company_df: pd.DataFrame, company_matrix, vectorizer: TfidfVectorizer) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    expected = claim.get("expected_direction", "unknown")
    top = select_current_matches(claim, company_df, company_matrix, vectorizer)

    debug: List[Dict[str, Any]] = []
    claim_query = make_claim_query(claim)
    for rank, (_, r) in enumerate(top.iterrows(), start=1):
        debug.append({
            "rank": rank,
            "similarity": round(float(r.get("similarity", 0.0)), 4),
            "table_company": r.get("company_name"),
            "table_year": int(r.get("year")),
            "source_document": r.get("source_document"),
            "page_number": r.get("page_number"),
            "table_row_id": r.get("table_row_id"),
            "metric_categories": r.get("metric_categories"),
            "row_text": clean_text(r.get("row_text"))[:400],
            "evidence_text": clean_text(r.get("evidence_text"))[:800],
        })

    if top.empty:
        return {
            "verdict": "unsupported",
            "reason": "no quantitative table evidence for same company/year",
            "expected_direction": expected,
            "evidence_confidence": "none",
        }, debug

    current = top.iloc[0]
    sim = float(current.get("similarity", 0.0))
    evidence_text = clean_text(current.get("evidence_text"))
    row_text = clean_text(current.get("row_text"))
    has_overlap = meaningful_metric_overlap(claim_query, evidence_text)

    common_base = {
        "similarity": round(sim, 4),
        "current_table_year": int(current.get("year")),
        "current_table_row_id": current.get("table_row_id"),
        "source_document": current.get("source_document"),
        "page_number": current.get("page_number"),
        "current_table_text": row_text[:700],
        "current_evidence_text": evidence_text[:1000],
        "table_metric_categories": current.get("metric_categories"),
    }

    if sim < MATCH_THRESHOLD:
        return {
            **common_base,
            "verdict": "unsupported",
            "reason": f"best table match below threshold ({sim:.3f} < {MATCH_THRESHOLD:.3f})",
            "expected_direction": expected,
            "evidence_confidence": "weak",
        }, debug

    if not has_overlap:
        return {
            **common_base,
            "verdict": "unsupported",
            "reason": "best table match lacks ESG metric-category overlap with the claim",
            "expected_direction": expected,
            "evidence_confidence": "weak",
        }, debug

    # Method 1: explicit table evidence direction, e.g. "WATER CONSUMPTION DECREASED BY | 41%".
    table_dir = explicit_direction_from_evidence(evidence_text)
    if table_dir != "unknown":
        method = "explicit_direction_in_table_evidence"
        confidence = confidence_from_similarity(sim, has_overlap, method)
        verdict = "aligned" if table_dir == expected else "contradicted"
        if confidence == "low":
            verdict = "review_needed"
        return {
            **common_base,
            "verdict": verdict,
            "reason": f"expected={expected}; table actual={table_dir}; method={method}",
            "expected_direction": expected,
            "actual_direction": table_dir,
            "evidence_method": method,
            "evidence_confidence": confidence,
        }, debug

    # Method 2: same table row contains multiple year columns.
    cur_val, prev_val, base_year, parse_method = extract_year_value_pair_from_context(
        context=clean_multiline(current.get("evidence_multiline")),
        row_text=row_text,
        report_year=claim.get("year"),
        baseline_year=claim.get("baseline_year_from_claim"),
    )
    if cur_val is not None and prev_val is not None:
        method = "same_table_year_columns"
        actual = actual_direction_from_values(cur_val, prev_val)
        confidence = confidence_from_similarity(sim, has_overlap, method)
        verdict = "aligned" if actual == expected else "contradicted"
        if confidence == "low":
            verdict = "review_needed"
        return {
            **common_base,
            "verdict": verdict,
            "reason": f"expected={expected}; table actual={actual}; method={method}; baseline_year={base_year}",
            "expected_direction": expected,
            "actual_direction": actual,
            "evidence_method": method,
            "evidence_confidence": confidence,
            "current_value": cur_val,
            "baseline_value": prev_val,
            "baseline_table_year": base_year,
        }, debug

    # Method 3: compare matching row to previous report year, but only if both rows are meaningful.
    baseline_row = select_baseline_match(claim, current, company_df, company_matrix, vectorizer)
    if baseline_row is not None:
        current_value = best_single_value_from_text(row_text)
        baseline_text = clean_text(baseline_row.get("row_text"))
        baseline_value = best_single_value_from_text(baseline_text)
        baseline_sim = float(baseline_row.get("similarity", 0.0)) if "similarity" in baseline_row else np.nan
        baseline_overlap = meaningful_metric_overlap(evidence_text, clean_text(baseline_row.get("evidence_text")))

        if current_value is not None and baseline_value is not None and baseline_overlap:
            method = "matched_row_vs_previous_report_year"
            actual = actual_direction_from_values(current_value, baseline_value)
            confidence = confidence_from_similarity(sim, has_overlap, method)
            # Previous-year row matching is noisier; require high confidence for contradiction.
            verdict = "aligned" if actual == expected else "contradicted"
            if confidence != "high":
                verdict = "review_needed"
            return {
                **common_base,
                "verdict": verdict,
                "reason": f"expected={expected}; table actual={actual}; method={method}",
                "expected_direction": expected,
                "actual_direction": actual,
                "evidence_method": method,
                "evidence_confidence": confidence,
                "current_value": current_value,
                "baseline_value": baseline_value,
                "baseline_table_year": int(baseline_row.get("year")),
                "baseline_table_row_id": baseline_row.get("table_row_id"),
                "baseline_table_text": baseline_text[:700],
                "baseline_evidence_text": clean_text(baseline_row.get("evidence_text"))[:1000],
            }, debug

    return {
        **common_base,
        "verdict": "no_quantitative_direction",
        "reason": "matched a relevant table window, but could not extract a reliable direction from numeric evidence",
        "expected_direction": expected,
        "evidence_confidence": "none",
    }, debug


# ============================================================
# Main
# ============================================================

def build_summary(per_claim_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for r in per_claim_results:
        grouped[(r.get("company_name"), r.get("year"))].append(r)

    summary: Dict[str, Dict[str, Any]] = {}
    for (company, year), rows in sorted(grouped.items(), key=lambda kv: (clean_text(kv[0][0]), kv[0][1] or 0)):
        counts = Counter(r.get("verdict") for r in rows)
        denom = counts["aligned"] + counts["contradicted"]
        score = round(counts["aligned"] / denom, 4) if denom >= MIN_VERDICTS_FOR_SCORE else None
        summary[f"{company}__{year}"] = {
            "company_name": company,
            "year": year,
            "n_total_claim_clauses_seen": len(rows),
            "n_aligned": counts["aligned"],
            "n_contradicted": counts["contradicted"],
            "n_review_needed": counts["review_needed"],
            "n_unsupported": counts["unsupported"],
            "n_no_quantitative_direction": counts["no_quantitative_direction"],
            "n_not_performance_claim": counts["not_performance_claim"],
            "n_directional_verdicts": denom,
            "internal_consistency_score": score,
            "note": None if score is not None else f"insufficient_evidence ({denom} aligned/contradicted verdicts)",
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal consistency check: qualitative LLM claims vs quantitative table evidence.")
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS_JSONL)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES_PARQUET)
    parser.add_argument("--out-claims", type=Path, default=DEFAULT_OUT_CLAIMS)
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    parser.add_argument("--out-debug", type=Path, default=DEFAULT_OUT_DEBUG)
    parser.add_argument("--match-threshold", type=float, default=MATCH_THRESHOLD)
    parser.add_argument("--high-conf-threshold", type=float, default=HIGH_CONF_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    global MATCH_THRESHOLD, HIGH_CONF_THRESHOLD
    args = parse_args()
    MATCH_THRESHOLD = args.match_threshold
    HIGH_CONF_THRESHOLD = args.high_conf_threshold

    t0 = time.time()

    print(f"Loading quantitative table evidence: {args.tables}")
    evidence = load_table_evidence(args.tables)
    print(f"  usable table evidence windows: {len(evidence):,}")
    print(f"  companies in table evidence: {evidence['company_name'].nunique():,}")

    print("\nFitting TF-IDF matcher on filtered table evidence...")
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95, stop_words="english")
    evidence_matrix = vectorizer.fit_transform(evidence["match_text"].tolist())
    print(f"  TF-IDF matrix shape: {evidence_matrix.shape}")

    company_to_positions: Dict[str, List[int]] = defaultdict(list)
    for pos, key in enumerate(evidence["company_key"].tolist()):
        company_to_positions[key].append(pos)

    results: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    print(f"\nStreaming and assessing assessable LLM claim clauses from: {args.claims}")
    i = 0
    for i, claim in enumerate(iter_assessable_claim_clauses(args.claims), start=1):
        positions = company_to_positions.get(claim.get("company_key"), [])
        if not positions:
            result = {
                "verdict": "unsupported",
                "reason": "no quantitative table evidence found for same company",
                "expected_direction": claim.get("expected_direction"),
                "evidence_confidence": "none",
            }
            top_debug: List[Dict[str, Any]] = []
        else:
            company_df = evidence.iloc[positions].copy().reset_index(drop=True)
            company_matrix = evidence_matrix[positions]
            result, top_debug = assess_claim(claim, company_df, company_matrix, vectorizer)

        out = {
            "company_name": claim.get("company_name"),
            "year": claim.get("year"),
            "qualitative_block_id": claim.get("block_id"),
            "claim_type": claim.get("claim_type"),
            "page_number_min": claim.get("page_number_min"),
            "qualitative_metric": claim.get("metric"),
            "qualitative_clause": claim.get("qualitative_clause"),
            "scope": claim.get("scope"),
            "geography": claim.get("geography"),
            **result,
        }
        results.append(out)

        for d in top_debug:
            debug_rows.append({
                "company_name": claim.get("company_name"),
                "year": claim.get("year"),
                "qualitative_block_id": claim.get("block_id"),
                "qualitative_metric": claim.get("metric"),
                "qualitative_clause": claim.get("qualitative_clause"),
                **d,
            })

        if i % 100 == 0:
            print(f"  processed {i:,} assessable clauses")

    args.out_claims.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_debug.parent.mkdir(parents=True, exist_ok=True)

    print("\nWriting outputs...")
    with open(args.out_claims, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False, default=json_safe) + "\n")

    summary = build_summary(results)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_safe)

    pd.DataFrame(debug_rows).to_csv(args.out_debug, index=False, encoding="utf-8-sig")

    verdict_counts = Counter(r.get("verdict") for r in results)
    print("\nVerdict counts:")
    for k, v in verdict_counts.most_common():
        print(f"  {k}: {v:,}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Claim-level output:  {args.out_claims}")
    print(f"Summary output:      {args.out_summary}")
    print(f"Debug matches CSV:   {args.out_debug}")
    print(f"Assessable clauses:  {i:,}")
    print(f"Rows written:        {len(results):,}")
    print(f"Total time:          {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
