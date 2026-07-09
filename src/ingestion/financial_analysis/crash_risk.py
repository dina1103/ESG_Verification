import json
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INTEGRITY_JSON = r"data\processed\integrity_score_company_year.json"
OUTPUT_CSV_Y   = r"data\processed\crash_risk_y.csv"
OUTPUT_PLOT_Y  = r"data\processed\crash_risk_y_scatter.png"
OUTPUT_CSV_Y1  = r"data\processed\crash_risk_y+1.csv"
OUTPUT_PLOT_Y1 = r"data\processed\crash_risk_y+1_scatter.png"

# one (csv, scatter) pair per window: same-year (Y) and forward (Y+1)
WINDOWS = {
    0: {"tag": "year Y (same-year)",
        "csv": OUTPUT_CSV_Y,
        "plot": OUTPUT_PLOT_Y,
        "title": "Crash Risk (year Y)"},
    1: {"tag": "year Y+1 (forward-year)",
        "csv": OUTPUT_CSV_Y1,
        "plot": OUTPUT_PLOT_Y1,
        "title": "Forward Crash Risk (year Y+1)"},
}

TICKERS = {
    "Aston Martin Lagonda Global Holdings PLC": ("AML.L",   "^FTSE"),
    "Bayerische Motoren Werke AG":              ("BMW.DE",  "^GDAXI"),
    "Ferrari N.V.":                             ("RACE",    "^GSPC"),
    "Ford Motor Company":                       ("F",       "^GSPC"),
    "General Motors Company":                   ("GM",      "^GSPC"),
    "Nissan Motor Co., Ltd.":                   ("7201.T",  "^N225"),
    "Subaru Corporation":                       ("7270.T",  "^N225"),
    "Tesla, Inc.":                              ("TSLA",    "^GSPC"),
    "Toyota Motor Corporation":                 ("7203.T",  "^N225"),
    "Volkswagen AG":                            ("VOW3.DE", "^GDAXI"),
}

MIN_WEEKS = 30
TODAY = date.today()
MEASURES = ["NCSKEW", "DUVOL"]


def get_series(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def firm_specific_weekly(close, idx):
    wk = close.resample("W-FRI").last().pct_change().dropna()
    iw = idx.resample("W-FRI").last().pct_change().dropna()
    j = pd.concat([wk, iw], axis=1, join="inner").dropna()
    if len(j) < MIN_WEEKS:
        return None
    y = j.iloc[:, 0].values
    x = j.iloc[:, 1].values
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.log1p(y - X @ beta)


def ncskew(W):
    n = len(W)
    return float(-(n * (n - 1) ** 1.5 * np.sum(W ** 3)) /
                 ((n - 1) * (n - 2) * np.sum(W ** 2) ** 1.5))


def duvol(W):
    m = W.mean()
    down, up = W[W < m], W[W > m]
    if len(down) < 2 or len(up) < 2:
        return None
    return float(np.log(((len(up) - 1) * np.sum(down ** 2)) /
                        ((len(down) - 1) * np.sum(up ** 2))))


def build_window(integ, cache, offset):
    rows = []
    for key, v in sorted(integ.items()):
        c, y, sc = v.get("company_name"), v.get("year"), v.get("integrity_score")
        if sc is None or c not in TICKERS:
            continue
        wy = y + offset                       # window year (Y for offset 0, Y+1 for offset 1)
        if date(wy, 12, 31) > TODAY:
            continue
        stk, idx = TICKERS[c]
        s0, s1 = f"{wy}-01-01", f"{wy + 1}-01-01"
        cl = cache[stk][(cache[stk].index >= s0) & (cache[stk].index < s1)] if cache[stk] is not None else None
        ic = cache[idx][(cache[idx].index >= s0) & (cache[idx].index < s1)] if cache[idx] is not None else None
        if cl is None or ic is None:
            continue
        W = firm_specific_weekly(cl, ic)
        if W is None:
            continue
        dv = duvol(W)
        rows.append({"company_name": c, "year": y, "integrity_score": sc,
                     "n_layers_available": v.get("n_layers_available", 0),
                     "window_year": wy, "n_weeks": len(W),
                     "NCSKEW": round(ncskew(W), 4),
                     "DUVOL": round(dv, 4) if dv is not None else None})
    return pd.DataFrame(rows)


def make_scatter(df, plot_path, title):
    cmap = {1: "#d9d9d9", 2: "#9ecae1", 3: "#4292c6", 4: "#08519c"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, col in zip(axes, MEASURES):
        s = df.dropna(subset=[col])
        for nl in sorted(s["n_layers_available"].unique()):
            g = s[s["n_layers_available"] == nl]
            ax.scatter(g["integrity_score"], g[col], s=70, c=cmap.get(nl, "#000"),
                       edgecolor="black", linewidth=0.4, label=f"{nl} layer(s)", zorder=3)
        for _, r in s.iterrows():
            ax.annotate(f"{r['company_name'].split()[0]} {str(r['year'])[2:]}",
                        (r["integrity_score"], r[col]), fontsize=8, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        rho, p = spearmanr(s["integrity_score"], s[col])
        b = np.polyfit(s["integrity_score"], s[col], 1)
        xs = np.linspace(s["integrity_score"].min(), s["integrity_score"].max(), 50)
        ax.plot(xs, b[0] * xs + b[1], "--", color="#d62728", linewidth=1.2, zorder=2)
        ax.set_xlabel("ESG Integrity Score", fontsize=14)
        ax.set_ylabel(f"{col} ", fontsize=14)
        ax.set_title(f"{col}   rho = {rho:+.2f}  p = {p:.3f}  n = {len(s)}", fontsize=16)
        ax.grid(True, alpha=0.3, zorder=0)
    axes[0].legend(title="layers (full sample)", fontsize=10, title_fontsize=10)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main():
    integ = json.load(open(INTEGRITY_JSON, encoding="utf-8"))
    need = {t for pair in TICKERS.values() for t in pair}
    print(f"Downloading {len(need)} series...")
    cache = {t: get_series(t, "2019-06-01", f"{TODAY.year}-{TODAY.month:02d}-{TODAY.day:02d}") for t in need}

    for offset, cfg in WINDOWS.items():
        df = build_window(integ, cache, offset)
        df.to_csv(cfg["csv"], index=False)
        print("\n" + "=" * 64)
        print(f"CRASH RISK  ({cfg['tag']}, full sample)  hypothesis: rho < 0")
        print("=" * 64)
        print(f"saved: {cfg['csv']}  ({len(df)} company-years)")
        for col in MEASURES:
            sub = df.dropna(subset=[col])
            rho, p = spearmanr(sub["integrity_score"], sub[col])
            print(f"   {col:8s}  n={len(sub):>2} ({sub['company_name'].nunique()} firms)   rho = {rho:+.3f}  (p = {p:.3f})")
        make_scatter(df, cfg["plot"], cfg["title"])
        print(f"saved scatter: {cfg['plot']}")
    print("\nNote: no statistical inference (small n)")


if __name__ == "__main__":
    main()