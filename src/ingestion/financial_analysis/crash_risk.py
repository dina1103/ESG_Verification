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
OUTPUT_CSV     = r"data\processed\crash_risk.csv"
OUTPUT_PLOT    = r"data\processed\crash_risk_scatter.png"

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

FORWARD = 1            # window = the single forward year t+1
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


def main():
    integ = json.load(open(INTEGRITY_JSON, encoding="utf-8"))
    need = {t for pair in TICKERS.values() for t in pair}
    print(f"Downloading {len(need)} series...")
    cache = {t: get_series(t, "2019-06-01", f"{TODAY.year}-{TODAY.month:02d}-{TODAY.day:02d}") for t in need}

    rows = []
    for key, v in sorted(integ.items()):
        c, y, sc = v.get("company_name"), v.get("year"), v.get("integrity_score")
        if sc is None or c not in TICKERS:
            continue
        we = y + FORWARD
        if date(we, 12, 31) > TODAY:
            continue
        stk, idx = TICKERS[c]
        s0, s1 = f"{we}-01-01", f"{we + 1}-01-01"
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
                     "forward_year": we, "n_weeks": len(W),
                     "NCSKEW": round(ncskew(W), 4),
                     "DUVOL": round(dv, 4) if dv is not None else None})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"saved: {OUTPUT_CSV}  ({len(df)} company-years)\n")

    print("=" * 64)
    print("CRASH RISK  (forward year t+1, full sample)  hypothesis: rho < 0")
    print("=" * 64)
    for col in MEASURES:
        sub = df.dropna(subset=[col])
        rho, p = spearmanr(sub["integrity_score"], sub[col])
        print(f"   {col:8s}  n={len(sub):>2} ({sub['company_name'].nunique()} firms)   rho = {rho:+.3f}  (p = {p:.3f})")
    print("\nNote: no statistical inference (small n)")

    # scatter: one panel per crash measure, FULL SAMPLE, points shaded by layer count
    # (correlation is full-sample; shading only lets you SEE the layer differences)
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
                        (r["integrity_score"], r[col]), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        rho, p = spearmanr(s["integrity_score"], s[col])
        b = np.polyfit(s["integrity_score"], s[col], 1)
        xs = np.linspace(s["integrity_score"].min(), s["integrity_score"].max(), 50)
        ax.plot(xs, b[0] * xs + b[1], "--", color="#d62728", linewidth=1.2, zorder=2)
        ax.set_xlabel("ESG Integrity Score")
        ax.set_ylabel(f"{col} ")
        ax.set_title(f"{col}   rho = {rho:+.2f}  p = {p:.3f}  n = {len(s)}", fontsize=10)
        ax.grid(True, alpha=0.3, zorder=0)
    axes[0].legend(title="layers (full sample)", fontsize=8)
    fig.suptitle("Forward Crash Risk (t+1)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=150)
    print(f"saved scatter: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()