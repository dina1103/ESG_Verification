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
OUTPUT_CSV_Y   = r"data\processed\annual_return_y.csv"
OUTPUT_PLOT_Y  = r"data\processed\annual_return_y_scatter.png"
OUTPUT_CSV_Y1  = r"data\processed\annual_return_y+1.csv"
OUTPUT_PLOT_Y1 = r"data\processed\annual_return_y+1_scatter.png"

# annual stock return. hypothesis: higher integrity -> higher return -> POSITIVE rho.
# same-year (Y) = contemporaneous association; forward (Y+1) = predictive test.
WINDOWS = {
    0: {"tag": "year Y (same-year)",
        "csv": OUTPUT_CSV_Y,
        "plot": OUTPUT_PLOT_Y,
        "title": "Annual Stock Return (year Y)"},
    1: {"tag": "year Y+1 (forward-year)",
        "csv": OUTPUT_CSV_Y1,
        "plot": OUTPUT_PLOT_Y1,
        "title": "Annual Stock Return (year Y+1)"},
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

MIN_DAYS = 60
TODAY = date.today()


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


def annual_return(close):
    if close is None or len(close) < MIN_DAYS:
        return None
    return round(float(close.iloc[-1] / close.iloc[0] - 1), 4)


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
        ar = annual_return(cl)
        if ar is None:
            continue
        rows.append({"company_name": c, "year": y, "integrity_score": sc,
                     "n_layers_available": v.get("n_layers_available", 0),
                     "window_year": wy, "annual_return": ar})
    return pd.DataFrame(rows)


def make_scatter(df, plot_path, title):
    cmap = {1: "#d9d9d9", 2: "#9ecae1", 3: "#4292c6", 4: "#08519c"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for nl in sorted(df["n_layers_available"].unique()):
        g = df[df["n_layers_available"] == nl]
        ax.scatter(g["integrity_score"], g["annual_return"], s=80, c=cmap.get(nl, "#000"),
                   edgecolor="black", linewidth=0.5, label=f"{nl} layer(s)", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(f"{r['company_name'].split()[0]} {str(r['year'])[2:]}",
                    (r["integrity_score"], r["annual_return"]), fontsize=8, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")
    rho, p = spearmanr(df["integrity_score"], df["annual_return"])
    b = np.polyfit(df["integrity_score"], df["annual_return"], 1)
    xs = np.linspace(df["integrity_score"].min(), df["integrity_score"].max(), 50)
    ax.plot(xs, b[0] * xs + b[1], "--", color="#d62728", linewidth=1.2, zorder=2)
    ax.axhline(0, color="grey", linewidth=0.6, zorder=1)
    ax.set_xlabel("ESG Integrity Score", fontsize=14)
    ax.set_ylabel("Annual Stock Return", fontsize=14)
    ax.set_title(f"{title}\nrho = {rho:+.2f}  p = {p:.3f}  n = {len(df)}  "
                 , fontsize=16)
    ax.legend(title="layers (full sample)", fontsize=10)
    ax.grid(True, alpha=0.3, zorder=0)
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
        print(f"ANNUAL RETURN  ({cfg['tag']}, full sample)  hypothesis: rho > 0")
        print("=" * 64)
        print(f"saved: {cfg['csv']}  ({len(df)} company-years)")
        rho, p = spearmanr(df["integrity_score"], df["annual_return"])
        print(f"   n={len(df):>2} ({df['company_name'].nunique()} firms)   rho = {rho:+.3f}  (p = {p:.3f})")
        make_scatter(df, cfg["plot"], cfg["title"])
        print(f"saved scatter: {cfg['plot']}")
    print("\nNote: same-year = association; forward = predictive")
    print("no statistical inference (small n)")


if __name__ == "__main__":
    main()