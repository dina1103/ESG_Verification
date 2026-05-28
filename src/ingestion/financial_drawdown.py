import json
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INTEGRITY_JSON = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\integrity_score_company_year.json"
OUTPUT_CSV     = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\drawdown.csv"
OUTPUT_PLOT    = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\drawdown_scatter_diagram.png"

TICKERS = {
    "Bayerische Motoren Werke AG": "BMW.DE",
    "Mercedes-Benz Group AG":      "MBG.DE",
    "Volkswagen AG":               "VOW3.DE",
    "Tesla Inc":                   "TSLA",
    "Toyota Industries Corp":      "6201.T",
}

# fixed 2-year forward window: t+1 to t+2 inclusive (24 months)
WINDOW_YEARS = 2

# need the WHOLE window to be in the past, otherwise that company-year is dropped
TODAY = date.today()

WELL_COVERED_MIN_LAYERS = 3


def max_drawdown(ticker, start_year, end_year):
    # max peak-to-trough drop in daily close over [start_year-01-01, end_year+1-01-01)
    start = f"{start_year}-01-01"
    end = f"{end_year + 1}-01-01"
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"    download failed for {ticker} {start_year}-{end_year}: {e}")
        return None
    if df is None or df.empty or len(df) < 60:
        return None
    # newer yfinance returns MultiIndex columns even for one ticker - flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < 60:
        return None
    # max drawdown: largest peak-to-trough drop as a fraction of the peak
    rolling_max = close.cummax()
    drawdown = (close - rolling_max) / rolling_max  # negative values
    mdd = drawdown.min()
    return float(mdd.item() if hasattr(mdd, "item") else mdd)


def main():
    print("Loading integrity scores...")
    integ = json.load(open(INTEGRITY_JSON, encoding="utf-8"))

    rows = []
    print(f"\nComputing max drawdown over t+1 to t+{WINDOW_YEARS} per company-year...")
    print(f"Today is {TODAY}; company-years whose forward window is not complete are dropped.\n")

    for key, v in sorted(integ.items()):
        company = v.get("company_name")
        year = v.get("year")
        iscore = v.get("integrity_score")
        nlayers = v.get("n_layers_available", 0)
        if iscore is None:
            continue

        ticker = TICKERS.get(company)
        if ticker is None:
            print(f"  no ticker for {company} - skipped")
            continue

        # forward window: year+1 to year+WINDOW_YEARS, inclusive
        window_start = year + 1
        window_end = year + WINDOW_YEARS

        # need the window to be entirely in the past
        if date(window_end, 12, 31) > TODAY:
            print(f"  {company} {year}: forward window {window_start}-{window_end} not yet complete - skipped")
            continue

        mdd = max_drawdown(ticker, window_start, window_end)
        # report drawdown as a positive magnitude for the plot/correlation
        # (e.g. -0.35 -> 0.35 means "35% peak-to-trough drop")
        mdd_magnitude = -mdd if mdd is not None else None
        print(f"  {company} {year}: integrity={iscore} layers={nlayers} "
              f"window={window_start}-{window_end} max_drawdown={mdd_magnitude}")

        rows.append({
            "company_name": company,
            "year": year,
            "integrity_score": iscore,
            "n_layers_available": nlayers,
            "window_start": window_start,
            "window_end": window_end,
            "max_drawdown_magnitude": mdd_magnitude,
        })

    df = pd.DataFrame(rows)
    df = df[df["max_drawdown_magnitude"].notna()].reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nsaved dataset: {OUTPUT_CSV}  ({len(df)} company-years)")

    def describe(subset, label):
        n = len(subset)
        if n < 3:
            print(f"\n[{label}] n={n} - too few points to compute a correlation")
            return
        rho, p = spearmanr(subset["integrity_score"], subset["max_drawdown_magnitude"])
        print(f"\n[{label}] n={n}")
        print(f"  Spearman rho = {rho:.3f}   (p = {p:.3f})")
        print(f"  integrity range:  {subset['integrity_score'].min():.3f} - {subset['integrity_score'].max():.3f}")
        print(f"  drawdown range:   {subset['max_drawdown_magnitude'].min():.3f} - {subset['max_drawdown_magnitude'].max():.3f}")
        print(f"  hypothesis predicts NEGATIVE rho (lower integrity -> bigger drawdown)")
        print(f"  NOTE: descriptive only - n is far too small for statistical inference")

    print("\n" + "=" * 70)
    print("EXPLORATORY ANALYSIS - integrity score vs max drawdown (t+1 to t+2)")
    print("=" * 70)
    describe(df, "ALL company-years")
    well = df[df["n_layers_available"] >= WELL_COVERED_MIN_LAYERS]
    describe(well, f">= {WELL_COVERED_MIN_LAYERS} layers (well-covered)")

    # scatter plot, points shaded by n_layers_available
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = {1: "#d9d9d9", 2: "#9ecae1", 3: "#4292c6", 4: "#08519c"}
    for nl in sorted(df["n_layers_available"].unique()):
        sub = df[df["n_layers_available"] == nl]
        ax.scatter(sub["integrity_score"], sub["max_drawdown_magnitude"],
                   c=cmap.get(nl, "#000000"), s=90, edgecolor="black",
                   linewidth=0.5, label=f"{nl} layer(s)", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(f"{r['company_name'].split()[0]} {r['year']}",
                    (r["integrity_score"], r["max_drawdown_magnitude"]),
                    fontsize=6, alpha=0.7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("ESG Integrity Score")
    ax.set_ylabel(f"Max Drawdown over t+1 to t+{WINDOW_YEARS} (magnitude)")
    ax.set_title("Exploratory: Integrity Score vs Forward Drawdown\n"
                 "(point shade = number of verification layers; small n - descriptive only)",
                 fontsize=10)
    ax.legend(title="Layers available", fontsize=8)
    ax.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nsaved scatter plot: {OUTPUT_PLOT}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Drawdown is a more direct test of the hypothesis than same-year volatility:")
    print(f"it measures the deepest peak-to-trough stock fall in the {WINDOW_YEARS} years")
    print(f"AFTER the disclosure, which is closer to 'markets eventually punished the company'.")
    print(f"Still exploratory, still descriptive, still constrained by sample size.")


if __name__ == "__main__":
    main()