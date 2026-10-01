"""
Step 1: Download and cache 1 year of daily data for the analysis.

Series used
-----------
^NSEI     Nifty 50 index, INR                (the dependent/Indian equity leg)
GC=F     COMEX gold front-month futures, USD  (gold in USD)
USDINR=X USD/INR spot                          (FX bridge, lets us build gold in INR)

Gold in INR is constructed as GC=F * USDINR. That is the correct way to express a
USD-denominated commodity for an INR investor, and it matches how MCX gold quotes
behave. Yahoo does not carry a clean XAU/INR spot series, so we build it.

Everything is cached to data/ so re-runs are offline-reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

TICKERS = {
    "NIFTY": "^NSEI",
    "GOLD_USD": "GC=F",
    "USDINR": "USDINR=X",
}

PERIOD = "1y"
INTERVAL = "1d"


def fetch(name: str, ticker: str) -> pd.DataFrame:
    cache = DATA / f"{name}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"  {name:9s} loaded from cache: {len(df):4d} rows  "
              f"{df.index[0].date()} -> {df.index[-1].date()}")
        return df

    print(f"  {name:9s} downloading {ticker} ...")
    raw = yf.Ticker(ticker).history(period=PERIOD, interval=INTERVAL, auto_adjust=True)

    if raw.empty:
        sys.exit(f"FATAL: no data returned for {ticker}")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.tz_localize(None).normalize()

    # Yahoo stamps daily bars with the session open time; drop anything that
    # fell on a weekend rather than silently carrying a stale close forward.
    df = df[df.index.dayofweek < 5]

    df.to_csv(cache, index_label="Date")
    print(f"  {name:9s} {len(df):4d} rows  {df.index[0].date()} -> {df.index[-1].date()}  "
          f"last={df['Close'].iloc[-1]:,.2f}")
    return df


def main() -> None:
    print("Fetching 1 year of daily data...\n")
    series = {name: fetch(name, tk) for name, tk in TICKERS.items()}

    # Build gold in INR. FX trades ~24h on a different calendar than NSE, so
    # align on the inner join of actual sessions and record the overlap loss.
    fx = series["USDINR"]["Close"].rename("USDINR")
    gold_usd = series["GOLD_USD"]["Close"].rename("GOLD_USD")

    gold_inr = (gold_usd * fx).rename("GOLD_INR")
    px = pd.concat([series["NIFTY"]["Close"].rename("NIFTY"),
                    gold_usd, fx, gold_inr], axis=1, sort=True).dropna()

    print("\nCalendar overlap")
    print(f"  NIFTY sessions          : {len(series['NIFTY'])}")
    print(f"  GOLD_USD sessions      : {len(series['GOLD_USD'])}")
    print(f"  USDINR sessions        : {len(series['USDINR'])}")
    print(f"  Common (all three)     : {len(px)}")
    print(f"  Days dropped by overlap: {len(series['NIFTY']) - len(px)}")

    px.to_csv(DATA / "prices_aligned.csv", index_label="Date")

    # Prices are what every downstream script consumes.
    summary = pd.DataFrame({
        "first": px.iloc[0],
        "last": px.iloc[-1],
        "min": px.min(),
        "max": px.max(),
        "n_obs": px.notna().sum(),
        "total_ret_%": ((px.iloc[-1] / px.iloc[0] - 1) * 100).round(2),
    })
    print("\nAligned price levels")
    print(summary.round(2).to_string())

    print(f"\nWrote {DATA / 'prices_aligned.csv'}")


if __name__ == "__main__":
    main()
