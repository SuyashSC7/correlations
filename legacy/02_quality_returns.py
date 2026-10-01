"""
Step 2: Data quality audit + return construction.

Nothing downstream is trustworthy unless this passes, so we check for the usual
data sins before computing any statistic:
  * calendar gaps / missing sessions
  * stale or zero returns (non-trading days leaking through)
  * implausible outliers
  * internal consistency of the synthetic gold-in-INR series
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

px = pd.read_csv(DATA / "prices_aligned.csv", index_col=0, parse_dates=True)

print("=" * 72)
print("DATA QUALITY AUDIT")
print("=" * 72)
print(f"Sample       : {px.index[0].date()} -> {px.index[-1].date()}")
print(f"Sessions     : {len(px)}")
print(f"Span (days)  : {(px.index[-1] - px.index[0]).days}")
print(f"Observations : {len(px) - 1} return periods")

# --- calendar completeness -------------------------------------------------
print("\n[1] Calendar")
gaps = px.index.to_series().diff().dt.days
big_gaps = gaps[gaps > 4]
print(f"    gaps > 4 calendar days: {len(big_gaps)}")
if len(big_gaps):
    for d, g in big_gaps.items():
        print(f"      {d.date()}  (+{int(g)}d)  <- likely NSE holiday / data gap")
non_weekend = [d for d in px.index if d.dayofweek >= 5]
print(f"    weekend bars present   : {len(non_weekend)}")
print(f"    monotonic increasing    : {px.index.is_monotonic_increasing}")

# --- consistency of synthetic series --------------------------------------
print("\n[2] Synthetic gold-in-INR consistency")
rebuilt = px["GOLD_USD"] * px["USDINR"]
resid = (rebuilt - px["GOLD_INR"]).abs().max()
print(f"    max |GOLD_USD*USDINR - GOLD_INR| = {resid:.6f}  (should be ~0)")

# --- returns and staleness -------------------------------------------------
rets = np.log(px / px.shift(1)).dropna()
rets.columns = [c.replace("_", "") for c in rets.columns]
rets.index.name = "Date"

print("\n[3] Return diagnostics (log returns, %)")
diag = pd.DataFrame({
    "mean": rets.mean() * 100,
    "median": rets.median() * 100,
    "std": rets.std() * 100,
    "ann_vol_%": rets.std() * np.sqrt(252) * 100,
    "min": rets.min() * 100,
    "max": rets.max() * 100,
    "skew": rets.skew(),
    "kurtosis": rets.kurtosis(),
    "zeros": (rets.abs() < 1e-12).sum(),
}).T
print(diag.round(3).to_string())

# Zero-return days mean a non-trading session was treated as a session.
zero_days = (rets.abs() < 1e-12).any(axis=1)
if zero_days.any():
    print(f"\n    WARNING: {zero_days.sum()} sessions with a zero return")
    print(rets[zero_days].round(6).to_string())

# --- outliers --------------------------------------------------------------
print("\n[4] Outlier screen (|return| > 5 sigma)")
flagged = {}
for c in rets.columns:
    z = (rets[c] - rets[c].mean()) / rets[c].std()
    out = rets.loc[z.abs() > 5, c]
    if len(out):
        flagged[c] = out
        print(f"    {c}: {len(out)} day(s)")
        for d, v in out.items():
            print(f"      {d.date()}  {v * 100:+.2f}%")
if not flagged:
    print("    none - no return exceeds 5 sigma")

# --- autocorrelation (matters for HAC/DCC assumptions) --------------------
print("\n[5] Return autocorrelation (ACF lag 1-5)")
for c in rets.columns:
    acf = [rets[c].autocorr(l) for l in range(1, 6)]
    print(f"    {c:9s} " + "  ".join(f"L{l}={a:+.3f}" for l, a in zip(range(1, 6), acf)))

rets.to_csv(DATA / "log_returns.csv", index_label="Date")
print(f"\nWrote {DATA / 'log_returns.csv'}  shape={rets.shape}")

# Persist the audit so the report can cite numbers rather than guesses.
lines = [
    f"sessions={len(px)}",
    f"span={px.index[0].date()}..{px.index[-1].date()}",
    f"n_returns={len(rets)}",
    f"zero_return_sessions={int(zero_days.sum())}",
    f"gaps_gt_4d={len(big_gaps)}",
    f"synthetic_max_resid={resid:.6f}",
]
for c in rets.columns:
    lines.append(
        f"{c}: ann_vol={rets[c].std() * np.sqrt(252) * 100:.3f}% "
        f"mean_daily={rets[c].mean() * 100:.4f}% "
        f"min={rets[c].min() * 100:.3f}% max={rets[c].max() * 100:.3f}% "
        f"acf1={rets[c].autocorr(1):.4f}"
    )
(ROOT / "outputs" / "quality_report.txt").write_text("\n".join(lines) + "\n")
print("Wrote outputs/quality_report.txt")
