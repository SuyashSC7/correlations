"""
Step 4: Time variation, regimes, and tail dependence.

A single correlation over a whole year is a summary of a relationship that
rarely stays still. This script asks:
  * how much does the rolling correlation move, and is that variation real
    (tested, not eyeballed)?
  * does the relationship change across volatility / market regimes?
  * is the hedge working where it matters - in Nifty drawdowns - or only
    on average? (This is the entire reason investors hold gold.)

Pearson correlation is linear and average-based, so tail behaviour needs a
copula view: lower/upper tail dependence coefficients.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRADING_DAYS = 252

rets = pd.read_csv(ROOT / "data" / "log_returns.csv", index_col=0, parse_dates=True)
px = pd.read_csv(ROOT / "data" / "prices_aligned.csv", index_col=0, parse_dates=True)
x, y = rets["NIFTY"], rets["GOLDINR"]
res: dict = {}

print("=" * 78)
print("ROLLING CORRELATION & REGIME ANALYSIS")
print("=" * 78)

# ---------------------------------------------------------------- rolling --
print("\n[1] Rolling correlation (Pearson, log returns)")
print("-" * 78)
roll = {}
for w in (20, 30, 60, 90, 120):
    rc = x.rolling(w).corr(y)
    roll[w] = rc.dropna()
    v = rc.dropna()
    print(f"  {w:>3}d window: mean={v.mean():+.4f}  sd={v.std():.4f}  "
          f"min={v.min():+.4f} ({v.idxmin().date()})  max={v.max():+.4f} ({v.idxmax().date()})  "
          f"sign flips={int((np.sign(v).diff().fillna(0) != 0).sum())}")

rc60 = roll[60]
roll_df = pd.concat({f"corr_{w}d": s for w, s in roll.items()}, axis=1)
roll_df.to_csv(OUT / "rolling_correlation.csv", index_label="Date")
res["rolling"] = {str(w): {"mean": float(s.mean()), "sd": float(s.std()),
                          "min": float(s.min()), "max": float(s.max())}
                  for w, s in roll.items()}

# Is the variation in the 60d rolling correlation bigger than sampling noise?
# If the rolling r is mostly noise, its sd should be close to the Fisher-z
# sampling spread expected under a constant true rho.
n_w = 60
se = 1.0 / np.sqrt(n_w - 3)
expected_sd = np.tanh(se)
print(f"\n  Fisher-z sampling SE for a {n_w}d window      : {se:.4f}")
print(f"  Expected sd of rolling r if rho were constant : {expected_sd:.4f}")
print(f"  Observed sd of rolling 60d r                  : {rc60.std():.4f}")
print(f"  Observed / expected                           : {rc60.std() / expected_sd:.2f}x")
genuine = rc60.std() > expected_sd * 2.5
print(f"  => rolling correlation is "
      f"{'GENUINELY TIME-VARYING' if genuine else 'mostly SAMPLING NOISE around a constant rho'}")
res["rolling_noise_ratio"] = float(rc60.std() / expected_sd)
res["rolling_time_varying"] = bool(genuine)

# Sign-flip frequency: how persistent is the relationship's sign?
signs = np.sign(rc60)
flips = int((signs.diff().fillna(0) != 0).sum())
print(f"  60d rolling correlation changes sign {flips} time(s) over {len(rc60)} windows")
res["sign_flips_60d"] = flips

# ------------------------------------------------------- vol / stress regime --
print("\n[2] Conditional correlation by volatility regime")
print("-" * 78)
rv = x.rolling(21).std().shift(1)          # trailing vol, lagged to avoid look-ahead
q = rv.dropna().quantile([1 / 3, 2 / 3])
lab = pd.Series("mid", index=rv.index)
lab[rv <= q.iloc[0]] = "low-vol"
lab[rv >= q.iloc[1]] = "high-vol"
regime = pd.DataFrame({"NIFTY": x, "GOLDINR": y, "regime": lab}).dropna()
print(f"{'Regime':<12}{'n':>5}{'r':>10}{'95% CI':>20}{'p':>10}")
for rg in ["low-vol", "mid", "high-vol"]:
    s = regime[regime["regime"] == rg]
    r, p = stats.pearsonr(s["NIFTY"], s["GOLDINR"])
    z = np.arctanh(r)
    se_ = 1 / np.sqrt(len(s) - 3)
    ci = (np.tanh(z - 1.96 * se_), np.tanh(z + 1.96 * se_))
    print(f"{rg:<12}{len(s):>5}{r:>10.4f}{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>20}{p:>10.4f}")
    res.setdefault("regime", {})[rg] = {"n": len(s), "r": r, "p": p,
                                        "ci": [float(ci[0]), float(ci[1])]}

# ------------------------------------------------- drawdown / tail behaviour --
print("\n[3] Conditional correlation by Nifty drawdown day")
print("-" * 78)
nq = x.quantile([0.05, 0.95])
buckets = {
    "worst 5% (crash days)": y[x <= nq.iloc[0]],
    "5-25% (down days)":     y[(x > nq.iloc[0]) & (x < x.quantile(0.25))],
    "25-75% (normal)":       y[(x >= x.quantile(0.25)) & (x <= x.quantile(0.75))],
    "75-95% (up days)":      y[(x > x.quantile(0.75)) & (x < nq.iloc[1])],
    "best 5% (rally days)":  y[x >= nq.iloc[1]],
}
print(f"{'Nifty return bucket':<26}{'n':>5}{'mean gold %':>13}{'median':>10}{'gold +ve %':>12}")
for nm, gy in buckets.items():
    print(f"{nm:<26}{len(gy):>5}{gy.mean() * 100:>13.3f}"
          f"{gy.median() * 100:>10.3f}{(gy > 0).mean() * 100:>11.1f}%")
    res.setdefault("drawdown", {})[nm] = {
        "n": len(gy), "mean_gold_pct": float(gy.mean() * 100),
        "pct_gold_up": float((gy > 0).mean() * 100)}

# Up/down capture: does gold actually pay when Nifty falls?
up = y[x > 0]
dn = y[x < 0]
print(f"\n  Gold's average move when Nifty RISES : {up.mean() * 100:+.3f}%  (n={len(up)})")
print(f"  Gold's average move when Nifty FALLS : {dn.mean() * 100:+.3f}%  (n={len(dn)})")
print(f"  Gold hit rate on Nifty down days    : {(dn > 0).mean() * 100:.1f}%")
print(f"  Gold hit rate on Nifty up days      : {(up > 0).mean() * 100:.1f}%")
res["capture"] = {"mean_up": float(up.mean() * 100), "mean_down": float(dn.mean() * 100),
                  "hit_down": float((dn > 0).mean() * 100), "hit_up": float((up > 0).mean() * 100)}

# The single best stress test: Nifty's worst day. What did gold do?
worst5 = x.nsmallest(5)
print("\n  Nifty's 5 worst days and gold's response on the same day:")
print(f"  {'date':<12}{'Nifty %':>10}{'Gold(INR) %':>13}{'Gold(USD) %':>13}")
g_usd = rets["GOLDUSD"]
for d, v in worst5.items():
    print(f"  {d.date()!s:<12}{v * 100:>10.2f}{y.loc[d] * 100:>13.2f}{g_usd.loc[d] * 100:>13.2f}")
res["worst_days"] = [{"date": str(d.date()), "nifty": float(v * 100),
                      "gold_inr": float(y.loc[d] * 100)} for d, v in worst5.items()]

# ---------------------------------------------------------- tail dependence --
z1 = (x - x.mean()) / x.std()
z2 = (y - y.mean()) / y.std()
r = np.corrcoef(z1, z2)[0, 1]


q = 1 - 0.05
zq = stats.norm.ppf(1 - q)          # standardised 5% tail cutoff


def cond_tail_dep(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """
    Conditional tail co-exceedance: of the observations where `a` is in its
    lower k% tail, what fraction also have `b` in its lower k% tail?
    Returns (lower, upper, n_in_a_tail) using the SAME statistic for both tails.
    """
    lo_a, lo_b = a <= zq, b <= zq
    hi_a, hi_b = a >= -zq, b >= -zq
    lower = float((lo_a & lo_b).sum() / max(lo_a.sum(), 1))
    upper = float((hi_a & hi_b).sum() / max(hi_a.sum(), 1))
    return lower, upper, float(lo_a.sum())


obs_lo, obs_hi, n_lo = cond_tail_dep(z1, z2)

# Null: full replicate samples of the same length drawn from a bivariate normal
# carrying the SAME correlation, so the null already embeds the dependence that
# rho implies. What it does NOT allow is extra clustering in the tails.
# Under independence this conditional probability would be 0.05.
rng = np.random.default_rng(11)
B, n = 5000, len(x)
zc = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], size=(B, n))
z1c, z2c = zc[:, :, 0], zc[:, :, 1]

lo_a, lo_b = z1c <= zq, z2c <= zq
hi_a, hi_b = z1c >= -zq, z2c >= -zq
null_lo_c = (lo_a & lo_b).sum(axis=1) / np.maximum(lo_a.sum(axis=1), 1)
null_hi_c = (hi_a & hi_b).sum(axis=1) / np.maximum(hi_a.sum(axis=1), 1)

null_lo_mean, null_hi_mean = float(null_lo_c.mean()), float(null_hi_c.mean())
p_lo = float((np.sum(null_lo_c >= obs_lo) + 1) / (B + 1))
p_hi = float((np.sum(null_hi_c >= obs_hi) + 1) / (B + 1))

print("\n[4] Conditional tail dependence, 5% tails (Nifty vs gold in INR)")
print("-" * 78)
print("  Question: when Nifty is in its worst 5% of days, is gold also there?")
print(f"  Nifty lower-tail observations: {int(n_lo)}   (independent baseline = 0.05)\n")
print(f"  {'':<34}{'observed':>10}{'null E[]':>11}{'p-value':>10}")
print(f"  {'P(gold down | Nifty in LL tail)':<34}{obs_lo:>10.4f}{null_lo_mean:>11.4f}{p_lo:>10.4f}")
print(f"  {'P(gold up   | Nifty in HH tail)':<34}{obs_hi:>10.4f}{null_hi_mean:>11.4f}{p_hi:>10.4f}")
print()
if p_lo < 0.05 and obs_lo > null_lo_mean:
    print("  => SIGNIFICANT POSITIVE lower-tail dependence: gold tends to fall when")
    print("     Nifty collapses. Over this year gold was a POOR same-day crash hedge.")
else:
    print("  => No significant extra lower-tail clustering; gold does not reliably")
    print("     fall with Nifty in a crash - it behaves as a genuine hedge.")
print(f"  CAVEAT: only {int(n_lo)} Nifty lower-tail observations, so this test is")
print("  LOW POWER. A non-significant p here is not proof of no tail dependence.")
res["tail_dep"] = {"lower": obs_lo, "upper": obs_hi, "n_nifty_lower_tail": int(n_lo),
                   "null_lower": null_lo_mean, "null_upper": null_hi_mean,
                   "p_lower": p_lo, "p_upper": p_hi}

# --------------------------------------------------------------- stability --
print("\n[5] Sub-period stability (is one number enough?)")
print("-" * 78)
third = len(rets) // 3
periods = {
    "Q1 (first third)":  slice(0, third),
    "Q2 (middle third)": slice(third, 2 * third),
    "Q3 (last third)":   slice(2 * third, len(rets)),
}
sub = rets.iloc[:]
for nm, sl in periods.items():
    s = sub.iloc[sl]
    r_, p_ = stats.pearsonr(s["NIFTY"], s["GOLDINR"])
    d0, d1 = s.index[0].date(), s.index[-1].date()
    print(f"  {nm:<20}{d0!s}..{d1!s}  n={len(s):>3}  r={r_:+.4f}  p={p_:.4f}")
    res.setdefault("subperiod", {})[nm] = {"r": r_, "p": p_, "n": len(s),
                                           "start": str(d0), "end": str(d1)}

# Fisher-z test: do the first and last third differ?
r1 = res["subperiod"]["Q1 (first third)"]["r"]
r3 = res["subperiod"]["Q3 (last third)"]["r"]
n1 = res["subperiod"]["Q1 (first third)"]["n"]
n3 = res["subperiod"]["Q3 (last third)"]["n"]
z = (np.arctanh(r3) - np.arctanh(r1)) / np.sqrt(1 / (n3 - 3) + 1 / (n1 - 3))
p_diff = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"\n  Fisher-z test, Q3 vs Q1: z={z:+.4f}, p={p_diff:.4f} -> "
      f"{'relationship CHANGED' if p_diff < 0.05 else 'no significant change'}")
res["subperiod_ztest"] = {"z": float(z), "p": float(p_diff)}

(OUT / "dynamics.json").write_text(json.dumps(res, indent=2, default=str))
print("\nWrote outputs/rolling_correlation.csv and outputs/dynamics.json")
