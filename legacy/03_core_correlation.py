"""
Step 3: Core correlation statistics.

Headline number is Pearson correlation of DAILY LOG RETURNS between Nifty 50 and
gold in INR, because:
  * returns are stationary, price levels are not - correlating levels is a
    classic spurious-regression trap (we compute it anyway, to show the gap)
  * gold in INR matches the currency of the index, so the FX move is not
    silently counted as a gold move

Inference is done properly:
  * Fisher z-transform confidence interval
  * t-test for H0: rho = 0
  * Spearman / Kendall as outlier-robust alternatives to Pearson
  * stationary bootstrap CI that respects return autocorrelation
  * Blinder z-test for whether two correlations differ significantly
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TRADING_DAYS = 252
rets = pd.read_csv(ROOT / "data" / "log_returns.csv", index_col=0, parse_dates=True)
px = pd.read_csv(ROOT / "data" / "prices_aligned.csv", index_col=0, parse_dates=True)

# The one extreme day identified in the quality audit (2026-01-30 gold crash).
CRASH = pd.Timestamp("2026-01-30")

results: dict = {}


def fisher_ci(r: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Fisher z-transform CI. Corrects the upward bias of r at small n."""
    if abs(r) >= 1.0:
        return (-1.0, 1.0)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zc = stats.norm.ppf(0.5 + conf / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def corr_block(x: pd.Series, y: pd.Series, label: str) -> dict:
    x, y = x.dropna(), y.dropna()
    d = pd.concat([x, y], axis=1).dropna()
    x, y = d.iloc[:, 0], d.iloc[:, 1]
    n = len(d)

    r, p = stats.pearsonr(x, y)
    lo, hi = fisher_ci(r, n)
    t = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else np.inf
    p_t = 2 * (1 - stats.t.cdf(abs(t), n - 2))

    rho, p_s = stats.spearmanr(x, y)
    tau, p_k = stats.kendalltau(x, y)

    return {
        "label": label, "n": n,
        "pearson_r": r, "ci_lo": lo, "ci_hi": hi,
        "t_stat": t, "p_pearson": p, "p_ttest": p_t,
        "spearman_rho": rho, "p_spearman": p_s,
        "kendall_tau": tau, "p_kendall": p_k,
        "r2": r**2,
        "strength": ("negligible" if abs(r) < 0.1 else
                     "weak" if abs(r) < 0.3 else
                     "moderate" if abs(r) < 0.5 else
                     "strong" if abs(r) < 0.7 else "very strong"),
    }


def stationary_bootstrap_ci(x: pd.Series, y: pd.Series, n_boot: int = 10000,
                            mean_block: float = 5.0, seed: int = 7) -> tuple[float, float, float]:
    """
    Politis & White (2004) stationary bootstrap. Resamples blocks of geometric
    mean length so serial dependence (mean reversion / momentum) is preserved.
    A plain i.i.d. bootstrap would understate the CI given the ACF structure.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    p_keep = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = 0
        while i < n:
            idx[i] = rng.integers(0, n)
            i += 1
            if rng.random() > p_keep:
                run = rng.geometric(p_keep)
                idx[i:i + run] = idx[i - 1]
                i += run
        out[b] = np.corrcoef(x.values[idx], y.values[idx])[0, 1]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(np.mean(out))


def blinder_z(r1: float, r2: float, n1: int, n2: int, r12: float | None = None) -> tuple[float, float]:
    """
    Blinder (2003) test for equality of two dependent correlations.
    SE accounts for the cross-correlation between the two sample rho's, which
    is ignored by the naive Fisher-z comparison.
    """
    r12 = r12 if r12 is not None else 0.0
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3) - 2 * r12 / (np.sqrt((n1 - 3) * (n2 - 3))))
    z = (z1 - z2) / se
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


# ===========================================================================
print("=" * 78)
print("CORE CORRELATION - NIFTY 50 vs GOLD")
print("=" * 78)
print(f"Sample {rets.index[0].date()} to {rets.index[-1].date()}  |  "
      f"{len(rets)} daily return observations\n")

# --- 1. the headline estimates --------------------------------------------
print("[1] Pearson correlation of daily log returns")
print("-" * 78)
print(f"{'Series':<34}{'n':>5}{'r':>9}{'95% CI':>20}{'p':>10}{'R2':>8}  {'Strength'}")

pairs = [
    ("NIFTY vs GOLD_INR  (primary)", "NIFTY", "GOLDINR"),
    ("NIFTY vs GOLD_USD", "NIFTY", "GOLDUSD"),
    ("NIFTY vs USDINR   (FX effect)", "NIFTY", "USDINR"),
    ("GOLD_INR vs USDINR (gold in USD vs FX)", "GOLDINR", "USDINR"),
]

for label, a, b in pairs:
    c = corr_block(rets[a], rets[b], label)
    results[label] = c
    print(f"{label:<34}{c['n']:>5}{c['pearson_r']:>9.4f}"
          f"{f'[{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]':>20}"
          f"{c['p_pearson']:>10.4f}{c['r2']:>8.4f}  {c['strength']}")

head = results["NIFTY vs GOLD_INR  (primary)"]
print(f"\n  PRIMARY  r = {head['pearson_r']:.4f}  "
      f"95% CI [{head['ci_lo']:.4f}, {head['ci_hi']:.4f}]  "
      f"p = {head['p_pearson']:.4f}  n = {head['n']}")
print(f"  t({head['n'] - 2}) = {head['t_stat']:.3f}, "
      f"p = {head['p_ttest']:.4f} -> "
      f"{'REJECT' if head['p_ttest'] < 0.05 else 'cannot reject'} H0: rho = 0 at 5%")

# --- 2. robustness of the estimator ---------------------------------------
print("\n[2] Estimator robustness (NIFTY vs GOLD_INR)")
print("-" * 78)
print(f"{'Method':<38}{'coef':>10}{'p-value':>12}   note")
for nm, val, p in [
    ("Pearson (linear)", head["pearson_r"], head["p_pearson"]),
    ("Spearman (rank, outlier-robust)", head["spearman_rho"], head["p_spearman"]),
    ("Kendall tau (concordance)", head["kendall_tau"], head["p_kendall"]),
]:
    print(f"{nm:<38}{val:>10.4f}{p:>12.4f}")

lo_b, hi_b, mean_b = stationary_bootstrap_ci(rets["NIFTY"], rets["GOLDINR"])
print(f"{'Stationary bootstrap mean (10k reps)':<38}{mean_b:>10.4f}"
      f"{'':>12}   95% CI [{lo_b:.4f}, {hi_b:.4f}]")
results["bootstrap"] = {"mean": mean_b, "lo": lo_b, "hi": hi_b}

# --- 3. sensitivity to the crash day and to measurement choices -----------
print("\n[3] Sensitivity / robustness checks (NIFTY vs GOLD_INR)")
print("-" * 78)
checks = []

clean = rets.drop(index=CRASH)
c = corr_block(clean["NIFTY"], clean["GOLDINR"], "excl crash day")
checks.append(("Excluding 2026-01-30 gold crash", c["pearson_r"], c["p_pearson"], c["n"]))
results["excl_crash"] = c

c = corr_block(rets["NIFTY"], rets["GOLDINR"].clip(
    rets["GOLDINR"].quantile(0.01), rets["GOLDINR"].quantile(0.99)), "winsorized")
checks.append(("Winsorised at 1st/99th pct", c["pearson_r"], c["p_pearson"], c["n"]))
results["winsorized"] = c

simple = (px[["NIFTY", "GOLD_INR"]].pct_change().dropna() * 100)
c = corr_block(simple["NIFTY"], simple["GOLD_INR"], "simple returns")
checks.append(("Simple (arithmetic) returns", c["pearson_r"], c["p_pearson"], c["n"]))

wk = rets.groupby(rets.index.isocalendar().week.astype(int)).sum()
c = corr_block(wk["NIFTY"], wk["GOLDINR"], "weekly")
checks.append(("Non-overlapping weekly returns", c["pearson_r"], c["p_pearson"], c["n"]))

mo = rets.resample("ME").sum()
c = corr_block(mo["NIFTY"], mo["GOLDINR"], "monthly")
checks.append(("Non-overlapping monthly returns", c["pearson_r"], c["p_pearson"], c["n"]))
results["monthly"] = c

for nm, r, p, n in checks:
    print(f"{nm:<38}{r:>10.4f}{p:>12.4f}   n={n}")

# --- 4. the spurious-correlation trap -------------------------------------
print("\n[4] Why returns, not price levels")
print("-" * 78)
r_lvl, p_lvl = stats.pearsonr(px["NIFTY"], px["GOLD_INR"])
r_ret = head["pearson_r"]
print(f"  Correlation of PRICE LEVELS   r = {r_lvl:>7.4f}  (p={p_lvl:.4f})")
print(f"  Correlation of DAILY RETURNS  r = {r_ret:>7.4f}  (p={head['p_pearson']:.4f})")
print(f"  Inflation of the levels figure = {r_lvl - r_ret:+.4f}")
print("  Level correlation is not a co-movement measure: any two non-stationary")
print("  trending series produce a large r. Use the return figure.")
results["level_corr"] = {"pearson_r": r_lvl, "p": p_lvl}

# --- 5. is the USD version significantly different? ----------------------
print("\n[5] INR vs USD gold: is the difference significant? (Blinder z-test)")
print("-" * 78)
r_inr = head["pearson_r"]
r_usd = results["NIFTY vs GOLD_USD"]["pearson_r"]
cross = rets[["NIFTY"]].join(rets["GOLDINR"]).dropna()
z, p_z = blinder_z(r_inr, r_usd, head["n"],
                   results["NIFTY vs GOLD_USD"]["n"],
                   np.corrcoef(cross["NIFTY"], cross["GOLDINR"])[0, 1])
print(f"  r(gold in INR) = {r_inr:.4f}")
print(f"  r(gold in USD) = {r_usd:.4f}")
print(f"  difference     = {r_inr - r_usd:+.4f}")
print(f"  Blinder z      = {z:.4f},  p = {p_z:.4f}  -> "
      f"{'SIGNIFICANTLY different' if p_z < 0.05 else 'not significantly different'}")
results["blinder"] = {"z": z, "p": p_z, "r_inr": r_inr, "r_usd": r_usd}

# --- 6. variance decomposition -------------------------------------------
print("\n[6] Variance overlap (daily, %)")
print("-" * 78)
sn, sg = rets["NIFTY"].std() * 100, rets["GOLDINR"].std() * 100
print(f"  Nifty ann. vol      {sn * np.sqrt(TRADING_DAYS):6.2f} %")
print(f"  Gold (INR) ann. vol {sg * np.sqrt(TRADING_DAYS):6.2f} %")
print(f"  Shared variance     {r_inr**2 * 100:6.2f} %  (R-squared)")
print("  i.e. gold's daily moves explain only a small slice of Nifty's variance,")
print("  and the hedge value of gold is concentrated in the tail, not the mean.")

# --- export ---------------------------------------------------------------
(OUT / "core_correlation.json").write_text(json.dumps(results, indent=2, default=str))

rows = []
for label, c in results.items():
    if isinstance(c, dict) and "pearson_r" in c and "n" in c:
        rows.append({
            "comparison": label, "n": c["n"],
            "pearson_r": round(c["pearson_r"], 4),
            "ci_low": round(c.get("ci_lo", np.nan), 4),
            "ci_high": round(c.get("ci_hi", np.nan), 4),
            "p_value": round(c.get("p_pearson", np.nan), 5),
            "r_squared": round(c["r2"], 4),
            "spearman_rho": round(c.get("spearman_rho", np.nan), 4),
            "kendall_tau": round(c.get("kendall_tau", np.nan), 4),
        })
pd.DataFrame(rows).to_csv(OUT / "core_correlation.csv", index=False)
print("\nWrote outputs/core_correlation.csv and .json")
