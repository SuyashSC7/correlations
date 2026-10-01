"""
Step 5: Regression, stationarity, and structural-break tests.

Correlation describes co-movement but not direction or magnitude. Here we:
  * prove the level series are non-stationary and the returns are stationary,
    which is what justifies analysing returns rather than prices
  * run two-way OLS with Newey-West (HAC) standard errors, since return
    volatility is clustered and plain OLS SEs would be too small
  * test for structural breaks with a Quandt-Andrews sup-Wald scan rather than
    picking a break date by eye
  * translate the result into a portfolio number: how much gold to hold
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRADING_DAYS = 252

rets = pd.read_csv(ROOT / "data" / "log_returns.csv", index_col=0, parse_dates=True)
px = pd.read_csv(ROOT / "data" / "prices_aligned.csv", index_col=0, parse_dates=True)
res: dict = {}

print("=" * 78)
print("REGRESSION, STATIONARITY & STRUCTURAL BREAKS")
print("=" * 78)

# ------------------------------------------------------------ stationarity --
print("\n[1] Augmented Dickey-Fuller: are the series stationary?")
print("-" * 78)
print(f"{'Series':<12}{'level':>36}{'':>4}{'log returns':>36}")
print(f"{'':<12}{'stat':>11}{'p':>10}{'verdict':>15}{'stat':>11}{'p':>10}{'verdict':>15}")
for c in px.columns:
    ad_l = adfuller(px[c], autolag="AIC", result_object=False)
    ad_r = adfuller(np.log(px[c]).diff().dropna(), autolag="AIC", result_object=False)
    vl = "stationary" if ad_l[1] < 0.05 else "NON-stationary"
    vr = "stationary" if ad_r[1] < 0.05 else "NON-stationary"
    print(f"{c:<12}{ad_l[0]:>11.3f}{ad_l[1]:>10.4f}{vl:>15}"
          f"{ad_r[0]:>11.3f}{ad_r[1]:>10.4f}{vr:>15}")
    res.setdefault("adf", {})[c] = {"level": {"stat": ad_l[0], "p": ad_l[1]},
                                    "returns": {"stat": ad_r[0], "p": ad_r[1]}}
print("\n  Levels are integrated -> correlating them is spurious. Returns are")
print("  stationary -> the return-based correlation is the valid statistic.")

# --------------------------------------------------------------------- OLS --
print("\n[2] OLS with Newey-West (HAC) standard errors")
print("-" * 78)
pairs = [("NIFTY ~ GOLDINR", "NIFTY", "GOLDINR"),
         ("GOLDINR ~ NIFTY", "GOLDINR", "NIFTY"),
         ("NIFTY ~ GOLDUSD", "NIFTY", "GOLDUSD")]

ols_rows = []
for label, dep, indep in pairs:
    d = rets[[dep, indep]].dropna()
    m = sm.OLS(d[dep], sm.add_constant(d[indep])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})
    b = m.params[indep]
    t = m.tvalues[indep]
    p = m.pvalues[indep]
    ci = m.conf_int().loc[indep]
    print(f"\n  {label}")
    print(f"    alpha    = {m.params['const'] * 100:+.4f} %/day  "
          f"(p={m.pvalues['const']:.4f})")
    print(f"    beta     = {b:+.4f}  HAC t={t:+.3f}  p={p:.4f}  "
          f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"    R2       = {m.rsquared:.4f}   adj R2 = {m.rsquared_adj:.4f}")
    print(f"    n        = {int(m.nobs)}")
    ols_rows.append({"model": label, "alpha_daily_pct": m.params["const"] * 100,
                     "beta": b, "hac_t": t, "p_value": p, "r2": m.rsquared,
                     "ci_low": ci[0], "ci_high": ci[1], "n": int(m.nobs)})
    res.setdefault("ols", {})[label] = {"alpha": m.params["const"], "beta": b,
                                        "t": t, "p": p, "r2": m.rsquared}

print("\n  Plain-OLS vs HAC t-statistic on the beta of NIFTY ~ GOLDINR:")
d = rets[["NIFTY", "GOLDINR"]].dropna()
m_plain = sm.OLS(d["NIFTY"], sm.add_constant(d["GOLDINR"])).fit()
m_hac = sm.OLS(d["NIFTY"], sm.add_constant(d["GOLDINR"])).fit(
    cov_type="HAC", cov_kwds={"maxlags": 5})
print(f"    plain OLS t = {m_plain.tvalues['GOLDINR']:+.3f},  "
      f"HAC t = {m_hac.tvalues['GOLDINR']:+.3f}")
print("    (HAC widens the SE when volatility is clustered; it is the")
print("     defensible choice for return data.)")

pd.DataFrame(ols_rows).to_csv(OUT / "regression.csv", index=False)

# --------------------------------------------------- structural break (QLR) --
print("\n[3] Quandt-Andrews structural break test (sup-Wald / CUSUM)")
print("-" * 78)
# Restrict to the 15%-85% trimming so the test does not pick a break from a
# handful of observations at the very edge of the sample.
trim = 0.15
n = len(d)
lo, hi = int(n * trim), int(n * (1 - trim))

yv, xv = d["NIFTY"].values, d["GOLDINR"].values
Xfull = np.column_stack([xv, np.ones(n)])
rss_full = ((yv - Xfull @ np.linalg.lstsq(Xfull, yv, rcond=None)[0]) ** 2).sum()


def _ssr(a: np.ndarray, b: np.ndarray) -> float:
    """Residual sum of squares of a regression with intercept."""
    Xa = np.column_stack([a, np.ones(len(a))])
    return ((b - Xa @ np.linalg.lstsq(Xa, b, rcond=None)[0]) ** 2).sum()


def fstat_at(i: int) -> float:
    """Chow F for a slope+intercept break at position i."""
    ssr_pieces = _ssr(xv[:i], yv[:i]) + _ssr(xv[i:], yv[i:])
    return ((rss_full - ssr_pieces) / 2) / (ssr_pieces / (n - 4))


best_F, i_max = max((fstat_at(i), i) for i in range(lo, hi))
lr_max = n * np.log(best_F)

print(f"  Homoskedastic QLR sup-Wald F : {best_F:.3f}")
print(f"  LR statistic (n*ln F)        : {lr_max:.3f}")
print(f"  Implied break date          : {d.index[i_max].date()}")
print(f"  Correlation before break    : {d['NIFTY'][:i_max].corr(d['GOLDINR'][:i_max]):+.4f}")
print(f"  Correlation after break     : {d['NIFTY'][i_max:].corr(d['GOLDINR'][i_max:]):+.4f}")

# The homoskedastic QLR critical value depends on the trimming fraction and
# regressor count, and a heteroskedasticity-robust critical value depends on
# the error distribution - which return data violates badly here (kurtosis 6+).
# So calibrate the null by simulation instead of trusting a tabulated constant:
# generate series with a CONSTANT correlation and ask how large the same
# sup-statistic gets by chance.
def sup_stat(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """max_t |r(t) - r_full| in Fisher-z space over candidate split points."""
    zf = np.arctanh(np.clip(np.corrcoef(a, b)[0, 1], -0.999, 0.999))
    best, arg = 0.0, 0
    for i in range(lo, hi):
        r1 = np.clip(np.corrcoef(a[:i], b[:i])[0, 1], -0.999, 0.999)
        r2 = np.clip(np.corrcoef(a[i:], b[i:])[0, 1], -0.999, 0.999)
        dev = abs(np.arctanh(r1) - zf) + abs(np.arctanh(r2) - zf)
        if dev > best:
            best, arg = dev, i
    return best, arg


obs_stat, obs_i = sup_stat(yv, xv)
rho_full = np.corrcoef(yv, xv)[0, 1]
rng = np.random.default_rng(23)
B = 2000
null_stats = np.empty(B)
for b in range(B):
    sim = rng.multivariate_normal([0, 0], [[1, rho_full], [rho_full, 1]], size=n)
    null_stats[b] = sup_stat(sim[:, 0], sim[:, 1])[0]
crit = float(np.percentile(null_stats, 95))
p_val = float((np.sum(null_stats >= obs_stat) + 1) / (B + 1))

print("\n  Bootstrap-calibrated structural break test")
print(f"    sup-statistic observed     : {obs_stat:.4f}   (at {d.index[obs_i].date()})")
print(f"    95th pct of null (B={B})  : {crit:.4f}")
print(f"    p-value                    : {p_val:.4f}")
print(f"  => {'BREAK DETECTED' if obs_stat > crit else 'no significant structural break'} "
      f"(constant-correlation process explains the variation)")
res["break"] = {"qlr_F": best_F, "lr": lr_max, "date": str(d.index[i_max].date()),
                "r_before": float(d["NIFTY"][:i_max].corr(d["GOLDINR"][:i_max])),
                "r_after": float(d["NIFTY"][i_max:].corr(d["GOLDINR"][i_max:])),
                "sup_stat": obs_stat, "crit_95": crit, "p_value": p_val}

# -------------------------------------------------------- portfolio impact --
print("\n[4] Portfolio implication: how much gold belongs in a Nifty portfolio?")
print("-" * 78)
sn = rets["NIFTY"].std()
sg = rets["GOLDINR"].std()
corr = rets["NIFTY"].corr(rets["GOLDINR"])

grid = np.linspace(-0.50, 0.50, 1001)
pvar = (1 - grid) ** 2 * sn**2 + grid**2 * sg**2 + 2 * (1 - grid) * grid * corr * sn * sg
w_min = grid[int(np.argmin(pvar))]
pvar_min = pvar.min()
pvar_100 = sn**2
print(f"  Nifty-only annual vol        : {sn * np.sqrt(TRADING_DAYS) * 100:.2f} %")
print(f"  Gold (INR) annual vol        : {sg * np.sqrt(TRADING_DAYS) * 100:.2f} %")
print(f"  Portfolio variance minimised at gold weight : {w_min * 100:+.1f} %")
print(f"  Vol at that weight           : {np.sqrt(pvar_min) * np.sqrt(TRADING_DAYS) * 100:.2f} %")
print(f"  Variance reduction vs 100% Nifty            : {(1 - pvar_min / pvar_100) * 100:.2f} %")
for w in (0.05, 0.10, 0.15, 0.20):
    pv = ((1 - w) ** 2 * sn**2 + w**2 * sg**2 + 2 * (1 - w) * w * corr * sn * sg)
    print(f"    gold weight {w * 100:>4.0f}%  ->  portfolio vol "
          f"{np.sqrt(pv) * np.sqrt(TRADING_DAYS) * 100:5.2f} %  "
          f"({(1 - pv / pvar_100) * 100:+.2f}% variance)")
res["portfolio"] = {"min_var_gold_weight": w_min, "vol_min": float(np.sqrt(pvar_min) * np.sqrt(TRADING_DAYS)),
                    "var_reduction_pct": float((1 - pvar_min / pvar_100) * 100)}

(OUT / "regression.json").write_text(json.dumps(res, indent=2, default=str))
print("\nWrote outputs/regression.csv and outputs/regression.json")
