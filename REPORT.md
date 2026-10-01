# Correlation: ^NSEI vs GC=F

> **Correlation of ^NSEI and GC=F (2025-09-30 to 2026-09-30, 238 log returns): r = +0.2046**

## What was compared

Input `^NSEI` resolved to `^NSEI` and `GC=F` to `GC=F`.

| | A | B |
|---|---|---|
| Input | `^NSEI` | `GC=F` |
| Resolved | `^NSEI` | `GC=F` |
| Matched | ^NSEI - NSI - INR | GC=F - CMX - USD |
| Quoted in | INR | USD |
| Resolution | direct-symbol | direct-symbol |

Sample: **2025-09-30 to 2026-09-30** — 239 aligned sessions, 238 log returns, 1y window at 1d frequency.

Both legs restated in **INR** (A: via USDINR=X; B: identity (already USD)).


## The number

| Statistic | Value |
|---|---|
| Pearson r | **+0.2046** |
| 95% CI (Fisher z) | [+0.0795, +0.3233] |
| t-statistic | +3.210 |
| p-value | 0.0015 (yes at 95%) |
| R² (variance shared) | 0.0418 |
| Spearman ρ | +0.1232 (p=0.0578) |
| Kendall τ | +0.0824 (p=0.0583) |
| Block-bootstrap 95% CI | [+0.0484, +0.3331] (block=5) |
| Annualised vol A / B | 13.62% / 30.01% |

**weak positive correlation (r=+0.2046) - treat as indicative, not established**

Reasons for caution:

- Spearman rho=+0.123 does not reach significance (p=0.058), so the linear coefficient may be picking up a few extreme pairs rather than a monotone relationship
- Kendall tau=+0.082 is also insignificant (p=0.058)

### Why not price levels?

Correlating the price levels instead of returns gives r = -0.3446 (p = 0.0000) versus +0.2046 for returns — **an opposite sign**.

Levels: 0/2 reject a unit root. Returns: 2/2 reject. Since levels are non-stationary, correlation between price levels is not interpretable as co-movement - only returns are used as the headline.

### Currency effect

| Measure | Value |
|---|---|
| Unconverted correlation | +0.2333 |
| After normalising to INR | +0.2046 |
| Difference (unconverted - normalised) | +0.0288 |
| A vs FX rate | -0.2423 |
| B vs FX rate | -0.0844 |
| FX-implied component (product) | +0.0204 |
| Difference 95% CI (paired bootstrap) | [-0.021, +0.079] |
| Is the difference significant? | p = 0.2470 (includes zero) |

The exchange rate carries a material loading on at least one leg (r=-0.24 / -0.08), so cross-currency correlations are partly a currency story. The normalised figure is the defensible one.

### Robustness

| Check | n | r | p |
|---|---|---|---|
| Baseline | 238 | +0.2046 | 0.0015 |
| drop largest moves | 235 | +0.1437 | 0.0277 |
| winsorised | 238 | +0.1851 | 0.0042 |

Dependence on extreme days: **low**.

| Sub-period | Window | n | r | 95% CI |
|---|---|---|---|---|
| 1 | 2025-10-01 to 2026-02-02 | 80 | +0.0727 | [-0.149, +0.288] |
| 2 | 2026-02-03 to 2026-06-04 | 79 | +0.3117 | [+0.097, +0.498] |
| 3 | 2026-06-05 to 2026-09-30 | 79 | +0.1915 | [-0.031, +0.396] |

### Does it change over time?

| Window | Mean | Std | Min | Max | Latest | Sign changes |
|---|---|---|---|---|---|---|
| 20d | +0.126 | +0.219 | -0.302 | +0.571 | +0.366 | 13 |
| 60d | +0.184 | +0.112 | -0.099 | +0.379 | +0.120 | 4 |
| 90d | +0.211 | +0.091 | +0.008 | +0.324 | +0.139 | 0 |
| 120d | +0.238 | +0.032 | +0.107 | +0.284 | +0.107 | 0 |

Correlation by volatility state:

| Regime | n | r | 95% CI | Significant |
|---|---|---|---|---|
| low vol | 80 | -0.0835 | [-0.298, +0.139] | no |
| mid vol | 79 | -0.0071 | [-0.228, +0.214] | no |
| high vol | 79 | +0.2844 | [+0.068, +0.476] | yes |

Correlation ranges -0.084 to +0.284 across volatility states (spread 0.368). Co-movement is regime-dependent, so a single annual number hides the conditions that matter.

**Structural break test.** largest split difference 0.158 at 2026-04-06 (+0.272 -> +0.113); block-bootstrap p=0.516, time-shift p=0.345. No evidence of a genuine regime change - the variation looks like noise.

Plain rolling windows (the practical dynamic estimate):

| Window | Latest | Mean | Std | Min | Max |
|---|---|---|---|---|---|
| 60d | +0.120 | +0.184 | +0.112 | -0.099 | +0.379 |
| 90d | +0.139 | +0.211 | +0.091 | +0.008 | +0.324 |

### Behaviour in the tails

| Tail | Joint rate | Expected | Ratio | n in tail | r within tail |
|---|---|---|---|---|---|
| lower | 0.8% | 5% | x0.167 | 12 | 0.5143 |
| upper | 1.7% | 5% | x0.333 | 12 | 0.3167 |

- **lower**: joint tail rate is 0.8% vs 5.0% expected (x0.17): the lower tail shows mild avoidance.
- **upper**: joint tail rate is 1.7% vs 5.0% expected (x0.33): the upper tail shows mild avoidance.

> At q=5% of 238 sessions each tail holds roughly 11 observations. Non-significant tail results here reflect low power, not evidence of independence.

**On ^NSEI's worst 5% of days** (12 sessions, mean -2.06% for A):

| Measure | Value |
|---|---|
| GC=F mean return | -1.01% |
| GC=F median return | -1.00% |
| GC=F worst | -5.20% |
| GC=F positive rate | +25.0% |
| GC=F mean, normal days | +0.12% |
| Difference vs normal days | p = 0.1385 |

On ^NSEI's worst 5% of days (12 sessions), GC=F averaged -1.01% and rose 25% of the time. GC=F fell alongside ^NSEI, so it did not offset the drawdown in this sample.

### Regression

| Model | alpha (%/day) | beta | HAC se | p | R² |
|---|---|---|---|---|---|
| ^NSEI ~ GC=F | -0.0415 | +0.0928 | 0.0354 | 0.0094 | 0.0418 |
| GC=F ~ ^NSEI | +0.0817 | +0.4509 | 0.1558 | 0.0042 | 0.0418 |

^NSEI beta on GC=F = +0.0928, but GC=F beta on ^NSEI = +0.4509. Correlation is symmetric by construction; regression betas are not, so the choice of direction reflects volatility, not causation.

### Portfolio implication

Minimum-variance weights: **88.9% ^NSEI / 11.1% GC=F**, giving 13.19% annualised volatility against 13.62% for ^NSEI alone — a 6.1% reduction in variance.

- At r=+0.20 the diversification benefit is limited: the minimum-variance mix cuts ^NSEI's variance by only 6.1%.
- Weights assume the sample covariance persists. Minimum-variance estimates are notoriously unstable, so this is a sample-specific number, not a recommended allocation.

### Conditional correlation (DCC / EWMA)

EWMA (λ=0.94): latest +0.3713, mean +0.1561, sd 0.1478, effective sample size ~32.3.

> Effective sample size ~32.3; still thin for a pair.

n=238 is below the ~500 observations DCC(1,1) is usually given. Estimation proceeds so the failure mode is visible, but parameters should be treated as unidentified unless the diagnostics pass.

Univariate GARCH(1,1) variance models:

| Leg | omega | alpha | beta | persistence | half-life (days) |
|---|---|---|---|---|---|
| ^NSEI | 0.0183 | 0.0875 | 0.8888 | 0.9763 | 28.9 |
| GC=F | 0.5876 | 0.0658 | 0.7696 | 0.8353 | 3.9 |

DCC(1,1) correlation equation:

| DCC(1,1) parameter | Value |
|---|---|
| a | 0.00000 |
| b | 0.00000 |
| mean conditional correlation | +0.0547 |
| sd of conditional correlation | 0.9983 |
| max abs correlation reached | 1.0000 |
| min eigenvalue of R(t) | 0.000000 |

DCC(1,1) is **not identified** on this sample: b=0.0000 is near zero, so the correlation equation collapses to a sign model and rho is pushed to the boundary (+/-1); correlation reaches 1.0000, indicating a boundary solution rather than an estimate; conditional correlation matrices are numerically singular; n=238 is short for a two-parameter dynamic correlation model. The parameters are reported as a diagnostic, not an estimate. Use the rolling window correlation instead.

### Data quality

Largest moves:

| Date | Leg | Return |
|---|---|---|
| 2026-01-30 | GC=F | -11.69% |


## Conclusion

- Treat the coefficient as indicative, not established: Spearman rho=+0.123 does not reach significance (p=0.058), so the linear coefficient may be picking up a few extreme pairs rather than a monotone relationship; Kendall tau=+0.082 is also insignificant (p=0.058).
- Note the level correlation is -0.345 - the opposite sign. That figure is a shared-trend artefact and is not the answer.
- When ^NSEI had its worst days, GC=F averaged -1.01%, so the pair did not act as a hedge over this window.
- Dynamic-correlation models (DCC/EWMA) are not reliably identified on this much data; the rolling-window figures are the usable dynamic estimate.

---

### Method and caveats

- Correlations are on log returns of prices adjusted for corporate actions, computed on the intersection of both trading calendars.
- Confidence intervals use Fisher's z; p-values are two-sided t-tests on H0: ρ = 0.
- The block-bootstrap interval uses the stationary bootstrap (Politis & Romano) so that volatility clustering is respected; an observation-level bootstrap would understate uncertainty.
- Regression standard errors are Newey-West (HAC).
- All figures are sample statistics over one historical window. A correlation is not a structural constant and can change.
- **1y of daily data is a small sample.** With 238 observations, tail and conditional-correlation results are underpowered and should be read as indicative.

Generated by corrkit. Reproduce: `python correlate.py "^NSEI" "GC=F" --period 1y --freq 1d`
