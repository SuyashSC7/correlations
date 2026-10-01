# corrkit — correlation index for any two instruments

Give it two company names or ticker symbols and it returns a correlation
analysis: a headline number, the confidence interval behind it, and an explicit
account of how much to trust it.

```bash
python correlate.py "tata motors" RELIANCE.NS
```

```
==========================================================================
  CORRELATION INDEX   TMCV.NS  vs  HDFCBANK.NS
==========================================================================
  window            2025-11-12 -> 2026-09-30  (1y, 1d)
  observations      223 log returns
  base currency     INR

  >>> PEARSON r    = +0.5470
      95% CI       = [+0.4479, +0.6329]
      p-value      = 0.0000      R² = 0.2993
      Spearman ρ   = +0.3961 (p=0.0000)
      Kendall τ    = +0.2742 (p=0.0000)
      block-boot   = [+0.2809, +0.7039]

  VERDICT: strong positive correlation (r=+0.5470) - stable across estimators
           and sub-periods
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add requirements-dev.txt to test
```

Python 3.11+. Data comes from Yahoo Finance; the first run of each symbol
downloads and caches, later runs are offline-fast.

## Usage

```bash
python correlate.py AAPL MSFT                      # symbols
python correlate.py "hdfc bank" RELIANCE.NS       # company names
python correlate.py ^NSEI GC=F --base INR          # normalise currencies
python correlate.py AAPL MSFT --period 5y          # longer window
python correlate.py AAPL MSFT --freq 1wk           # weekly returns
python correlate.py VOD.L SAP.DE --json            # machine-readable
python correlate.py "hdfc bank" --list             # what does this match?
```

| Option | Default | Effect |
|---|---|---|
| `--period` | `1y` | `6mo` `1y` `2y` `3y` `5y` `10y` `max` |
| `--freq` | `1d` | `1d` `1wk` `1mo` |
| `--base` | `auto` | currency for both legs (`INR`, `USD`, …) |
| `--no-fx` | off | skip currency normalisation |
| `--pick N` / `--pick-b N` | — | choose the Nth name match |
| `--list` | — | show all matches for an input and exit |
| `--simple-returns` | off | simple instead of log returns |
| `--conf` | `0.95` | confidence level |
| `--boot` | `10000` | block-bootstrap draws |
| `--tail` | `0.05` | tail quantile |
| `--lam` | `0.94` | EWMA lambda |
| `--no-dcc` | off | skip DCC-GARCH |
| `--out` | `output` | output directory |
| `--json` | off | print summary JSON only |

Each run writes `report.md`, `results.json`, `prices.csv`, `returns.csv` and
eight figures to `--out`.

## What it computes

**Headline.** Pearson r on log returns, Fisher-z confidence interval, two-sided
t-test, R².

**Robustness.** Spearman ρ and Kendall τ (which assume less than a linear
relationship); a stationary block-bootstrap interval (Politis & Romano) that
respects volatility clustering, where an observation-level bootstrap would
understate uncertainty; sub-period correlations; and sensitivity to dropping flat
or extreme days and to winsorising the tails.

**Currency.** When the two instruments quote in different currencies both legs
are restated in one currency using the *time-varying* FX series, and the report
shows the unconverted figure alongside so the currency's contribution is
visible. A single spot rate would freeze the exchange rate for the whole sample
and invent a trend.

**Time variation.** Rolling correlations at 20/60/90/120 days, correlations
split by volatility regime, and a structural-break test whose p-value is
calibrated by both a block bootstrap and circularly time-shifted surrogates
(because the usual Chow assumptions are exactly the ones return data violates).

**Tails and stress.** Joint tail-exceedance rates against chance, plus what the
second asset actually did on the first's worst days — the number that actually
matters for a hedge.

**Models.** ADF stationarity tests on levels and returns; OLS in both directions
with Newey-West errors; minimum-variance portfolio weights and the variance
reduction they buy; EWMA and DCC-GARCH conditional correlation.

## Two rules the tool will not break

**Returns, never levels.** Two non-stationary series always correlate
strongly — levels can give a confident p-value for a pure artefact. The level
figure is computed only to show the trap (in the gold/Nifty case it was −0.34
against +0.20 for returns, an outright sign flip).

**A p-value is not a verdict.** `judge()` in `corrkit/core.py` checks rank
agreement, sub-period sign stability and the block-bootstrap interval, and will
report "treat as indicative, not established" on a coefficient that is
statistically significant but not stable. The gold/Nifty correlation is
p = 0.0015 by Pearson and insignificant by both Spearman and Kendall — the tool
says so rather than quoting the flattering number.

## Known limits

- **DCC-GARCH is usually unidentified here.** With a year of daily data, `b`
  collapses toward zero, the correlation path is pushed to ±1 and the model
  degenerates into a sign model. The tool fits it, runs identification
  diagnostics, and reports the failure rather than a number. Rolling windows are
  the usable dynamic estimate.
- **Low-powered tails.** At `q = 0.05` a 250-session sample gives ~12 points per
  tail. "Not significant" there means "not enough evidence", and the report says
  so.
- **Minimum-variance weights are sample-specific.** They assume the sample
  covariance persists and are notoriously unstable. The report says so.
- **A correlation is not a constant.** It is a statistic over one window and can
  change; that is what the rolling and regime sections are for.
- **Yahoo data is adjusted and back-filled**, and futures roll. Flat days are
  flagged and tested for.

## Layout

```
correlate.py          CLI
corrkit/
  universe.py         name -> symbol resolution, currency table, FX series
  data.py             download, FX normalisation, alignment, quality audit
  core.py             correlation statistics and inference
  dynamics.py         rolling, regimes, tails, stress, break test
  models.py           ADF, OLS/HAC, DCC-GARCH, EWMA, portfolio
  pipeline.py         orchestration
  report.py           markdown, figures, JSON
tests/                51 offline tests
examples/gold_nifty/  the full worked example, regenerated
legacy/               the original hardcoded scripts (superseded - see its README)
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q     # 51 tests, no network needed
```

Offline and fast. They pin the traps this code has actually hit: pence-to-major
scaling, FX quote conventions (`USDEUR=X` is EUR per USD, `EURUSD=X` is the
reverse, and the two do not reciprocate exactly), a double-counted `sigma²` in
the HAC sandwich, the minimum-variance closed form, EWMA effective sample size,
and the `judge()` fragility rules.

## Worked example

`REPORT.md` is the full gold vs Nifty 50 study, the case this tool was built
from. Reproduce it with:

```bash
python correlate.py "^NSEI" "GC=F" --base INR --period 1y --out output/gold_nifty
```
