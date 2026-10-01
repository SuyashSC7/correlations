# Superseded

These are the original hardcoded scripts from the gold/Nifty study. They have
been replaced by the `corrkit` package and `correlate.py`, which is the
supported entry point:

```bash
python correlate.py "^NSEI" "GC=F" --base INR --period 1y
```

They are kept only for reference. Two reasons not to use them:

- **They are hardcoded** to NIFTY / GC=F / USDINR and cannot be pointed at any
  other pair.
- **`06_dcc_garch.py` has an EWMA bug.** Its recursion is written
  `S = (1 - lam) * S + lam * outer(z_t)`, which weights *today's* observation
  by `lam` and decays backwards. The standard RiskMetrics form is
  `S = (1 - lam) * outer(z_t) + lam * S`, which `corrkit/models.py` implements.
  The old form produced a "correlation" of -0.374 with an effective sample size
  quoted as 1.13; the corrected one gives +0.373 with n_eff = 32.3. Its
  effective-sample-size formula `(2 - lam) / lam` is also not the standard
  `(1 + lam) / (1 - lam)`.

`data_original/` holds the raw CSVs the original study used. The new tool
caches to `.cache/` instead.
