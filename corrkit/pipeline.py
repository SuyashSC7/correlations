"""
End-to-end analysis pipeline.

Runs every module over a loaded `Dataset` and returns one flat result tree, so
the renderer and the CLI never have to know how a number was produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import core, dynamics, models
from .data import Dataset
from .universe import _fx_series


def run(ds: Dataset, *, conf: float = 0.95, n_boot: int = 10_000,
        n_sub: int = 3, tail_q: float = 0.05, stress_q: float = 0.05,
        lam: float = 0.94, with_dcc: bool = True,
        progress=None) -> dict:
    """Full correlation study for one pair."""
    say = progress or (lambda *_: None)
    short = (ds.a.symbol, ds.b.symbol)
    res: dict = {
        "meta": {
            "symbols": list(short),
            "resolved": [ds.a.res.describe(), ds.b.res.describe()],
            "query": [ds.a.res.query, ds.b.res.query],
            "resolution_method": [ds.a.res.method, ds.b.res.method],
            "candidates": [ds.a.res.candidates, ds.b.res.candidates],
            "base_currency": ds.base_currency,
            "currency_a": ds.a.currency, "currency_b": ds.b.currency,
            "fx_conversion": [ds.a.conversion, ds.b.conversion],
            "period": ds.period, "freq": ds.freq,
            "return_type": ds.audit.get("return_type", "log"),
            "n_sessions": ds.audit["n_sessions"], "n_returns": ds.audit["n_returns"],
            "start": ds.audit["start"], "end": ds.audit["end"],
            "conf": conf,
        },
        "notes": list(ds.notes),
        "audit": ds.audit,
    }

    say("[3] Correlation")
    res["core"] = core.analyse(ds.returns["a"], ds.returns["b"], short[0], short[1],
                               conf=conf, freq=ds.freq, n_boot=n_boot, n_sub=n_sub)
    res["levels"] = core.levels_vs_returns(ds.prices, ds.returns)
    res["sensitivity"] = core.sensitivity_checks(
        ds.returns["a"], ds.returns["b"],
        flat_dates=ds.audit.get("flat_days"), drop_top=3, winsorise=0.01)

    say("[4] FX confounding")
    fx_ret = None
    if ds.audit.get("fx_normalised"):
        sym = ds.a.fx_symbol or "USDINR=X"
        s = _fx_series(sym, None, 3, ds.period)
        if s is not None:
            fx_ret = np.log(s).diff()
    res["fx"] = core.fx_diagnosis(ds.native_returns, ds.returns, fx_ret)

    say("[5] Time variation")
    res["rolling"] = dynamics.rolling_correlation(ds.returns, label_a=short[0],
                                                   label_b=short[1])
    res["rolling_summary"] = res["rolling"].attrs.get("summary", [])
    res["autocorr"] = dynamics.autocorrelation_check(ds.returns)
    res["regimes"] = dynamics.volatility_regimes(ds.returns)
    res["break"] = dynamics.structural_break(ds.returns, *short)

    say("[6] Tails and stress")
    res["tails"] = dynamics.tail_dependence(ds.returns, q=tail_q)
    res["stress"] = dynamics.stress_behaviour(ds.returns, tail_q=stress_q,
                                               label_a=short[0], label_b=short[1])

    say("[7] Regression and portfolio")
    res["stationarity"] = models.stationarity(ds.prices, short)
    res["regression"] = models.bidirection(ds.returns, short)
    res["portfolio"] = models.portfolio_frontier(ds.returns, short)

    say("[8] Conditional correlation")
    res["ewma"] = models.ewma_correlation(ds.returns, lam=lam, labels=short)
    res["rolling_diag"] = models.rolling_diagnostics(ds.returns)
    if with_dcc:
        res["dcc"] = models.dcc_garch(ds.returns, short)

    say("[9] Conclusion")
    res["conclusion"] = conclude(ds, res)
    return res


def conclude(ds: Dataset, res: dict) -> dict:
    """
    Turn the numbers into a short, honest answer.

    Written to be quotable on its own: what the number is, whether to trust it,
    and the one thing most likely to mislead a reader of it.
    """
    c = res["core"]
    short = (ds.a.symbol, ds.b.symbol)
    bits: list[str] = []

    if c.robust:
        bits.append("The coefficient survives rank-based estimators and does not "
                    "flip sign across sub-periods.")
    else:
        reasons = "; ".join(c.fragility[:2]) if c.fragility else \
            "estimator checks did not fully agree"
        bits.append(f"Treat the coefficient as indicative, not established: {reasons}.")

    fx = res.get("fx", {})
    if fx.get("fx_induced_r") is not None and abs(fx.get("difference", 0)) >= 0.05:
        bits.append(
            f"Currency matters here: the unconverted correlation is "
            f"{fx['r_unconverted']:+.3f} versus {fx['r_normalised']:+.3f} once both "
            f"legs are restated in {ds.base_currency}, so part of the raw number "
            f"was the exchange rate, not the assets.")

    lv = res["levels"]
    if lv.get("sign_flip"):
        bits.append(
            f"Note the level correlation is {lv['levels_r']:+.3f} - the opposite sign. "
            f"That figure is a shared-trend artefact and is not the answer.")

    st = res.get("stress", {})
    if st.get("n_stress_days", 0) >= 5 and st.get("mean_b_on_stress") is not None:
        m = st["mean_b_on_stress"]
        if m < 0:
            bits.append(
                f"When {short[0]} had its worst days, {short[1]} averaged {m:+.2%}, "
                f"so the pair did not act as a hedge over this window.")

    d = res.get("dcc")
    if d and not d.get("identified", True):
        bits.append(
            "Dynamic-correlation models (DCC/EWMA) are not reliably identified on "
            "this much data; the rolling-window figures are the usable dynamic "
            "estimate.")

    sens = res.get("sensitivity", {})
    if sens.get("outlier_dependence") == "high":
        bits.append("The result is highly sensitive to a few extreme days.")

    return {
        "headline": (f"Correlation of {short[0]} and {short[1]} "
                     f"({ds.audit['start']} to {ds.audit['end']}, "
                     f"{ds.audit['n_returns']} {ds.audit['return_type']} returns): "
                     f"r = {c.r:+.4f}"),
        "verdict": c.verdict,
        "robust": c.robust,
        "points": bits,
        "headline_number": round(c.r, 4),
    }


def headline(res: dict) -> pd.Series:
    """One-row summary, handy for scripting."""
    c = res["core"]
    return pd.Series({
        "symbol_a": c.label_a, "symbol_b": c.label_b,
        "n": c.n, "correlation": round(c.r, 4),
        "p_value": round(c.p_value, 4),
        "ci_low": round(c.ci_low, 4), "ci_high": round(c.ci_high, 4),
        "r_squared": round(c.r2, 4),
        "spearman": round(c.spearman_rho, 4),
        "kendall": round(c.kendall_tau, 4),
        "verdict": c.verdict, "robust": c.robust,
        "start": res["meta"]["start"], "end": res["meta"]["end"],
    })
