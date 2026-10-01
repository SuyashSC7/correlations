"""
Time-varying behaviour: rolling correlation, regimes, tails and stability.

A single correlation coefficient summarises a year as though one number could
describe every regime inside it. These functions check whether that is fair -
when co-movement switches on, whether it flips sign, and whether it survives
being asked for.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .core import _as_date, _stationary_resample, corr_test, fisher_ci

DEFAULT_WINDOWS = (20, 60, 90, 120)


def rolling_correlation(returns: pd.DataFrame, windows=DEFAULT_WINDOWS,
                       label_a: str = "A", label_b: str = "B") -> pd.DataFrame:
    """Rolling correlation over several lookbacks, for stability comparison."""
    out = pd.DataFrame(index=returns.index)
    for w in windows:
        out[f"roll_{w}"] = returns["a"].rolling(w).corr(returns["b"])
    out = out.dropna(how="all")
    summary = []
    for w in windows:
        s = out[f"roll_{w}"].dropna()
        if s.empty:
            continue
        summary.append({
            "window": w, "n": len(s),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std(ddof=1)), 4) if len(s) > 2 else None,
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "latest": round(float(s.iloc[-1]), 4),
            "sign_changes": int((np.sign(s).diff().fillna(0) != 0).sum()),
            "frac_positive": round(float((s > 0).mean()), 3),
        })
    out.attrs["summary"] = summary
    return out


def volatility_regimes(returns: pd.DataFrame, n_bins: int = 3,
                       per_year: int = 252) -> dict:
    """
    Correlation inside low / mid / high volatility states.

    Co-movement between two assets is usually not constant: it tends to appear
    when volatility is elevated, which is exactly when a hedge matters.
    """
    pair = returns.dropna().copy()
    if len(pair) < 40:
        return {"available": False, "reason": "sample too small for regime splits"}

    # Pooled realised volatility across both legs, so "high vol" means volatile
    # for the pair rather than for one asset alone.
    rv = np.sqrt((pair["a"] ** 2 + pair["b"] ** 2) / 2)
    q = rv.quantile([1 / n_bins, 2 / n_bins]).to_numpy()
    labels = np.where(rv <= q[0], "low", np.where(rv <= q[1], "mid", "high"))
    pair["regime"] = labels

    out: dict = {"available": True, "n_bins": n_bins, "regimes": [],
                 "spread_note": None}
    overall = float(pair["a"].corr(pair["b"]))
    means = []
    for name in ("low", "mid", "high"):
        seg = pair[pair["regime"] == name]
        if len(seg) < 10:
            continue
        r = float(seg["a"].corr(seg["b"]))
        lo, hi = fisher_ci(r, len(seg))
        means.append(r)
        out["regimes"].append({
            "regime": name, "n": len(seg), "r": round(r, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "p": round(corr_test(r, len(seg))[1], 4),
            "significant": bool(corr_test(r, len(seg))[1] < 0.05),
        })
    if len(means) >= 2:
        spread = max(means) - min(means)
        out["spread"] = round(spread, 4)
        out["spread_note"] = (
            f"Correlation ranges {min(means):+.3f} to {max(means):+.3f} across "
            f"volatility states (spread {spread:.3f}). "
            + ("Co-movement is regime-dependent, so a single annual number hides "
               "the conditions that matter." if spread > 0.3 else
               "Reasonably stable across volatility states."))
    out["overall_r"] = round(overall, 4)
    return out


def tail_dependence(returns: pd.DataFrame, q: float = 0.05) -> dict:
    """
    Does one asset move with the other in its tails?

    Reported with an explicit power warning: at 5% of ~250 sessions the tail
    subsample holds about a dozen points, so a non-significant result here means
    "not enough evidence", not "no relationship".
    """
    pair = returns.dropna()
    out: dict = {"q": q, "tails": {}}
    n = len(pair)
    for side, label in (("lower", "tail"), ("upper", "tail")):
        if side == "lower":
            ta, tb = pair["a"].quantile(q), pair["b"].quantile(q)
            sa, sb = pair["a"] <= ta, pair["b"] <= tb
        else:
            ta, tb = pair["a"].quantile(1 - q), pair["b"].quantile(1 - q)
            sa, sb = pair["a"] >= ta, pair["b"] >= tb
        joint = float((sa & sb).mean())
        marg = float(sa.mean())
        seg = pair[sa]
        k = len(seg)
        rec = {
            "threshold_a": round(float(ta), 5), "threshold_b": round(float(tb), 5),
            "joint_rate": round(joint, 4), "marginal_rate": round(marg, 4),
            "n_in_tail": k,
            "exceedance_ratio": round(joint / marg, 3) if marg else None,
        }
        if k >= 8 and seg["b"].std(ddof=1) > 0:
            r = float(seg["a"].corr(seg["b"]))
            _, p, _ = corr_test(r, k)
            rec["r_in_tail"] = round(r, 4)
            rec["p_in_tail"] = round(p, 4)
        rec["verdict"] = _tail_verdict(rec, side, q, n)
        out["tails"][side] = rec

    out["power_warning"] = (
        f"At q={q:.0%} of {n} sessions each tail holds roughly {int(n*q)} observations. "
        f"Non-significant tail results here reflect low power, not evidence of "
        f"independence.")
    return out


def _tail_verdict(rec: dict, side: str, q: float, n: int) -> str:
    expected = q
    ratio = rec["exceedance_ratio"]
    k = rec["n_in_tail"]
    if ratio is None:
        return "insufficient data"
    if ratio > 1.8:
        # Both instruments are in the *lower* tail means both are falling.
        direction = "moves down" if side == "lower" else "moves up"
        return (f"{rec['joint_rate']:.1%} of the time both sit in the {side} tail "
                f"jointly, vs {expected:.1%} expected by chance "
                f"(x{ratio:.2f}): a genuine tendency for {rec.get('label', 'the pair')} "
                f"to {direction} together.")
    if ratio < 0.6:
        return (f"joint tail rate is {rec['joint_rate']:.1%} vs {expected:.1%} expected "
                f"(x{ratio:.2f}): the {side} tail shows mild avoidance.")
    return (f"joint tail rate {rec['joint_rate']:.1%} vs {expected:.1%} expected "
            f"(x{ratio:.2f}): indistinguishable from independence.")
    _ = k


def stress_behaviour(returns: pd.DataFrame, tail_q: float = 0.05,
                     label_a: str = "A", label_b: str = "B") -> dict:
    """
    What does B do on A's worst days? The practical hedging question.

    Distinguished from `tail_dependence` on purpose: that asks whether the two
    co-occur in tails at all, this asks what the payoff of holding B was in A's
    crisis, which is the number a portfolio actually needs.
    """
    pair = returns.dropna()
    thr = pair["a"].quantile(tail_q)
    bad = pair[pair["a"] <= thr]
    good = pair[pair["a"] > thr]

    n_bad = len(bad)
    up = float((bad["b"] > 0).mean()) if n_bad else float("nan")
    out = {
        "tail_q": tail_q,
        "n_stress_days": n_bad,
        "threshold_a": round(float(thr), 5),
        "mean_a_on_stress": round(float(bad["a"].mean()), 5) if n_bad else None,
        "mean_b_on_stress": round(float(bad["b"].mean()), 5) if n_bad else None,
        "median_b_on_stress": round(float(bad["b"].median()), 5) if n_bad else None,
        "worst_b_on_stress": round(float(bad["b"].min()), 5) if n_bad else None,
        "b_positive_rate_on_stress": round(up, 4) if n_bad else None,
        "mean_b_normal": round(float(good["b"].mean()), 5),
        "mean_b_stress": round(float(bad["b"].mean()), 5) if n_bad else None,
        "annualised_b_on_stress": None,
    }
    if n_bad:
        per_year = 252 / len(pair)
        out["annualised_b_on_stress"] = round(
            float(bad["b"].mean()) * per_year, 4)

    if n_bad >= 5:
        tr = stats.ttest_ind(bad["b"], good["b"], equal_var=False)
        t, p = float(tr.statistic), float(tr.pvalue)
        out["t_stat"] = round(t, 4)
        out["p_value"] = round(p, 4)
        out["significantly_different"] = bool(p < 0.05)

    if n_bad == 0:
        out["verdict"] = "No stress days at this threshold."
    else:
        hedges = out["b_positive_rate_on_stress"] or 0
        m = out["mean_b_on_stress"]
        if m is not None and m < 0:
            out["verdict"] = (
                f"On {label_a}'s worst {tail_q:.0%} of days ({n_bad} sessions), "
                f"{label_b} averaged {m:+.2%} and rose {hedges:.0%} of the time. "
                f"{label_b} fell alongside {label_a}, so it did not offset the drawdown "
                f"in this sample.")
        elif hedges >= 0.6:
            out["verdict"] = (
                f"On {label_a}'s worst {tail_q:.0%} of days ({n_bad} sessions), "
                f"{label_b} averaged {m:+.2%} and rose {hedges:.0%} of the time: "
                f"partial offset, not a reliable hedge.")
        else:
            out["verdict"] = (
                f"On {label_a}'s worst {tail_q:.0%} of days ({n_bad} sessions), "
                f"{label_b} averaged {m:+.2%}.")
    out["power_warning"] = (
        f"Only {n_bad} stress days at q={tail_q:.0%}; treat this as indicative.")
    return out


def _row_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation for stacked rows (no per-row Python loop)."""
    a = A - A.mean(axis=1, keepdims=True)
    b = B - B.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    return np.where(den > 0, num / den, np.nan)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(a) != len(b):
        return float("nan")
    return float(_row_corr(a[None, :], b[None, :])[0])


def structural_break(returns: pd.DataFrame, label_a: str = "A",
                     label_b: str = "B", n_boot: int = 2000) -> dict:
    """
    Test whether the correlation genuinely changed partway through the sample.

    The naive Chow statistic assumes homoskedastic, independent residuals, which
    return data violates badly. So the observed statistic is compared against a
    null distribution built two ways:

      1. stationary block bootstrap on the residuals (keeps volatility clusters)
      2. circularly time-shifted surrogates (destroys any temporal alignment
         while preserving the full serial and distributional structure)

    Reporting both makes the p-value hard to accuse of being an artefact of the
    one assumption most likely to be wrong.
    """
    pair = returns.dropna()
    n = len(pair)
    out: dict = {"n": n, "tests": [], "note": None}
    if n < 60:
        out["note"] = "Sample too short to test for a break."
        return out

    y = pair["a"].to_numpy()
    x = pair["b"].to_numpy()
    r_full = float(np.corrcoef(y, x)[0, 1])
    _, p_full, _ = corr_test(r_full, n)

    # Candidate breakpoints at 40-60% of the sample.
    best = None
    for frac in (0.4, 0.45, 0.5, 0.55, 0.6):
        k = int(n * frac)
        r1 = float(np.corrcoef(y[:k], x[:k])[0, 1])
        r2 = float(np.corrcoef(y[k:], x[k:])[0, 1])
        lo1, hi1 = fisher_ci(r1, k)
        lo2, hi2 = fisher_ci(r2, n - k)
        diff = abs(r1 - r2)
        # Overlap of the two CIs: non-overlapping intervals are the clean signal.
        overlap = not (hi1 < lo2 or hi2 < lo1)
        row = {
            "break_fraction": frac, "index": k,
            "date": _as_date(pair.index[k]),
            "r_before": round(r1, 4), "r_after": round(r2, 4),
            "ci_before": [round(lo1, 4), round(hi1, 4)],
            "ci_after": [round(lo2, 4), round(hi2, 4)],
            "diff": round(diff, 4), "ci_overlap": overlap,
        }
        out["tests"].append(row)
        if best is None or diff > best["diff"]:
            best = row

    out["full_r"] = round(r_full, 4)
    out["full_p"] = round(p_full, 4)
    if not best:
        return out
    out["largest"] = best

    # Calibrated p-value: max difference across random splits, then circular
    # shifts of the pairing to strip out real autocorrelation structure.
    f = np.full((n_boot, 5), np.nan)
    bs = _stationary_resample(n, max(5, n // 20), n_boot, rng=np.random.default_rng(7))
    ys, xs = y[bs], x[bs]
    for j, frac in enumerate((0.4, 0.45, 0.5, 0.55, 0.6)):
        k = int(n * frac)
        f[:, j] = np.abs(_row_corr(ys[:, :k], xs[:, :k])
                         - _row_corr(ys[:, k:], xs[:, k:]))
    p_boot = float((np.nanmax(f, axis=1) >= best["diff"]).mean())

    shifts = []
    rng = np.random.default_rng(11)
    for _ in range(n_boot):
        s = int(rng.integers(1, n))
        # Circular shift: pair y[i] with x[(i+s) % n], splitting at n-s.
        r1 = _safe_corr(y[:n - s], x[s:])
        r2 = _safe_corr(y[n - s:], x[:s])
        if np.isfinite(r1) and np.isfinite(r2):
            shifts.append(abs(r1 - r2))
    p_shift = (float((np.asarray(shifts) >= best["diff"]).mean())
               if len(shifts) >= n_boot * 0.5 else float("nan"))

    out["p_blockbootstrap"] = round(p_boot, 4)
    out["p_timeshift"] = round(p_shift, 4)
    out["detected"] = bool(min(p_boot, p_shift) < 0.05)
    out["note"] = (
        f"largest split difference {best['diff']:.3f} at {best['date']} "
        f"({best['r_before']:+.3f} -> {best['r_after']:+.3f}); "
        f"block-bootstrap p={p_boot:.3f}, time-shift p={p_shift:.3f}. "
        + ("No evidence of a genuine regime change - the variation looks like noise."
           if not out["detected"] else
           "Evidence of a real change in the relationship over the sample."))
    return out


def autocorrelation_check(returns: pd.DataFrame, max_lag: int = 10) -> dict:
    """
    Serial correlation of returns, which is why an i.i.d. bootstrap is unsafe.
    """
    out: dict = {}
    for col in ("a", "b"):
        s = returns[col].dropna()
        n = len(s)
        denom = float((s ** 2).sum())
        if denom == 0:
            continue
        acf = [round(float((s.iloc[1:] * s.iloc[:-1]).sum() / denom), 4)]
        for lag in range(2, max_lag + 1):
            acf.append(round(float((s.iloc[lag:] * s.iloc[:-lag]).sum() / denom), 4))
        ci = 1.96 / math.sqrt(n)
        out[col] = {"acf": acf, "ci95": round(ci, 4),
                    "first_significant_lag": next(
                        (i + 1 for i, v in enumerate(acf) if abs(v) > ci), None)}
    out["note"] = ("Non-negligible serial correlation means single-observation "
                   "resampling understates uncertainty; a block bootstrap is used.")
    return out
