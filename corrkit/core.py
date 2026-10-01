"""
Correlation statistics: estimation, inference and robustness.

The governing choice is that everything is computed on *returns*, never on price
levels. Two non-stationary series will always "correlate" strongly - levels can
give a confident p-value for a relationship that is pure artefact - so the level
figure is computed only to demonstrate the trap, never as the headline.

Where a coefficient's significance is fragile, this module tries to say so
rather than reporting one confident number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = {"1d": 252, "1wk": 52, "1mo": 12}


# --------------------------------------------------------------------- core --

def fisher_ci(r: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Confidence interval for a correlation via Fisher's z transform."""
    if n < 4 or not np.isfinite(r):
        return (float("nan"), float("nan"))
    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    crit = stats.norm.ppf(0.5 + conf / 2)
    lo, hi = z - crit * se, z + crit * se
    return float(np.tanh(lo)), float(np.tanh(hi))


def corr_test(r: float, n: int) -> tuple[float, float, float]:
    """t statistic and p-value for H0: rho = 0. Returns (t, p, df)."""
    if n < 3 or not np.isfinite(r):
        return float("nan"), float("nan"), 0
    r = float(np.clip(r, -0.999999, 0.999999))
    df = n - 2
    t = r * math.sqrt(df / (1 - r ** 2))
    p = float(2 * stats.t.sf(abs(t), df))
    return float(t), p, df


def rank_correlations(x: pd.Series, y: pd.Series) -> dict:
    """Spearman rho and Kendall tau, which do not assume linear form."""
    sp = stats.spearmanr(x, y)
    kt = stats.kendalltau(x, y)
    return {
        "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
        "kendall_tau": float(kt.statistic), "kendall_p": float(kt.pvalue),
    }


@dataclass
class CorrelationResult:
    """Headline correlation plus everything needed to judge how solid it is."""

    label_a: str
    label_b: str
    n: int
    r: float
    r2: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    conf: float
    spearman_rho: float = float("nan")
    spearman_p: float = float("nan")
    kendall_tau: float = float("nan")
    kendall_p: float = float("nan")
    boot_ci: tuple[float, float] = (float("nan"), float("nan"))
    boot_block: int = 0
    vol_a: float = float("nan")
    vol_b: float = float("nan")
    covariance: float = float("nan")
    subperiods: dict = field(default_factory=dict)
    fragility: list[str] = field(default_factory=list)
    verdict: str = ""
    robust: bool = False

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["boot_ci"] = list(self.boot_ci)
        return d


def analyse(x: pd.Series, y: pd.Series, label_a: str, label_b: str, *,
            conf: float = 0.95, freq: str = "1d",
            n_boot: int = 10_000, blocks: tuple[int, ...] = (5, 10, 20),
            n_sub: int = 3) -> CorrelationResult:
    """
    Full correlation battery for one pair of return series.

    Reports the linear coefficient *and* the reasons to distrust it: a
    block-bootstrap interval that respects autocorrelation, rank-based
    estimates, and sub-period variation.
    """
    pair = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna()
    n = len(pair)
    if n < 10:
        raise ValueError(f"Only {n} overlapping returns - too few to estimate a correlation.")

    r = float(pair["a"].corr(pair["b"]))
    r2 = r * r
    t, p, _ = corr_test(r, n)
    lo, hi = fisher_ci(r, n, conf)
    rank = rank_correlations(pair["a"], pair["b"])

    per_year = TRADING_DAYS.get(freq, 252)
    boot_ci, block = block_bootstrap_ci(
        pair["a"].to_numpy(), pair["b"].to_numpy(), conf=conf, n_boot=n_boot,
        block_lengths=blocks)

    res = CorrelationResult(
        label_a=label_a, label_b=label_b, n=n, r=r, r2=r2, t_stat=t, p_value=p,
        ci_low=lo, ci_high=hi, conf=conf, boot_ci=boot_ci, boot_block=block,
        vol_a=float(pair["a"].std(ddof=1) * math.sqrt(per_year)),
        vol_b=float(pair["b"].std(ddof=1) * math.sqrt(per_year)),
        covariance=float(pair["a"].cov(pair["b"])),
        **{k: rank[k] for k in ("spearman_rho", "spearman_p", "kendall_tau", "kendall_p")},
    )
    res.subperiods = subperiod_stability(pair, n_sub=n_sub)
    res.fragility, res.verdict, res.robust = judge(res, conf=conf)
    return res


# -------------------------------------------------------------- robustness --

def block_bootstrap_ci(x: np.ndarray, y: np.ndarray, *, conf: float = 0.95,
                       n_boot: int = 10_000,
                       block_lengths: tuple[int, ...] = (5, 10, 20)
                       ) -> tuple[tuple[float, float], int]:
    """
    Stationary block-bootstrap CI for the correlation (Politis & Romano).

    Resampling contiguous blocks rather than single observations preserves
    volatility clustering, which an i.i.d. bootstrap would destroy - important
    here because apparent co-movement is largely driven by a few large joint
    moves. The block length is chosen from the candidates, and the widest
    resulting interval is reported so the answer is the most conservative.
    """
    n = len(x)
    alpha = 1 - conf
    best: tuple[float, float] | None = None
    best_L = 0
    for L in block_lengths:
        if L >= n:
            continue
        draws = _stationary_resample(n, L, n_boot, rng=np.random.default_rng(20240617))
        xs, ys = x[draws], y[draws]
        with np.errstate(invalid="ignore", divide="ignore"):
            xc = xs - xs.mean(axis=1, keepdims=True)
            yc = ys - ys.mean(axis=1, keepdims=True)
            den = np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))
            rb = np.where(den > 0, (xc * yc).sum(axis=1) / den, np.nan)
        rb = rb[np.isfinite(rb)]
        if len(rb) < n_boot * 0.5:
            continue
        lo, hi = np.percentile(rb, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        width = hi - lo
        if best is None or width > (best[1] - best[0]):
            best, best_L = (float(lo), float(hi)), L
    if best is None:
        return (float("nan"), float("nan")), 0
    return best, best_L


def _stationary_resample(n: int, block: int, n_boot: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices, shape (n_boot, n)."""
    p = 1.0 / block
    starts = rng.random((n_boot, n)) < p
    idx = np.empty((n_boot, n), dtype=np.int64)
    cur = rng.integers(0, n, size=n_boot)
    for t in range(n):
        restart = starts[:, t]
        cur = np.where(restart, rng.integers(0, n, size=n_boot), cur)
        idx[:, t] = cur
        cur = (cur + 1) % n
    return idx


def _as_date(idx) -> str:
    """Date string from an index that may or may not be temporal."""
    try:
        return str(pd.Timestamp(idx).date())
    except (ValueError, TypeError):
        return str(idx)


def subperiod_stability(pair: pd.DataFrame, n_sub: int = 3) -> dict:
    """
    Correlation within contiguous slices of the sample.

    A headline number that only holds in one sub-period is regime-specific, not
    a structural relationship.
    """
    out: dict = {"n_slices": n_sub, "slices": []}
    if len(pair) < n_sub * 20:
        return out
    bounds = np.array_split(np.arange(len(pair)), n_sub)
    for i, b in enumerate(bounds, start=1):
        seg = pair.iloc[b]
        if len(seg) < 10:
            continue
        r = float(seg["a"].corr(seg["b"]))
        lo, hi = fisher_ci(r, len(seg))
        out["slices"].append({
            "slice": i,
            "start": _as_date(seg.index.min()),
            "end": _as_date(seg.index.max()),
            "n": len(seg),
            "r": round(r, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "sign_flip": bool(np.sign(r) != np.sign(pair["a"].corr(pair["b"]))),
        })
    return out


def sensitivity_checks(x: pd.Series, y: pd.Series, flat_dates: dict | None = None,
                       drop_top: int = 0, winsorise: float = 0.01) -> dict:
    """
    Does the result survive removing suspicious or dominant observations?

    A correlation driven by a handful of joint crashes is a different claim from
    one that holds across ordinary days, so we test both.
    """
    pair = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna()
    base = float(pair["a"].corr(pair["b"]))
    out: dict = {"baseline_r": round(base, 4), "checks": {}}

    if flat_dates:
        bad = sorted({d for ds in flat_dates.values() for d in ds})
        if bad:
            clean = pair[~pair.index.strftime("%Y-%m-%d").isin(bad)]
            if len(clean) >= 15:
                rr, pp, nn = _quick(clean["a"], clean["b"])
                out["checks"]["drop_flat_days"] = {
                    "n": nn, "r": round(rr, 4), "p": pp,
                    "removed": len(pair) - nn}

    if drop_top:
        keep = pair.copy()
        drop_idx = (keep["a"].abs() + keep["b"].abs()).nlargest(drop_top).index
        keep = keep.drop(index=drop_idx)
        if len(keep) >= 15:
            rr, pp, nn = _quick(keep["a"], keep["b"])
            out["checks"]["drop_largest_moves"] = {
                "n": nn, "r": round(rr, 4), "p": pp, "removed": int(drop_top)}

    if winsorise > 0:
        w = pair.copy()
        lo_a, hi_a = w["a"].quantile([winsorise, 1 - winsorise])
        lo_b, hi_b = w["b"].quantile([winsorise, 1 - winsorise])
        w["a"] = w["a"].clip(lo_a, hi_a)
        w["b"] = w["b"].clip(lo_b, hi_b)
        rr, pp, nn = _quick(w["a"], w["b"])
        out["checks"]["winsorised"] = {"n": nn, "r": round(rr, 4), "p": pp,
                                       "tail": winsorise}

    # Winsorising the joint extremes is the most informative single check: if
    # the coefficient collapses, the "relationship" was really two or three days.
    if "winsorised" in out["checks"]:
        drift = abs(out["checks"]["winsorised"]["r"] - base)
        out["outlier_dependence"] = ("high" if drift > 0.15
                                     else "moderate" if drift > 0.07 else "low")
    return out


def _quick(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    r = float(x.corr(y))
    _, p, _ = corr_test(r, len(x))
    return r, round(p, 4), len(x)


def levels_vs_returns(prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """
    Correlate levels and returns side by side.

    Reported to show how misleading the level figure is, not as a result.
    """
    clean = prices.dropna()
    r_lvl = float(clean["a"].corr(clean["b"]))
    _, p_lvl, n_lvl = corr_test(r_lvl, len(clean))
    r_ret = float(returns["a"].corr(returns["b"]))
    _, p_ret, _ = corr_test(r_ret, len(returns))
    return {
        "levels_r": round(r_lvl, 4), "levels_p": round(p_lvl, 4), "levels_n": n_lvl,
        "returns_r": round(r_ret, 4), "returns_p": round(p_ret, 4),
        "sign_flip": bool(np.sign(r_lvl) != np.sign(r_ret)),
        "caveat": ("Level correlation of non-stationary series is not a co-movement "
                   "measure; it can be highly significant and still meaningless."),
    }


def fx_diagnosis(native: pd.DataFrame, normalised: pd.DataFrame,
                 fx_returns: pd.Series | None) -> dict:
    """
    Quantify how much of an unconverted correlation is really currency.

    Re-expresses the problem in Blinder terms: each leg has a loading on the
    exchange rate, and the product of those loadings is the FX-induced part of
    any correlation measured across currencies. The difference between the
    unconverted and normalised coefficients is tested by bootstrap, which
    respects the fact that the two estimates come from the same days.
    """
    n_pair = pd.concat([native["a"].rename("a"), native["b"].rename("b")], axis=1).dropna()
    r_native = float(n_pair["a"].corr(n_pair["b"]))
    r_norm = float(normalised["a"].corr(normalised["b"]))

    out: dict = {
        "r_unconverted": round(r_native, 4),
        "r_normalised": round(r_norm, 4),
        "difference": round(r_native - r_norm, 4),
    }

    # The two coefficients are compared on the days they share. FX alignment can
    # drop dates, so the raw point estimates above may rest on different samples
    # and their difference would not be a like-for-like comparison.
    aligned = pd.concat([n_pair["a"].rename("na"), n_pair["b"].rename("nb"),
                         normalised["a"].rename("xa"), normalised["b"].rename("xb")],
                        axis=1).dropna()
    d_native = float(aligned["na"].corr(aligned["nb"]))
    d_norm = float(aligned["xa"].corr(aligned["xb"]))
    out["aligned_n"] = len(aligned)
    out["aligned_r_unconverted"] = round(d_native, 4)
    out["aligned_r_normalised"] = round(d_norm, 4)
    out["aligned_difference"] = round(d_native - d_norm, 4)

    if fx_returns is not None:
        j = pd.concat([n_pair["a"], n_pair["b"], fx_returns.rename("fx")], axis=1).dropna()
        r_af = float(j["a"].corr(j["fx"]))
        r_bf = float(j["b"].corr(j["fx"]))
        out["r_a_vs_fx"] = round(r_af, 4)
        out["r_b_vs_fx"] = round(r_bf, 4)
        out["fx_induced_r"] = round(r_af * r_bf, 4)
        if abs(r_af) > 0.15 or abs(r_bf) > 0.15:
            out["note"] = (
                f"The exchange rate carries a material loading on at least one leg "
                f"(r={r_af:+.2f} / {r_bf:+.2f}), so cross-currency correlations are "
                f"partly a currency story. The normalised figure is the defensible one.")

    # Paired stationary block bootstrap: one set of resampled day-indices drives
    # all four legs, so both coefficients are always measured on identical days.
    # Bootstrap the *difference*; a difference of two correlations is not itself
    # a correlation and must not be fed to a correlation t-test.
    n = len(aligned)
    if n >= 30:
        draws = _stationary_resample(n, max(5, n // 20), 4000,
                                     rng=np.random.default_rng(991))
        arr = aligned[["na", "nb", "xa", "xb"]].to_numpy()
        idx = np.asarray(draws)

        def _corr_rows(blocks: np.ndarray) -> np.ndarray:
            x = blocks[:, :, 0]
            y = blocks[:, :, 1]
            xc = x - x.mean(axis=1, keepdims=True)
            yc = y - y.mean(axis=1, keepdims=True)
            den = np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))
            return np.where(den > 0, (xc * yc).sum(axis=1) / den, np.nan)

        r_bn = _corr_rows(arr[idx][:, :, 0:2])
        r_bx = _corr_rows(arr[idx][:, :, 2:4])
        with np.errstate(invalid="ignore"):
            r_bd = r_bn - r_bx
        ok = np.isfinite(r_bd)
        if ok.sum() > 100:
            diffs, rnorms = r_bd[ok], r_bx[ok]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            out["difference_ci"] = [round(float(lo), 4), round(float(hi), 4)]
            # Two-sided bootstrap p-value against H0: the two coefficients agree.
            frac_le = float((diffs <= 0).mean())
            p_boot = 2 * min(frac_le, 1 - frac_le)
            out["difference_p"] = round(min(1.0, p_boot), 4)
            nlo, nhi = np.percentile(rnorms, [2.5, 97.5])
            out["r_normalised_ci"] = [round(float(nlo), 4), round(float(nhi), 4)]
            out["bootstrap_method"] = "paired stationary block bootstrap"
    else:
        out["difference_p"] = None
        out["note_too_short"] = "Too few overlapping days to test the difference."
    return out


# ------------------------------------------------------------- interpretation --

def strength_label(r: float) -> str:
    a = abs(r)
    if not np.isfinite(a):
        return "undefined"
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "weak"
    if a < 0.50:
        return "moderate"
    if a < 0.70:
        return "strong"
    return "very strong"


def judge(res: CorrelationResult, conf: float = 0.95) -> tuple[list[str], str, bool]:
    """
    Decide whether the headline correlation deserves to be trusted.

    A coefficient can be statistically significant and still be the wrong answer:
    rank measures disagree, the block bootstrap straddles zero, or the sign
    flips across sub-periods. Each of those is recorded as a fragility.
    """
    frag: list[str] = []
    alpha = 1 - conf

    if res.p_value >= alpha:
        frag.append(f"not significant at {int(alpha*100)}% (p={res.p_value:.4f})")

    # Rank measures assume less than a linear relation.
    if np.isfinite(res.spearman_p) and res.spearman_p >= alpha:
        frag.append(
            f"Spearman rho={res.spearman_rho:+.3f} does not reach significance "
            f"(p={res.spearman_p:.3f}), so the linear coefficient may be picking up "
            f"a few extreme pairs rather than a monotone relationship")
    if np.isfinite(res.kendall_p) and res.kendall_p >= alpha:
        frag.append(f"Kendall tau={res.kendall_tau:+.3f} is also insignificant "
                    f"(p={res.kendall_p:.3f})")

    # Autocorrelation-aware interval.
    if np.isfinite(res.boot_ci[0]) and (res.boot_ci[0] <= 0 <= res.boot_ci[1]):
        frag.append(f"block-bootstrap 95% CI [{res.boot_ci[0]:+.3f}, {res.boot_ci[1]:+.3f}] "
                    f"includes zero once volatility clustering is respected")

    flips = [s for s in res.subperiods.get("slices", []) if s["sign_flip"]]
    if flips:
        frag.append("sign flips across sub-periods "
                    f"({', '.join(s['start'][:7] for s in flips)}) - the relationship "
                    f"is regime-dependent")

    # A rank coefficient that is much larger than the linear one means the
    # relationship is monotone but not proportional, so r understates it.
    if np.isfinite(res.spearman_rho) and abs(res.spearman_rho - res.r) > 0.15:
        direction = "understates" if res.spearman_rho > res.r else "overstates"
        frag.append(f"Spearman rho ({res.spearman_rho:+.3f}) differs enough from "
                    f"r ({res.r:+.3f}) that the linear coefficient {direction} the "
                    f"strength of the monotone relationship")

    vals = [s["r"] for s in res.subperiods.get("slices", [])]
    if len(vals) >= 2 and (max(vals) - min(vals)) > 0.4:
        frag.append(f"sub-period correlations span {min(vals):+.2f} to {max(vals):+.2f}")

    direction = "positive" if res.r > 0 else "negative"
    verdict = f"{strength_label(res.r)} {direction} correlation (r={res.r:+.4f})"

    # "Robust" is a strong word: require significance, rank agreement in sign
    # and significance, no sub-period sign flip, and a stable block interval.
    signs_agree = (np.sign(res.spearman_rho) == np.sign(res.r)
                   and np.sign(res.kendall_tau) == np.sign(res.r))
    ranks_sig = (np.isfinite(res.spearman_p) and res.spearman_p < alpha
                 and np.isfinite(res.kendall_p) and res.kendall_p < alpha)
    boot_excludes_zero = (np.isfinite(res.boot_ci[0])
                          and not (res.boot_ci[0] <= 0 <= res.boot_ci[1]))
    robust = (res.p_value < alpha and signs_agree and ranks_sig
              and not flips and boot_excludes_zero)

    if robust:
        verdict += " - stable across estimators and sub-periods"
    else:
        verdict += " - treat as indicative, not established"
    return frag, verdict, robust
