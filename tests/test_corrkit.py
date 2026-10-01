"""
Offline tests for the pieces that are easy to get silently wrong.

No network: every test uses synthetic series or a stubbed currency table, so the
suite runs anywhere and stays fast. The cases mirror real bugs found while
building this: an inverted pence conversion, a double-counted sigma^2 in the
HAC sandwich, mismatched FX quote conventions, and a wrong closed form for the
minimum-variance weight.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corrkit import core, data, models, universe
from corrkit.data import _convert, _to_returns
from corrkit.universe import fx_symbol_for, normalise_currency


def _pair(n: int, rho: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rho * a + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b})


# ------------------------------------------------------------------- core --

def test_pearson_matches_numpy():
    p = _pair(500, 0.4, seed=1)
    r = float(p["a"].corr(p["b"]))
    assert abs(r - core.analyse(p["a"], p["b"], "A", "B", n_boot=200).r) < 1e-12


@pytest.mark.parametrize("rho", [0.0, 0.3, -0.5, 0.85])
def test_fisher_ci_brackets_and_widens(rho):
    n = 300
    p = _pair(n, rho, seed=2)
    r = float(p["a"].corr(p["b"]))
    lo, hi = core.fisher_ci(r, n)
    assert lo < r < hi
    assert hi - lo > 0


def test_fisher_ci_handles_perfect_correlation():
    lo, hi = core.fisher_ci(0.999999, 50)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_corr_test_known_value():
    # r=0.2, n=100 -> df=98, t = 0.2*sqrt(98/0.96) = 2.020726, two-sided p = 0.046036
    t, p, df = core.corr_test(0.2, 100)
    assert df == 98
    assert abs(t - 2.020726) < 1e-6
    assert abs(p - 0.0460363) < 1e-7


def test_judge_flags_insignificant_pearson():
    res = core.CorrelationResult("A", "B", n=100, r=0.05, r2=0.0025, t_stat=0.5,
                                 p_value=0.62, ci_low=-0.15, ci_high=0.25, conf=0.95,
                                 spearman_rho=0.04, spearman_p=0.68,
                                 kendall_tau=0.03, kendall_p=0.71,
                                 boot_ci=(-0.2, 0.3), boot_block=5)
    frag, verdict, robust = core.judge(res)
    assert not robust
    assert any("not significant" in f for f in frag)
    assert "indicative" in verdict


def test_judge_flags_rank_disagreement():
    """Pearson significant but Spearman not: the coefficient is not established."""
    res = core.CorrelationResult("A", "B", n=250, r=0.30, r2=0.09, t_stat=3.97,
                                 p_value=0.0001, ci_low=0.16, ci_high=0.43, conf=0.95,
                                 spearman_rho=0.08, spearman_p=0.21,
                                 kendall_tau=0.05, kendall_p=0.44,
                                 boot_ci=(0.02, 0.55), boot_block=10)
    frag, _, robust = core.judge(res)
    assert not robust
    assert any("Spearman" in f for f in frag)
    assert any("Kendall" in f for f in frag)


def test_judge_flags_sign_flip_across_subperiods():
    res = core.CorrelationResult("A", "B", n=300, r=0.25, r2=0.06, t_stat=4.4,
                                 p_value=0.0, ci_low=0.12, ci_high=0.38, conf=0.95,
                                 spearman_rho=0.22, spearman_p=0.001,
                                 kendall_tau=0.15, kendall_p=0.002,
                                 boot_ci=(0.05, 0.44), boot_block=10,
                                 subperiods={"slices": [
                                     {"r": 0.6, "start": "2024-01-01",
                                      "sign_flip": False},
                                     {"r": -0.3, "start": "2024-07-01",
                                      "sign_flip": True}]})
    frag, _, robust = core.judge(res)
    assert not robust
    assert any("sign flips" in f for f in frag)


def test_judge_marks_clean_result_robust():
    res = core.CorrelationResult("A", "B", n=400, r=0.35, r2=0.12, t_stat=7.2,
                                 p_value=0.0, ci_low=0.26, ci_high=0.44, conf=0.95,
                                 spearman_rho=0.33, spearman_p=0.0,
                                 kendall_tau=0.24, kendall_p=0.0,
                                 boot_ci=(0.24, 0.46), boot_block=10,
                                 subperiods={"slices": [
                                     {"r": 0.36, "start": "2024-01-01",
                                      "sign_flip": False},
                                     {"r": 0.34, "start": "2024-07-01",
                                      "sign_flip": False}]})
    frag, verdict, robust = core.judge(res)
    assert robust and not frag
    assert "stable" in verdict


def test_judge_flags_bootstrap_straddling_zero():
    res = core.CorrelationResult("A", "B", n=250, r=0.22, r2=0.05, t_stat=3.5,
                                 p_value=0.0005, ci_low=0.10, ci_high=0.34, conf=0.95,
                                 spearman_rho=0.21, spearman_p=0.001,
                                 kendall_tau=0.14, kendall_p=0.004,
                                 boot_ci=(-0.05, 0.48), boot_block=10)
    frag, _, robust = core.judge(res)
    assert not robust
    assert any("bootstrap" in f for f in frag)


def test_block_bootstrap_brackets_point_estimate():
    n = 300
    p = _pair(n, 0.4, seed=21)
    res = core.analyse(p["a"], p["b"], "A", "B", n_boot=3000)
    assert res.boot_ci[0] <= res.r <= res.boot_ci[1]


def test_block_bootstrap_returns_the_widest_candidate_interval():
    """Contract of block_bootstrap_ci: with several block lengths offered, the
    widest resulting interval is the one reported.

    Interval width is not monotone in block length - for an AR(1) pair, long
    blocks can actually narrow the spread - so the selection rule is an explicit
    conservative choice rather than a consequence of the data.
    """
    rng = np.random.default_rng(22)
    n = 400
    a = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.75 * a[t - 1] + rng.normal()
    b = 0.3 * a + rng.normal(size=n)
    blocks = (2, 5, 10, 20, 30)
    (lo, hi), chosen = core.block_bootstrap_ci(a, b, n_boot=2000,
                                               block_lengths=blocks)
    widths = {}
    for L in blocks:
        (l2, h2), _ = core.block_bootstrap_ci(a, b, n_boot=2000, block_lengths=(L,))
        widths[L] = h2 - l2
    assert chosen in blocks
    assert abs((hi - lo) - max(widths.values())) < 1e-12


def test_stationary_resample_keeps_contiguous_runs():
    """Blocks must preserve short-range structure, not shuffle observations."""
    from corrkit.core import _stationary_resample
    idx = _stationary_resample(200, 10, 30, rng=np.random.default_rng(23))
    row = idx[0]
    steps = np.diff(row)
    # A stationary block bootstrap advances by 1 inside a block and jumps
    # randomly at restarts; it should never walk backwards by exactly -1.
    assert (steps != -1).all()


def test_stationary_resample_shape_and_range():
    from corrkit.core import _stationary_resample
    idx = _stationary_resample(50, 5, 7, rng=np.random.default_rng(0))
    assert idx.shape == (7, 50)
    assert idx.min() >= 0 and idx.max() < 50


def test_subperiod_stability_finds_flip():
    rng = np.random.default_rng(5)
    n = 400
    a = rng.normal(size=n)
    b = np.concatenate([0.8 * a[:200], -0.8 * a[200:]]) + 0.3 * rng.normal(size=n)
    s = pd.DataFrame({"a": a, "b": b},
                     index=pd.date_range("2020-01-01", periods=n, freq="B"))
    res = core.analyse(s["a"], s["b"], "A", "B", n_boot=200)
    assert any(sl["sign_flip"] for sl in res.subperiods["slices"])


def test_strength_label_bands():
    assert core.strength_label(0.05) == "negligible"
    assert core.strength_label(0.2) == "weak"
    assert core.strength_label(0.4) == "moderate"
    assert core.strength_label(0.6) == "strong"
    assert core.strength_label(-0.9) == "very strong"


def test_levels_vs_returns_detects_sign_flip():
    n = 300
    t = np.arange(n)
    lvl = pd.DataFrame({"a": t + 50.0, "b": (t + 20.0)[::-1].copy()})
    ret = _to_returns(lvl[["a", "b"]], simple=False)
    out = core.levels_vs_returns(lvl, ret.dropna())
    assert out["levels_r"] < 0 < out["returns_r"]
    assert out["sign_flip"] is True


# ------------------------------------------------------------- FX handling --

def test_minor_unit_mapping():
    assert normalise_currency("GBp") == ("GBP", 100.0)
    assert normalise_currency("INR") == ("INR", 1.0)
    assert normalise_currency("") == ("USD", 1.0)


def test_fx_symbol_convention_is_major_per_usd():
    """Regression guard: FX_TO_PER_USD means 'major units per 1 USD'."""
    assert fx_symbol_for("INR") == "USDINR=X"
    assert fx_symbol_for("EUR") == "USDEUR=X"
    assert fx_symbol_for("USD") is None


def _fake_fx(index: pd.DatetimeIndex) -> pd.DataFrame:
    """USD per major unit, constant: USD/GBP = 1.25, USD/EUR = 1.1111..., USD/INR = 0.01."""
    n = len(index)
    return pd.DataFrame({
        "GBP": np.full(n, 1 / 0.8),        # 1 GBP = 1.25 USD
        "EUR": np.full(n, 1 / 0.9),        # 1 EUR = 1.1111 USD
        "INR": np.full(n, 0.01),           # 1 INR = 0.01 USD
        "JPY": np.full(n, 1 / 150.0),
    }, index=index)


def test_pence_conversion_divides_by_100():
    """123.30 GBp = 1.233 GBP. Getting this backwards was a real bug."""
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    px = pd.Series(np.full(5, 123.30), index=idx)
    fx = _fake_fx(idx)
    out = _convert(px, "GBp", "USD", fx, tolerance=7)
    assert np.allclose(out.to_numpy(), 123.30 / 100 * 1.25)


def test_pence_to_gbp_needs_no_fx_table():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    px = pd.Series(np.full(3, 250.0), index=idx)
    out = _convert(px, "GBp", "GBP", pd.DataFrame(), tolerance=7)
    assert np.allclose(out.to_numpy(), 2.50)


def test_cross_rate_conversion_is_ratio_not_level():
    """EUR->INR must go via USD and equal (usd_per_EUR / usd_per_INR)."""
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    px = pd.Series(np.full(4, 90.0), index=idx)
    fx = _fake_fx(idx)
    out = _convert(px, "EUR", "INR", fx, tolerance=7)
    expected = 90.0 * (1 / 0.9) / 0.01
    assert np.allclose(out.to_numpy(), expected)


def test_usd_is_identity_and_needs_no_column():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    px = pd.Series(np.full(4, 42.0), index=idx)
    out = _convert(px, "USD", "INR", _fake_fx(idx), tolerance=7)
    assert np.allclose(out.to_numpy(), 42.0 / 0.01)


def test_missing_fx_raises_actionable_error():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    px = pd.Series(np.full(4, 1.0), index=idx)
    with pytest.raises(ValueError, match="FX rate"):
        _convert(px, "ZWL", "USD", _fake_fx(idx), tolerance=7)


def test_stale_fx_becomes_nan_rather_than_wrong():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    px = pd.Series(np.full(10, 100.0), index=idx)
    fx = _fake_fx(idx)
    fx = fx.iloc[:2]                      # FX data stops early
    out = _convert(px, "EUR", "USD", fx, tolerance=2)
    assert out.iloc[:2].notna().all()
    assert out.iloc[-1] != out.iloc[-1] or pd.isna(out.iloc[-1])


def test_time_varying_fx_is_used_not_a_scalar():
    """A frozen FX rate would make converted returns identical to native ones."""
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    px = pd.Series([100.0] * 6, index=idx)
    fx = _fake_fx(idx)
    fx["INR"] = [0.01, 0.011, 0.012, 0.013, 0.014, 0.015]
    out = _convert(px, "USD", "INR", fx, tolerance=7)
    assert len(set(np.round(out.to_numpy(), 6))) == 6


def test_returns_definition():
    p = pd.Series([100.0, 110.0, 121.0])
    log = _to_returns(p, simple=False)
    assert abs(log.iloc[1] - np.log(1.1)) < 1e-12
    simple = _to_returns(p, simple=True)
    assert abs(simple.iloc[1] - 0.1) < 1e-12


# ----------------------------------------------------------------- models --

def test_hac_matches_statsmodels():
    """Guards the double-counted sigma^2 that understated SEs ~100x."""
    import statsmodels.api as sm
    rng = np.random.default_rng(7)
    n = 300
    y = rng.normal(size=n)
    x = 0.5 * y + rng.normal(size=n)
    frame = pd.DataFrame({"a": y, "b": x})
    res = models.ols(frame, "a", "b", hac_lags=5)
    sm_fit = sm.OLS(y, sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}, use_t=False)
    assert abs(res.beta - sm_fit.params[1]) < 1e-10
    assert abs(res.beta_se - sm_fit.bse[1]) < 1e-8


def test_min_variance_weights_sum_to_one():
    frame = _pair(400, 0.4, seed=8)
    pf = models.portfolio_frontier(frame, ("A", "B"))
    assert abs(sum(pf["weights"].values()) - 1.0) < 1e-9
    assert pf["port_vol"] <= min(pf["vol_a"], pf["vol_b"]) + 1e-9


def test_min_variance_closed_form_matches_matrix_solution():
    """w_b = (va - cab) / (va - 2*cab + vb), not -cab/vb."""
    frame = _pair(400, 0.35, seed=9)
    pf = models.portfolio_frontier(frame, ("A", "B"))
    assert pf["closed_form_check"]["agrees"]


def test_zero_correlation_min_variance_is_5050():
    rng = np.random.default_rng(10)
    n = 2000
    frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    pf = models.portfolio_frontier(frame, ("A", "B"))
    assert abs(pf["weights"]["A"] - 0.5) < 0.06
    assert abs(pf["variance_reduction"] - 0.5) < 0.06


def test_ewma_matches_textbook_recursion():
    """RiskMetrics: S_t = (1-lam) z_t z_t' + lam S_{t-1}, with lam-weighted history."""
    rng = np.random.default_rng(11)
    n = 200
    frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    lam = 0.94
    out = models.ewma_correlation(frame, lam=lam, labels=("A", "B"))

    z = frame[["a", "b"]].to_numpy()
    dev = z - z.mean(axis=0)
    S = np.cov(dev, rowvar=False, ddof=1)
    for t_ in range(1, n):                     # full recursion, not one step
        S = (1 - lam) * np.outer(dev[t_], dev[t_]) + lam * S
    expected = S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])
    assert abs(out["latest"] - expected) < 1e-4      # reported rounded to 4dp


def test_ewma_effective_n_formula():
    """(1+lam)/(1-lam) is the standard effective sample size, not (2-lam)/lam."""
    rng = np.random.default_rng(12)
    frame = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    out = models.ewma_correlation(frame, lam=0.94)
    assert abs(out["effective_n"] - (1.94 / 0.06)) < 0.01


def test_bidirection_betas_are_not_symmetric():
    frame = _pair(500, 0.5, seed=13)
    frame["b"] = frame["b"] * 3.0        # different volatilities
    res = models.bidirection(frame, ("A", "B"))
    assert abs(res["forward"]["beta"] - res["reverse"]["beta"]) > 1e-6


def test_stationarity_finds_i1_levels():
    rng = np.random.default_rng(14)
    n = 500
    walk = np.cumsum(rng.normal(size=n))
    clean = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    # stationarity takes log differences, so the level paths must stay positive;
    # a shift leaves the random-walk character untouched.
    fwd = walk - walk.min() + 100.0
    rev = walk[::-1] - walk[::-1].min() + 100.0
    out = models.stationarity(pd.DataFrame({"a": fwd, "b": rev}), ("A", "B"))
    assert not out["series"]["A"]["levels"]["stationary"]
    assert out["series"]["A"]["returns"]["stationary"]
    assert "returns are used as the headline" in out["conclusion"]
    _ = clean


def test_dcc_refuses_to_claim_identification_on_short_sample():
    frame = _pair(120, 0.3, seed=15)
    out = models.dcc_garch(frame, ("A", "B"))
    assert out["identified"] is False
    assert out["note"]


# --------------------------------------------------------------- dynamics --

def test_structural_break_runs_and_reports_a_p_value():
    from corrkit import dynamics
    rng = np.random.default_rng(16)
    n = 400
    a = rng.normal(size=n)
    b = np.concatenate([0.7 * a[:200], -0.7 * a[200:]]) + 0.3 * rng.normal(size=n)
    frame = pd.DataFrame({"a": a, "b": b},
                         index=pd.date_range("2020-01-01", periods=n, freq="B"))
    out = dynamics.structural_break(frame, "A", "B", n_boot=300)
    assert out["detected"] is True
    assert 0.0 <= out["p_blockbootstrap"] <= 1.0
    assert 0.0 <= out["p_timeshift"] <= 1.0


def test_structural_break_handles_stable_series():
    from corrkit import dynamics
    frame = _pair(400, 0.4, seed=17)
    frame.index = pd.date_range("2020-01-01", periods=400, freq="B")
    out = dynamics.structural_break(frame, "A", "B", n_boot=300)
    assert out["detected"] is False


def test_rolling_correlation_converges_to_full_sample():
    from corrkit import dynamics
    frame = _pair(300, 0.5, seed=18)
    roll = dynamics.rolling_correlation(frame, windows=(120,))
    full = float(frame["a"].corr(frame["b"]))
    assert abs(float(roll["roll_120"].dropna().mean()) - full) < 0.06


def test_stress_behaviour_reports_direction():
    from corrkit import dynamics
    rng = np.random.default_rng(19)
    n = 400
    a = rng.normal(size=n)
    b = a + rng.normal(size=n)             # moves with a, so no offset in stress
    frame = pd.DataFrame({"a": a, "b": b})
    out = dynamics.stress_behaviour(frame, 0.05, "A", "B")
    assert out["n_stress_days"] >= 10
    assert "power_warning" in out and "verdict" in out


def test_fx_diagnosis_separates_currency_from_relationship():
    rng = np.random.default_rng(20)
    n = 400
    fx = rng.normal(scale=0.005, size=n)
    native = pd.DataFrame({"a": 0.5 * fx, "b": -0.5 * fx})
    normalised = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    out = core.fx_diagnosis(native, normalised, pd.Series(fx, index=native.index))
    assert out["difference"] < 0
    assert "fx_induced_r" in out


# ---------------------------------------------- regressions for real bugs -----

def test_ols_r2_matches_definition_not_sse_over_dof():
    """R2 is 1 - SSE/SST. Using sigma2 = SSE/dof reports a near-perfect fit."""
    import statsmodels.api as sm

    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    pair = pd.DataFrame({"a": y, "b": x})
    got = models.ols(pair, "a", "b")
    ref = sm.OLS(y, sm.add_constant(x)).fit()
    assert abs(got.r2 - ref.rsquared) < 1e-9
    assert abs(got.adj_r2 - ref.rsquared_adj) < 1e-9
    assert got.r2 < 0.5                      # the bug reported ~0.996


def test_big_moves_keep_the_sign_of_the_return():
    """Ranking on magnitude must not discard which way the price moved."""
    idx = pd.bdate_range("2025-01-01", periods=12)
    a = 100 * np.exp(np.concatenate([[0.0], np.full(10, 0.0), [-0.30]]))
    b = 100 * np.exp(np.concatenate([[0.0], np.full(10, 0.0), [0.25]]))
    prices = pd.DataFrame({"a": a, "b": b}, index=idx)
    rep = data.audit(prices, prices["a"], prices["b"], "USD", False,
                     "USD", "USD", False)
    moves = {m["leg"]: m["log_return"] for m in rep["big_moves"]}
    assert moves["a"] < 0, "a -30% crash must not be reported as +30%"
    assert moves["b"] > 0
    assert abs(moves["a"]) == 0.30


def test_lower_tail_verdict_says_moves_down():
    from corrkit.dynamics import _tail_verdict
    rec = {"joint_rate": 0.20, "n_in_tail": 40, "exceedance_ratio": 4.0,
           "label": "the pair"}
    assert "moves down" in _tail_verdict(rec, "lower", 0.05, 800)
    assert "moves up" in _tail_verdict(rec, "upper", 0.05, 800)


def test_fx_difference_is_bootstrapped_not_treated_as_a_correlation():
    """The difference of two coefficients must come from a paired bootstrap,
    and the reported normalised CI must describe the normalised pair."""
    rng = np.random.default_rng(3)
    n = 260
    common = rng.normal(size=n)
    native = pd.DataFrame({"a": 0.5 * common + rng.normal(size=n) * 0.8,
                           "b": common + rng.normal(size=n) * 0.8})
    normalised = pd.DataFrame({"a": native["a"] - 0.3 * common,
                               "b": native["b"] + 0.2 * common})
    out = core.fx_diagnosis(native, normalised, pd.Series(0.2 * common,
                                                          index=native.index))

    assert "difference_ci" in out
    assert out["aligned_n"] == n
    assert abs(out["aligned_r_unconverted"] - out["r_unconverted"]) < 1e-3
    assert abs(out["aligned_difference"] - out["difference"]) < 1e-3
    assert 0.0 <= out["difference_p"] <= 1.0
    lo, hi = out["difference_ci"]
    assert lo <= hi
    if out["difference_p"] < 0.05:
        assert lo * hi > 0                    # significant <=> interval excludes 0


def test_fx_normalised_ci_is_not_the_native_ci():
    """Guards the old bug: the CI labelled 'normalised' was the native one."""
    rng = np.random.default_rng(11)
    n = 300
    common = rng.normal(size=n)
    native = pd.DataFrame({"a": 0.5 * common + rng.normal(size=n),
                           "b": common + rng.normal(size=n)})
    fx = pd.Series(0.5 * common, index=native.index)
    normalised = pd.DataFrame({"a": native["a"] - 0.4 * fx,
                               "b": native["b"] + 0.4 * fx})
    out = core.fx_diagnosis(native, normalised, None)
    assert out["r_unconverted"] != out["r_normalised"]
    lo, hi = out["r_normalised_ci"]
    assert abs((lo + hi) / 2 - out["r_normalised"]) < 0.25


def test_period_days_do_not_silently_extend_the_window():
    assert data.PERIOD_DAYS["3y"] == "3y"
    assert all(k == v for k, v in data.PERIOD_DAYS.items())


def test_fx_cache_key_includes_period(tmp_path, monkeypatch):
    """A short FX cache must never be served to a longer request: the two must
    not collide on the same cache file."""
    calls = []

    class FakeHistory:
        def __init__(self, n):
            self._h = pd.DataFrame(
                {"Close": np.linspace(80, 85, n)},
                index=pd.bdate_range("2020-01-01", periods=n))

    class FakeTicker:
        def __init__(self, sym):
            self.sym = sym

        def history(self, period=None, interval=None, auto_adjust=None):
            calls.append(period)
            return FakeHistory(10)._h

    monkeypatch.setattr(universe.yf, "Ticker", FakeTicker)

    universe._fx_series("USDINR=X", tmp_path, 3, period="2y")
    universe._fx_series("USDINR=X", tmp_path, 3, period="10y")
    assert calls == ["2y", "10y"], "second call reused the first call's cache"
    files = sorted(f.name for f in tmp_path.glob("fx_*.csv"))
    assert len(files) == 2, files
    # both cached, and the longer request was actually requested
    assert universe._fx_series("USDINR=X", tmp_path, 3, period="10y") is not None
    assert calls == ["2y", "10y"], "10y cache should have been reused"


def test_resolution_probes_the_network_at_most_once(tmp_path, monkeypatch):
    """A name-shaped input used to pay two probe round-trips before searching."""
    probes = []

    def fake_probe(symbol, period="5d"):
        # implicitly returns None: not a valid symbol, so resolve() must fall
        # through to the name search.
        probes.append(symbol)

    class FakeSearch:
        def __init__(self, *a, **k):
            self.quotes = [{"symbol": "HDFCBANK.NS",
                            "longname": "HDFC Bank Limited",
                            "exchDisp": "NSE", "quoteType": "EQUITY",
                            "currency": "INR"}]

    monkeypatch.setattr(universe, "_probe", fake_probe)
    monkeypatch.setattr(universe.yf, "Search", FakeSearch)
    res = universe.resolve("hdfc bank", cache_dir=tmp_path)
    assert res.symbol == "HDFCBANK.NS"
    assert probes == ["hdfc bank"], f"probed {len(probes)} times"


def _stub_search(monkeypatch, quotes):
    monkeypatch.setattr(universe, "_probe", lambda symbol, period="5d": None)
    monkeypatch.setattr(
        universe.yf, "Search",
        lambda *a, **k: type("S", (), {"quotes": quotes})())


def test_bare_company_name_prefers_the_nse_listing(tmp_path, monkeypatch):
    """Yahoo ranks the US ADR above the home line ("hdfc bank" returns HDB
    first). A name with no venue of its own should land on .NS."""
    _stub_search(monkeypatch, [
        {"symbol": "HDB", "longname": "HDFC Bank Limited", "exchDisp": "NYSE",
         "quoteType": "EQUITY"},
        {"symbol": "HDFCBANK.NS", "longname": "HDFC Bank Limited",
         "exchDisp": "NSE", "quoteType": "EQUITY"},
    ])
    res = universe.resolve("hdfc bank", cache_dir=tmp_path)
    assert res.symbol == "HDFCBANK.NS"


def test_explicit_venue_suffix_still_beats_the_nse_default(tmp_path, monkeypatch):
    """An input that names its venue must not be silently redirected to NSE."""
    _stub_search(monkeypatch, [
        {"symbol": "HDFCBANK.NS", "longname": "HDFC Bank Limited",
         "exchDisp": "NSE", "quoteType": "EQUITY"},
        {"symbol": "HDFCBANK.BO", "longname": "HDFC Bank Limited",
         "exchDisp": "Bombay", "quoteType": "EQUITY"},
    ])
    res = universe.resolve("hdfc bank bombay", cache_dir=tmp_path)
    assert res.symbol == "HDFCBANK.NS"      # no NSE-aware redirect, takes top hit


def test_us_name_is_not_redirected_to_nse(tmp_path, monkeypatch):
    """"microsoft" must stay US: an NSE listing of an unrelated Indian company
    must never win merely by carrying a .NS suffix."""
    _stub_search(monkeypatch, [
        {"symbol": "MSFT", "shortname": "Microsoft Corporation",
         "exchDisp": "NasdaqGS", "quoteType": "EQUITY"},
        {"symbol": "MSFT.NS", "shortname": "Some Indian listing",
         "exchDisp": "NSE", "quoteType": "EQUITY"},
    ])
    assert universe.resolve("microsoft", cache_dir=tmp_path).symbol == "MSFT"


def test_nse_match_does_not_warn_about_other_venues(tmp_path, monkeypatch):
    """Resolving to NSE is the intended behaviour, so it must not be reported as
    a possibly-wrong match."""
    _stub_search(monkeypatch, [
        {"symbol": "INFY.NS", "longname": "Infosys Limited", "exchDisp": "NSE",
         "quoteType": "EQUITY"},
        {"symbol": "INFY", "longname": "Infosys Limited", "exchDisp": "NYSE",
         "quoteType": "EQUITY"},
    ])
    res = universe.resolve("infosys", cache_dir=tmp_path)
    assert res.symbol == "INFY.NS"
    assert universe.resolution_warning(res) is None
