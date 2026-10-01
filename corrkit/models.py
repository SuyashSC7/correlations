"""
Model-based companions to the raw correlation.

Correlation is symmetric and scale-free, which hides two things worth knowing:
which asset drives which (regression betas are not symmetric), and how much
diversification is actually available (a covariance matrix, not a correlation).

The GARCH/DCC section exists mainly to be honest about when it cannot deliver.
Conditional correlation models need far more data than a year of daily prices
provides, and reporting a number anyway would be worse than reporting the
failure - so identification diagnostics gate the output.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller

from .core import corr_test

TRADING_DAYS = {"1d": 252, "1wk": 52, "1mo": 12}


# ------------------------------------------------------------ stationarity --

def stationarity(prices: pd.DataFrame, labels: tuple[str, str] = ("A", "B")) -> dict:
    """
    ADF tests on price levels and returns.

    The point is not the p-values themselves but the contrast: if levels fail to
    reject a unit root while returns reject it strongly, then a "significant"
    level correlation is a shared trend, not a relationship.
    """
    out: dict = {"series": {}, "conclusion": None}
    levels_reject, returns_reject = 0, 0
    counted = 0

    for col, lab in zip(prices.columns, labels):
        rec: dict = {}
        s = prices[col].dropna()
        try:
            st = adfuller(s, autolag="AIC", result_object=False)
            rec["levels"] = {"adf": round(float(st[0]), 4),
                             "p": round(float(st[1]), 4),
                             "stationary": bool(st[1] < 0.05)}
            levels_reject += int(st[1] < 0.05)
        except Exception as exc:
            rec["levels"] = {"error": f"{type(exc).__name__}: {exc}"}

        r = np.log(s).diff().dropna()
        try:
            st = adfuller(r, autolag="AIC", result_object=False)
            rec["returns"] = {"adf": round(float(st[0]), 4),
                              "p": round(float(st[1]), 4),
                              "stationary": bool(st[1] < 0.05)}
            returns_reject += int(st[1] < 0.05)
        except Exception as exc:
            rec["returns"] = {"error": f"{type(exc).__name__}: {exc}"}
        counted += 1
        out["series"][lab] = rec

    if counted:
        out["conclusion"] = (
            f"Levels: {levels_reject}/{counted} reject a unit root. "
            f"Returns: {returns_reject}/{counted} reject. "
            + ("Since levels are non-stationary, correlation between price levels is "
               "not interpretable as co-movement - only returns are used as the "
               "headline."
               if levels_reject < counted else
               "Levels appear stationary here, though this does not by itself make a "
               "level correlation a co-movement measure."))
    return out


# --------------------------------------------------------------- regression --

@dataclass
class RegressionResult:
    dependent: str
    independent: str
    alpha: float
    alpha_p: float
    beta: float
    beta_se: float
    beta_p: float
    r2: float
    adj_r2: float
    n: int
    hac_lags: int
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def ols(returns: pd.DataFrame, dependent: str, independent: str, *,
        hac_lags: int | None = None, per_year: int = 252,
        label_dependent: str = "A", label_independent: str = "B") -> RegressionResult:
    """
    OLS with Newey-West standard errors.

    HAC errors are the minimum for return data; ordinary OLS errors assume the
    shocks are independent across days, which they are not.
    """
    pair = returns.dropna()
    y = pair[dependent].to_numpy()
    X = np.column_stack([np.ones(len(pair)), pair[independent].to_numpy()])
    n, k = X.shape

    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat
    dof = n - k
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    xtx_inv = np.linalg.inv(X.T @ X)
    # R2 is 1 - SSE/SST, so the residual variance must be rescaled by dof here;
    # using sigma2 = SSE/dof directly would report a near-perfect fit.
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / dof

    lags = hac_lags if hac_lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    # S already carries the squared residuals (X'diag(e^2)X), so no extra sigma^2
    # factor belongs here - doing both understates the SE by ~sqrt(sigma^2).
    S = _hac_cov(X, resid, lags)
    cov = xtx_inv @ S @ xtx_inv
    cov = (cov + cov.T) / 2
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_a = beta_hat[0] / se[0] if se[0] > 0 else np.nan
        t_b = beta_hat[1] / se[1] if se[1] > 0 else np.nan
    from scipy import stats as _st
    p_a = float(2 * _st.t.sf(abs(t_a), dof)) if np.isfinite(t_a) else np.nan
    p_b = float(2 * _st.t.sf(abs(t_b), dof)) if np.isfinite(t_b) else np.nan

    return RegressionResult(
        dependent=label_dependent, independent=label_independent,
        alpha=float(beta_hat[0]), alpha_p=p_a,
        beta=float(beta_hat[1]), beta_se=float(se[1]), beta_p=p_b,
        r2=float(r2), adj_r2=float(adj), n=n, hac_lags=lags,
        notes=[(
            f"Newey-West lags={lags}; beta is a return-on-return sensitivity - "
            f"a 1% move in {label_independent} goes with {beta_hat[1]:+.4f}% in "
            f"{label_dependent}. It is unitless and unchanged by annualising "
            f"the horizon."
        )],
    )


def _hac_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    n, _ = X.shape
    u = resid[:, None] * X
    S = u.T @ u
    for l in range(1, min(lags, n - 1) + 1):
        w = 1 - l / (lags + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    return S


def bidirection(returns: pd.DataFrame, labels: tuple[str, str]) -> dict:
    """
    Regress each leg on the other.

    Correlation is symmetric; betas are not. Reporting only one direction invites
    the reader to treat it as a structural relationship when it is not.
    """
    fwd = ols(returns, "a", "b", label_dependent=labels[0], label_independent=labels[1])
    rev = ols(returns, "b", "a", label_dependent=labels[1], label_independent=labels[0])
    return {
        "forward": fwd.as_dict(),
        "reverse": rev.as_dict(),
        "note": (f"{labels[0]} beta on {labels[1]} = {fwd.beta:+.4f}, but "
                 f"{labels[1]} beta on {labels[0]} = {rev.beta:+.4f}. "
                 f"Correlation is symmetric by construction; regression betas are "
                 f"not, so the choice of direction reflects volatility, not causation."),
    }


# ---------------------------------------------------------------- portfolio --

def portfolio_frontier(returns: pd.DataFrame, labels: tuple[str, str],
                       per_year: int = 252) -> dict:
    """
    Minimum-variance weights and what diversification is actually worth.

    For two assets the minimum-variance weight has a closed form,
    w_b = -cov(a,b) / var(b), which is a useful check on the matrix result.
    """
    pair = returns.dropna()
    X = pair.to_numpy()
    cov = np.cov(X, rowvar=False) * per_year
    vols = np.sqrt(np.diag(cov))
    inv = np.linalg.inv(cov)
    ones = np.ones(2)
    w = inv @ ones / (ones @ inv @ ones)

    port_var = float(w @ cov @ w)
    port_vol = float(np.sqrt(port_var))
    r = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])

    closed = float((cov[0, 0] - cov[0, 1]) / (cov[0, 0] - 2 * cov[0, 1] + cov[1, 1]))
    grid = []
    for b in np.linspace(-0.5, 1.5, 201):
        ww = np.array([1 - b, b])
        grid.append(float(np.sqrt(max(ww @ cov @ ww, 0))))
    out = {
        "weights": {labels[0]: round(float(w[0]), 4), labels[1]: round(float(w[1]), 4)},
        "port_var": round(port_var, 6),
        "port_vol": round(port_vol, 6),
        "vol_a": round(float(vols[0]), 6), "vol_b": round(float(vols[1]), 6),
        "variance_reduction": round(1 - port_var / float(vols[0] ** 2), 4),
        "variance_reduction_vs_b": round(1 - port_var / float(vols[1] ** 2), 4),
        "r": round(r, 4),
        "short_allowed": bool((w < 0).any()),
        "grid_vol": grid,
        "closed_form_check": {
            "w_b_closed_form": round(closed, 6),
            "w_b_matrix": round(float(w[1]), 6),
            "agrees": bool(abs(closed - w[1]) < 1e-6),
        },
    }
    notes = []
    if abs(r) < 0.3:
        notes.append(
            f"At r={r:+.2f} the diversification benefit is limited: the "
            f"minimum-variance mix cuts {labels[0]}'s variance by only "
            f"{out['variance_reduction']:.1%}.")
    else:
        notes.append(
            f"At r={r:+.2f} the legs move together, so holding both is an effective "
            f"levered bet on the same factor; minimum-variance still trims variance "
            f"by {out['variance_reduction']:.1%} relative to {labels[0]} alone.")
    if out["short_allowed"]:
        notes.append(
            f"The unconstrained optimum requires a short position in "
            f"{labels[1]}; a long-only mandate cannot reach it.")
    notes.append(
        "Weights assume the sample covariance persists. Minimum-variance "
        "estimates are notoriously unstable, so this is a sample-specific "
        "number, not a recommended allocation.")
    out["notes"] = notes
    return out


# ------------------------------------------------- conditional correlation --

def ewma_correlation(returns: pd.DataFrame, lam: float = 0.94,
                     labels: tuple[str, str] = ("A", "B")) -> dict:
    """
    RiskMetrics EWMA correlation, with a fit for purpose diagnostic.

    An EWMA correlation is not an estimate of a population parameter - it is a
    deterministic function of the path taken, with an effective sample size that
    collapses as lambda rises. Reporting it without that number invites reading
    noise as signal.
    """
    pair = returns.dropna()
    X = pair.to_numpy()
    n = len(X)
    mean = X.mean(axis=0)
    dev = X - mean

    # Recurse on the full 2x2 covariance, not variances alone: the covariance
    # term is what carries the EWMA correlation.
    S = np.cov(dev, rowvar=False, ddof=1)
    path = np.empty(n)
    path[0] = (S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])
               if S[0, 0] > 0 and S[1, 1] > 0 else np.nan)
    for t in range(1, n):
        outer = np.outer(dev[t], dev[t])
        S = (1 - lam) * outer + lam * S
        den = np.sqrt(S[0, 0] * S[1, 1])
        path[t] = S[0, 1] / den if den > 0 else np.nan

    n_eff = (1 + lam) / (1 - lam)
    series = pd.Series(path, index=pair.index, name="ewma")
    raw_r = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    out = {
        "lambda": lam,
        "effective_n": round(float(n_eff), 3),
        "latest": round(float(series.iloc[-1]), 4) if len(series) else None,
        "mean": round(float(series.mean()), 4),
        "std": round(float(series.std(ddof=1)), 4) if len(series) > 2 else None,
        "min": round(float(series.min()), 4), "max": round(float(series.max()), 4),
        "series": series,
        "raw_estimate": round(raw_r, 4),
    }
    if n_eff < 10:
        out["usable"] = False
        out["note"] = (
            f"At lambda={lam} the EWMA correlation carries an effective sample size "
            f"of about {n_eff:.1f} observations. For a *pair* the effective size for "
            f"the correlation itself is smaller still. Values near +/-1 are a "
            f"mechanical consequence of the smoothing, not evidence of anything. "
            f"Reported for completeness; not used for inference.")
    else:
        out["usable"] = True
        out["note"] = (f"Effective sample size ~{n_eff:.1f}; still thin for a pair.")
    return out


def univariate_garch(returns: pd.DataFrame, labels: tuple[str, str]) -> dict:
    """Fit GARCH(1,1) to each leg separately and report the parameters."""
    out: dict = {"fitted": [], "note": None}
    for col, lab in zip(("a", "b"), labels):
        s = returns[col].dropna() * 100.0
        try:
            from arch import arch_model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = arch_model(s, vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(
                    disp="off", show_warning=False)
            pv = res.params
            persistence = float(pv.get("alpha[1]", np.nan) + pv.get("beta[1]", np.nan))
            out["fitted"].append({
                "label": lab, "converged": bool(res.convergence_flag == 0),
                "omega": round(float(pv.get("omega", np.nan)), 6),
                "alpha": round(float(pv.get("alpha[1]", np.nan)), 6),
                "beta": round(float(pv.get("beta[1]", np.nan)), 6),
                "persistence": round(persistence, 4),
                "half_life_days": round(math.log(0.5) / math.log(persistence), 1)
                if 0 < persistence < 1 else None,
                "df": round(float(pv.get("nu", np.nan)), 2),
            })
        except Exception as exc:
            out["fitted"].append({"label": lab, "error": f"{type(exc).__name__}: {exc}"})
    return out


def dcc_garch(returns: pd.DataFrame, labels: tuple[str, str],
              n_starts: int = 8) -> dict:
    """
    Two-step DCC(1,1) with explicit identification diagnostics.

    Step 1 fits univariate GARCH per leg; step 2 maximises the DCC
    quasi-likelihood over (a, b). The point of the diagnostics is that this
    frequently fails on short samples: when b collapses towards zero the
    correlation path is pushed to +/-1 and the model is really just a sign
    model dressed up as a conditional correlation. That failure is reported
    rather than hidden.
    """
    pair = returns.dropna()
    n = len(pair)
    res: dict = {
        "n": n, "params": None, "correlation_path": None,
        "identified": False, "diagnostics": {}, "note": None,
        "garch": univariate_garch(pair, labels),
    }
    if n < 150:
        res["note"] = (f"n={n} is far too short to attempt DCC(1,1) reliably.")
        return res
    if n < 500:
        res["pre_note"] = (f"n={n} is below the ~500 observations DCC(1,1) is usually "
                           f"given. Estimation proceeds so the failure mode is visible, "
                           f"but parameters should be treated as unidentified unless the "
                           f"diagnostics pass.")

    std = pair.to_numpy()
    s = pair.std(axis=0).to_numpy()
    if not np.all(s > 0):
        res["note"] = "A leg has zero variance; DCC is undefined."
        return res
    z = std / s

    def nll(theta: np.ndarray) -> float:
        a, b = theta
        if not (0 <= a < 1) or not (0 <= b < 1) or a + b >= 0.9999:
            return 1e6
        q = np.empty((n, 2, 2))
        q[0] = np.eye(2)
        for t in range(1, n):
            qt = (1 - a - b) * np.outer(z[t - 1], z[t - 1]) + a * q[t - 1]
            q[t] = qt + b * np.eye(2)
        R = np.empty((n, 2, 2))
        for t in range(n):
            d = np.sqrt(np.clip(np.diag(q[t]), 1e-12, None))
            R[t] = q[t] / np.outer(d, d)
            R[t] = (R[t] + R[t].T) / 2
            np.fill_diagonal(R[t], 1.0)
        ll = 0.0
        for t in range(n):
            try:
                Li = np.linalg.cholesky(R[t])
                sol = np.linalg.solve(Li, z[t])
                ll += np.log(np.diag(Li)).sum() + 0.5 * sol @ sol
            except np.linalg.LinAlgError:
                return 1e6
        return -ll

    bounds = [(1e-6, 0.4), (1e-6, 0.995)]
    best, best_val = None, np.inf
    for k in range(n_starts):
        rng = np.random.default_rng(1000 + k)
        x0 = [rng.uniform(*bounds[0]), rng.uniform(0.5, 0.98)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
            if o.fun < best_val:
                best_val, best = float(o.fun), o
        except Exception:
            continue

    if best is None:
        res["note"] = "DCC optimisation failed from every starting point."
        return res

    a, b = float(best.x[0]), float(best.x[1])
    R = np.empty((n, 2, 2))
    q = np.eye(2)
    for t in range(n):
        if t > 0:
            q = (1 - a - b) * np.outer(z[t - 1], z[t - 1]) + a * q + b * np.eye(2)
        d = np.sqrt(np.clip(np.diag(q), 1e-12, None))
        Rt = q / np.outer(d, d)
        R[t] = (Rt + Rt.T) / 2
        np.fill_diagonal(R[t], 1.0)
    corr_path = R[:, 0, 1]
    eig = np.array([np.linalg.eigvalsh(R[t]) for t in range(n)])

    diag = {
        "a": round(a, 6), "b": round(b, 6), "a_plus_b": round(a + b, 6),
        "corr_mean": round(float(np.mean(corr_path)), 4),
        "corr_std": round(float(np.std(corr_path, ddof=1)), 4),
        "corr_min": round(float(np.min(corr_path)), 4),
        "corr_max": round(float(np.max(corr_path)), 4),
        "min_eigenvalue": round(float(np.min(eig[:, 0])), 6),
        "optimiser_converged": bool(best.success),
        "at_boundary": bool(a <= bounds[0][1] * 1.02 or b >= bounds[0][1] * 0.98),
    }
    res["params"] = {"a": a, "b": b, "loglik": -best_val}
    res["correlation_path"] = pd.Series(
        corr_path, index=pair.index, name="dcc")
    res["diagnostics"] = diag

    # Identification tests.
    reasons = []
    if b < 0.05:
        reasons.append(f"b={b:.4f} is near zero, so the correlation equation collapses "
                       f"to a sign model and rho is pushed to the boundary (+/-1)")
    if abs(corr_path).max() > 0.995:
        reasons.append(f"correlation reaches {corr_path[np.argmax(np.abs(corr_path))]:.4f}, "
                       f"indicating a boundary solution rather than an estimate")
    if float(np.min(eig[:, 0])) < 1e-4:
        reasons.append("conditional correlation matrices are numerically singular")
    if n < 500:
        reasons.append(f"n={n} is short for a two-parameter dynamic correlation model")

    res["identified"] = not reasons
    if reasons:
        res["note"] = (
            "DCC(1,1) is **not identified** on this sample: "
            + "; ".join(reasons)
            + ". The parameters are reported as a diagnostic, not an estimate. Use the "
              "rolling window correlation instead.")
    else:
        res["note"] = (f"DCC(1,1) fitted: a={a:.4f}, b={b:.4f}, mean conditional "
                       f"correlation {np.mean(corr_path):+.3f}.")
    return res


def rolling_diagnostics(returns: pd.DataFrame, windows=(60, 90)) -> dict:
    """Cross-check DCC/EWMA against plain rolling windows."""
    out: dict = {}
    for w in windows:
        s = returns["a"].rolling(w).corr(returns["b"]).dropna()
        if len(s) < 10:
            continue
        _, p, _ = corr_test(float(s.iloc[-1]), len(s))
        out[f"roll_{w}"] = {
            "latest": round(float(s.iloc[-1]), 4),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std(ddof=1)), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "p_at_latest": round(p, 4),
            "n": len(s),
        }
    return out
