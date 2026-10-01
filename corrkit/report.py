"""
Rendering: markdown report, figures and machine-readable output.

Every claim written here is read off the result tree rather than recomputed, so
the prose cannot drift away from the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import Dataset

NAVY = "#1f3a5f"
GOLD = "#b8860b"
RED = "#a4262c"
GREEN = "#1a6b4a"
GREY = "#6b7280"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


# ------------------------------------------------------------------ markdown --

def _num(x: float | None, dp: int = 2) -> str:
    """Fixed-point float that tolerates None and NaN (model output can carry
    either, e.g. a half-life that does not exist because persistence >= 1)."""
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(v) else f"{v:.{dp}f}"


def _pct(x: float, dp: int = 2, signed: bool = True) -> str:
    """Percentage, sign-optional. Volatility is a magnitude, so it reads better
    unsigned than as a +13.6% 'gain'."""
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x*100:+.{dp}f}%" if signed else f"{x*100:.{dp}f}%"


def _f(x, dp: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(v) else f"{v:+.{dp}f}"


def _sig(p: float, alpha: float) -> str:
    return "yes" if (np.isfinite(p) and p < alpha) else "no"


def render_markdown(res: dict) -> str:
    m, c = res["meta"], res["core"]
    alpha = 1 - m["conf"]
    sa, sb = m["symbols"]
    L: list[str] = []
    A = L.append

    A(f"# Correlation: {sa} vs {sb}")
    A("")
    A(f"> **{res['conclusion']['headline']}**")
    A("")

    # ---------------------------------------------------------- resolution --
    A("## What was compared")
    A("")
    A(f"Input `{m['query'][0]}` resolved to `{sa}` and `{m['query'][1]}` to `{sb}`.")
    A("")
    A("| | A | B |")
    A("|---|---|---|")
    A(f"| Input | `{m['query'][0]}` | `{m['query'][1]}` |")
    A(f"| Resolved | `{sa}` | `{sb}` |")
    A(f"| Matched | {m['resolved'][0]} | {m['resolved'][1]} |")
    A(f"| Quoted in | {m['currency_a']} | {m['currency_b']} |")
    A(f"| Resolution | {m['resolution_method'][0]} | {m['resolution_method'][1]} |")
    A("")
    for side, (cands, q) in enumerate(zip(m["candidates"], m["query"])):
        if cands and len(cands) > 1:
            A(f"Other matches for `{q}` (disambiguation matters here):")
            A("")
            for cd in cands[:8]:
                A(f"- `{cd['symbol']}` — {cd['name']} ({cd['exchange']}, "
                  f"{cd['quote_type']})")
            A("")

    A(f"Sample: **{m['start']} to {m['end']}** — {m['n_sessions']} aligned sessions, "
      f"{m['n_returns']} {m['return_type']} returns, "
      f"{m['period']} window at {m['freq']} frequency.")
    A("")
    if m["fx_conversion"][0] != "none" or m["fx_conversion"][1] != "none":
        A(f"Both legs restated in **{m['base_currency']}** "
          f"(A: {m['fx_conversion'][0]}; B: {m['fx_conversion'][1]}).")
        A("")
    for n in res["notes"]:
        A(f"- {n}")
    A("")

    # ------------------------------------------------------------- headline --
    A("## The number")
    A("")
    A("| Statistic | Value |")
    A("|---|---|")
    A(f"| Pearson r | **{_f(c.r)}** |")
    A(f"| 95% CI (Fisher z) | [{_f(c.ci_low)}, {_f(c.ci_high)}] |")
    A(f"| t-statistic | {c.t_stat:+.3f} |")
    A(f"| p-value | {c.p_value:.4f} ({_sig(c.p_value, alpha)} at {int(m['conf']*100)}%) |")
    A(f"| R² (variance shared) | {c.r2:.4f} |")
    A(f"| Spearman ρ | {_f(c.spearman_rho)} (p={c.spearman_p:.4f}) |")
    A(f"| Kendall τ | {_f(c.kendall_tau)} (p={c.kendall_p:.4f}) |")
    A(f"| Block-bootstrap 95% CI | [{_f(c.boot_ci[0])}, {_f(c.boot_ci[1])}] "
      f"(block={c.boot_block}) |")
    A(f"| Annualised vol A / B | {_pct(c.vol_a, signed=False)} / "
      f"{_pct(c.vol_b, signed=False)} |")
    A("")
    A(f"**{c.verdict}**")
    A("")
    if c.fragility:
        A("Reasons for caution:")
        A("")
        for f in c.fragility:
            A(f"- {f}")
        A("")

    # ------------------------------------------------------------ levels --
    lv = res["levels"]
    A("### Why not price levels?")
    A("")
    A(f"Correlating the price levels instead of returns gives r = "
      f"{lv['levels_r']:+.4f} (p = {lv['levels_p']:.4f}) versus "
      f"{lv['returns_r']:+.4f} for returns"
      + (" — **an opposite sign**." if lv["sign_flip"] else "."))
    A("")
    stn = res["stationarity"]
    concl = stn.get("conclusion")
    if concl:
        A(concl)
        A("")

    # ------------------------------------------------------------- fx --
    fx = res.get("fx") or {}
    if fx:
        A("### Currency effect")
        A("")
        A("| Measure | Value |")
        A("|---|---|")
        A(f"| Unconverted correlation | {fx['r_unconverted']:+.4f} |")
        A(f"| After normalising to {m['base_currency']} | {fx['r_normalised']:+.4f} |")
        A(f"| Difference (unconverted - normalised) | {fx['difference']:+.4f} |")
        if "fx_induced_r" in fx:
            A(f"| A vs FX rate | {fx['r_a_vs_fx']:+.4f} |")
            A(f"| B vs FX rate | {fx['r_b_vs_fx']:+.4f} |")
            A(f"| FX-implied component (product) | {fx['fx_induced_r']:+.4f} |")
        if "difference_ci" in fx:
            A(f"| Difference 95% CI (paired bootstrap) | "
              f"[{fx['difference_ci'][0]:+.3f}, {fx['difference_ci'][1]:+.3f}] |")
            p = fx.get("difference_p")
            if p is not None:
                sig = "excludes" if fx["difference_ci"][0] * fx["difference_ci"][1] > 0 \
                    else "includes"
                A(f"| Is the difference significant? | p = {p:.4f} ({sig} zero) |")
        A("")
        if fx.get("note"):
            A(fx["note"])
            A("")

    # -------------------------------------------------------- robustness --
    A("### Robustness")
    A("")
    A("| Check | n | r | p |")
    A("|---|---|---|---|")
    A(f"| Baseline | {c.n} | {c.r:+.4f} | {c.p_value:.4f} |")
    for name, chk in (res["sensitivity"]["checks"]).items():
        A(f"| {name.replace('_', ' ')} | {chk['n']} | {chk['r']:+.4f} | "
          f"{chk['p']:.4f} |")
    A("")
    if res["sensitivity"].get("outlier_dependence"):
        A(f"Dependence on extreme days: **{res['sensitivity']['outlier_dependence']}**.")
        A("")

    sp = res["core"].subperiods.get("slices", [])
    if sp:
        A("| Sub-period | Window | n | r | 95% CI |")
        A("|---|---|---|---|---|")
        for s in sp:
            A(f"| {s['slice']} | {s['start']} to {s['end']} | {s['n']} | {s['r']:+.4f} "
              f"| [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}] |")
        A("")

    # -------------------------------------------------------- dynamics --
    A("### Does it change over time?")
    A("")
    if res["rolling_summary"]:
        A("| Window | Mean | Std | Min | Max | Latest | Sign changes |")
        A("|---|---|---|---|---|---|---|")
        for w in res["rolling_summary"]:
            A(f"| {w['window']}d | {w['mean']:+.3f} | {w['std']:+.3f} | "
              f"{w['min']:+.3f} | {w['max']:+.3f} | {w['latest']:+.3f} | "
              f"{w['sign_changes']} |")
        A("")
    rg = res.get("regimes", {})
    if rg.get("available") and rg.get("regimes"):
        A("Correlation by volatility state:")
        A("")
        A("| Regime | n | r | 95% CI | Significant |")
        A("|---|---|---|---|---|")
        for r_ in rg["regimes"]:
            A(f"| {r_['regime']} vol | {r_['n']} | {r_['r']:+.4f} | "
              f"[{r_['ci_low']:+.3f}, {r_['ci_high']:+.3f}] | "
              f"{_sig(r_['p'], alpha)} |")
        A("")
        if rg.get("spread_note"):
            A(rg["spread_note"])
            A("")

    br = res.get("break", {})
    if br.get("note"):
        A("**Structural break test.** " + br["note"])
        A("")

    rd = res.get("rolling_diag", {})
    if rd:
        A("Plain rolling windows (the practical dynamic estimate):")
        A("")
        A("| Window | Latest | Mean | Std | Min | Max |")
        A("|---|---|---|---|---|---|")
        for k, v in rd.items():
            A(f"| {k.replace('roll_','')}d | {v['latest']:+.3f} | {v['mean']:+.3f} | "
              f"{v['std']:+.3f} | {v['min']:+.3f} | {v['max']:+.3f} |")
        A("")

    # ------------------------------------------------------------ tails --
    A("### Behaviour in the tails")
    A("")
    tl = res.get("tails", {})
    if tl.get("tails"):
        A("| Tail | Joint rate | Expected | Ratio | n in tail | r within tail |")
        A("|---|---|---|---|---|---|")
        for side in ("lower", "upper"):
            t = tl["tails"][side]
            A(f"| {side} | {t['joint_rate']:.1%} | {tl['q']:.0%} | "
              f"x{t['exceedance_ratio']} | {t['n_in_tail']} | "
              f"{t.get('r_in_tail', 'n/a')} |")
        A("")
        for side in ("lower", "upper"):
            A(f"- **{side}**: {tl['tails'][side]['verdict']}")
        A("")
        A(f"> {tl['power_warning']}")
        A("")

    st = res.get("stress", {})
    if st.get("n_stress_days"):
        A(f"**On {sa}'s worst {st['tail_q']:.0%} of days** ({st['n_stress_days']} "
          f"sessions, mean {st.get('mean_a_on_stress', 0):+.2%} for A):")
        A("")
        A("| Measure | Value |")
        A("|---|---|")
        A(f"| {sb} mean return | {_pct(st['mean_b_on_stress'])} |")
        A(f"| {sb} median return | {_pct(st['median_b_on_stress'])} |")
        A(f"| {sb} worst | {_pct(st['worst_b_on_stress'])} |")
        A(f"| {sb} positive rate | {_pct(st['b_positive_rate_on_stress'], 1)} |")
        A(f"| {sb} mean, normal days | {_pct(st['mean_b_normal'])} |")
        if "p_value" in st:
            A(f"| Difference vs normal days | p = {st['p_value']:.4f} |")
        A("")
        A(st["verdict"])
        A("")

    # -------------------------------------------------------- regression --
    reg = res.get("regression")
    if reg:
        A("### Regression")
        A("")
        A("| Model | alpha (%/day) | beta | HAC se | p | R² |")
        A("|---|---|---|---|---|---|")
        for key in ("forward", "reverse"):
            r_ = reg[key]
            A(f"| {r_['dependent']} ~ {r_['independent']} | "
              f"{r_['alpha']*100:+.4f} | {r_['beta']:+.4f} | {r_['beta_se']:.4f} | "
              f"{r_['beta_p']:.4f} | {r_['r2']:.4f} |")
        A("")
        A(reg["note"])
        A("")

    pf = res.get("portfolio")
    if pf:
        A("### Portfolio implication")
        A("")
        A(f"Minimum-variance weights: **{pf['weights'][sa]:.1%} {sa} / "
          f"{pf['weights'][sb]:.1%} {sb}**, giving "
          f"{pf['port_vol']:.2%} annualised volatility against "
          f"{pf['vol_a']:.2%} for {sa} alone — a "
          f"{pf['variance_reduction']:.1%} reduction in variance.")
        A("")
        for nt in pf["notes"]:
            A(f"- {nt}")
        A("")

    # ------------------------------------------------ conditional corr --
    A("### Conditional correlation (DCC / EWMA)")
    A("")
    ew = res.get("ewma", {})
    if ew:
        A(f"EWMA (λ={ew['lambda']}): latest {ew['latest']:+.4f}, mean "
          f"{ew['mean']:+.4f}, sd {ew['std']:.4f}, effective sample size "
          f"~{ew['effective_n']:.1f}.")
        A("")
        A(f"> {ew['note']}")
        A("")
    d = res.get("dcc")
    if d:
        if d.get("pre_note"):
            A(d["pre_note"])
            A("")
        # Univariate stage of the two-step: the variance models whose standardised
        # residuals feed the correlation equation.
        g = d.get("garch") or {}
        rows = [f for f in g.get("fitted", []) if f.get("converged")]
        if rows:
            A("Univariate GARCH(1,1) variance models:")
            A("")
            A("| Leg | omega | alpha | beta | persistence | half-life (days) |")
            A("|---|---|---|---|---|---|")
            for f in rows:
                A(f"| {f['label']} | {_num(f.get('omega'), 4)} | "
                  f"{_num(f.get('alpha'), 4)} | {_num(f.get('beta'), 4)} | "
                  f"{_num(f.get('persistence'), 4)} | "
                  f"{_num(f.get('half_life_days'), 1)} |")
            A("")
            if any(f.get("half_life_days") is None for f in rows):
                A("> A half-life of n/a means the fitted persistence fell outside "
                  "(0, 1), so a mean-reversion timescale does not exist for that "
                  "leg - shock decay is effectively permanent over this sample.")
                A("")
        if d.get("params"):
            dg = d["diagnostics"]
            A("DCC(1,1) correlation equation:")
            A("")
            A("| DCC(1,1) parameter | Value |")
            A("|---|---|")
            A(f"| a | {dg['a']:.5f} |")
            A(f"| b | {dg['b']:.5f} |")
            A(f"| mean conditional correlation | {dg['corr_mean']:+.4f} |")
            A(f"| sd of conditional correlation | {dg['corr_std']:.4f} |")
            A(f"| max abs correlation reached | "
              f"{max(abs(dg['corr_min']), abs(dg['corr_max'])):.4f} |")
            A(f"| min eigenvalue of R(t) | {dg['min_eigenvalue']:.6f} |")
            A("")
        if d.get("note"):
            A(d["note"])
            A("")

    # --------------------------------------------------------- data quality --
    au = res["audit"]
    A("### Data quality")
    A("")
    if au.get("big_moves"):
        A("Largest moves:")
        A("")
        A("| Date | Leg | Return |")
        A("|---|---|---|")
        for bm in au["big_moves"][:6]:
            A(f"| {bm['date']} | {sa if bm['leg'] == 'a' else sb} | "
              f"{bm['log_return']:+.2%} |")
        A("")
    if au.get("flat_days", {}).get("a") or au.get("flat_days", {}).get("b"):
        A(f"Flat days detected (A: {len(au['flat_days'].get('a', []))}, "
          f"B: {len(au['flat_days'].get('b', []))}) — see the robustness table.")
        A("")
    for w in au.get("warnings", []):
        A(f"- {w}")
    A("")

    # ----------------------------------------------------------- conclusion --
    A("## Conclusion")
    A("")
    for p in res["conclusion"]["points"]:
        A(f"- {p}")
    A("")

    A("---")
    A("")
    A("### Method and caveats")
    A("")
    A(f"- Correlations are on {m['return_type']} returns of prices adjusted for "
      f"corporate actions, computed on the intersection of both trading calendars.")
    A("- Confidence intervals use Fisher's z; p-values are two-sided t-tests on "
      "H0: ρ = 0.")
    A("- The block-bootstrap interval uses the stationary bootstrap "
      "(Politis & Romano) so that volatility clustering is respected; an "
      "observation-level bootstrap would understate uncertainty.")
    A("- Regression standard errors are Newey-West (HAC).")
    A("- All figures are sample statistics over one historical window. A "
      "correlation is not a structural constant and can change.")
    if m["period"] in ("1y", "6mo"):
        A(f"- **{m['period']} of daily data is a small sample.** With "
          f"{m['n_returns']} observations, tail and conditional-correlation "
          f"results are underpowered and should be read as indicative.")
    A("")
    A(f"Generated by corrkit. Reproduce: "
      f"`python correlate.py \"{m['query'][0]}\" \"{m['query'][1]}\" "
      f"--period {m['period']} --freq {m['freq']}`")
    A("")
    return "\n".join(L)


# ------------------------------------------------------------------ figures --

def make_figures(ds: Dataset, res: dict, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    sa, sb = res["meta"]["symbols"]
    made: list[Path] = []

    def save(fig, name):
        p = outdir / name
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        made.append(p)

    # 1. normalised prices
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    norm = ds.prices / ds.prices.iloc[0] * 100
    ax.plot(norm.index, norm["a"], color=NAVY, lw=1.3, label=sa)
    ax.plot(norm.index, norm["b"], color=GOLD, lw=1.3, label=sb)
    ax.set_title(f"{sa} vs {sb}, rebased to 100 ({ds.base_currency}-normalised)")
    ax.set_ylabel("index, start = 100")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "01_prices.png")

    # 2. levels vs returns scatter
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, (frame, title) in zip(axes, [
            (ds.prices, "price levels"),
            (np.log(ds.prices).diff().dropna(), "log returns")]):
        ax.scatter(frame["a"], frame["b"], s=9, alpha=0.45,
                   color=NAVY, edgecolors="none")
        r = frame["a"].corr(frame["b"])
        ax.set_title(f"{title}\nr = {r:+.3f}")
        ax.set_xlabel(sa)
        ax.set_ylabel(sb)
    save(fig, "02_levels_vs_returns.png")

    # 3. rolling correlation
    roll = res["rolling"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for col, style in zip(roll.columns, ["-", "--", "-.", ":"]):
        s = roll[col].dropna()
        ax.plot(s.index, s, style, lw=1.1,
                label=col.replace("roll_", "") + "d")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axhline(res["core"].r, color=RED, lw=0.9, ls="-.", label="full sample")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("rolling correlation")
    ax.set_title(f"Rolling correlation, {sa} vs {sb}")
    ax.legend(frameon=False, fontsize=7.5, ncol=3)
    save(fig, "03_rolling.png")

    # 4. scatter with fit, highlighting stress days
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    r = ds.returns
    thr = r["a"].quantile(res["stress"]["tail_q"])
    norm_d, stress = r[r["a"] > thr], r[r["a"] <= thr]
    ax.scatter(norm_d["a"], norm_d["b"], s=11, alpha=0.4, color=NAVY,
               edgecolors="none", label="normal days")
    ax.scatter(stress["a"], stress["b"], s=34, color=RED, marker="D",
               edgecolors="none", label=f"A worst {res['stress']['tail_q']:.0%}")
    z = np.polyfit(r["a"], r["b"], 1)
    xs = np.linspace(r["a"].min(), r["a"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=GREY, lw=1.0, ls="--")
    ax.set_xlabel(f"{sa} return")
    ax.set_ylabel(f"{sb} return")
    ax.set_title(f"Daily returns, r = {res['core'].r:+.3f}")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "04_scatter_stress.png")

    # 5. return distributions by A quintile
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    q = pd.qcut(r["a"], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    data = [g["b"].to_numpy() for _, g in r.groupby(q, observed=True)]
    bp = ax.boxplot(data, tick_labels=list(q.cat.categories), patch_artist=True,
                    widths=0.55,
                    medianprops={"color": "white", "lw": 1.5},
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.55})
    for patch in bp["boxes"]:
        patch.set_facecolor(NAVY)
        patch.set_alpha(0.6)
    for i, arr in enumerate(data, start=1):
        ax.plot(i, float(np.mean(arr)), "o", color=GOLD, ms=5, zorder=3)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel(f"{sb} return")
    ax.set_title(f"{sb} return within each {sa} quintile (dot = mean)")
    save(fig, "05_quintiles.png")

    # 6. cumulative and drawdown
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    cum = (1 + r).cumprod()
    axes[0].plot(cum.index, cum["a"], color=NAVY, lw=1.3, label=sa)
    axes[0].plot(cum.index, cum["b"], color=GOLD, lw=1.3, label=sb)
    axes[0].set_ylabel("growth of 1")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Cumulative return")
    dd = r.apply(lambda s: s.cumsum() - s.cumsum().cummax())
    axes[1].fill_between(dd.index, dd["a"], 0, color=NAVY, alpha=0.30, label=sa)
    axes[1].fill_between(dd.index, dd["b"], 0, color=GOLD, alpha=0.30, label=sb)
    axes[1].set_ylabel("drawdown (log)")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "06_cumulative_drawdown.png")

    # 7. efficient frontier
    pf = res["portfolio"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(pf["grid_vol"], np.linspace(-0.5, 1.5, len(pf["grid_vol"])),
            color=GREY, lw=1.2, label="efficient frontier (shorts allowed)")
    w = list(pf["weights"].values())
    ax.scatter([pf["vol_a"], pf["vol_b"]], [1, 0], s=55, color=NAVY,
               edgecolors="none", zorder=3)
    ax.annotate(sa, (pf["vol_a"], 1), textcoords="offset points", xytext=(6, 4),
                fontsize=8)
    ax.annotate(sb, (pf["vol_b"], 0), textcoords="offset points", xytext=(6, -12),
                fontsize=8)
    ax.scatter([pf["port_vol"]], [w[1]], s=70, color=GOLD, marker="*",
               zorder=4, label="minimum variance")
    ax.set_xlabel("annualised volatility")
    ax.set_ylabel(f"weight in {sb}")
    ax.set_title(f"Diversification at r = {pf['r']:+.2f}")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "07_frontier.png")

    # 8. estimator comparison
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    c = res["core"]
    labels_ = ["Pearson", "Spearman", "Kendall", "Rolling (latest)", "EWMA"]
    vals = [c.r, c.spearman_rho, c.kendall_tau,
            (res["rolling_diag"].get("roll_60") or {}).get("latest", np.nan),
            res["ewma"]["latest"]]
    cols = [NAVY, NAVY, NAVY, GOLD, GREY]
    ax.bar(labels_, vals, color=cols, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(c.r, color=RED, ls="--", lw=0.9)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v + (0.02 if v >= 0 else -0.05), f"{v:+.3f}",
                    ha="center", fontsize=8)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("The same pair, five estimators - spread is the honest uncertainty")
    save(fig, "08_estimators.png")

    return made


# --------------------------------------------------------------- json export --

def _jsonable(o):
    """Convert numpy/pandas types into something json.dumps accepts."""
    if isinstance(o, (pd.Series, pd.DataFrame, pd.Index)):
        if isinstance(o, pd.DataFrame):
            return [{str(k): _jsonable(v) for k, v in rec.items()}
                    for rec in o.to_dict(orient="records")]
        if isinstance(o, pd.Index):
            return [_jsonable(v) for v in o.tolist()]
        return [{str(k): _jsonable(v) for k, v in o.items()}]
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(o)
    return o


def write_outputs(res: dict, outdir: Path, ds: Dataset | None = None) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict = {}
    (outdir / "report.md").write_text(render_markdown(res))
    paths["report"] = str(outdir / "report.md")
    (outdir / "results.json").write_text(
        json.dumps(_jsonable(res), indent=2, default=str))
    paths["json"] = str(outdir / "results.json")

    if ds is not None:
        ds.prices.to_csv(outdir / "prices.csv", index_label="Date")
        ds.returns.to_csv(outdir / "returns.csv", index_label="Date")
        paths["prices"] = str(outdir / "prices.csv")
        paths["returns"] = str(outdir / "returns.csv")
    return paths
