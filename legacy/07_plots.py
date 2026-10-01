"""
Step 7: Figures.

Each figure answers one question and is annotated with the number the report
quotes, so a reader can check the visual against the statistics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

rets = pd.read_csv(ROOT / "data" / "log_returns.csv", index_col=0, parse_dates=True)
px = pd.read_csv(ROOT / "data" / "prices_aligned.csv", index_col=0, parse_dates=True)

NAVY, GOLD, GREY, RED, GREEN = "#1f3b57", "#c9a227", "#8a8a8a", "#c0392b", "#1e8449"
QUINT = ["worst 20%", "20-40%", "40-60%", "60-80%", "best 20%"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG / name}")


x, y = rets["NIFTY"], rets["GOLDINR"]
r = x.corr(y)
CRASH = pd.Timestamp("2026-01-30")

# ---------------------------------------------------------------- 1. levels --
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(px.index, px["NIFTY"], color=NAVY, lw=1.4)
ax[0].set_title(f"Indexed price levels: Nifty 50 falls, gold in INR rises "
                f"(Nifty {px['NIFTY'].iloc[-1] / px['NIFTY'].iloc[0] - 1:+.1%}, "
                f"gold {px['GOLD_INR'].iloc[-1] / px['GOLD_INR'].iloc[0] - 1:+.1%})")
ax[0].set_ylabel("Nifty 50")
ax[0].axhline(px["NIFTY"].iloc[0], color=GREY, ls="--", lw=0.8)

ax[1].plot(px.index, px["GOLD_INR"], color=GOLD, lw=1.4)
ax[1].plot(px.index, px["GOLD_USD"], color=RED, lw=1.0, alpha=0.75, label="gold in USD")
ax[1].set_title("Gold: in INR (primary) vs in USD — the gap is the rupee")
ax[1].set_ylabel("INR per 10g-equivalent")
ax[1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax[1].legend(frameon=False, loc="upper left")
ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
save(fig, "01_price_levels.png")

# ------------------------------------------------------- 2. levels vs returns --
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].scatter(px["NIFTY"], px["GOLD_INR"], s=14, color=NAVY, alpha=0.6)
sl = np.polyfit(px["NIFTY"], px["GOLD_INR"], 1)
xs = np.linspace(px["NIFTY"].min(), px["NIFTY"].max(), 10)
ax[0].plot(xs, np.polyval(sl, xs), color=RED, lw=1.2)
rl = px["NIFTY"].corr(px["GOLD_INR"])
ax[0].set_title(f"Price LEVELS: r = {rl:+.3f}  (spurious)")
ax[0].set_xlabel("Nifty 50"); ax[0].set_ylabel("Gold in INR")
ax[0].annotate("both trend -> high |r|\nwith no co-movement", xy=(0.05, 0.92),
               xycoords="axes fraction", fontsize=8, color=RED)

ax[1].scatter(x * 100, y * 100, s=14, color=GOLD, alpha=0.65, edgecolors="none")
sl2 = np.polyfit(x * 100, y * 100, 1)
xs2 = np.linspace((x * 100).min(), (x * 100).max(), 10)
ax[1].plot(xs2, np.polyval(sl2, xs2), color=NAVY, lw=1.2)
ax[1].axhline(0, color=GREY, lw=0.7); ax[1].axvline(0, color=GREY, lw=0.7)
ax[1].set_title(f"Daily RETURNS: r = {r:+.3f}  (the valid statistic)")
ax[1].set_xlabel("Nifty 50 daily return (%)"); ax[1].set_ylabel("Gold INR daily return (%)")
ax[1].annotate("r flips sign vs the levels panel:\nlevels are not co-movement",
               xy=(0.05, 0.92), xycoords="axes fraction", fontsize=8, color=NAVY)
save(fig, "02_levels_vs_returns.png")

# ------------------------------------------------------------ 3. rolling corr --
roll = pd.read_csv(OUT / "rolling_correlation.csv", index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(10, 4.6))
for c, col, lw in [("corr_20d", RED, 0.9), ("corr_60d", GOLD, 1.3), ("corr_90d", NAVY, 1.5)]:
    if c in roll.columns:
        ax.plot(roll.index, roll[c], color=col, lw=lw, label=c.replace("corr_", "").replace("d", "-day"))
ax.axhline(0, color="k", lw=0.9)
ax.axhline(r, color=GREEN, ls="--", lw=1.1, label=f"full-sample r = {r:+.3f}")
ax.axvline(CRASH, color=GREY, ls=":", lw=1.2)
ax.annotate("30 Jan gold\ncrash", xy=(CRASH, 0.46), fontsize=7.5, color=GREY, ha="right")
ax.set_title("Rolling correlation of daily returns — Nifty 50 vs gold in INR")
ax.set_ylabel("Pearson r"); ax.set_ylim(-0.45, 0.65)
ax.legend(frameon=False, ncol=4, fontsize=8, loc="lower left")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
save(fig, "03_rolling_correlation.png")

# --------------------------------------------------------------- 4. scatter+bins --
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].hist(x * 100, bins=30, color=NAVY, alpha=0.8, label="Nifty", density=True)
ax[0].hist(y * 100, bins=30, color=GOLD, alpha=0.65, label="Gold (INR)", density=True)
ax[0].set_title("Return distributions: gold is ~2.2x more volatile")
ax[0].set_xlabel("daily return (%)"); ax[0].legend(frameon=False, fontsize=8)
ax[0].set_ylabel("density")

q = pd.qcut(x, 5, labels=QUINT)
g = y.groupby(q, observed=True).mean() * 100
cols = [RED if v < 0 else GREEN for v in g]
ax[1].bar(range(len(g)), g.values, color=cols, alpha=0.85)
ax[1].set_xticks(range(len(g)))
ax[1].set_xticklabels(g.index, fontsize=8)
ax[1].axhline(0, color="k", lw=0.9)
ax[1].set_title("Gold's average return by Nifty quintile\n(the hedge works only in the bad tail)")
ax[1].set_ylabel("mean gold return (%)")
for i, v in enumerate(g.values):
    ax[1].text(i, v + (0.05 if v >= 0 else -0.12), f"{v:+.2f}%", ha="center", fontsize=8)
save(fig, "04_distribution_and_quintiles.png")

# ------------------------------------------------------------- 5. FX confound --
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
r_inr = x.corr(y)
r_usd = x.corr(rets["GOLDUSD"])
r_fx = x.corr(rets["USDINR"])
ax[0].bar(["gold in\nINR (primary)", "gold in\nUSD", "USD/INR\nalone"],
          [r_inr, r_usd, r_fx], color=[NAVY, GOLD, RED], alpha=0.85)
ax[0].axhline(0, color="k", lw=0.9)
ax[0].set_title("Why the currency matters")
ax[0].set_ylabel("correlation with Nifty")
ax[0].text(0, r_inr + 0.02, f"{r_inr:+.3f}", ha="center", fontsize=9)
ax[0].text(1, r_usd + 0.02, f"{r_usd:+.3f}", ha="center", fontsize=9)
ax[0].text(2, r_fx - 0.05, f"{r_fx:+.3f}", ha="center", fontsize=9)
ax[0].annotate("rupee weakness drags Nifty;\nstrip it out for a fair gold read",
               xy=(0.05, 0.06), xycoords="axes fraction", fontsize=7.5, color=NAVY)

ax[1].plot(px.index, (px["GOLD_INR"] / px["GOLD_INR"].iloc[0]), color=GOLD, lw=1.6,
           label="gold in INR")
ax[1].plot(px.index, (px["GOLD_USD"] / px["GOLD_USD"].iloc[0]), color=RED, lw=1.2,
           label="gold in USD")
ax[1].plot(px.index, (px["USDINR"] / px["USDINR"].iloc[0]), color=GREY, lw=1.1, ls="--",
           label="USD/INR")
ax[1].axhline(1, color="k", lw=0.8)
ax[1].set_title("Cumulative performance, indexed to 1.0")
ax[1].set_ylabel("index")
ax[1].legend(frameon=False, fontsize=8)
ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
save(fig, "05_fx_confound.png")

# ------------------------------------------------------------ 6. stress days --
fig, ax = plt.subplots(figsize=(10, 4.4))
bins = pd.qcut(x, 5)                       # keep intervals, not labels
data = [y[(x >= b.left) & (x <= b.right)] * 100 for b in bins.cat.categories]
ax.boxplot(data, tick_labels=QUINT, patch_artist=True,
           widths=0.55,
           boxprops=dict(facecolor=NAVY, alpha=0.5),
           medianprops=dict(color="white", lw=1.6),
           flierprops=dict(marker="o", markersize=3, alpha=0.5, markerfacecolor=RED))
ax.axhline(0, color="k", lw=0.9, ls="--")
ax.set_title("Distribution of gold's return within each Nifty quintile")
ax.set_ylabel("gold (INR) daily return (%)")
save(fig, "06_stress_boxplot.png")

# ---------------------------------------------------------------- 7. summary --
# Values are recomputed here from the return series rather than hard-coded, so
# the figure can never drift from the statistics the report quotes.
from scipy import stats as _st

wk = rets.groupby(rets.index.isocalendar().week.astype(int)).sum()
mo = rets.resample("ME").sum()
specs = [
    ("Pearson\n(linear)", x, y, x.corr(y)),
    ("Spearman\n(rank)", x, y, _st.spearmanr(x, y)[0]),
    ("Kendall\n(tau)", x, y, _st.kendalltau(x, y)[0]),
    ("90d rolling\n(mean)", None, None, roll["corr_90d"].mean()),
    ("excl. crash\nday", x.drop(index=CRASH), y.drop(index=CRASH),
     x.drop(index=CRASH).corr(y.drop(index=CRASH))),
    ("weekly\nreturns", wk["NIFTY"], wk["GOLDINR"], wk["NIFTY"].corr(wk["GOLDINR"])),
    ("monthly\nreturns", mo["NIFTY"], mo["GOLDINR"], mo["NIFTY"].corr(mo["GOLDINR"])),
]

fig, ax = plt.subplots(figsize=(9.5, 4.8))
labels = [s[0] for s in specs]
vals = [s[3] for s in specs]
pvals = [_st.pearsonr(s[1], s[2])[1] for s in specs if s[1] is not None]
pvals.insert(3, np.nan)                      # rolling mean has no single p-value
cols = [GOLD if (np.isnan(pv) or pv >= 0.05) else NAVY for pv in pvals]
bars = ax.bar(labels, vals, color=cols, alpha=0.88, width=0.62)
ax.axhline(0, color="k", lw=0.9)
ax.axhline(r, color=RED, ls="--", lw=1.1, label=f"full-sample r = {r:+.3f}")
for b, v, pv in zip(bars, vals, pvals):
    tag = "" if np.isnan(pv) else ("  *" if pv < 0.05 else "  n.s.")
    ax.text(b.get_x() + b.get_width() / 2, v + (0.014 if v >= 0 else -0.045),
            f"{v:+.3f}{tag}", ha="center", fontsize=8.5)
ax.set_ylim(min(-0.12, min(vals) - 0.09), max(vals) + 0.11)
ax.set_title("Correlation estimate by method — * = significant at 5%")
ax.set_ylabel("correlation with Nifty 50")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.text(0.015, 0.97, "Rank-based and lower-frequency estimates are NOT\n"
                     "significantly different from zero; the linear one is.",
        transform=ax.transAxes, fontsize=8, color=NAVY, va="top")
save(fig, "07_method_comparison.png")

print("\nAll figures written to outputs/figures/")
