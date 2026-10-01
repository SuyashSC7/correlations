#!/usr/bin/env python3
"""
correlate.py - correlation analysis for any two instruments.

    python correlate.py "tata motors" RELIANCE.NS
    python correlate.py "tata steel" "hcl"           # two NSE company names
    python correlate.py AAPL MSFT --period 2y
    python correlate.py RELIANCE.NS AAPL --base INR
    python correlate.py VOD.L SAP.DE --json          # machine-readable only

Accepts Yahoo symbols or company names. Names are resolved through Yahoo's
search index and the match is printed, because "tata motors" and
"TATAMOTORS.NS" are not the same instrument any more.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from corrkit.data import load
from corrkit.pipeline import headline, run
from corrkit.report import make_figures, write_outputs

PERIODS = ["6mo", "1y", "2y", "3y", "5y", "10y", "max"]
FREQ = ["1d", "1wk", "1mo"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="correlate.py",
        description="Correlation index for any two instruments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("__doc__")[0] if False else None,
    )
    p.add_argument("asset_a", help="symbol or company name")
    p.add_argument("asset_b", help="symbol or company name")
    p.add_argument("--period", default="1y", choices=PERIODS,
                   help="history window (default: 1y)")
    p.add_argument("--freq", default="1d", choices=FREQ,
                   help="return frequency (default: 1d)")
    p.add_argument("--base", default="auto",
                   help="currency to express both legs in, e.g. INR, USD. "
                        "'auto' keeps A's currency (default)")
    p.add_argument("--no-fx", action="store_true",
                   help="skip currency normalisation; compare native prices")
    p.add_argument("--simple-returns", action="store_true",
                   help="use simple rather than log returns")
    p.add_argument("--conf", type=float, default=0.95,
                   help="confidence level for intervals (default: 0.95)")
    p.add_argument("--boot", type=int, default=10_000,
                   help="block-bootstrap draws (default: 10000)")
    p.add_argument("--sub", type=int, default=3,
                   help="number of sub-periods for the stability test "
                        "(default: 3)")
    p.add_argument("--tail", type=float, default=0.05,
                   help="tail quantile for dependence tests (default: 0.05)")
    p.add_argument("--lam", type=float, default=0.94,
                   help="EWMA lambda (default: 0.94)")
    p.add_argument("--no-dcc", action="store_true",
                   help="skip the DCC-GARCH fit (it is often unidentified anyway)")
    p.add_argument("--out", default="output",
                   help="output directory (default: output)")
    p.add_argument("--cache", default=".cache",
                   help="download cache directory (default: .cache)")
    p.add_argument("--no-plot", action="store_true", help="skip figures")
    p.add_argument("--json", action="store_true",
                   help="print the summary as JSON and nothing else")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="suppress progress output")
    p.add_argument("--pick", type=int, metavar="N", default=None,
                   help="choose the Nth match when an input is a company name "
                        "that exists on several exchanges (see --list)")
    p.add_argument("--pick-b", type=int, metavar="N", default=None,
                   help="same, for the second input")
    p.add_argument("--list", action="store_true",
                   help="list candidate matches for the inputs and exit")
    return p


def print_summary(res: dict, ds, paths: dict, quiet: bool = False) -> None:
    c, con, m = res["core"], res["conclusion"], res["meta"]
    sa, sb = m["symbols"]
    print()
    print("=" * 74)
    print(f"  CORRELATION INDEX   {sa}  vs  {sb}")
    print("=" * 74)
    print(f"  window            {m['start']} -> {m['end']}  ({m['period']}, {m['freq']})")
    print(f"  observations      {m['n_returns']} {m['return_type']} returns")
    print(f"  base currency     {m['base_currency']}")
    print()
    print(f"  >>> PEARSON r    = {c.r:+.4f}")
    print(f"      95% CI       = [{c.ci_low:+.4f}, {c.ci_high:+.4f}]")
    print(f"      p-value      = {c.p_value:.4f}      R² = {c.r2:.4f}")
    print(f"      Spearman ρ   = {c.spearman_rho:+.4f} (p={c.spearman_p:.4f})")
    print(f"      Kendall τ    = {c.kendall_tau:+.4f} (p={c.kendall_p:.4f})")
    print(f"      block-boot   = [{c.boot_ci[0]:+.4f}, {c.boot_ci[1]:+.4f}]")
    print()
    print(f"  VERDICT: {c.verdict}")
    if c.fragility and not quiet:
        print()
        print("  Caveats:")
        for f in c.fragility:
            print(f"    - {f}")
    print()
    print("  Reading:")
    for p in con["points"]:
        print(f"    * {p}")
    if paths:
        print()
        print(f"  report  {paths.get('report')}")
        print(f"  json    {paths.get('json')}")
        if paths.get("figures"):
            print(f"  figures {paths['figures']}")
    print("=" * 74)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    quiet = args.quiet or args.json
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    if args.list:
        from corrkit.universe import resolve
        for spec in (args.asset_a, args.asset_b):
            r = resolve(spec, cache_dir=cache)
            print(f"\n{spec!r} -> {r.describe()}   [{r.method}]")
            if len(r.candidates) > 1:
                print("  candidates:")
                for c in r.candidates[:10]:
                    print(f"    {c['symbol']:14s} {c['name'][:44]:44s} "
                          f"({c['exchange']}, {c['quote_type']})")
        return 0

    def say(msg: str) -> None:
        if not quiet:
            print(msg)

    try:
        ds = load(args.asset_a, args.asset_b, period=args.period, freq=args.freq,
                  base=args.base, cache_dir=cache, no_fx=args.no_fx,
                  simple_returns=args.simple_returns,
                  pick_a=args.pick, pick_b=args.pick_b, verbose=not quiet)
    except (ValueError, LookupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    res = run(ds, conf=args.conf, n_boot=args.boot, n_sub=args.sub,
              tail_q=args.tail, stress_q=args.tail, lam=args.lam,
              with_dcc=not args.no_dcc, progress=say)

    outdir = Path(args.out)
    paths = write_outputs(res, outdir, ds)
    if not args.no_plot:
        figs = make_figures(ds, res, outdir / "figures")
        paths["figures"] = f"{outdir / 'figures'} ({len(figs)} files)"

    if args.json:
        import json

        from corrkit.report import _jsonable
        print(json.dumps(_jsonable(headline(res).to_dict()), indent=2))
        return 0

    print_summary(res, ds, paths, quiet=quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
