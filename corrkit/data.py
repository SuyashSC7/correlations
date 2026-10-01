"""Data acquisition, FX normalisation, calendar alignment and quality audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .universe import (
    Resolution,
    fx_symbol_for,
    fx_usd_per_series,
    normalise_currency,
    quiet_yfinance,
    resolution_warning,
    resolve,
)

PERIOD_DAYS = {
    "6mo": "6mo", "1y": "1y", "2y": "2y", "3y": "3y",
    "5y": "5y", "10y": "10y", "max": "max",
}
FREQ_INTERVAL = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}

FLAT_RETURN_EPS = 1e-9      # near-zero log return => stale or rolled quote
MAX_HOLIDAY_GAP = 5         # business days
FX_TOLERANCE_DAYS = {"1d": 7, "1wk": 21, "1mo": 62}


@dataclass
class Asset:
    """One resolved instrument plus its price history."""

    res: Resolution
    close: pd.Series                     # native currency
    base_close: pd.Series | None = None  # converted to the analysis base currency
    fx_symbol: str = ""
    conversion: str = "none"             # human-readable description
    notes: list[str] = field(default_factory=list)

    @property
    def symbol(self) -> str:
        return self.res.symbol

    @property
    def currency(self) -> str:
        return self.res.currency or "USD"


@dataclass
class Dataset:
    """Aligned, FX-normalised prices for a pair of assets."""

    a: Asset
    b: Asset
    base_currency: str
    prices: pd.DataFrame                   # base-currency closes, aligned
    returns: pd.DataFrame                  # returns of prices
    native_returns: pd.DataFrame           # unconverted returns (FX-effect check)
    freq: str = "1d"
    period: str = "1y"
    audit: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def symbols(self) -> tuple[str, str]:
        return self.a.symbol, self.b.symbol


def _to_returns(s: pd.Series, simple: bool) -> pd.Series:
    return s.pct_change(fill_method=None) if simple else np.log(s).diff()


def _history(symbol: str, period: str, interval: str,
             cache_dir: Path | None) -> pd.Series:
    """Close series for `symbol`, cached to disk by symbol/interval/period."""
    cache_file = None
    if cache_dir:
        safe = symbol.replace("=", "_").replace("^", "")
        cache_file = cache_dir / f"px_{safe}_{interval}_{period}.csv"
        if cache_file.exists():
            try:
                s = pd.read_csv(cache_file, index_col=0, parse_dates=True)["Close"].dropna()
                if len(s) > 5:
                    s.index = pd.to_datetime(s.index).normalize()
                    return s
            except Exception:
                cache_file.unlink(missing_ok=True)

    with quiet_yfinance():
        h = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    if h is None or h.empty or "Close" not in h:
        raise ValueError(
            f"No price history for '{symbol}' (period={period}, interval={interval}). "
            f"The symbol may be delisted or misspelled.")

    s = h["Close"].dropna().astype(float)
    s.index = pd.to_datetime(s.index)
    try:
        s.index = s.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    s.index = s.index.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()

    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            s.to_csv(cache_file, index_label="Date")
        except Exception:
            pass
    return s


def _convert(px: pd.Series, own: str, base: str, fx: pd.DataFrame,
             tolerance: int) -> pd.Series:
    """
    Restate a native-currency price series in `base`, date by date.

    FX is applied as a time-varying factor. Using one spot rate for the whole
    history would freeze the exchange rate and invent a trend that never
    happened.
    """
    maj_o, mult_o = normalise_currency(own)
    maj_b, mult_b = normalise_currency(base)

    if maj_o == maj_b:                       # only a units change (e.g. GBp->GBP)
        return px * (mult_b / mult_o)

    def leg(maj: str) -> pd.Series | None:
        if maj == "USD":                      # USD is the pivot, no series needed
            return pd.Series(1.0, index=px.index)
        if fx.empty or maj not in fx.columns:
            return None
        col = fx.reindex(fx.index.union(px.index)).ffill().reindex(px.index)
        return col[maj]

    num, den = leg(maj_o), leg(maj_b)
    if num is None or den is None:
        missing = [c for c, v in ((maj_o, num), (maj_b, den)) if v is None]
        raise ValueError(
            f"No FX rate available for {missing}. Try --base with a currency that "
            f"has a liquid market, or --no-fx to compare unconverted prices.")

    ratio = num / den
    # Blank conversions that lean on a stale FX print.
    if maj_o != "USD":
        known = fx[maj_o].dropna()
        if not known.empty:
            pos = np.clip(known.index.searchsorted(px.index, side="right") - 1,
                          0, len(known) - 1)
            ages = (px.index.to_numpy()
                    - known.index.to_numpy()[pos]) / np.timedelta64(1, "D")
            if (ages > tolerance).any():
                ratio = ratio.mask(pd.Series(ages > tolerance, index=px.index))
    return px * (ratio * (mult_b / mult_o))


def load(spec_a: str, spec_b: str, *, period: str = "1y", freq: str = "1d",
         base: str = "auto", cache_dir: Path | None = None,
         no_fx: bool = False, simple_returns: bool = False,
         pick_a: int | None = None, pick_b: int | None = None,
         verbose: bool = True) -> Dataset:
    """
    Resolve, download, FX-normalise and align a pair of instruments.

    base='auto' keeps asset A's currency, so the result is expressed in the
    currency the user named first.
    """
    period = PERIOD_DAYS.get(period.lower(), period)
    interval = FREQ_INTERVAL[freq.lower()]
    freq = freq.lower()

    notes: list[str] = []
    res_a = resolve(spec_a, cache_dir=cache_dir, pick=pick_a)
    res_b = resolve(spec_b, cache_dir=cache_dir, pick=pick_b)
    for r in (res_a, res_b):
        w = resolution_warning(r)
        if w:
            notes.append(w)
    if res_a.symbol.upper() == res_b.symbol.upper():
        raise ValueError(f"Both inputs resolved to '{res_a.symbol}'. Pick two different assets.")

    px_a = _history(res_a.symbol, period, interval, cache_dir)
    px_b = _history(res_b.symbol, period, interval, cache_dir)
    cur_a = res_a.currency or "USD"
    cur_b = res_b.currency or "USD"

    tol = FX_TOLERANCE_DAYS.get(freq, 7)

    if no_fx:
        base_ccy = "n/a (native currencies, unconverted)"
        fx_needed = False
        conv_a = conv_b = "disabled via --no-fx"
        notes.append("FX normalisation disabled: the two series are compared in their "
                     "native currencies, so any exchange-rate move is attributed to the asset.")
    elif cur_a == cur_b:
        base_ccy = cur_a
        fx_needed = False
        conv_a = conv_b = f"none (both in {cur_a})"
        notes.append(f"Both instruments quote in {cur_a}; no FX conversion required.")
    else:
        base_ccy = cur_a if base.lower() == "auto" else base.upper()
        fx_needed = True
        fx = fx_usd_per_series([cur_a, cur_b, base_ccy], cache_dir=cache_dir,
                               period=period)
        conv_a = (f"via {fx_symbol_for(cur_a)}"
                  if fx_symbol_for(cur_a) else "identity (already USD)")
        conv_b = (f"via {fx_symbol_for(cur_b)}"
                  if fx_symbol_for(cur_b) else "identity (already USD)")
        base_maj = normalise_currency(base_ccy)[0]
        known = set(fx.columns) | {normalise_currency(cur_a)[0],
                                   normalise_currency(cur_b)[0], "USD"}
        if base_maj not in known:
            raise ValueError(f"Unsupported base currency '{base_ccy}'.")

    def build(p: pd.Series, own: str, conv: str) -> tuple[pd.Series, pd.Series]:
        if not fx_needed:
            return p.copy(), p
        return _convert(p, own, base_ccy, fx, tol), p

    base_a, _ = build(px_a, cur_a, conv_a)
    base_b, _ = build(px_b, cur_b, conv_b)

    prices = pd.DataFrame({"a": base_a, "b": base_b}).dropna(how="any")
    # Fewer bars are needed at lower frequency: 24 monthly closes is a usable
    # sample, whereas 24 daily closes is not.
    min_obs = {"1d": 30, "1wk": 18, "1mo": 18}.get(freq, 30)
    if len(prices) < min_obs:
        raise ValueError(
            f"Only {len(prices)} overlapping sessions between {res_a.symbol} and "
            f"{res_b.symbol} at {freq} frequency (need {min_obs}). Their calendars "
            f"may barely intersect, or one listing may be too new - try a longer "
            f"--period or a higher --freq.")

    rets = _to_returns(prices, simple_returns).dropna(how="any")
    native = pd.DataFrame({"a": _to_returns(px_a, simple_returns),
                           "b": _to_returns(px_b, simple_returns)}).dropna(how="any")

    ds = Dataset(
        a=Asset(res_a, px_a, base_a, fx_symbol_for(cur_a) or "", conv_a),
        b=Asset(res_b, px_b, base_b, fx_symbol_for(cur_b) or "", conv_b),
        base_currency=base_ccy,
        prices=prices,
        returns=rets,
        native_returns=native,
        freq=freq,
        period=period,
        notes=notes,
    )
    ds.audit = audit(prices, px_a, px_b, base_ccy, fx_needed, cur_a, cur_b, simple_returns)
    if verbose:
        report_to_console(ds)
    return ds


def audit(prices: pd.DataFrame, px_a: pd.Series, px_b: pd.Series,
          base_ccy: str, fx_needed: bool, cur_a: str, cur_b: str,
          simple_returns: bool) -> dict:
    """Coverage, staleness and outlier checks on the aligned sample."""
    idx = prices.index
    rets = (prices.pct_change(fill_method=None) if simple_returns
            else np.log(prices).diff()).dropna(how="any")

    rep: dict = {
        "base_currency": base_ccy,
        "return_type": "simple" if simple_returns else "log",
        "n_sessions": len(prices),
        "n_returns": len(rets),
        "start": str(idx.min().date()) if len(idx) else None,
        "end": str(idx.max().date()) if len(idx) else None,
        "currency_a": cur_a, "currency_b": cur_b,
        "fx_normalised": bool(fx_needed),
        "flat_days": {}, "big_moves": [], "calendar_gaps": [], "warnings": [],
    }

    for col in rets.columns:
        flat = rets.index[rets[col].abs() < FLAT_RETURN_EPS]
        rep["flat_days"][col] = [str(d.date()) for d in flat]

    for col in rets.columns:
        # rank on magnitude, but report the *signed* return: a -11.7% crash and a
        # +11.7% rally rank identically and are not the same event.
        for d in rets[col].abs().nlargest(3).index:
            v = float(rets.loc[d, col])
            if abs(v) > 0.08:
                rep["big_moves"].append({"date": str(d.date()), "leg": col,
                                         "log_return": round(v, 4)})
    rep["big_moves"].sort(key=lambda r: -abs(r["log_return"]))

    if len(idx) > 3:
        gaps = idx.to_series().diff().dt.days
        for d, g in gaps[gaps > MAX_HOLIDAY_GAP].items():
            rep["calendar_gaps"].append({"after": str(d.date()), "days": int(g)})

    if rep["flat_days"].get("a") or rep["flat_days"].get("b"):
        rep["warnings"].append(
            "Flat (zero-return) days present - usually a rolled or stale feed. "
            "A drop-those-days sensitivity check is reported.")
    if rep["calendar_gaps"]:
        rep["warnings"].append(
            "Calendar gaps where a venue was shut; the sample is the intersection "
            "of both trading calendars.")
    if rep["n_returns"] < 60:
        rep["warnings"].append(
            f"Only {rep['n_returns']} overlapping returns - estimates are thin.")
    return rep


def report_to_console(ds: Dataset) -> None:
    print("[1] Resolved inputs")
    print(f"    {ds.a.res.describe()}")
    print(f"    {ds.b.res.describe()}")
    print(f"    base currency: {ds.base_currency}"
          f"  | A: {ds.a.conversion} | B: {ds.b.conversion}")
    au = ds.audit
    print(f"\n[2] Aligned sample: {au['n_sessions']} sessions, "
          f"{au['n_returns']} {au['return_type']} returns")
    print(f"    {au['start']} -> {au['end']}")
    for n in ds.notes:
        print(f"    - {n}")
    for w in au["warnings"]:
        print(f"    ! {w}")
