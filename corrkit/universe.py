"""
Symbol resolution and currency handling.

Two jobs:

1. Turn whatever the user typed ("tata motors", "AAPL", "7203.T") into a Yahoo
   symbol, and say exactly what it resolved to. Silent resolution is dangerous:
   "tata motors" now matches de-merger entities TMPV.NS / TMCV.NS, not the old
   TATAMOTORS.NS listing.

2. Put two assets in a common currency. This is not cosmetic. A Nifty vs
   gold-in-USD comparison lets the rupee move leak into the "gold" signal - in the
   gold/Nifty study USD/INR alone correlated -0.26 with the index, which inflated
   the measured relationship. Normalising both legs to one currency removes it.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yfinance as yf


@contextlib.contextmanager
def quiet_yfinance():
    """
    Silence yfinance's probe chatter.

    Deciding whether a string is already a valid symbol means deliberately
    requesting data that does not exist, and yfinance narrates each failure on
    stderr. That noise is expected here and would otherwise bury the results.
    """
    loggers = [logging.getLogger(n) for n in ("yfinance", "yfinance.data",
                                              "yfinance.utils", "peewee")]
    prev = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.CRITICAL)
    buf = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                old_fd = os.dup(2)
                os.dup2(buf.fileno(), 2)
            except OSError:
                old_fd = None
            try:
                yield
            finally:
                if old_fd is not None:
                    sys_stderr_flush()
                    os.dup2(old_fd, 2)
                    os.close(old_fd)
    finally:
        for lg, lv in zip(loggers, prev):
            lg.setLevel(lv)


def sys_stderr_flush() -> None:
    import sys
    try:
        sys.stderr.flush()
    except Exception:
        pass

# A string that already looks like a ticker: short, optional suffix/prefix.
SYMBOL_RE = re.compile(r"^[A-Za-z0-9^][A-Za-z0-9.\-^]{0,19}$")

# Minor-unit quote currencies: (major currency, major units per minor unit).
MINOR_UNITS = {
    "GBp": ("GBP", 100.0),   # British pence
    "ZAc": ("ZAR", 100.0),   # South African cents
    "ILa": ("IDR", 100.0),
}

# Currency -> Yahoo FX symbol quoted as "major units per 1 USD".
FX_TO_PER_USD = {
    "USD": None,
    "INR": "USDINR=X", "JPY": "USDJPY=X", "EUR": "USDEUR=X", "GBP": "USDGBP=X",
    "CNY": "USDCNY=X", "CNH": "USDCNH=X", "HKD": "USDHKD=X", "KRW": "USDKRW=X",
    "TWD": "USDTWD=X", "SGD": "USDSGD=X", "CHF": "USDCHF=X", "CAD": "USDCAD=X",
    "AUD": "USDAUD=X", "NZD": "USDNZD=X", "BRL": "USDBRL=X", "MXN": "USDMXN=X",
    "ZAR": "USDZAR=X", "TRY": "USDTRY=X", "SEK": "USDSEK=X", "NOK": "USDNOK=X",
    "DKK": "USDDKK=X", "PLN": "USDPLN=X", "THB": "USDTHB=X", "IDR": "USDIDR=X",
    "MYR": "USDMYR=X", "PHP": "USDPHP=X", "VND": "USDVND=X", "PKR": "USDPKR=X",
    "BDT": "USDBDT=X", "LKR": "USDLKR=X", "SAR": "USDSAR=X", "AED": "USDAED=X",
    "QAR": "USDQAR=X", "ILS": "USDILS=X", "CLP": "USDCLP=X", "COP": "USDCOP=X",
    "ARS": "USDARS=X", "PEN": "USDPEN=X", "UAH": "USDUAH=X", "CZK": "USDCZK=X",
    "HUF": "USDHUF=X", "NGN": "USDNGN=X", "EGP": "USDEGP=X", "KWD": "USDKWD=X",
    "BHD": "USDBHD=X", "OMR": "USDOMR=X", "JOD": "USDJOD=X",
}


@dataclass
class Resolution:
    """Outcome of turning user input into a concrete instrument."""

    query: str
    symbol: str
    name: str
    exchange: str = ""
    currency: str = ""
    quote_type: str = ""
    last_price: float | None = None
    candidates: list[dict] = field(default_factory=list)
    method: str = "direct-symbol"

    def describe(self) -> str:
        # " - " rather than " | ": this string is dropped into markdown table
        # cells, where a pipe would open a spurious column.
        bits = [self.symbol]
        if self.name and self.name.lower() != self.query.lower():
            bits.append(self.name)
        if self.exchange:
            bits.append(str(self.exchange))
        if self.currency:
            bits.append(str(self.currency))
        return " - ".join(str(b) for b in bits)

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol)
        self.name = str(self.name or self.symbol)
        self.currency = str(self.currency or "")


def looks_like_symbol(q: str) -> bool:
    q = q.strip()
    return bool(SYMBOL_RE.match(q)) and (q.isupper() or "." in q or "^" in q or "=" in q)


def quick_info(symbol: str) -> dict:
    """
    Metadata via yfinance's fast_info (currency, name, exchange, last price).

    fast_info is the only cheap, auth-free source for quote currency that works
    across equities, indices, futures and crypto. Returns {} on failure so
    callers can degrade gracefully instead of crashing.
    """
    try:
        with quiet_yfinance():
            fi = yf.Ticker(symbol).fast_info
            return {
                "currency": fi.get("currency") or "",
                "quote_type": fi.get("quoteType") or "",
                "exchange": fi.get("exchange") or "",
                "last_price": fi.get("lastPrice"),
            }
    except Exception:
        return {}


def _probe(symbol: str, period: str = "5d") -> pd.Series | None:
    try:
        with quiet_yfinance():
            h = yf.Ticker(symbol).history(period=period, interval="1d",
                                          auto_adjust=True)
        if h is None or h.empty or "Close" not in h:
            return None
        s = h["Close"].dropna()
        return s if not s.empty else None
    except Exception:
        return None


def _save(cache_file: Path | None, res: Resolution) -> None:
    if not cache_file:
        return
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(res.__dict__, indent=2, default=str))
    except Exception:
        pass


def resolve(query: str, cache_dir: Path | None = None,
            allow_search: bool = True, pick: int | None = None) -> Resolution:
    """
    Resolve a ticker symbol or company name to a Yahoo symbol.

    Direct symbols are probed first (cheap, no search index). If that fails we
    fall back to Yahoo's search. `pick` selects the Nth candidate (1-based) so a
    user can choose a specific listing rather than taking the top hit - which
    matters because Yahoo often ranks a US ADR above the home-market line
    ("hdfc bank" returns HDB before HDFCBANK.NS).
    """
    query = query.strip()
    key = re.sub(r"[^A-Za-z0-9]", "_", query) + (f"_pick{pick}" if pick else "")
    cache_file = (cache_dir / f"resolve_{key}.json") if cache_dir else None
    if cache_file and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            data["candidates"] = data.get("candidates", [])
            return Resolution(**data)
        except Exception:
            cache_file.unlink(missing_ok=True)

    # 1. Direct symbol probe. Probe at most once - a name-shaped input would
    #    otherwise pay two network round-trips before falling through to search.
    if pick is None and _probe(query) is not None:
        info = quick_info(query)
        res = Resolution(
            query=query, symbol=query, name=query,
            currency=info.get("currency", ""),
            quote_type=info.get("quote_type", ""),
            exchange=info.get("exchange", ""),
            last_price=info.get("last_price"),
            method="direct-symbol")
        _save(cache_file, res)
        return res

    if not allow_search:
        raise LookupError(f"'{query}' is not a valid Yahoo symbol and search is disabled")

    # 2. Name search.
    try:
        with quiet_yfinance():
            quotes = yf.Search(query, max_results=12).quotes or []
    except Exception as exc:
        raise LookupError(f"Yahoo search failed for '{query}': {exc}") from exc

    cands = [{
        "symbol": q.get("symbol"),
        "name": q.get("longname") or q.get("shortname") or "",
        "exchange": q.get("exchDisp") or "",
        "quote_type": q.get("quoteType") or "",
    } for q in quotes if q.get("symbol")]

    if not cands:
        raise LookupError(
            f"No instrument found for '{query}'.\n"
            f"  Use an explicit Yahoo symbol: RELIANCE.NS (NSE), 7203.T (Tokyo), "
            f"VOD.L (LSE), ^NSEI (index), GC=F (gold future), BTC-USD (crypto).")

    # An explicit choice by the user always wins.
    if pick is not None:
        if pick < 1 or pick > len(cands):
            raise LookupError(
                f"--pick {pick} is out of range; '{query}' has {len(cands)} "
                f"matches:\n" + _fmt_candidates(cands))
        chosen = cands[pick - 1]
        info = quick_info(chosen["symbol"])
        res = Resolution(query=query, symbol=chosen["symbol"], name=chosen["name"],
                         exchange=chosen["exchange"] or info.get("exchange", ""),
                         currency=info.get("currency", ""),
                         quote_type=chosen["quote_type"],
                         last_price=info.get("last_price"),
                         candidates=cands, method="search-pick")
        _save(cache_file, res)
        return res

    # Exact symbol match wins outright.
    for c in cands:
        if str(c["symbol"]).upper() == query.upper():
            info = quick_info(c["symbol"])
            res = Resolution(query=query, symbol=c["symbol"], name=c["name"],
                             exchange=c["exchange"] or info.get("exchange", ""),
                             currency=info.get("currency", ""),
                             quote_type=c["quote_type"],
                             last_price=info.get("last_price"),
                             candidates=cands, method="search-exact")
            _save(cache_file, res)
            return res

    top = cands[0]
    info = quick_info(top["symbol"])
    res = Resolution(query=query, symbol=top["symbol"], name=top["name"],
                     exchange=top["exchange"] or info.get("exchange", ""),
                     currency=info.get("currency", ""),
                     quote_type=top["quote_type"],
                     last_price=info.get("last_price"),
                     candidates=cands, method="search-fuzzy")
    _save(cache_file, res)
    return res


def _fmt_candidates(cands: list[dict], limit: int = 12) -> str:
    return "\n".join(
        f"  {i}. {c['symbol']:14s} {str(c['name'])[:42]:42s} "
        f"({c['exchange']}, {c['quote_type']})"
        for i, c in enumerate(cands[:limit], start=1))


def resolution_warning(res: Resolution) -> str | None:
    """
    Flag a match that a user should confirm before trusting the numbers.

    Search resolves to whatever ranks highest, and for many companies that is a
    foreign listing rather than the one meant. Silently analysing the wrong
    instrument would be worse than asking.
    """
    if res.method != "search-fuzzy" or len(res.candidates) < 2:
        return None
    others = [c for c in res.candidates if str(c["symbol"]) != res.symbol]
    venues = {str(c["exchange"]).strip() for c in res.candidates if c["exchange"]}
    if len(venues) > 1:
        alt = ", ".join(f"{c['symbol']} ({c['exchange']})" for c in others[:4])
        return (f"'{res.query}' was matched by name to {res.symbol} "
                f"({res.exchange}). Matches exist on other venues: {alt}. "
                f"These are different securities with different liquidity and "
                f"prices. Re-run with an explicit symbol (e.g. RELIANCE.NS) or use "
                f"--list to see all matches.")
    return None


# --------------------------------------------------------------- FX plumbing --

def normalise_currency(cur: str) -> tuple[str, float]:
    """Map a Yahoo quote currency to (major currency, minor units per major)."""
    cur = (cur or "USD").strip()
    if cur in MINOR_UNITS:
        return MINOR_UNITS[cur]
    if len(cur) == 3 and cur.isupper():
        return cur, 1.0
    return "USD", 1.0


def fx_symbol_for(currency: str) -> str | None:
    """Yahoo FX symbol quoted as "major units per 1 USD" (None for USD itself)."""
    maj, _ = normalise_currency(currency)
    return FX_TO_PER_USD.get(maj)


def fx_usd_per_series(currencies: list[str], cache_dir: Path | None = None,
                      max_age_days: int = 3, period: str = "2y") -> pd.DataFrame:
    """
    Time-varying USD value of one major currency unit, indexed by date.

    Deliberately a *series*, not a scalar. Converting a price history with a
    single spot rate would hold FX fixed for the whole sample, which fabricates
    a constant drift and discards the very FX variation we are trying to control
    for. FX_TO_PER_USD symbols quote "major per USD", so USD per major = 1/rate.
    """
    want = {normalise_currency(c)[0] for c in currencies}
    cols: dict[str, pd.Series] = {}
    for ccy in sorted(want):
        sym = FX_TO_PER_USD.get(ccy)
        if sym is None:                       # USD: identity, handled by caller
            continue
        s = _fx_series(sym, cache_dir, max_age_days, period)
        if s is not None and not s.empty:
            cols[ccy] = (1.0 / s).rename(ccy)
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index().ffill()


def _fx_series(symbol: str, cache_dir: Path | None, max_age_days: int,
               period: str = "2y") -> pd.Series | None:
    """Daily FX close for `symbol`, cached to disk for `max_age_days`.

    `period` must span at least the requested price window: a short FX series
    silently truncates the aligned sample, quietly reporting a shorter history
    than the user asked for.
    """
    cache_file = None
    if cache_dir:
        tag = symbol.replace("=", "_").replace("^", "")
        cache_file = cache_dir / f"fx_{tag}_{period}.csv"
        if cache_file.exists():
            try:
                age = (pd.Timestamp.now().normalize()
                       - pd.Timestamp(cache_file.stat().st_mtime, unit="s").normalize()).days
                if age <= max_age_days:
                    s = pd.read_csv(cache_file, index_col=0,
                                    parse_dates=True)["Close"].dropna()
                    if not s.empty:
                        return s
            except Exception:
                cache_file.unlink(missing_ok=True)

    try:
        with quiet_yfinance():
            h = yf.Ticker(symbol).history(period=period, interval="1d",
                                          auto_adjust=True)
        if h is None or h.empty:
            return None
        s = h["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        if cache_file:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                s.to_csv(cache_file, index_label="Date")
            except Exception:
                pass
        return s
    except Exception:
        return None
