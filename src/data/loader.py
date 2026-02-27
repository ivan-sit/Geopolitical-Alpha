"""
Data loading utilities for CRSP and commodity data.

Three data sources supported (auto-detected by load_crsp_auto):
  1. Dropbox (new, recommended) — streams per-date files directly from the
     professor's shared Dropbox folder, caches locally in .crsp_cache/
  2. OLD matrix format — pvCLCL_YYYYMMDD_YYYYMMDD.csv (one matrix per field)
  3. NEW per-date local — YYYYMMDD.csv.gz  (downloaded locally)

New columns available in the per-date format:
  SHRCD    - Share code: 10/11 = common shares, 73/74 = ETFs
  PRIMEXCH - Primary exchange (N=NYSE, A=AMEX, Q=Nasdaq)
  BID/ASK  - Closing bid/ask (for transaction cost estimation)
  DLRET    - Delisting return (survivorship bias correction)
  PRC      - Closing price
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ──────────────────────────────────────────────────────────────────────────────
# CRSP loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_crsp_matrix(data_path: str | Path,
                     field: str = "pvCLCL",
                     sector_csv: Optional[str | Path] = None,
                     sector: str = "Energy") -> pd.DataFrame:
    """
    Load the OLD matrix-format CRSP data.

    Args:
        data_path:  Path to Matrix_Format_SubsetUniverse/
        field:      'pvCLCL' (prev-close-to-close returns), 'OPCL', 'volume'
        sector_csv: Path to Sectors CSV; if given, filter to energy tickers
        sector:     Sector label to keep (Wikipedia column)

    Returns:
        DataFrame: rows=dates, cols=tickers, values=returns (decimal)
    """
    data_path = Path(data_path)
    files = list(data_path.glob(f"{field}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No {field}_*.csv found in {data_path}")

    raw = pd.read_csv(files[0], index_col=0)
    df = raw.T.copy()
    df.index = pd.to_datetime(df.index.str.replace("X", ""), format="%Y%m%d")
    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce")

    if sector_csv is not None:
        sectors = pd.read_csv(sector_csv)
        tickers = sectors.loc[sectors["Sector_Wikipedia"] == sector, "Ticker"].tolist()
        avail = [t for t in tickers if t in df.columns]
        df = df[avail]

    return df


def load_crsp_perdate(data_path: str | Path,
                      start: str = "1990-01-01",
                      end: str = "2024-12-31",
                      shrcd: tuple[int, ...] = (10, 11),
                      sector_sic: Optional[tuple[int, ...]] = None,
                      return_col: str = "pvCLCL") -> pd.DataFrame:
    """
    Load the NEW per-date CRSP format (individual YYYYMMDD.csv.gz files).

    Each file contains one row per ticker for that trading day.
    Filters to common shares (SHRCD 10/11 by default).

    Args:
        data_path:   Directory containing YYYYMMDD.csv.gz files
        start/end:   Date range (inclusive)
        shrcd:       Share codes to keep (10,11 = common shares)
        sector_sic:  Optional tuple of SIC major groups (HSICMG) to filter
        return_col:  Column to use as return; 'pvCLCL' or 'RET_fwd' (forward)

    Returns:
        DataFrame: rows=dates, cols=tickers, values=returns (decimal)
    """
    data_path = Path(data_path)
    files = sorted(data_path.glob("*.csv.gz"))
    if not files:
        files = sorted(data_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No csv.gz files found in {data_path}")

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    frames = []
    for f in files:
        # Parse date from filename
        try:
            date_str = f.stem.replace(".csv", "")  # handles .csv.gz
            dt = pd.Timestamp(date_str)
        except Exception:
            continue
        if dt < start_dt or dt > end_dt:
            continue

        day = pd.read_csv(f)

        # Filter to common shares
        if "SHRCD" in day.columns and shrcd:
            day = day[day["SHRCD"].isin(shrcd)]

        # Filter by SIC major group
        if sector_sic is not None and "HSICMG" in day.columns:
            day = day[day["HSICMG"].isin(sector_sic)]

        if return_col not in day.columns:
            continue

        ticker_col = "TICKER" if "TICKER" in day.columns else "ticker"
        day = day.set_index(ticker_col)[[return_col]]
        day.columns = [dt]
        frames.append(day.T)

    if not frames:
        raise ValueError("No data loaded — check date range and file format.")

    df = pd.concat(frames).sort_index()
    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_crsp_auto(data_path: str | Path,
                   sector_csv: Optional[str | Path] = None,
                   **kwargs) -> pd.DataFrame:
    """
    Auto-detect CRSP format (matrix or per-date) and load.

    Args:
        data_path:  Either Matrix_Format_SubsetUniverse/ or per-date directory
        sector_csv: Optional sector filter CSV (for matrix format)
        **kwargs:   Passed to the appropriate loader

    Returns:
        DataFrame of returns
    """
    data_path = Path(data_path)
    if list(data_path.glob("pvCLCL_*.csv")):
        return load_crsp_matrix(data_path, sector_csv=sector_csv, **kwargs)
    elif list(data_path.glob("*.csv.gz")) or list(data_path.glob("*.csv")):
        return load_crsp_perdate(data_path, **kwargs)
    else:
        raise FileNotFoundError(f"Cannot detect CRSP format in {data_path}")


def load_crsp_dropbox(
    start: str = "2008-01-01",
    end: str = "2024-12-31",
    sector: str = "energy",
    cache_dir: str | Path = ".crsp_cache",
    token: Optional[str] = None,
    return_col: str = "pvCLCL",
) -> pd.DataFrame:
    """
    Load CRSP data directly from the professor's Dropbox shared folder.

    No manual download required — files are fetched automatically and cached
    locally in `cache_dir` so each date is only downloaded once.

    This uses the new per-date format with the extended 1990-2024 universe.

    Args:
        start / end:  Date range (e.g. "2008-01-01" to "2024-12-31")
        sector:       'energy' (SIC 13+29 only) or 'all' (entire universe)
        cache_dir:    Local folder for caching (default: .crsp_cache/)
        token:        Dropbox API token for reliable access.
                      Can also set env var DROPBOX_TOKEN.
                      Get one free at https://www.dropbox.com/developers/apps
        return_col:   'pvCLCL' (prev-close-to-close) or 'RET_fwd' (forward)

    Returns:
        DataFrame: rows=dates, cols=tickers, values=returns (decimal)

    Example:
        stock_ret = load_crsp_dropbox("2008-01-01", "2020-12-31")
    """
    from src.data.dropbox_fetcher import load_from_dropbox
    return load_from_dropbox(
        start=start, end=end, sector=sector,
        cache_dir=cache_dir, token=token, return_col=return_col,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Commodity loader
# ──────────────────────────────────────────────────────────────────────────────

COMMODITY_TICKERS = {
    "CL=F": "WTI",
    "BZ=F": "Brent",
    "NG=F": "NatGas",
    "HO=F": "HeatOil",
}


def load_commodity_futures(equity_calendar: pd.DatetimeIndex,
                           start: str = "2000-01-01",
                           end: str = "2024-12-31",
                           tickers: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """
    Download commodity futures via yfinance and forward-fill to the equity calendar.

    Forward-filling is critical: futures markets close on some days when equity
    markets trade (and vice versa). Without ffill, inner-joining creates holes
    in the return series and silently corrupts HAR lagged features.

    Args:
        equity_calendar: DatetimeIndex from the CRSP stock data (equity trading days)
        start / end:     Date range for download
        tickers:         {yfinance_symbol: short_name}; defaults to 4 main energy futures

    Returns:
        DataFrame of PRICES (not returns), aligned to equity_calendar, forward-filled
    """
    if tickers is None:
        tickers = COMMODITY_TICKERS

    prices = {}
    for symbol, name in tickers.items():
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if len(df) == 0:
            print(f"  WARNING: no data for {symbol} ({name})")
            continue
        close = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        prices[name] = close
        print(f"  {name} ({symbol}): {len(close)} rows  "
              f"[{close.index[0].date()} – {close.index[-1].date()}]")

    # Forward-fill to equity calendar: reindex → ffill → restrict
    aligned = {}
    for name, ser in prices.items():
        all_dates = ser.index.union(equity_calendar)
        aligned[name] = ser.reindex(all_dates).ffill().reindex(equity_calendar)

    comm_prices = pd.DataFrame(aligned)
    comm_prices.index.name = "date"
    return comm_prices


def load_market_proxy(equity_calendar: pd.DatetimeIndex,
                      symbol: str = "SPY",
                      start: str = "2000-01-01",
                      end: str = "2024-12-31") -> pd.Series:
    """
    Download market proxy (SPY) and forward-fill to equity calendar.

    Returns:
        Series of SPY PRICES aligned to equity_calendar
    """
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    close = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = symbol

    all_dates = close.index.union(equity_calendar)
    return close.reindex(all_dates).ffill().reindex(equity_calendar)
