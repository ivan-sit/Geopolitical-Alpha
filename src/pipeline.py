"""
pipeline.py — Shared data-loading wrapper used by all notebooks.

Priority order for CRSP data (auto-detected):
  1. Dropbox  — fetches new per-date format directly, caches in .crsp_cache/
  2. Local per-date  — YYYYMMDD.csv.gz files already downloaded
  3. Local matrix    — old pvCLCL_*.csv format

Usage in any notebook (cell 1):
    import sys; sys.path.insert(0, '..')
    from src.pipeline import load_all

    data = load_all()
    X, Y_idio, Y_raw = data['X'], data['Y_idio'], data['Y_raw']

To use Dropbox:
    data = load_all(source='dropbox')

To get more reliable Dropbox access, set your token:
    export DROPBOX_TOKEN=your_token_here       # in terminal
    data = load_all(source='dropbox', dropbox_token='sl.xxx')

How to get a free Dropbox token (2 min):
    1. https://www.dropbox.com/developers/apps
    2. Create App → Scoped → Full Dropbox → any name
    3. Permissions: check files.content.read + sharing.read
    4. Settings → Generate access token → copy it
"""
from __future__ import annotations

from pathlib import Path
import sys
import os

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import (
    load_crsp_matrix, load_crsp_auto, load_crsp_dropbox,
    load_commodity_futures, load_market_proxy,
)
from src.features.har import build_full_feature_matrix, residualize_against_market


def load_all(
    source: str = "auto",
    # Dropbox settings
    dropbox_token: str | None = None,
    dropbox_cache: str | Path = ROOT / ".crsp_cache",
    # Local file settings
    data_path: str = "US_CRSP_NYSE/Matrix_Format_SubsetUniverse",
    sectors_csv: str = "US_CRSP_NYSE/Sectors/Sectors_SP500_YahooNWikipedia.csv",
    # Date range
    start: str = "2000-01-01",
    end: str = "2024-12-31",
    # Pipeline settings
    sector: str = "Energy",
    beta_window: int = 252,
    verbose: bool = True,
) -> dict:
    """
    Run the full data pipeline and return all aligned datasets.

    Args:
        source:   'dropbox' | 'local' | 'auto' (tries dropbox first, then local)
        dropbox_token: Dropbox API token (or set DROPBOX_TOKEN env var)
        dropbox_cache: Local cache directory for Dropbox files
        data_path:     Path to local CRSP data (used if source='local'/'auto')
        sectors_csv:   Path to sector labels CSV (for matrix format)
        start / end:   Date range
        sector:        Sector to filter to (default: 'Energy')
        beta_window:   Rolling window for market beta (default: 252 days)
        verbose:       Print progress

    Returns dict with keys:
        X           - Feature matrix (HAR + derived, all lagged, no look-ahead)
        Y_idio      - Market-residualized idiosyncratic returns  ← Lasso target
        Y_raw       - Raw stock returns                          ← PnL calculation
        spy_ret     - SPY daily returns
        comm_prices - Forward-filled commodity prices (aligned to equity calendar)
        equity_cal  - Full equity trading calendar
    """
    # ── Step 1: Load stock returns ─────────────────────────────────────────
    stock_ret = _load_stock_returns(
        source=source,
        dropbox_token=dropbox_token,
        dropbox_cache=dropbox_cache,
        data_path=data_path,
        sectors_csv=sectors_csv,
        start=start, end=end,
        sector=sector,
        verbose=verbose,
    )
    equity_cal = stock_ret.index

    # ── Step 2: Commodity futures (forward-filled to equity calendar) ──────
    if verbose:
        print("\nLoading commodity futures (yfinance, forward-filling)...")
    comm_prices = load_commodity_futures(equity_cal, start=start, end=end)

    # ── Step 3: Market proxy (SPY) ─────────────────────────────────────────
    if verbose:
        print("Loading SPY...")
    spy_prices = load_market_proxy(equity_cal, start=start, end=end)
    spy_ret = spy_prices.pct_change(fill_method=None)

    # ── Step 4: HAR + derived features ────────────────────────────────────
    if verbose:
        print("Building HAR + derived features...")
    X_full = build_full_feature_matrix(comm_prices, clip=1.0)
    if verbose:
        print(f"  Feature matrix: {X_full.shape[0]} days × {X_full.shape[1]} features")

    # ── Step 5: Align all on common dates ─────────────────────────────────
    common = (
        X_full.index
        .intersection(stock_ret.index)
        .intersection(spy_ret.dropna().index)
    )
    X_aligned  = X_full.loc[common]
    Y_raw_all  = stock_ret.loc[common]
    spy_common = spy_ret.loc[common]

    # ── Step 6: Market residualization ────────────────────────────────────
    if verbose:
        print(f"Residualizing against market (rolling {beta_window}d beta)...")
    Y_idio_all = residualize_against_market(Y_raw_all, spy_common, window=beta_window)

    # Drop burn-in rows
    valid = Y_idio_all.dropna(how="all").index
    X      = X_aligned.loc[valid]
    Y_idio = Y_idio_all.loc[valid]
    Y_raw  = Y_raw_all.loc[valid]
    spy_v  = spy_common.loc[valid]

    if verbose:
        print(f"\nPipeline ready:")
        print(f"  X      = {X.shape}   (features)")
        print(f"  Y_idio = {Y_idio.shape}   (idiosyncratic returns)")
        print(f"  Y_raw  = {Y_raw.shape}   (raw returns for PnL)")
        print(f"  Range  = {valid[0].date()} → {valid[-1].date()}")

    return {
        "X":           X,
        "Y_idio":      Y_idio,
        "Y_raw":       Y_raw,
        "spy_ret":     spy_v,
        "comm_prices": comm_prices.loc[comm_prices.index.intersection(valid)],
        "equity_cal":  equity_cal,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_stock_returns(
    source, dropbox_token, dropbox_cache,
    data_path, sectors_csv,
    start, end, sector, verbose,
) -> pd.DataFrame:
    """Try loading from Dropbox, then fall back to local files."""

    token = dropbox_token or os.environ.get("DROPBOX_TOKEN")
    # Fall back to config.py if still no token
    if not token:
        try:
            from config import DROPBOX_TOKEN as _cfg_token
            token = _cfg_token or None
        except ImportError:
            pass

    # ── Cache-only source (no Dropbox calls) ──────────────────────────────
    if source == "cache":
        if verbose:
            print("Loading CRSP from local cache (no Dropbox)...")
        from src.data.dropbox_fetcher import DropboxCRSPFetcher, _ENERGY_SIC
        fetcher = DropboxCRSPFetcher(
            cache_dir=dropbox_cache, token=None,
            verbose=verbose, readonly=True,
        )
        if sector.lower() == "energy":
            df = fetcher.load_energy_stocks(start, end)
        else:
            df = fetcher.load_date_range(start, end)
        if verbose:
            print(f"  {df.shape[1]} stocks × {len(df)} days  "
                  f"[{df.index[0].date()} → {df.index[-1].date()}]")
        return df

    # ── Dropbox source ────────────────────────────────────────────────────
    if source in ("dropbox", "auto"):
        if verbose:
            print("Loading CRSP from Dropbox (new per-date format, 1990-2024)...")
        try:
            df = load_crsp_dropbox(
                start=start, end=end,
                sector=sector.lower(),
                cache_dir=dropbox_cache,
                token=token,
            )
            if verbose:
                print(f"  {df.shape[1]} stocks × {len(df)} days  "
                      f"[{df.index[0].date()} → {df.index[-1].date()}]")
            return df
        except Exception as e:
            if source == "dropbox":
                raise
            if verbose:
                print(f"  Dropbox failed ({e}), falling back to local data...")

    # ── Local fallback ────────────────────────────────────────────────────
    local_path = ROOT / data_path
    sectors_path = ROOT / sectors_csv if Path(ROOT / sectors_csv).exists() else None

    if verbose:
        print(f"Loading CRSP from local files: {local_path} ...")

    df = load_crsp_auto(local_path, sector_csv=sectors_path, sector=sector)
    if verbose:
        print(f"  {df.shape[1]} stocks × {len(df)} days")
    return df
