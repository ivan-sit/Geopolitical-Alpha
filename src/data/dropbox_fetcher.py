"""
Dropbox CRSP Fetcher
====================
Stream per-date CRSP .csv.gz files from the professor's shared Dropbox folder.
Files are cached locally so each date is only downloaded once.

Quick start (no token needed for public links):
    from src.data.dropbox_fetcher import DropboxCRSPFetcher
    fetcher = DropboxCRSPFetcher()
    df = fetcher.load_date_range("2020-01-01", "2020-03-31")

If the public link stops working, get a free token in 2 minutes:
    1. Go to https://www.dropbox.com/developers/apps
    2. Create App → Scoped → Full Dropbox → any name
    3. Permissions tab → check "sharing.read" and "files.content.read"
    4. Settings tab → Generated access token → copy it
    5. Set DROPBOX_TOKEN=<token> in your environment or .env file
"""
from __future__ import annotations

import gzip
import io
import os
import re
import time
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

# ── Shared folder URL from the professor ──────────────────────────────────────
_FOLDER_URL = (
    "https://www.dropbox.com/scl/fo/bx9jv5x2fnu9j02i5j77a/"
    "AGljwCnU10aplUA6BZ3AwrU?rlkey=7jeselv0gnki38q4dvrzbhbsc"
)
_RLKEY = "7jeselv0gnki38q4dvrzbhbsc"
_FOLDER_ID = "bx9jv5x2fnu9j02i5j77a"
_FOLDER_HASH = "AGljwCnU10aplUA6BZ3AwrU"

# Direct-download URL templates to try (in order)
_URL_TEMPLATES = [
    # Template 1: dl.dropboxusercontent.com CDN (most common for public links)
    "https://dl.dropboxusercontent.com/scl/fo/{folder_id}/{folder_hash}/{filename}"
    "?rlkey={rlkey}",
    # Template 2: www.dropbox.com with dl=1
    "https://www.dropbox.com/scl/fo/{folder_id}/{folder_hash}/{filename}"
    "?rlkey={rlkey}&dl=1",
]

# Energy sector SIC major groups (for filtering the large universe)
_ENERGY_SIC = {13, 29}   # 13=Oil & Gas, 29=Petroleum Refining


# ──────────────────────────────────────────────────────────────────────────────


class DropboxCRSPFetcher:
    """
    Fetch and cache CRSP per-date files from the professor's shared Dropbox.

    Args:
        cache_dir:   Local directory for cached files (default: .crsp_cache/)
        token:       Dropbox API access token (optional — tries public access first)
        max_workers: Parallel download threads (default: 8)
        verbose:     Print progress messages
    """

    def __init__(
        self,
        cache_dir: str | Path = ".crsp_cache",
        token: Optional[str] = None,
        max_workers: int = 8,
        verbose: bool = True,
        readonly: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.verbose = verbose
        self.readonly = readonly   # if True, never download — read cache only

        # Try env var, then constructor arg
        self.token = token or os.environ.get("DROPBOX_TOKEN")

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0"})

    # ── Public API ────────────────────────────────────────────────────────────

    def load_date_range(
        self,
        start: str,
        end: str,
        shrcd: tuple[int, ...] = (10, 11),
        sector_sic: Optional[set[int]] = None,
        return_col: str = "pvCLCL",
        tickers: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load a date range of CRSP data, fetching from Dropbox as needed.

        Args:
            start / end:  Date strings (inclusive), e.g. "2020-01-01"
            shrcd:        Share codes to keep (10,11 = common shares)
            sector_sic:   Set of SIC major groups; None = all sectors
            return_col:   Return column to use ('pvCLCL', 'RET', 'RET_fwd')
            tickers:      If given, restrict to these tickers

        Returns:
            DataFrame: rows=dates, cols=tickers, values=returns (decimal)
        """
        dates = self._trading_dates(start, end)
        if self.verbose:
            print(f"Loading {len(dates)} trading days ({start} → {end}) ...")

        # Download any missing files in parallel
        self._ensure_cached(dates)

        # Read and assemble
        frames = []
        for dt in dates:
            day = self._read_cached(dt, shrcd=shrcd, sector_sic=sector_sic,
                                    return_col=return_col, tickers=tickers)
            if day is not None:
                frames.append(day)

        if not frames:
            raise ValueError("No data loaded — check date range and connectivity.")

        df = pd.concat(frames).sort_index()
        df.index.name = "date"
        if self.verbose:
            print(f"Loaded: {df.shape[1]} tickers × {df.shape[0]} days")
        return df

    def load_energy_stocks(
        self,
        start: str,
        end: str,
        return_col: str = "pvCLCL",
    ) -> pd.DataFrame:
        """
        Convenience wrapper: load only energy sector stocks.
        Filters to SHRCD=10/11 and SIC groups 13 (oil & gas) + 29 (refining).
        """
        return self.load_date_range(
            start, end,
            shrcd=(10, 11),
            sector_sic=_ENERGY_SIC,
            return_col=return_col,
        )

    def peek(self, date: str = "2020-01-02") -> pd.DataFrame:
        """
        Download and display one day of data to inspect columns.

        Useful for understanding the new dataset structure before loading all.
        """
        dt = pd.Timestamp(date)
        self._ensure_cached([dt])
        path = self._cache_path(dt)
        if not path.exists():
            raise FileNotFoundError(f"Could not fetch {date}")
        df = pd.read_csv(path)
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSHRCD distribution:\n{df['SHRCD'].value_counts().head() if 'SHRCD' in df.columns else 'N/A'}")
        print(f"\nSample rows:")
        return df.head(10)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _trading_dates(self, start: str, end: str) -> list[pd.Timestamp]:
        """Generate US business days in range (approximate equity calendar)."""
        return list(pd.bdate_range(start=start, end=end, freq="B"))

    def _cache_path(self, dt: pd.Timestamp) -> Path:
        return self.cache_dir / f"{dt.strftime('%Y%m%d')}.csv.gz"

    def _ensure_cached(self, dates: list[pd.Timestamp]):
        """Download any dates not already in cache (parallel)."""
        missing = [dt for dt in dates if not self._cache_path(dt).exists()]
        if not missing:
            if self.verbose:
                print(f"  All {len(dates)} files already in local cache — skipping download")
            return
        if self.readonly:
            if self.verbose:
                print(f"  Cache-only mode: skipping download of {len(missing)} missing files")
            return

        if self.verbose:
            cached = len(dates) - len(missing)
            if cached:
                print(f"  {cached} files already cached, fetching {len(missing)} new files from Dropbox ...")
            else:
                print(f"  Fetching {len(missing)} files from Dropbox ...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self._fetch_one, dt): dt for dt in missing}
            ok, fail = 0, 0
            for fut in as_completed(futures):
                dt = futures[fut]
                try:
                    fut.result()
                    ok += 1
                except Exception as e:
                    fail += 1
                    if self.verbose:
                        print(f"  WARN: could not fetch {dt.date()}: {e}")

        if self.verbose:
            print(f"  Downloaded: {ok} OK, {fail} failed")
            if fail > 0:
                print("  ⚠  Some files could not be fetched.")
                print("  → Get a free token: https://www.dropbox.com/developers/apps")
                print("    Then: export DROPBOX_TOKEN=sl.xxx  (or set in config.py)")
            if ok == 0 and fail > 0:
                raise RuntimeError(
                    "All downloads failed — Dropbox public access is blocked.\n"
                    "Set DROPBOX_TOKEN env var and retry."
                )

    def _fetch_one(self, dt: pd.Timestamp, retries: int = 3):
        """Download one date file and save to cache."""
        filename = f"{dt.strftime('%Y%m%d')}.csv.gz"
        out_path = self._cache_path(dt)

        # Dropbox API (most reliable — use token if available)
        if self.token:
            try:
                self._fetch_via_api(filename, out_path)
                return
            except Exception as e:
                last_err = e

        # Public URL fallback: visit folder first to set session cookies,
        # then request the individual file with dl=1
        last_err = None
        try:
            # Step 1: establish session cookie via folder URL
            self._session.get(_FOLDER_URL, timeout=15)
        except requests.RequestException:
            pass  # best-effort — proceed anyway

        for template in _URL_TEMPLATES:
            url = template.format(
                folder_id=_FOLDER_ID,
                folder_hash=_FOLDER_HASH,
                filename=filename,
                rlkey=_RLKEY,
            )
            try:
                resp = self._session.get(url, timeout=30)
                # Validate: must be a real gzip file (magic bytes \x1f\x8b)
                if (resp.status_code == 200
                        and len(resp.content) > 100
                        and resp.content[:2] == b'\x1f\x8b'):
                    out_path.write_bytes(resp.content)
                    return
            except requests.RequestException as e:
                last_err = e

        raise RuntimeError(
            f"Failed to fetch {filename}. Last error: {last_err}. "
            "Set DROPBOX_TOKEN env var for reliable access: "
            "https://www.dropbox.com/developers/apps"
        )

    def _fetch_via_api(self, filename: str, out_path: Path):
        """Download using Dropbox API files/download endpoint.
        Files live at /US_CRSP_NYSE/Yearly/{year}/{filename}.
        """
        import json

        # Extract year from filename (format: YYYYMMDD.csv.gz)
        year = filename[:4]
        dropbox_path = f"/US_CRSP_NYSE/Yearly/{year}/{filename}"

        url = "https://content.dropboxapi.com/2/files/download"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Dropbox-API-Arg": json.dumps({"path": dropbox_path}),
        }
        resp = requests.post(url, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Dropbox API returned {resp.status_code}: {resp.text[:200]}"
            )
        if resp.content[:2] != b'\x1f\x8b':
            raise RuntimeError(
                f"Downloaded file is not gzip (got {resp.content[:20]})"
            )
        out_path.write_bytes(resp.content)

    def _read_cached(
        self,
        dt: pd.Timestamp,
        shrcd: tuple[int, ...],
        sector_sic: Optional[set[int]],
        return_col: str,
        tickers: Optional[list[str]],
    ) -> Optional[pd.Series]:
        """Read one cached CSV.gz and return a Series (ticker → return)."""
        path = self._cache_path(dt)
        if not path.exists():
            return None

        # Validate gzip magic bytes before trying to read
        if path.read_bytes()[:2] != b'\x1f\x8b':
            path.unlink()   # delete corrupt file so it gets re-fetched next run
            return None

        try:
            day = pd.read_csv(path)
        except Exception:
            path.unlink()   # delete unreadable file
            return None

        # Filter share codes
        if "SHRCD" in day.columns and shrcd:
            day = day[day["SHRCD"].isin(shrcd)]

        # Filter sector by SIC major group
        if sector_sic is not None and "HSICMG" in day.columns:
            day = day[day["HSICMG"].isin(sector_sic)]

        # Filter to specific tickers
        if tickers is not None:
            ticker_col = next(
                (c for c in ["TICKER", "ticker", "Ticker"] if c in day.columns), None
            )
            if ticker_col:
                day = day[day[ticker_col].isin(tickers)]

        if return_col not in day.columns:
            return None

        ticker_col = next(
            (c for c in ["TICKER", "ticker", "Ticker"] if c in day.columns), None
        )
        if ticker_col is None:
            return None

        ret = (
            day.set_index(ticker_col)[return_col]
            .apply(pd.to_numeric, errors="coerce")
        )
        ret.name = dt
        return pd.DataFrame({dt: ret}).T


# ──────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ──────────────────────────────────────────────────────────────────────────────


def load_from_dropbox(
    start: str = "2008-01-01",
    end: str = "2024-12-31",
    sector: str = "energy",
    cache_dir: str | Path = ".crsp_cache",
    token: Optional[str] = None,
    return_col: str = "pvCLCL",
) -> pd.DataFrame:
    """
    One-line loader: fetch CRSP data from Dropbox for a date range.

    Example:
        from src.data.dropbox_fetcher import load_from_dropbox
        stock_ret = load_from_dropbox("2010-01-01", "2020-12-31")

    Args:
        start / end:  Date range
        sector:       'energy' (default) or 'all'
        cache_dir:    Local cache directory
        token:        Dropbox API token (env var DROPBOX_TOKEN also works)
        return_col:   Column for returns

    Returns:
        DataFrame: rows=dates, cols=tickers
    """
    fetcher = DropboxCRSPFetcher(cache_dir=cache_dir, token=token)

    if sector.lower() == "energy":
        return fetcher.load_energy_stocks(start, end, return_col=return_col)
    else:
        return fetcher.load_date_range(start, end, return_col=return_col)
