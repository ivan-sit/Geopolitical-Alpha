"""
HAR (Heterogeneous Autoregressive) feature construction + derived features.

HAR decomposition:
    r^(d)_t  = r_{t-1}                          (daily, lag 1)
    r^(w)_t  = mean(r_{t-5..t-1})               (weekly, 5-day avg)
    r^(m)_t  = mean(r_{t-22..t-1})              (monthly, 22-day avg)

Derived features:
    CrackSpread  = HeatOil_return - WTI_return   (refining margin proxy)
    OilGasRatio  = Δlog(WTI / NatGas)            (substitution signal)
    RV_{comm}    = rolling 22d std of daily return (volatility regime)
    mom63_{comm} = rolling 63d mean return         (quarterly momentum)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_har_features(comm_prices: pd.DataFrame,
                       clip: float = 1.0) -> pd.DataFrame:
    """
    Build the core HAR feature matrix from commodity prices.

    Args:
        comm_prices: DataFrame of commodity PRICES aligned to equity calendar
                     (already forward-filled). Columns = commodity names.
        clip:        Clip returns to ±clip (handles WTI negative-price artifacts)

    Returns:
        DataFrame with columns {Commodity}_{d/w/m}, index = trading dates.
        All features are lagged by at least 1 day (no look-ahead).
    """
    comm_ret = comm_prices.pct_change(fill_method=None).clip(-clip, clip)

    frames = []
    for commodity in comm_ret.columns:
        r = comm_ret[commodity]
        f = pd.DataFrame(index=comm_ret.index)
        f[f"{commodity}_d"] = r.shift(1)                      # yesterday
        f[f"{commodity}_w"] = r.rolling(5).mean().shift(1)    # 5-day avg
        f[f"{commodity}_m"] = r.rolling(22).mean().shift(1)   # 22-day avg
        frames.append(f)

    return pd.concat(frames, axis=1)


def build_derived_features(comm_prices: pd.DataFrame,
                           clip: float = 1.0) -> pd.DataFrame:
    """
    Build engineered cross-commodity features.

    Args:
        comm_prices: Same aligned price DataFrame as build_har_features()
        clip:        Return clip level

    Returns:
        DataFrame with derived feature columns, index = trading dates.
    """
    comm_ret = comm_prices.pct_change(fill_method=None).clip(-clip, clip)
    derived = pd.DataFrame(index=comm_ret.index)

    # ── Crack spread (HeatOil − WTI) ─────────────────────────────────────
    # Proxy for refining margin; especially relevant for refiners like VLO.
    if "HeatOil" in comm_ret.columns and "WTI" in comm_ret.columns:
        crack = comm_ret["HeatOil"] - comm_ret["WTI"]
        derived["CrackSpread_d"] = crack.shift(1)
        derived["CrackSpread_w"] = crack.rolling(5).mean().shift(1)

    # ── Oil / NatGas ratio ────────────────────────────────────────────────
    # Energy substitution and relative demand signal.
    if "WTI" in comm_prices.columns and "NatGas" in comm_prices.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log(
                comm_prices["WTI"] / comm_prices["NatGas"]
            ).replace([np.inf, -np.inf], np.nan)
        derived["OilGasRatio_d"] = ratio.diff().shift(1)

    # ── Realized volatility (22-day) ─────────────────────────────────────
    # Volatility regime: high RV often precedes large energy stock moves.
    for commodity in comm_ret.columns:
        derived[f"{commodity}_RV"] = comm_ret[commodity].rolling(22).std().shift(1)

    # ── Quarterly momentum (63-day) ───────────────────────────────────────
    # Trend-following signal on most liquid oil benchmarks.
    for commodity in ["WTI", "Brent"]:
        if commodity in comm_ret.columns:
            derived[f"{commodity}_mom63"] = comm_ret[commodity].rolling(63).mean().shift(1)

    # ── Brent-WTI spread (geopolitical risk premium) ──────────────────────
    # Widens during Middle East / Russia supply disruptions → key geo signal.
    if "Brent" in comm_ret.columns and "WTI" in comm_ret.columns:
        bwti = comm_ret["Brent"] - comm_ret["WTI"]
        derived["BrentWTI_d"] = bwti.shift(1)
        derived["BrentWTI_w"] = bwti.rolling(5).mean().shift(1)

    # ── Short-term momentum (5-day) ───────────────────────────────────────
    # Captures fast-moving geopolitical shock momentum (news → price lag).
    for commodity in ["WTI", "Brent"]:
        if commodity in comm_ret.columns:
            derived[f"{commodity}_mom5"] = comm_ret[commodity].rolling(5).mean().shift(1)

    # ── Volatility regime: short/long RV ratio ────────────────────────────
    # Rising ratio = vol spike → regime change signal for energy stocks.
    if "WTI" in comm_ret.columns:
        rv5  = comm_ret["WTI"].rolling(5).std()
        rv22 = comm_ret["WTI"].rolling(22).std().replace(0, np.nan)
        derived["WTI_VolRatio"] = (rv5 / rv22).shift(1)

    # ── Cross-commodity dispersion ─────────────────────────────────────────
    # High dispersion = commodities disagree → uncertain supply/demand.
    comm_5d_mean = pd.DataFrame({c: comm_ret[c].rolling(5).mean() for c in comm_ret.columns})
    derived["CommDispersion"] = comm_5d_mean.std(axis=1).shift(1)

    return derived


def build_full_feature_matrix(comm_prices: pd.DataFrame,
                               clip: float = 1.0) -> pd.DataFrame:
    """
    Combine HAR + derived features and drop rows with any NaN.

    Returns:
        Clean feature matrix X ready for Lasso, index = trading dates.
    """
    har = build_har_features(comm_prices, clip=clip)
    derived = build_derived_features(comm_prices, clip=clip)
    X = pd.concat([har, derived], axis=1).dropna()
    return X


def residualize_against_market(stock_ret: pd.DataFrame,
                                market_ret: pd.Series,
                                window: int = 252) -> pd.DataFrame:
    """
    Remove market factor from stock returns via rolling OLS beta.

    Y_idio[t] = r_stock[t] - β_rolling(t) × r_market[t]

    This ensures the Lasso looks for commodity-specific (geopolitical) alpha,
    not general market co-movement.

    Args:
        stock_ret:  DataFrame of raw stock returns (rows=dates, cols=tickers)
        market_ret: Series of market proxy returns (e.g. SPY)
        window:     Rolling window for beta estimation (default 252 trading days)

    Returns:
        DataFrame of idiosyncratic returns, same shape as stock_ret.
        First `window` rows will be NaN (burn-in period).
    """
    Y_idio = stock_ret.copy()
    mkt = market_ret.reindex(stock_ret.index)

    for stock in stock_ret.columns:
        betas = pd.Series(np.nan, index=stock_ret.index)
        y_s = stock_ret[stock]
        for t in range(window, len(stock_ret)):
            y_w = y_s.iloc[t - window:t].values
            x_w = mkt.iloc[t - window:t].values
            mask = ~(np.isnan(y_w) | np.isnan(x_w))
            if mask.sum() < 50:
                continue
            cov = np.cov(y_w[mask], x_w[mask])
            betas.iloc[t] = cov[0, 1] / (np.var(x_w[mask]) + 1e-10)

        Y_idio[stock] = stock_ret[stock] - betas * mkt

    return Y_idio
