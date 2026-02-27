"""
Long-short backtest engine.

Signal generation: cross-sectional ranking of OOS Lasso predictions.
  - Long top-N stocks (highest predicted idiosyncratic return)
  - Short bottom-N stocks (lowest predicted idiosyncratic return)
  - Market neutral: equal dollar weight within longs and shorts

PnL: earned on RAW returns (signals from idio predictions, but you hold the stock).
Transaction cost: charged on weight changes (one-way).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signals(predictions: pd.DataFrame,
                     top_pct: float = 0.25,
                     bottom_pct: float = 0.25) -> pd.DataFrame:
    """
    Convert cross-sectional predictions to long/short signals.

    Each day:
      - Rank stocks by predicted return
      - Long top `top_pct` (highest predicted idiosyncratic return)
      - Short bottom `bottom_pct` (lowest predicted idiosyncratic return)
      - Flat otherwise

    Args:
        predictions: DataFrame of OOS predicted returns (dates × tickers)
        top_pct:     Fraction of stocks to go long
        bottom_pct:  Fraction of stocks to go short

    Returns:
        DataFrame of signals: 1=long, -1=short, 0=flat
    """
    signals = pd.DataFrame(0, index=predictions.index, columns=predictions.columns)

    for date in predictions.index:
        row = predictions.loc[date].dropna()
        if len(row) < 4:
            continue
        n = len(row)
        n_long  = max(1, int(n * top_pct))
        n_short = max(1, int(n * bottom_pct))

        ranked = row.sort_values()
        signals.loc[date, ranked.head(n_short).index] = -1   # short losers
        signals.loc[date, ranked.tail(n_long).index]  =  1   # long winners

    return signals


def compute_weights(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Convert signals to dollar-neutral portfolio weights.

    Equal weight within longs and equal weight within shorts.
    Total long = +1, total short = -1 (zero net exposure).

    Args:
        signals: DataFrame with values in {-1, 0, 1}

    Returns:
        DataFrame of weights (same shape)
    """
    weights = signals.copy().astype(float)

    for date in weights.index:
        row = weights.loc[date]
        n_long  = (row == 1).sum()
        n_short = (row == -1).sum()
        if n_long  > 0: weights.loc[date, row == 1]  =  1.0 / n_long
        if n_short > 0: weights.loc[date, row == -1] = -1.0 / n_short

    return weights


def run_backtest(signals: pd.DataFrame,
                 raw_returns: pd.DataFrame,
                 transaction_cost: float = 0.0005) -> pd.Series:
    """
    Run the long-short backtest.

    PnL attribution:
      daily_PnL = Σ_i  w_i[t] × r_raw_i[t]  − TC[t]
    where TC[t] = transaction_cost × Σ_i |Δw_i[t]|

    Args:
        signals:          DataFrame from generate_signals()
        raw_returns:      DataFrame of actual stock returns (same shape)
        transaction_cost: One-way cost per unit weight changed (default 5bps)

    Returns:
        Series of daily portfolio returns
    """
    weights = compute_weights(signals)

    # Align returns to weight dates
    ret_aligned = raw_returns.reindex(index=weights.index, columns=weights.columns)

    portfolio_ret = (weights * ret_aligned).sum(axis=1)
    tc = weights.diff().abs().sum(axis=1) * transaction_cost
    portfolio_ret = portfolio_ret - tc

    return portfolio_ret


def compute_metrics(returns: pd.Series,
                    start: pd.Timestamp | None = None,
                    periods_per_year: int = 252) -> dict:
    """
    Compute standard performance metrics for a daily return series.

    Args:
        returns:          Daily return Series
        start:            If given, restrict to returns >= start (OOS period)
        periods_per_year: 252 for daily

    Returns:
        Dict with: Total Return, Ann. Return, Ann. Volatility,
                   Sharpe Ratio, Max Drawdown, Win Rate, Num Days
    """
    r = returns.dropna()
    if start is not None:
        r = r[r.index >= start]
    if len(r) == 0:
        return {}

    total_ret  = (1 + r).prod() - 1
    ann_ret    = (1 + r).prod() ** (periods_per_year / len(r)) - 1
    ann_vol    = r.std() * np.sqrt(periods_per_year)
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum       = (1 + r).cumprod()
    max_dd    = ((cum - cum.cummax()) / cum.cummax()).min()
    win_rate  = (r > 0).mean()

    return {
        "Total Return":    f"{total_ret:.2%}",
        "Ann. Return":     f"{ann_ret:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio":    f"{sharpe:.3f}",
        "Max Drawdown":    f"{max_dd:.2%}",
        "Win Rate":        f"{win_rate:.2%}",
        "Num Days":        len(r),
    }


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling annualized Sharpe ratio.

    Args:
        returns: Daily return Series
        window:  Rolling window in trading days

    Returns:
        Series of rolling Sharpe values
    """
    mu  = returns.rolling(window).mean() * 252
    vol = returns.rolling(window).std() * np.sqrt(252)
    return mu / vol
