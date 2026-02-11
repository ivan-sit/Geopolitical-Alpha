"""
Long-short strategy based on residuals.
"""

def compute_residuals(actual_returns, predicted_returns):
    """Compute residuals: epsilon = r - r_hat."""
    raise NotImplementedError


def generate_signals(residuals, threshold=1.0):
    """
    Generate long/short signals based on residual z-scores.
    Long if residual < -threshold, Short if residual > +threshold.
    """
    raise NotImplementedError


def backtest_strategy(signals, returns, transaction_cost=0.001):
    """
    Run backtest and compute PnL.
    Returns: DataFrame with daily PnL, cumulative returns.
    """
    raise NotImplementedError


def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    """Compute annualized Sharpe ratio."""
    raise NotImplementedError
