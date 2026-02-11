"""
HAR (Heterogeneous Autoregressive) feature construction.
"""

def compute_daily_returns(prices):
    """Compute daily returns r^(d)."""
    raise NotImplementedError


def compute_weekly_returns(prices):
    """Compute 5-day average returns r^(w)."""
    raise NotImplementedError


def compute_monthly_returns(prices):
    """Compute 22-day average returns r^(m)."""
    raise NotImplementedError


def build_har_features(commodity_prices):
    """
    Build HAR feature matrix X_c = [r^(d), r^(w), r^(m)] for each commodity.
    Returns: DataFrame with columns like 'CL_daily', 'CL_weekly', 'CL_monthly', etc.
    """
    raise NotImplementedError
