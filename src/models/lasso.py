"""
Lasso regression for lead-lag discovery.
"""

def fit_lasso_for_stock(stock_returns, har_features, lambda_param):
    """
    Fit Lasso regression for a single stock:
    min ||r_i - X @ beta||^2 + lambda * ||beta||_1

    Returns: beta coefficients (sparse)
    """
    raise NotImplementedError


def fit_all_stocks(stock_returns_df, har_features, lambda_param):
    """
    Fit Lasso for all stocks in the universe.
    Returns: Dict[stock_ticker, beta_coefficients]
    """
    raise NotImplementedError


def select_lambda_cv(stock_returns, har_features, n_folds=5):
    """Cross-validation to select optimal lambda."""
    raise NotImplementedError
