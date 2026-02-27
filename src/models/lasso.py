"""
Lasso regression for HAR-based lead-lag discovery.

Two modes:
  - fit_insample:   LassoCV on full data (in-sample, for network visualization)
  - rolling_predict: Strictly out-of-sample rolling window predictions
                     (for backtesting — no look-ahead bias)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler


def fit_insample(X: pd.DataFrame,
                 Y: pd.DataFrame,
                 n_alphas: int = 100,
                 cv: int = 5,
                 max_iter: int = 20000,
                 random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    """
    Fit LassoCV for each stock on the full dataset (in-sample).

    Used for:
      - Network visualization (bipartite graph)
      - Lambda stability analysis
      - NOT for backtesting (use rolling_predict instead)

    Args:
        X:    Feature matrix (features already standardized recommended but
              this function standardizes internally)
        Y:    Target returns (ideally market-residualized)
        ...   sklearn LassoCV parameters

    Returns:
        beta_matrix: DataFrame (features × stocks) of Lasso coefficients
        models:      Dict {ticker: fitted LassoCV model}
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), index=X.index, columns=X.columns
    )

    betas = {}
    models = {}

    for stock in Y.columns:
        y = Y[stock].values
        mask = ~np.isnan(y)
        if mask.sum() < 50:
            betas[stock] = np.zeros(X.shape[1])
            continue

        model = LassoCV(cv=cv, max_iter=max_iter, n_alphas=n_alphas,
                        random_state=random_state)
        model.fit(X_scaled.values[mask], y[mask])

        betas[stock] = model.coef_
        models[stock] = model

    beta_matrix = pd.DataFrame(betas, index=X.columns)
    return beta_matrix, models


def rolling_predict(X: pd.DataFrame,
                    Y: pd.DataFrame,
                    train_window: int = 252,
                    alpha: float = 0.001,
                    max_iter: int = 5000) -> pd.DataFrame:
    """
    Generate strictly out-of-sample predictions via rolling Lasso.

    For each day t:
      1. Train on [t-window, t-1] using idiosyncratic returns
      2. Standardize features within training window (no leakage)
      3. Predict for day t

    These predictions are used for cross-sectional signal generation.

    Args:
        X:             Feature matrix (any scale; standardized per window)
        Y:             Idiosyncratic return targets (market-residualized)
        train_window:  Days of history for training (default 252 = 1 year)
        alpha:         Lasso regularization (fixed; use LassoCV in fit_insample
                       for selecting alpha, then plug in here)
        max_iter:      sklearn Lasso max_iter

    Returns:
        DataFrame of OOS predictions, same index/columns as Y.
        First `train_window` rows are NaN.
    """
    n_days = len(X)
    predictions = pd.DataFrame(
        np.nan, index=X.index, columns=Y.columns, dtype=float
    )

    print(f"Rolling OOS Lasso | window={train_window}d | alpha={alpha}")
    print(f"Generating {n_days - train_window} predictions across "
          f"{len(Y.columns)} stocks...")

    for t in range(train_window, n_days):
        if t % 500 == 0:
            print(f"  Day {t}/{n_days}  ({X.index[t].date()})")

        X_train_raw = X.iloc[t - train_window:t].values
        X_test_raw = X.iloc[t:t + 1].values

        # Standardize within window — no future information
        mu = X_train_raw.mean(axis=0)
        sd = X_train_raw.std(axis=0) + 1e-8
        X_train_z = (X_train_raw - mu) / sd
        X_test_z = (X_test_raw - mu) / sd

        for stock in Y.columns:
            y_train = Y[stock].iloc[t - train_window:t].values
            mask = ~np.isnan(y_train)
            if mask.sum() < 50:
                continue
            model = Lasso(alpha=alpha, max_iter=max_iter)
            model.fit(X_train_z[mask], y_train[mask])
            predictions.loc[X.index[t], stock] = model.predict(X_test_z)[0]

    print("Done.")
    return predictions


def fit_lambda_path(X: pd.DataFrame,
                    Y_stock: pd.Series,
                    lambdas: np.ndarray | None = None) -> pd.DataFrame:
    """
    Fit Lasso over a grid of lambda values for one stock.

    Used for lambda stability analysis (notebook 06).

    Args:
        X:          Feature matrix (will be standardized)
        Y_stock:    Single stock return series
        lambdas:    Array of alpha values; defaults to log-spaced 1e-5..1e-1

    Returns:
        DataFrame of shape (n_features, n_lambdas) — beta at each lambda
    """
    if lambdas is None:
        lambdas = np.logspace(-5, -1, 50)

    scaler = StandardScaler()
    y = Y_stock.values
    mask = ~np.isnan(y)
    X_z = scaler.fit_transform(X.values[mask])
    y_m = y[mask]

    results = {}
    for lam in lambdas:
        model = Lasso(alpha=lam, max_iter=10000)
        model.fit(X_z, y_m)
        results[lam] = model.coef_

    return pd.DataFrame(results, index=X.columns)
