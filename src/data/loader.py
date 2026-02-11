"""
Data loading utilities for CRSP and commodity data.
"""

def load_crsp_energy_stocks(data_path: str):
    """Load S&P 500 Energy sector stocks from CRSP data."""
    raise NotImplementedError


def load_commodity_futures(data_path: str):
    """Load commodity futures data (Crude, Brent, NatGas, Heating Oil)."""
    raise NotImplementedError


def merge_stock_commodity_data(stocks_df, commodities_df):
    """Align stock and commodity data on dates."""
    raise NotImplementedError
