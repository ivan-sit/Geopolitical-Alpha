"""
Daily Order Flow Imbalance proxy.
"""

def compute_daily_ofi(open_price, high, low, close, volume):
    """
    Compute daily OFI proxy:
    OFI = Volume * (Close - Open) / (High - Low)
    """
    raise NotImplementedError
