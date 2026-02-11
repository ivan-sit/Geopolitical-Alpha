"""
Project configuration.
"""

# Data paths
CRSP_DATA_PATH = "US_CRSP_NYSE/"
DATA_PROCESSED = "data/processed/"

# Commodities
COMMODITIES = ["CL", "BZ", "NG", "HO"]  # Crude, Brent, NatGas, Heating Oil

# Date range
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# Model
DEFAULT_LAMBDA = 0.01

# Event study
VENEZUELA_EVENT_DATE = "2026-01-15"
