"""
Project configuration — edit this file to change global settings.
"""

# ── CRSP Data source ──────────────────────────────────────────────────────────
# 'dropbox' : stream from professor's shared Dropbox (recommended, no manual download)
# 'local'   : read from local files (set CRSP_DATA_PATH below)
# 'auto'    : try Dropbox first, fall back to local
DATA_SOURCE = "auto"

# Dropbox shared folder (professor's link — do not change)
DROPBOX_FOLDER_URL = (
    "https://www.dropbox.com/scl/fo/bx9jv5x2fnu9j02i5j77a/"
    "AGljwCnU10aplUA6BZ3AwrU?rlkey=7jeselv0gnki38q4dvrzbhbsc"
)

# Dropbox API token for reliable access (optional but recommended).
# Get yours free in 2 minutes:
#   1. https://www.dropbox.com/developers/apps
#   2. Create App → Scoped → Full Dropbox
#   3. Permissions: files.content.read + sharing.read
#   4. Settings → Generate access token → paste below (or set env var)
#
# Alternative: set  export DROPBOX_TOKEN=sl.xxx  in your shell profile.
DROPBOX_TOKEN = "sl.u.AGUeWwvdQ1CNPMKmrsa-P8oQkenhsWw_JZZuJAwPjxluOrGD98modpCX_9yQJ9IigHsE-zI8KkbNMJ8wXHmuWKFVksgFKcHpci7vtgkfbTVNy_Ejka5CXrWaMhhlaSQ9ojWJnT7jYQitWbVUC_T1DG7JG64JsU8WF0pEhg7ApKkv9SH1hT7kXmcyCDL00ycaTdNAEI3fbSeRo3kprnLNiosq6TAqEzpw8A0pND7EHkNh4i_z4e2TrCVeuohBCL-kTlSTWobSDNpN5QOlWkNox4HHZfd51WQdw4T88JwX30UR8VHhkgHZdAjSpmJHXTjcwldLLZGbGOC1bn2DJ4mG5-VTXw8MsSm0t3vqIjMsdRDmxtPcQVj3BUK02MprLd8gCcIIDPoi1hk9-ZDeQF93SYaPxfHmR8iBbT-cqecs4iYDAzJj5ZzXWukj_yO9P-1pAeD_GdYr29BbEVgYG1AdcAshkIwD0SkSUb81xYLNM9guPf6GinAP7wCznG_gyPjYDY0Nxyl1JfminaiQklYy4v-b3Hwo0gqTsJgqczQhOYiIC75B1dM3azFTU--mDrrpq5ecUN4KrD6J9pvu0JYHWlWN9dzmC3Tib4zWyTOu4O68--kvR3kejlt9MIfGP-typwcB7xQzrPnLf2CpS32y3QivKrPWwboIlOfRpFQRH1DRRz4cOZ4H-iPybqre-0mtG2ws80z6jCWIK5mDl1MaXwn1AaT3xCs95UQu0lZVEiDdsRsQbbs5Bjd_tz08AN9o_VhVvWn-crT4pkY59iYskKt5qJ1gqL5zEFxrctW4c_JAtpKzyI2hyj8Lpmr98qpYpPQrDmm_Z7GQCR1a0xxucAs0i8DnNXolDxXWsbxm6QDtVQq8XGbXzYPAMf-ibDhAtf0wuERXK1sVSAUuheu6keaDQ6fjtWbAL7B_6mdNDyvJd5JkCjNQ8c3cZgOVZ8-xboe7p5Q60ullscuhQEVDJnWc73dPmIZYXyGZ9V_qwIKyMUfA_iRGXa8nGh9v-Q_00BeeAJ7MAPJMS1JEohC1Lf0ej5vOG8mvMuzBnicBxF2TOqc1ndPRW5UNw0BswjTHeWV9U_aC0qKfRXVYp2OURSBz22BQ7kAiwbE1pPrkVnWhWjudaZVpErkDQ2QJYQCqxszLHlfCyWvlaecGaTGh26gE4LcRvLPfuUWS1p70Cdr9BfbTe6bD5gsDNSnt1M0sbLMVLmZAI1oddwiivmSAMNNZyt0HGUWWcIoquwlzOpSsQxnIGAbEgcXQP6Kd-voZVvabL6u1X_AJVbOe1z9BtbW7UPBmI5Gyeq38pwU3oKqLe-Wb8_52u2UDKEyV9yYROZICappfcH58RDY3TthIDqzaAJKUAf3D0fqBKUNhT_o2XhWNb35qJu8ONi-HQUsrRzAZAfGCcGPplORa6gYYgNZA"
# Local cache for Dropbox downloads (files downloaded once, reused thereafter)
DROPBOX_CACHE = ".crsp_cache"

# ── Local CRSP paths (used when DATA_SOURCE='local') ─────────────────────────
CRSP_DATA_PATH = "US_CRSP_NYSE/Matrix_Format_SubsetUniverse"
CRSP_SECTORS   = "US_CRSP_NYSE/Sectors/Sectors_SP500_YahooNWikipedia.csv"

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE = "2000-01-01"
END_DATE   = "2024-12-31"

# ── Universe ──────────────────────────────────────────────────────────────────
SECTOR = "Energy"         # sector label (Wikipedia column in sectors CSV)
SHRCD  = (10, 11)         # common shares only (exclude ETFs 73/74)

# ── Commodities ───────────────────────────────────────────────────────────────
COMMODITY_TICKERS = {
    "CL=F": "WTI",
    "BZ=F": "Brent",
    "NG=F": "NatGas",
    "HO=F": "HeatOil",
}

# ── Model ─────────────────────────────────────────────────────────────────────
LASSO_ALPHA    = 0.001     # fixed alpha for rolling OOS (tune via fit_insample LassoCV)
TRAIN_WINDOW   = 252       # rolling training window (1 year)
BETA_WINDOW    = 252       # rolling market beta window for residualization

# ── Backtest ──────────────────────────────────────────────────────────────────
TRANSACTION_COST = 0.0005  # 5 bps one-way (realistic for S&P 500 large-caps)
TOP_PCT          = 0.25    # fraction of stocks to go long
BOTTOM_PCT       = 0.25    # fraction of stocks to go short

# ── Event study ───────────────────────────────────────────────────────────────
EVENTS = {
    "Lehman":        "2008-09-15",
    "OPEC_Cut":      "2016-11-30",
    "Venezuela":     "2019-01-23",
    "Aramco_Attack": "2019-09-14",
    "Saudi_Russia":  "2020-03-08",
}
EVENT_WINDOW = 120   # days before/after each event
