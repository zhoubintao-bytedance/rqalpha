# -*- coding: utf-8 -*-

from collections import OrderedDict


ETF_CODE = "512890"
INDEX_CODE = "H30269"

PERCENTILE_WINDOW = 730
PERCENTILE_MIN_DATA = 120
TRADING_DAYS_A_YEAR = 252

LABEL_WINDOW_N = 60
SUBSAMPLE_INTERVAL_S = 60
IC_ROLLING_K = 30
IC_IR_THRESHOLD = 0.3
IC_PVALUE_THRESHOLD = 0.1
IC_MIN_SURVIVING = 5
IC_MIN_POINTS = 6
WEIGHT_CONCENTRATION_MAX = 0.6
DIVIDEND_WEIGHT_MIN = 0.1
COSINE_SIM_MIN = 0.5

SCORE_BUY_THRESHOLD = 3.5
SCORE_SELL_THRESHOLD = 6.5

CONFIDENCE_EXTREME_LOW = 0.1
CONFIDENCE_EXTREME_HIGH = 0.9

CACHE_DB_PATH = "~/.rqalpha/dividend_scorer/cache.db"
CACHE_STALE_DAYS = 3
CACHE_EXPIRED_DAYS = 5
API_CALL_INTERVAL = 0.5

DATA_GAP_THRESHOLD = 0.05
MIN_DIMENSIONS = 3
STRATEGY_SCORE_TARGET = 6.0
SAMPLE_OUT_YEARS = 1
PARAMS_VERSION = "v1_20260319"

FEATURES = OrderedDict([
    ("dividend_yield_pct", {"dimension": "dividend", "inverted": True, "source": "stock_indicator"}),
    ("yield_spread", {"dimension": "dividend", "inverted": True, "source": "stock_indicator"}),
    ("pe_percentile", {"dimension": "pe", "inverted": False, "source": "index_daily"}),
    ("ma250_deviation", {"dimension": "price", "inverted": False, "source": "etf_daily"}),
    ("price_percentile", {"dimension": "price", "inverted": False, "source": "etf_daily"}),
    ("rsi_20", {"dimension": "price", "inverted": False, "source": "etf_daily"}),
    ("premium_rate", {"dimension": "premium", "inverted": False, "source": "etf_nav"}),
    ("premium_rate_ma20", {"dimension": "premium", "inverted": False, "source": "etf_nav"}),
])

CONFIDENCE_FEATURES = (
    "volatility_percentile",
    "volume_ratio",
)

VALUATION_FEATURES = tuple(FEATURES.keys())
ALL_FEATURES = VALUATION_FEATURES + CONFIDENCE_FEATURES

DIMENSION_FEATURES = OrderedDict([
    ("dividend", ("dividend_yield_pct", "yield_spread")),
    ("pe", ("pe_percentile",)),
    ("price", ("ma250_deviation", "price_percentile", "rsi_20")),
    ("premium", ("premium_rate", "premium_rate_ma20")),
])

DOMAIN_WEIGHTS = OrderedDict([
    ("dividend_yield_pct", 0.175),
    ("yield_spread", 0.175),
    ("pe_percentile", 0.20),
    ("ma250_deviation", 0.10),
    ("price_percentile", 0.10),
    ("rsi_20", 0.10),
    ("premium_rate", 0.075),
    ("premium_rate_ma20", 0.075),
])

REQUIRED_HISTORY_COLUMNS = (
    "etf_close",
    "etf_close_hfq",
    "etf_volume",
    "etf_nav",
    "pe_ttm",
    "dividend_yield",
    "bond_10y",
    "premium_rate",
)
