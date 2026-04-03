# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from skyeye.products.tx1.evaluator import CANDIDATE_FEATURE_COLUMNS, FUNDAMENTAL_FEATURE_COLUMNS


REQUIRED_COLUMNS = ("date", "order_book_id", "close", "volume", "benchmark_close")

# Optional columns for extended features (gracefully skipped if absent)
EXTENDED_COLUMNS = ("total_turnover", "sector") + tuple(FUNDAMENTAL_FEATURE_COLUMNS)

CORE_REQUIRED_FEATURES = (
    "mom_40d",
    "volatility_20d",
    "reversal_5d",
)


def _safe_ratio(numerator, denominator):
    if hasattr(denominator, "replace"):
        denominator = denominator.replace(0, np.nan)
    ratio = numerator / denominator
    if isinstance(ratio, pd.Series):
        return ratio.replace([np.inf, -np.inf], np.nan)
    return ratio


def _price_position(close, rolling_low, rolling_high):
    spread = (rolling_high - rolling_low).replace(0, np.nan)
    position = (close - rolling_low) / spread
    return position.fillna(0.5)


class DatasetBuilder(object):
    def __init__(self, input_window=60):
        self.input_window = int(input_window)

    def build(self, raw_df):
        if raw_df is None or len(raw_df) == 0:
            raise ValueError("raw_df must not be empty")
        missing = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
        if missing:
            raise ValueError("raw_df missing required columns: {}".format(", ".join(missing)))

        has_turnover = "total_turnover" in raw_df.columns

        frame = raw_df.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(["order_book_id", "date"]).reset_index(drop=True)

        parts = []
        for _, asset_df in frame.groupby("order_book_id", sort=False):
            asset_df = asset_df.sort_values("date").copy()
            asset_df["return_1d"] = asset_df["close"].pct_change(fill_method=None)
            asset_df["benchmark_return_1d"] = asset_df["benchmark_close"].pct_change(fill_method=None)

            turnover_proxy = asset_df["total_turnover"] if has_turnover else asset_df["close"] * asset_df["volume"]
            turnover_proxy = turnover_proxy.astype(float).clip(lower=1.0)

            asset_df["mom_20d"] = asset_df["close"].pct_change(20, fill_method=None)
            asset_df["mom_40d"] = asset_df["close"].pct_change(40, fill_method=None)
            asset_df["mom_60d"] = asset_df["close"].pct_change(60, fill_method=None)
            asset_df["excess_mom_20d"] = asset_df["mom_20d"] - asset_df["benchmark_close"].pct_change(20, fill_method=None)
            asset_df["excess_mom_60d"] = asset_df["mom_60d"] - asset_df["benchmark_close"].pct_change(60, fill_method=None)
            asset_df["volatility_20d"] = asset_df["return_1d"].rolling(window=20, min_periods=20).std() * np.sqrt(252.0)
            downside_returns = asset_df["return_1d"].clip(upper=0.0)
            asset_df["downside_volatility_20d"] = np.sqrt(
                downside_returns.pow(2).rolling(window=20, min_periods=20).mean()
            ) * np.sqrt(252.0)
            asset_df["reversal_5d"] = -asset_df["close"].pct_change(5, fill_method=None)
            asset_df["return_skew_20d"] = asset_df["return_1d"].rolling(window=20, min_periods=20).skew()

            ma_10d = asset_df["close"].rolling(window=10, min_periods=10).mean()
            ma_20d = asset_df["close"].rolling(window=20, min_periods=20).mean()
            ma_40d = asset_df["close"].rolling(window=40, min_periods=40).mean()
            ma_60d = asset_df["close"].rolling(window=60, min_periods=60).mean()
            asset_df["ma_gap_20d"] = _safe_ratio(asset_df["close"], ma_20d) - 1.0
            asset_df["ma_gap_60d"] = _safe_ratio(asset_df["close"], ma_60d) - 1.0
            asset_df["ma_crossover_10_40d"] = _safe_ratio(ma_10d, ma_40d) - 1.0

            rolling_low_20d = asset_df["close"].rolling(window=20, min_periods=20).min()
            rolling_high_20d = asset_df["close"].rolling(window=20, min_periods=20).max()
            rolling_low_60d = asset_df["close"].rolling(window=60, min_periods=60).min()
            rolling_high_60d = asset_df["close"].rolling(window=60, min_periods=60).max()
            asset_df["price_position_20d"] = _price_position(asset_df["close"], rolling_low_20d, rolling_high_20d)
            asset_df["price_position_60d"] = _price_position(asset_df["close"], rolling_low_60d, rolling_high_60d)
            asset_df["distance_to_high_60d"] = _safe_ratio(asset_df["close"], rolling_high_60d) - 1.0

            drawdown_20d = _safe_ratio(asset_df["close"], rolling_high_20d) - 1.0
            asset_df["max_drawdown_20d"] = drawdown_20d.rolling(window=20, min_periods=20).min()

            volume_mean_5d = asset_df["volume"].rolling(window=5, min_periods=5).mean()
            volume_mean_20d = asset_df["volume"].rolling(window=20, min_periods=20).mean()
            asset_df["volume_ratio_20d"] = _safe_ratio(asset_df["volume"], volume_mean_20d)
            asset_df["volume_trend_5_20d"] = _safe_ratio(volume_mean_5d, volume_mean_20d) - 1.0

            turnover_mean_20d = turnover_proxy.rolling(window=20, min_periods=20).mean()
            turnover_std_20d = turnover_proxy.rolling(window=20, min_periods=20).std()
            asset_df["turnover_ratio_20d"] = _safe_ratio(turnover_proxy, turnover_mean_20d)
            asset_df["turnover_stability_20d"] = _safe_ratio(turnover_mean_20d, turnover_std_20d)
            asset_df["dollar_volume_20d_log"] = np.log1p(turnover_mean_20d)
            asset_df["vol_adj_turnover_20d"] = _safe_ratio(np.log1p(turnover_mean_20d), asset_df["volatility_20d"])

            benchmark_var_60d = asset_df["benchmark_return_1d"].rolling(window=60, min_periods=60).var()
            covariance_60d = asset_df["return_1d"].rolling(window=60, min_periods=60).cov(asset_df["benchmark_return_1d"])
            asset_df["beta_60d"] = _safe_ratio(covariance_60d, benchmark_var_60d)

            if has_turnover:
                asset_df["amihud_daily"] = asset_df["return_1d"].abs() / asset_df["total_turnover"].clip(lower=1.0)
                asset_df["amihud_20d"] = asset_df["amihud_daily"].rolling(window=20, min_periods=20).mean()

            parts.append(asset_df)

        dataset = pd.concat(parts, ignore_index=True)

        required_non_null = list(CORE_REQUIRED_FEATURES)
        if has_turnover:
            required_non_null.append("amihud_20d")

        dataset = dataset.dropna(subset=required_non_null).sort_values(["date", "order_book_id"]).reset_index(drop=True)

        ordered_columns = [
            "date",
            "order_book_id",
            "close",
            "benchmark_close",
            "volume",
        ]
        ordered_columns.extend(CANDIDATE_FEATURE_COLUMNS)
        if "sector" in dataset.columns:
            ordered_columns.append("sector")

        return dataset.loc[:, [c for c in ordered_columns if c in dataset.columns]]
