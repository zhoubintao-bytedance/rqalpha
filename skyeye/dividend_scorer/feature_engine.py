# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
import pandas as pd

from skyeye.dividend_scorer.config import (
    ALL_FEATURES,
    CONFIDENCE_FEATURES,
    FEATURES,
    PERCENTILE_MIN_DATA,
    PERCENTILE_WINDOW,
    REQUIRED_HISTORY_COLUMNS,
    VALUATION_FEATURES,
)


class FeatureEngine(object):
    def __init__(self, percentile_window=PERCENTILE_WINDOW, percentile_min_data=PERCENTILE_MIN_DATA):
        self.percentile_window = percentile_window
        self.percentile_min_data = percentile_min_data
        self.raw_matrix = None
        self.percentile_matrix = None
        self.normalized_matrix = None
        self.count_matrix = None

    def precompute(self, history_df):
        self._validate_history_df(history_df)
        raw_features = self._compute_raw_features(history_df)
        percentile_matrix = pd.DataFrame(index=raw_features.index)
        normalized_matrix = pd.DataFrame(index=raw_features.index)
        count_matrix = pd.DataFrame(index=raw_features.index)

        for feature_name in ALL_FEATURES:
            percentile_series, count_series = self._rolling_hazen_percentile(raw_features[feature_name])
            percentile_matrix[feature_name] = percentile_series
            count_matrix[feature_name] = count_series
            if feature_name in FEATURES and FEATURES[feature_name]["inverted"]:
                normalized_matrix[feature_name] = 1.0 - percentile_series
            else:
                normalized_matrix[feature_name] = percentile_series

        self.raw_matrix = raw_features
        self.percentile_matrix = percentile_matrix
        self.normalized_matrix = normalized_matrix
        self.count_matrix = count_matrix
        return normalized_matrix

    def compute_single(self, date, history_df=None):
        if history_df is not None and self.normalized_matrix is None:
            self.precompute(history_df)
        if self.normalized_matrix is None:
            raise RuntimeError("precompute must be called before compute_single")
        ts = self._resolve_date(date)
        result = OrderedDict()
        for feature_name in ALL_FEATURES:
            raw_value = self._maybe_float(self.raw_matrix.at[ts, feature_name])
            percentile = self._maybe_float(self.percentile_matrix.at[ts, feature_name])
            normalized = self._maybe_float(self.normalized_matrix.at[ts, feature_name])
            sample_size = self._maybe_float(self.count_matrix.at[ts, feature_name])
            feature_meta = FEATURES.get(feature_name, {})
            result[feature_name] = {
                "raw": raw_value,
                "percentile": percentile,
                "normalized": normalized,
                "inverted": bool(feature_meta.get("inverted", False)),
                "dimension": feature_meta.get("dimension", "confidence"),
                "sample_size": int(sample_size) if sample_size is not None else None,
                "under_sampled": bool(sample_size is not None and sample_size < self.percentile_min_data),
            }
        return result

    def _compute_raw_features(self, history_df):
        close = pd.to_numeric(history_df["etf_close"], errors="coerce")
        close_hfq = pd.to_numeric(history_df["etf_close_hfq"], errors="coerce")
        volume = pd.to_numeric(history_df["etf_volume"], errors="coerce")
        dividend_yield = pd.to_numeric(history_df["dividend_yield"], errors="coerce")
        bond_10y = pd.to_numeric(history_df["bond_10y"], errors="coerce")
        premium_rate = pd.to_numeric(history_df["premium_rate"], errors="coerce")

        raw = pd.DataFrame(index=history_df.index)
        raw["dividend_yield_pct"] = dividend_yield
        raw["yield_spread"] = dividend_yield - bond_10y
        raw["pe_percentile"] = pd.to_numeric(history_df["pe_ttm"], errors="coerce")
        raw["ma250_deviation"] = close / close.rolling(window=250, min_periods=20).mean() - 1.0
        raw["price_percentile"] = close
        raw["rsi_20"] = self._compute_rsi(close, period=20)
        raw["premium_rate"] = premium_rate
        raw["premium_rate_ma20"] = premium_rate.rolling(window=20, min_periods=5).mean()
        daily_returns = close_hfq.pct_change(fill_method=None)
        raw["volatility_percentile"] = daily_returns.rolling(window=20, min_periods=5).std() * np.sqrt(252.0)
        raw["volume_ratio"] = volume / volume.rolling(window=20, min_periods=5).mean()
        return raw

    def _validate_history_df(self, history_df):
        missing_columns = [column for column in REQUIRED_HISTORY_COLUMNS if column not in history_df.columns]
        if missing_columns:
            raise RuntimeError("history dataframe missing required columns: {}".format(", ".join(missing_columns)))
        if not isinstance(history_df.index, pd.DatetimeIndex):
            raise RuntimeError("history dataframe index must be DatetimeIndex")

    def _resolve_date(self, date):
        ts = pd.Timestamp(date)
        index = self.normalized_matrix.index
        if ts in index:
            return ts
        prior = index[index <= ts]
        if len(prior) == 0:
            raise KeyError("no feature data on or before {}".format(ts.strftime("%Y-%m-%d")))
        return prior[-1]

    def _rolling_hazen_percentile(self, series):
        counts = series.rolling(window=self.percentile_window, min_periods=1).count()
        percentile = series.rolling(window=self.percentile_window, min_periods=1).apply(
            self._hazen_percentile_of_last,
            raw=False,
        )
        return percentile, counts

    @staticmethod
    def _hazen_percentile_of_last(window):
        valid = pd.Series(window).dropna()
        if len(valid) == 0:
            return np.nan
        current = valid.iloc[-1]
        ranks = valid.rank(method="average")
        rank = ranks.iloc[-1]
        return (rank - 0.5) / float(len(valid))

    @staticmethod
    def _compute_rsi(close, period=20):
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _maybe_float(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return float(value)
