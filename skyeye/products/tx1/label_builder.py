# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np
import pandas as pd


DEFAULT_TARGET_CONFIG = {
    "volatility": {
        "transform": "rank",
    },
    "max_drawdown": {
        "transform": "rank",
    },
}


class LabelBuilder(object):
    def __init__(self, horizon=20, transform="raw", winsorize=None, target_config=None):
        self.horizon = int(horizon)
        self.transform = transform
        # winsorize: tuple (lower, upper) percentiles e.g. (0.01, 0.99), or None
        self.winsorize = winsorize
        self.target_config = self._normalize_target_config(target_config)

    def build(self, dataset_df):
        if dataset_df is None or len(dataset_df) == 0:
            raise ValueError("dataset_df must not be empty")
        frame = dataset_df.sort_values(["order_book_id", "date"]).copy()

        benchmark = frame[["date", "benchmark_close"]].drop_duplicates(subset=["date"]).sort_values("date").copy()
        benchmark["benchmark_forward_return"] = benchmark["benchmark_close"].shift(-self.horizon) / benchmark["benchmark_close"] - 1.0
        benchmark = benchmark[["date", "benchmark_forward_return"]]

        parts = []
        for _, asset_df in frame.groupby("order_book_id", sort=False):
            asset_df = asset_df.sort_values("date").copy()
            future_close = asset_df["close"].shift(-self.horizon)
            asset_df["asset_forward_return"] = future_close / asset_df["close"] - 1.0
            future_returns = []
            future_drawdowns = []
            closes = asset_df["close"].to_numpy(dtype=float)
            for i in range(len(asset_df)):
                if i + self.horizon >= len(asset_df):
                    future_returns.append(np.nan)
                    future_drawdowns.append(np.nan)
                    continue
                window = closes[i + 1:i + self.horizon + 1]
                returns = pd.Series(window).pct_change(fill_method=None).dropna()
                if len(returns) == 0:
                    future_returns.append(np.nan)
                else:
                    future_returns.append(float(returns.std() * np.sqrt(252.0)))
                running_max = np.maximum.accumulate(window)
                drawdown = 1.0 - window / running_max
                future_drawdowns.append(float(np.max(drawdown)))
            asset_df["label_volatility_horizon"] = future_returns
            asset_df["label_max_drawdown_horizon"] = future_drawdowns
            parts.append(asset_df)

        labeled = pd.concat(parts, ignore_index=True)
        labeled = labeled.merge(benchmark, on="date", how="left")
        labeled["label_return_raw"] = labeled["asset_forward_return"] - labeled["benchmark_forward_return"]
        labeled = labeled.dropna(subset=["label_return_raw", "label_volatility_horizon", "label_max_drawdown_horizon"]).copy()
        labeled = labeled.sort_values(["date", "order_book_id"]).reset_index(drop=True)

        return_base_column = "label_return_raw"
        if self.winsorize is not None:
            lo, hi = self.winsorize
            labeled["_winsorized_return"] = labeled.groupby("date")["label_return_raw"].transform(
                lambda s: s.clip(lower=s.quantile(lo), upper=s.quantile(hi))
            )
            return_base_column = "_winsorized_return"

        labeled["target_return"] = self._transform_by_date(
            labeled,
            return_base_column,
            transform=self.transform,
        )
        labeled["target_label"] = labeled["target_return"]
        labeled["target_volatility"] = self._transform_by_date(
            labeled,
            "label_volatility_horizon",
            transform=self.target_config["volatility"]["transform"],
        )
        labeled["target_max_drawdown"] = self._transform_by_date(
            labeled,
            "label_max_drawdown_horizon",
            transform=self.target_config["max_drawdown"]["transform"],
        )

        if "_winsorized_return" in labeled.columns:
            labeled = labeled.drop(columns=["_winsorized_return"])
        return labeled

    @classmethod
    def _normalize_target_config(cls, target_config):
        normalized = deepcopy(DEFAULT_TARGET_CONFIG)
        if not target_config:
            return normalized
        for target_name, overrides in target_config.items():
            if target_name not in normalized or not isinstance(overrides, dict):
                continue
            normalized[target_name].update(overrides)
        return normalized

    def _transform_by_date(self, frame, column, transform=None):
        transform = transform or self.transform
        if transform == "raw":
            return frame[column]
        if transform == "rank":
            return frame.groupby("date")[column].rank(method="average", pct=True)
        if transform == "quantile":
            return frame.groupby("date")[column].transform(self._quantile_transform)
        if transform == "log1p":
            return frame.groupby("date")[column].transform(self._log1p_transform)
        if transform == "robust":
            return frame.groupby("date")[column].transform(self._robust_transform)
        raise ValueError("unsupported transform: {}".format(transform))

    @staticmethod
    def _quantile_transform(series):
        valid = series.dropna()
        if valid.empty:
            return pd.Series(index=series.index, dtype=float)
        quantized = pd.qcut(valid.rank(method="first"), q=min(5, len(valid)), labels=False, duplicates="drop")
        if quantized is None:
            return pd.Series(index=series.index, dtype=float)
        scaled = (quantized.astype(float) + 0.5) / float(int(quantized.max()) + 1)
        result = pd.Series(index=series.index, dtype=float)
        result.loc[valid.index] = scaled.to_numpy()
        return result

    @staticmethod
    def _log1p_transform(series):
        valid = series.dropna().astype(float)
        result = pd.Series(index=series.index, dtype=float)
        if valid.empty:
            return result
        result.loc[valid.index] = np.log1p(valid.clip(lower=0.0))
        return result

    @staticmethod
    def _robust_transform(series):
        valid = series.dropna().astype(float)
        result = pd.Series(index=series.index, dtype=float)
        if valid.empty:
            return result
        median = float(valid.median())
        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        scale = q3 - q1
        if scale <= 0:
            scale = float((valid - median).abs().median() * 1.4826)
        if scale <= 0:
            result.loc[valid.index] = valid - median
            return result
        result.loc[valid.index] = (valid - median) / scale
        return result
