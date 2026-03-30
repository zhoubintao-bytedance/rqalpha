# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


class LabelBuilder(object):
    def __init__(self, horizon=20, transform="raw", winsorize=None):
        self.horizon = int(horizon)
        self.transform = transform
        # winsorize: tuple (lower, upper) percentiles e.g. (0.01, 0.99), or None
        self.winsorize = winsorize

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
        if self.winsorize is not None:
            lo, hi = self.winsorize
            clipped = labeled.groupby("date")["label_return_raw"].transform(
                lambda s: s.clip(lower=s.quantile(lo), upper=s.quantile(hi))
            )
            labeled["_winsorized"] = clipped
            labeled["target_label"] = self._transform_by_date(labeled, "_winsorized")
            labeled = labeled.drop(columns=["_winsorized"])
        else:
            labeled["target_label"] = self._transform_by_date(labeled, "label_return_raw")
        return labeled

    def _transform_by_date(self, frame, column):
        if self.transform == "raw":
            return frame[column]
        if self.transform == "rank":
            return frame.groupby("date")[column].rank(method="average", pct=True)
        if self.transform == "quantile":
            return frame.groupby("date")[column].transform(self._quantile_transform)
        raise ValueError("unsupported transform: {}".format(self.transform))

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
