# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("date", "order_book_id", "close", "volume", "benchmark_close")


class DatasetBuilder(object):
    def __init__(self, input_window=60):
        self.input_window = int(input_window)

    def build(self, raw_df):
        if raw_df is None or len(raw_df) == 0:
            raise ValueError("raw_df must not be empty")
        missing = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
        if missing:
            raise ValueError("raw_df missing required columns: {}".format(", ".join(missing)))

        frame = raw_df.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(["order_book_id", "date"]).reset_index(drop=True)

        benchmark = frame[["date", "benchmark_close"]].drop_duplicates(subset=["date"]).sort_values("date")
        benchmark["benchmark_return_1d"] = benchmark["benchmark_close"].pct_change(fill_method=None)
        for window in (20, 40, 60):
            benchmark["benchmark_mom_{}d".format(window)] = benchmark["benchmark_close"].pct_change(window, fill_method=None)
        benchmark["benchmark_volatility_20d"] = benchmark["benchmark_return_1d"].rolling(window=20, min_periods=20).std() * np.sqrt(252.0)
        benchmark = benchmark.drop(columns=["benchmark_close"])

        parts = []
        for _, asset_df in frame.groupby("order_book_id", sort=False):
            asset_df = asset_df.sort_values("date").copy()
            asset_df["return_1d"] = asset_df["close"].pct_change(fill_method=None)
            for window in (20, 40, 60):
                asset_df["mom_{}d".format(window)] = asset_df["close"].pct_change(window, fill_method=None)
            asset_df["volatility_20d"] = asset_df["return_1d"].rolling(window=20, min_periods=20).std() * np.sqrt(252.0)
            asset_df["volume_ratio_20d"] = asset_df["volume"] / asset_df["volume"].rolling(window=20, min_periods=20).mean()
            parts.append(asset_df)

        dataset = pd.concat(parts, ignore_index=True)
        dataset = dataset.merge(benchmark, on="date", how="left")
        dataset["excess_mom_20d"] = dataset["mom_20d"] - dataset["benchmark_mom_20d"]
        dataset["regime_support"] = dataset["benchmark_mom_20d"] - dataset["benchmark_volatility_20d"].fillna(0.0) / 10.0

        required_non_null = [
            "mom_20d",
            "mom_40d",
            "mom_60d",
            "excess_mom_20d",
            "volatility_20d",
            "volume_ratio_20d",
            "benchmark_mom_20d",
        ]
        dataset = dataset.dropna(subset=required_non_null).sort_values(["date", "order_book_id"]).reset_index(drop=True)
        ordered_columns = [
            "date",
            "order_book_id",
            "close",
            "benchmark_close",
            "volume",
            "mom_20d",
            "mom_40d",
            "mom_60d",
            "excess_mom_20d",
            "volatility_20d",
            "volume_ratio_20d",
            "benchmark_mom_20d",
            "benchmark_mom_40d",
            "benchmark_mom_60d",
            "benchmark_volatility_20d",
            "regime_support",
        ]
        return dataset.loc[:, ordered_columns]
