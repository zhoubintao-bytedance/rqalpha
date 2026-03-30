# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("date", "order_book_id", "close", "volume", "benchmark_close")

# Optional columns for extended features (gracefully skipped if absent)
EXTENDED_COLUMNS = ("total_turnover", "sector")


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
            asset_df["mom_40d"] = asset_df["close"].pct_change(40, fill_method=None)
            asset_df["volatility_20d"] = asset_df["return_1d"].rolling(window=20, min_periods=20).std() * np.sqrt(252.0)
            asset_df["reversal_5d"] = -asset_df["close"].pct_change(5, fill_method=None)

            # Amihud illiquidity (requires total_turnover for correct calculation)
            if has_turnover:
                asset_df["amihud_daily"] = asset_df["return_1d"].abs() / asset_df["total_turnover"].clip(lower=1.0)
                asset_df["amihud_20d"] = asset_df["amihud_daily"].rolling(20, min_periods=20).mean()

            parts.append(asset_df)

        dataset = pd.concat(parts, ignore_index=True)

        required_non_null = [
            "mom_40d",
            "volatility_20d",
            "reversal_5d",
        ]
        if has_turnover:
            required_non_null.append("amihud_20d")

        dataset = dataset.dropna(subset=required_non_null).sort_values(["date", "order_book_id"]).reset_index(drop=True)

        ordered_columns = [
            "date",
            "order_book_id",
            "close",
            "benchmark_close",
            "volume",
            "mom_40d",
            "volatility_20d",
            "reversal_5d",
        ]
        if has_turnover:
            ordered_columns.append("amihud_20d")
        # Carry through sector if present
        if "sector" in dataset.columns:
            ordered_columns.append("sector")

        return dataset.loc[:, [c for c in ordered_columns if c in dataset.columns]]
