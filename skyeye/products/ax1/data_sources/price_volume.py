"""Implemented AX1 price/volume data source wrapper."""

from __future__ import annotations

import pandas as pd

from skyeye.products.ax1.data_sources.base import AX1DataSource, DataSourceCapability


class PriceVolumeDataSource(AX1DataSource):
    source_family = "price_volume"

    def __init__(self, raw_df: pd.DataFrame | None = None) -> None:
        self.raw_df = raw_df

    def capabilities(self) -> list[DataSourceCapability]:
        return [
            DataSourceCapability(
                name="price_volume.daily_ohlcv",
                source_family="price_volume",
                asset_type="both",
                point_in_time=True,
                observable_lag_days=0,
                requires_as_of_date=False,
                status="implemented",
                reason_code="implemented_daily_panel",
                description="Daily OHLCV panel supplied to AX1 runner.",
            )
        ]

    def load_panel(self, raw_df: pd.DataFrame | None = None) -> pd.DataFrame:
        frame = raw_df if raw_df is not None else self.raw_df
        if frame is None:
            raise ValueError("PriceVolumeDataSource requires raw_df")
        columns = [
            column
            for column in ("date", "order_book_id", "open", "high", "low", "close", "adjusted_close", "volume")
            if column in frame.columns
        ]
        missing = [column for column in ("date", "order_book_id", "close", "volume") if column not in frame.columns]
        if missing:
            raise ValueError(f"price_volume panel missing required columns: {missing}")
        result = frame.loc[:, columns].copy()
        result["date"] = pd.to_datetime(result["date"])
        result["order_book_id"] = result["order_book_id"].astype(str)
        return result
