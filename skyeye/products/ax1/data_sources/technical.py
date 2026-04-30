"""AX1 technical indicator data source implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from skyeye.products.ax1.data_sources.base import AX1DataSource, DataSourceCapability

if TYPE_CHECKING:
    from skyeye.data.facade import DataFacade


TECHNICAL_FEATURE_COLUMNS = ["feature_rsi_14d", "feature_macd"]


def build_technical_indicator_features(frame: pd.DataFrame) -> list[str]:
    """Build RSI/MACD features in place from `close` prices."""
    grouped = frame.groupby("order_book_id", sort=False)

    delta = grouped["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.groupby(frame["order_book_id"], sort=False).transform(
        lambda s: s.rolling(14, min_periods=7).mean()
    )
    avg_loss = loss.groupby(frame["order_book_id"], sort=False).transform(
        lambda s: s.rolling(14, min_periods=7).mean()
    )
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    frame["feature_rsi_14d"] = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0).clip(lower=0.0, upper=100.0)

    ema12 = grouped["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = grouped["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    dif = ema12 - ema26
    dea = dif.groupby(frame["order_book_id"], sort=False).transform(
        lambda s: s.ewm(span=9, adjust=False).mean()
    )
    frame["feature_macd"] = (dif - dea).fillna(0.0)
    return list(TECHNICAL_FEATURE_COLUMNS)


class TechnicalIndicatorDataSource(AX1DataSource):
    source_family = "technical"

    def capabilities(self) -> list[DataSourceCapability]:
        return [
            DataSourceCapability(
                name="technical.indicators",
                source_family="technical",
                asset_type="both",
                point_in_time=True,
                observable_lag_days=0,
                requires_as_of_date=True,
                status="implemented",
                reason_code="derived_from_daily_close_prices",
                description="RSI(14) and MACD histogram derived from daily close prices.",
            )
        ]

    def load_panel(
        self,
        order_book_ids: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        data_facade: Optional["DataFacade"] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if data_facade is None:
            from skyeye.data.facade import DataFacade

            data_facade = DataFacade()

        raw = data_facade.get_daily_bars(
            order_book_ids,
            start_date=start_date,
            end_date=end_date,
            fields=["close"],
            adjust_type="pre",
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["date", "order_book_id", *TECHNICAL_FEATURE_COLUMNS])

        frame = raw.copy()
        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.reset_index()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["order_book_id"] = frame["order_book_id"].astype(str)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.sort_values(["order_book_id", "date"]).reset_index(drop=True)
        build_technical_indicator_features(frame)
        return frame[["date", "order_book_id", *TECHNICAL_FEATURE_COLUMNS]].copy()
