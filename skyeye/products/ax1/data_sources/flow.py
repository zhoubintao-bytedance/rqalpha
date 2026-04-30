"""AX1 capital flow data source implementation."""

from __future__ import annotations

import logging
from functools import reduce
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from skyeye.products.ax1.data_sources.base import AX1DataSource, DataSourceCapability

if TYPE_CHECKING:
    from skyeye.data.facade import DataFacade

logger = logging.getLogger(__name__)


class FlowDataSource(AX1DataSource):
    """Capital flow data source for margin financing and northbound flow."""

    source_family = "flow"

    def capabilities(self) -> list[DataSourceCapability]:
        return [
            DataSourceCapability(
                name="flow.capital_flow",
                source_family="flow",
                asset_type="both",
                point_in_time=True,
                observable_lag_days=1,
                requires_as_of_date=True,
                status="implemented",
                reason_code="connected_to_rqdatac_margin_and_stock_connect",
                description="Margin balance from get_securities_margin, Northbound flow from get_stock_connect",
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
        """Load capital flow data with point-in-time compliance.

        Args:
            order_book_ids: List of stock/ETF codes
            start_date: Start date for the panel
            end_date: End date for the panel
            data_facade: DataFacade instance for cached access

        Returns:
            DataFrame with columns: date, order_book_id, feature_margin_financing_balance,
                                    feature_northbound_net_flow
        """
        if data_facade is None:
            from skyeye.data.facade import DataFacade

            data_facade = DataFacade()

        # Separate stocks and ETFs (northbound only applies to stocks)
        stock_ids = [id for id in order_book_ids if self._is_stock(id)]
        etf_ids = [id for id in order_book_ids if not self._is_stock(id)]

        result_parts = []

        # Part 1: Load margin financing balance (stocks + ETFs)
        margin_df = self._load_margin_data(
            order_book_ids=order_book_ids,
            start_date=start_date,
            end_date=end_date,
            data_facade=data_facade,
        )
        if margin_df is not None and not margin_df.empty:
            result_parts.append(margin_df)

        # Part 2: Load northbound capital flow (stocks only)
        if stock_ids:
            northbound_df = self._load_northbound_data(
                order_book_ids=stock_ids,
                start_date=start_date,
                end_date=end_date,
                data_facade=data_facade,
            )
            if northbound_df is not None and not northbound_df.empty:
                result_parts.append(northbound_df)

        # Merge results
        if not result_parts:
            logger.warning("No flow data available")
            return pd.DataFrame(
                columns=[
                    "date",
                    "order_book_id",
                    "feature_margin_financing_balance",
                    "feature_northbound_net_flow",
                    "feature_institutional_holding_ratio",
                ]
            )

        result = reduce(
            lambda left, right: pd.merge(left, right, on=["date", "order_book_id"], how="outer"),
            result_parts,
        )

        # Normalize data types
        result["date"] = pd.to_datetime(result["date"])
        result["order_book_id"] = result["order_book_id"].astype(str)

        return result.sort_values(["date", "order_book_id"]).reset_index(drop=True)

    def _load_margin_data(
        self,
        order_book_ids: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        data_facade: "DataFacade",
    ) -> pd.DataFrame | None:
        """Load margin financing balance from DataFacade.get_securities_margin."""
        try:
            # Use DataFacade's unified interface with caching
            margin_df = data_facade.get_securities_margin(
                order_book_ids,
                start_date=start_date,
                end_date=end_date,
                fields="margin_balance",
            )

            if margin_df is None or margin_df.empty:
                logger.warning("No margin data returned")
                return None

            # Normalize format
            if isinstance(margin_df.index, pd.MultiIndex):
                margin_df = margin_df.reset_index()
            elif isinstance(margin_df.index, pd.DatetimeIndex):
                margin_df = margin_df.reset_index()
                margin_df = margin_df.melt(
                    id_vars="index", var_name="order_book_id", value_name="margin_balance"
                )
                margin_df = margin_df.rename(columns={"index": "date"})

            # Ensure date column exists
            if "date" not in margin_df.columns:
                for col in ["datetime", "index"]:
                    if col in margin_df.columns:
                        margin_df = margin_df.rename(columns={col: "date"})
                        break

            # Ensure order_book_id column exists
            if "order_book_id" not in margin_df.columns:
                for col in ["symbol", "ticker"]:
                    if col in margin_df.columns:
                        margin_df = margin_df.rename(columns={col: "order_book_id"})
                        break

            # Rename margin_balance to feature name
            if "margin_balance" in margin_df.columns:
                margin_df["feature_margin_financing_balance"] = np.log1p(
                    pd.to_numeric(margin_df["margin_balance"], errors="coerce")
                    .fillna(0.0)
                    .clip(lower=0.0)
                )

            # Select required columns
            required_cols = ["date", "order_book_id", "feature_margin_financing_balance"]
            available_cols = [col for col in required_cols if col in margin_df.columns]

            if len(available_cols) < 3:
                logger.warning(f"Insufficient margin columns: {margin_df.columns.tolist()}")
                return None

            return margin_df[available_cols]

        except Exception as exc:
            logger.error(f"Failed to load margin data: {exc}")
            return None

    def _load_northbound_data(
        self,
        order_book_ids: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        data_facade: "DataFacade",
    ) -> pd.DataFrame | None:
        """Load northbound capital flow from DataFacade.get_stock_connect."""
        try:
            # Use DataFacade's unified interface with caching
            connect_df = data_facade.get_stock_connect(
                order_book_ids,
                start_date=start_date,
                end_date=end_date,
                fields=["shares_holding", "holding_ratio"],
            )

            if connect_df is None or connect_df.empty:
                logger.warning("No northbound data returned")
                return None

            # Normalize format
            if isinstance(connect_df.index, pd.MultiIndex):
                connect_df = connect_df.reset_index()

            # Ensure date column exists
            if "date" not in connect_df.columns:
                for col in ["datetime", "index"]:
                    if col in connect_df.columns:
                        connect_df = connect_df.rename(columns={col: "date"})
                        break

            # Ensure order_book_id column exists
            if "order_book_id" not in connect_df.columns:
                for col in ["symbol", "ticker"]:
                    if col in connect_df.columns:
                        connect_df = connect_df.rename(columns={col: "order_book_id"})
                        break

            # Sort by date and order_book_id
            connect_df["date"] = pd.to_datetime(connect_df["date"])
            connect_df = connect_df.sort_values(["order_book_id", "date"])

            # Calculate net flow: change in shares_holding
            if "shares_holding" in connect_df.columns:
                connect_df["feature_northbound_net_flow"] = connect_df.groupby("order_book_id")[
                    "shares_holding"
                ].diff()

                # Normalize as percentage of current holding
                connect_df["feature_northbound_net_flow"] = (
                    connect_df["feature_northbound_net_flow"]
                    / connect_df["shares_holding"].replace(0, np.nan)
                ).fillna(0.0)

            if "holding_ratio" in connect_df.columns:
                connect_df["feature_institutional_holding_ratio"] = pd.to_numeric(
                    connect_df["holding_ratio"], errors="coerce"
                )
                median_val = connect_df["feature_institutional_holding_ratio"].median()
                if pd.notna(median_val) and abs(median_val) > 1.0:
                    connect_df["feature_institutional_holding_ratio"] = (
                        connect_df["feature_institutional_holding_ratio"] / 100.0
                    )

            # Select required columns
            required_cols = [
                "date",
                "order_book_id",
                "feature_northbound_net_flow",
                "feature_institutional_holding_ratio",
            ]
            available_cols = [col for col in required_cols if col in connect_df.columns]

            if len(available_cols) < 3:
                logger.warning(f"Insufficient northbound columns: {connect_df.columns.tolist()}")
                return None

            return connect_df[available_cols]

        except Exception as exc:
            logger.error(f"Failed to load northbound data: {exc}")
            return None

    @staticmethod
    def _is_stock(order_book_id: str) -> bool:
        """Check if an order_book_id is a stock (not ETF).

        Simple heuristic: ETF codes typically start with 51, 159, 58, or contain .XSHG/.XSHE
        """
        if not order_book_id:
            return False

        # Remove exchange suffix for pattern matching
        code = order_book_id.split(".")[0]

        # ETF code patterns
        etf_patterns = [
            code.startswith("51"),  # Shanghai ETFs
            code.startswith("159"),  # Shenzhen ETFs
            code.startswith("58"),  # Shanghai ETFs (alternative)
        ]

        return not any(etf_patterns)
