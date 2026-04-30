"""AX1 fundamental data source implementation."""

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


class FundamentalDataSource(AX1DataSource):
    """Fundamental data source for valuation, quality, and growth factors with PIT compliance."""

    source_family = "fundamental"

    def capabilities(self) -> list[DataSourceCapability]:
        return [
            DataSourceCapability(
                name="fundamental.valuation_quality",
                source_family="fundamental",
                asset_type="stock",
                point_in_time=True,
                observable_lag_days=1,
                requires_as_of_date=True,
                status="implemented",
                reason_code="connected_to_rqdatac_factor_and_pit_financials",
                description="PE/PB/dividend_yield/gross_profit_margin from get_factor, ROE/net_profit_growth from get_pit_financials",
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
        """Load fundamental data with point-in-time compliance.

        Args:
            order_book_ids: List of stock codes
            start_date: Start date for the panel
            end_date: End date for the panel (as-of date for PIT)
            data_facade: DataFacade instance for cached access

        Returns:
            DataFrame with columns: date, order_book_id, feature_pe_ttm, feature_pb_ratio, feature_roe_ttm
        """
        if data_facade is None:
            from skyeye.data.facade import DataFacade

            data_facade = DataFacade()

        # Step 1: Load valuation + quality factors from get_factor
        factor_df = self._load_valuation_factors(
            order_book_ids=order_book_ids,
            start_date=start_date,
            end_date=end_date,
            data_facade=data_facade,
        )

        # Step 2: Load ROE from get_pit_financials
        roe_df = self._load_roe_ttm(
            order_book_ids=order_book_ids,
            end_date=end_date,
            data_facade=data_facade,
        )

        # Step 3: Load net profit growth YoY
        growth_df = self._load_net_profit_growth_yoy(
            order_book_ids=order_book_ids,
            end_date=end_date,
            data_facade=data_facade,
        )

        # Step 4: Merge and normalize
        result = self._merge_and_normalize(factor_df, roe_df, growth_df)

        return result

    def _load_valuation_factors(
        self,
        order_book_ids: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        data_facade: "DataFacade",
    ) -> pd.DataFrame | None:
        """Load PE and PB ratios from DataFacade.get_factor API."""
        try:
            # rqdatac factor names
            factors = ["pe_ratio", "pb_ratio", "dividend_yield", "gross_profit_margin"]

            factor_df = data_facade.get_factor(
                order_book_ids=order_book_ids,
                factors=factors,
                start_date=start_date,
                end_date=end_date,
            )

            if factor_df is None or factor_df.empty:
                logger.warning("No valuation factor data returned")
                return None

            # Normalize format
            if isinstance(factor_df.index, pd.MultiIndex):
                factor_df = factor_df.reset_index()

            # Rename columns to feature names
            column_mapping = {
                "pe_ratio": "feature_pe_ttm",
                "pb_ratio": "feature_pb_ratio",
                "dividend_yield": "feature_dividend_yield",
                "gross_profit_margin": "feature_gross_profit_margin",
            }
            factor_df = factor_df.rename(columns=column_mapping)

            # Ensure date column exists
            if "date" not in factor_df.columns:
                for col in ["datetime", "index"]:
                    if col in factor_df.columns:
                        factor_df = factor_df.rename(columns={col: "date"})
                        break

            # Ensure order_book_id column exists
            if "order_book_id" not in factor_df.columns:
                for col in ["symbol", "ticker"]:
                    if col in factor_df.columns:
                        factor_df = factor_df.rename(columns={col: "order_book_id"})
                        break

            # Select required columns
            required_cols = ["date", "order_book_id", "feature_pe_ttm", "feature_pb_ratio", "feature_dividend_yield", "feature_gross_profit_margin"]
            available_cols = [col for col in required_cols if col in factor_df.columns]

            if len(available_cols) < 3:  # At least date, order_book_id, and one feature
                logger.warning(f"Insufficient columns: {factor_df.columns.tolist()}")
                return None

            return factor_df[available_cols]

        except Exception as exc:
            logger.error(f"Failed to load valuation factors: {exc}")
            return None

    def _load_roe_ttm(
        self,
        order_book_ids: list[str],
        end_date: pd.Timestamp,
        data_facade: "DataFacade",
    ) -> pd.DataFrame | None:
        """Calculate ROE TTM from point-in-time financial statements via DataFacade."""
        try:
            # Use DataFacade's unified interface
            financials = data_facade.get_pit_financials(
                order_book_ids=order_book_ids,
                fields=["net_profit", "total_owner_equities"],
                count=4,
                statements="latest",
            )

            if financials is None or financials.empty:
                logger.warning("No financial data returned for ROE calculation")
                return None

            # Normalize format
            if isinstance(financials.index, pd.MultiIndex):
                financials = financials.reset_index()

            # Calculate ROE TTM per stock
            roe_records = []
            for order_book_id, group in financials.groupby("order_book_id"):
                # Sort by end_date (most recent first)
                if "end_date" in group.columns:
                    group = group.sort_values("end_date", ascending=False)
                elif "date" in group.columns:
                    group = group.sort_values("date", ascending=False)

                # TTM net profit = sum of last 4 quarters
                net_profit_values = pd.to_numeric(group["net_profit"], errors="coerce")
                net_profit_ttm = net_profit_values.sum()

                # Average equity = mean of last 2 quarters
                equity_values = pd.to_numeric(group["total_owner_equities"], errors="coerce")
                equity_avg = equity_values.iloc[:2].mean() if len(equity_values) >= 2 else None

                # Calculate ROE
                if equity_avg and equity_avg > 0 and pd.notna(net_profit_ttm):
                    roe_ttm = net_profit_ttm / equity_avg
                else:
                    roe_ttm = None

                roe_records.append(
                    {
                        "date": end_date,
                        "order_book_id": order_book_id,
                        "feature_roe_ttm": roe_ttm,
                    }
                )

            if not roe_records:
                return None

            return pd.DataFrame(roe_records)

        except Exception as exc:
            logger.error(f"Failed to calculate ROE TTM: {exc}")
            return None

    def _load_net_profit_growth_yoy(
        self,
        order_book_ids: list[str],
        end_date: pd.Timestamp,
        data_facade: "DataFacade",
    ) -> pd.DataFrame | None:
        """Calculate net profit YoY growth from point-in-time financial statements."""
        try:
            financials = data_facade.get_pit_financials(
                order_book_ids=order_book_ids,
                fields=["net_profit"],
                count=8,
                statements="latest",
            )

            if financials is None or financials.empty:
                logger.warning("No financial data returned for net profit growth")
                return None

            if isinstance(financials.index, pd.MultiIndex):
                financials = financials.reset_index()

            growth_records = []
            for order_book_id, group in financials.groupby("order_book_id"):
                sort_col = "end_date" if "end_date" in group.columns else "date"
                group = group.sort_values(sort_col, ascending=False)
                net_profit_values = pd.to_numeric(group["net_profit"], errors="coerce")

                # Current TTM = sum of most recent 4 quarters
                current_ttm = net_profit_values.iloc[:4].sum()
                # Prior TTM = sum of prior 4 quarters
                prior_ttm = net_profit_values.iloc[4:8].sum()

                if pd.notna(current_ttm) and pd.notna(prior_ttm) and prior_ttm != 0:
                    growth_yoy = (current_ttm - prior_ttm) / abs(prior_ttm)
                else:
                    growth_yoy = None

                growth_records.append(
                    {
                        "date": end_date,
                        "order_book_id": order_book_id,
                        "feature_net_profit_growth_yoy": growth_yoy,
                    }
                )

            if not growth_records:
                return None

            return pd.DataFrame(growth_records)

        except Exception as exc:
            logger.error(f"Failed to calculate net profit growth YoY: {exc}")
            return None

    def _merge_and_normalize(
        self,
        factor_df: pd.DataFrame | None,
        roe_df: pd.DataFrame | None,
        growth_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Merge valuation factors, ROE, and growth into a single panel."""
        result_parts = []

        if factor_df is not None and not factor_df.empty:
            result_parts.append(factor_df)

        if roe_df is not None and not roe_df.empty:
            result_parts.append(roe_df)

        if growth_df is not None and not growth_df.empty:
            result_parts.append(growth_df)

        if not result_parts:
            logger.warning("No fundamental data available")
            return pd.DataFrame(
                columns=["date", "order_book_id", "feature_pe_ttm", "feature_pb_ratio", "feature_roe_ttm", "feature_dividend_yield", "feature_gross_profit_margin", "feature_net_profit_growth_yoy"]
            )

        # Merge all parts
        result = reduce(
            lambda left, right: pd.merge(left, right, on=["date", "order_book_id"], how="outer"),
            result_parts,
        )

        # Normalize data types
        result["date"] = pd.to_datetime(result["date"])
        result["order_book_id"] = result["order_book_id"].astype(str)

        # Validate and clip extreme values
        for col in ["feature_pe_ttm", "feature_pb_ratio", "feature_roe_ttm", "feature_dividend_yield", "feature_gross_profit_margin", "feature_net_profit_growth_yoy"]:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")
                result[col] = result[col].clip(lower=-1000, upper=10000)

        return result.sort_values(["date", "order_book_id"]).reset_index(drop=True)
