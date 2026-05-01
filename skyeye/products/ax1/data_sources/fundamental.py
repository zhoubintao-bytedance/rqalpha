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
            start_date=start_date,
            end_date=end_date,
            data_facade=data_facade,
        )

        # Step 3: Load net profit growth YoY
        growth_df = self._load_net_profit_growth_yoy(
            order_book_ids=order_book_ids,
            start_date=start_date,
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
            factors = ["pe_ratio_ttm", "pb_ratio", "dividend_yield", "gross_profit_margin"]

            factor_df = data_facade.get_factor(
                order_book_ids=order_book_ids,
                factors=factors,
                start_date=start_date,
                end_date=end_date,
            )
            if factor_df is None or factor_df.empty:
                fallback_factors = ["pe_ratio", "pb_ratio", "dividend_yield", "gross_profit_margin"]
                factor_df = data_facade.get_factor(
                    order_book_ids=order_book_ids,
                    factors=fallback_factors,
                    start_date=start_date,
                    end_date=end_date,
                )

            if factor_df is None or factor_df.empty:
                logger.warning("No valuation factor data returned")
                return None

            factor_df = self._normalize_panel_keys(factor_df)
            has_ttm_pe = "pe_ratio_ttm" in factor_df.columns and factor_df["pe_ratio_ttm"].notna().any()
            if not has_ttm_pe:
                fallback_pe = data_facade.get_factor(
                    order_book_ids=order_book_ids,
                    factors=["pe_ratio"],
                    start_date=start_date,
                    end_date=end_date,
                )
                if fallback_pe is not None and not fallback_pe.empty:
                    fallback_pe = self._normalize_panel_keys(fallback_pe)
                    pe_cols = [col for col in ["date", "order_book_id", "pe_ratio"] if col in fallback_pe.columns]
                    if set(["date", "order_book_id", "pe_ratio"]).issubset(pe_cols):
                        factor_df = factor_df.merge(
                            fallback_pe[pe_cols].drop_duplicates(["date", "order_book_id"], keep="last"),
                            on=["date", "order_book_id"],
                            how="left",
                        )

            # Rename columns to feature names
            column_mapping = {
                "pe_ratio_ttm": "feature_pe_ttm",
                "pe_ratio": "feature_pe_ttm",
                "pb_ratio": "feature_pb_ratio",
                "dividend_yield": "feature_dividend_yield",
                "gross_profit_margin": "feature_gross_profit_margin",
            }
            factor_df = factor_df.rename(columns=column_mapping)

            # Select required columns
            required_cols = ["date", "order_book_id", "feature_pe_ttm", "feature_pb_ratio", "feature_dividend_yield", "feature_gross_profit_margin"]
            available_cols = [col for col in required_cols if col in factor_df.columns]

            if len(available_cols) < 3:  # At least date, order_book_id, and one feature
                logger.warning(f"Insufficient columns: {factor_df.columns.tolist()}")
                return None

            result = factor_df[available_cols].copy()
            for ratio_col in ("feature_dividend_yield", "feature_gross_profit_margin"):
                if ratio_col in result.columns:
                    result[ratio_col] = self._normalize_percent_like_series(result[ratio_col])
            return result

        except Exception as exc:
            logger.error(f"Failed to load valuation factors: {exc}")
            return None

    @staticmethod
    def _normalize_panel_keys(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if isinstance(result.index, pd.MultiIndex):
            result = result.reset_index()
        elif "date" not in result.columns or "order_book_id" not in result.columns:
            result = result.reset_index()

        if "date" not in result.columns:
            for col in ["datetime", "DateTime", "trading_date", "index"]:
                if col in result.columns:
                    result = result.rename(columns={col: "date"})
                    break
        if "order_book_id" not in result.columns:
            for col in ["symbol", "ticker", "order_book_ids"]:
                if col in result.columns:
                    result = result.rename(columns={col: "order_book_id"})
                    break
        if "date" in result.columns:
            result["date"] = pd.to_datetime(result["date"]).dt.normalize()
        if "order_book_id" in result.columns:
            result["order_book_id"] = result["order_book_id"].astype(str)
        return result

    def _load_roe_ttm(
        self,
        order_book_ids: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        data_facade: "DataFacade",
    ) -> pd.DataFrame | None:
        """Calculate ROE TTM from point-in-time financial statements via DataFacade."""
        try:
            # Use DataFacade's unified interface
            financials = data_facade.get_pit_financials(
                order_book_ids=order_book_ids,
                fields=["return_on_equity_weighted_average", "net_profitTTM", "equity_parent_company"],
                count=8,
                statements="latest",
                date=pd.Timestamp(end_date),
            )

            if financials is None or financials.empty:
                logger.warning("No financial data returned for ROE calculation")
                return None

            financials = self._normalize_financial_frame(financials)
            if "return_on_equity_weighted_average" in financials.columns:
                financials["feature_roe_ttm"] = self._normalize_percent_like_series(
                    financials["return_on_equity_weighted_average"]
                )
            elif {"net_profitTTM", "equity_parent_company"}.issubset(financials.columns):
                equity = pd.to_numeric(financials["equity_parent_company"], errors="coerce").replace(0.0, np.nan)
                financials["feature_roe_ttm"] = pd.to_numeric(financials["net_profitTTM"], errors="coerce") / equity
            else:
                logger.warning(f"Insufficient financial columns for ROE: {financials.columns.tolist()}")
                return None

            return self._expand_financial_feature_to_daily(
                financials,
                value_column="feature_roe_ttm",
                start_date=start_date,
                end_date=end_date,
            )

        except Exception as exc:
            logger.error(f"Failed to calculate ROE TTM: {exc}")
            return None

    def _load_net_profit_growth_yoy(
        self,
        order_book_ids: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        data_facade: "DataFacade",
    ) -> pd.DataFrame | None:
        """Calculate net profit YoY growth from point-in-time financial statements."""
        try:
            financials = data_facade.get_pit_financials(
                order_book_ids=order_book_ids,
                fields=["return_on_equity_weighted_average", "net_profitTTM", "equity_parent_company"],
                count=8,
                statements="latest",
                date=pd.Timestamp(end_date),
            )

            if financials is None or financials.empty:
                logger.warning("No financial data returned for net profit growth")
                return None

            financials = self._normalize_financial_frame(financials)
            if "net_profitTTM" not in financials.columns:
                logger.warning(f"Insufficient financial columns for net profit growth: {financials.columns.tolist()}")
                return None

            result_parts = []
            for order_book_id, group in financials.groupby("order_book_id"):
                group = group.sort_values("_quarter_order").copy()
                current = pd.to_numeric(group["net_profitTTM"], errors="coerce")
                prior = current.shift(4).replace(0.0, np.nan)
                group["feature_net_profit_growth_yoy"] = (current - prior) / prior.abs()
                result_parts.append(group)

            if not result_parts:
                return None

            return self._expand_financial_feature_to_daily(
                pd.concat(result_parts, ignore_index=True),
                value_column="feature_net_profit_growth_yoy",
                start_date=start_date,
                end_date=end_date,
            )

        except Exception as exc:
            logger.error(f"Failed to calculate net profit growth YoY: {exc}")
            return None

    @staticmethod
    def _normalize_percent_like_series(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        clean = values.dropna()
        if clean.empty:
            return values
        median_abs = clean.abs().median()
        if pd.notna(median_abs):
            if median_abs > 100.0:
                values = values / 10000.0
            elif median_abs > 1.0:
                values = values / 100.0
        return values

    @staticmethod
    def _quarter_order(value) -> int:
        text = str(value).lower().replace(" ", "")
        if "q" in text:
            year_text, quarter_text = text.split("q", 1)
            try:
                return int(year_text) * 4 + int(quarter_text)
            except ValueError:
                return -1
        return -1

    def _normalize_financial_frame(self, financials: pd.DataFrame) -> pd.DataFrame:
        result = financials.copy()
        if isinstance(result.index, pd.MultiIndex):
            result = result.reset_index()
        if "order_book_id" not in result.columns:
            raise ValueError("financials missing order_book_id")
        if "quarter" not in result.columns:
            result["quarter"] = result.get("end_date", result.get("date", pd.NaT))
        if "info_date" not in result.columns:
            result["info_date"] = result.get("date", result.get("end_date", pd.NaT))
        result["order_book_id"] = result["order_book_id"].astype(str)
        result["info_date"] = pd.to_datetime(result["info_date"], errors="coerce")
        result["_quarter_order"] = result["quarter"].map(self._quarter_order)
        if (result["_quarter_order"] < 0).all():
            result["_quarter_order"] = pd.to_datetime(result["info_date"], errors="coerce").rank(method="first").astype(int)
        return result

    def _expand_financial_feature_to_daily(
        self,
        financials: pd.DataFrame,
        *,
        value_column: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame | None:
        records = []
        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        calendar = pd.DataFrame({"date": pd.date_range(start_ts, end_ts, freq="D")})
        for order_book_id, group in financials.groupby("order_book_id"):
            events = group[["info_date", value_column]].copy()
            events["date"] = (pd.to_datetime(events["info_date"], errors="coerce") + pd.Timedelta(days=1)).dt.normalize()
            events[value_column] = pd.to_numeric(events[value_column], errors="coerce")
            events = (
                events.dropna(subset=["date", value_column])
                .loc[lambda frame: frame["date"] <= end_ts, ["date", value_column]]
                .sort_values("date")
                .drop_duplicates("date", keep="last")
            )
            if events.empty:
                continue
            daily = pd.merge_asof(calendar, events, on="date", direction="backward")
            daily = daily.dropna(subset=[value_column])
            if daily.empty:
                continue
            daily["order_book_id"] = str(order_book_id)
            records.append(daily[["date", "order_book_id", value_column]])
        if not records:
            return None
        return pd.concat(records, ignore_index=True)

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
