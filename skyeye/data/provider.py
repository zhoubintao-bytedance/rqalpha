# -*- coding: utf-8 -*-
"""
RQDataProvider — 天眼统一数据接口

底层调用 rqdatac，为所有 skyeye 产品提供统一、可靠的数据访问层。
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import pandas as pd
import rqdatac

logger = logging.getLogger(__name__)

DateLike = Union[str, int, "pd.Timestamp"]


class RQDataProvider:
    """Skyeye unified data provider backed by rqdatac."""

    _initialized: bool = False

    def __init__(self) -> None:
        if not RQDataProvider._initialized:
            rqdatac.init()
            RQDataProvider._initialized = True
            logger.info("rqdatac initialized")

    # ------------------------------------------------------------------
    # Price / OHLCV
    # ------------------------------------------------------------------

    def get_price(
        self,
        order_book_ids: Union[str, list[str]],
        start_date: DateLike,
        end_date: DateLike,
        frequency: str = "1d",
        adjust_type: str = "pre",
        fields: Optional[list[str]] = None,
        skip_suspended: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Stock/ETF/Index OHLCV data.

        Replaces: h5py reads from stocks.h5/indexes.h5, ak.fund_etf_hist_em.
        """
        return rqdatac.get_price(
            order_book_ids,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            fields=fields,
            adjust_type=adjust_type,
            skip_suspended=skip_suspended,
            expect_df=True,
        )

    def get_etf_price(
        self,
        etf_code: str,
        start_date: DateLike,
        end_date: DateLike,
        with_iopv: bool = True,
    ) -> Optional[pd.DataFrame]:
        """ETF price data including IOPV (indicative NAV).

        Replaces: ak.fund_etf_hist_em + ak.fund_etf_fund_info_em.
        """
        fields = None  # get all fields including iopv
        df = rqdatac.get_price(
            etf_code,
            start_date=start_date,
            end_date=end_date,
            frequency="1d",
            fields=fields,
            adjust_type="none",
            expect_df=True,
        )
        if df is None:
            return None

        # Also get post-adjusted close for close_hfq
        df_hfq = rqdatac.get_price(
            etf_code,
            start_date=start_date,
            end_date=end_date,
            frequency="1d",
            fields=["close"],
            adjust_type="post",
            expect_df=True,
        )
        if df_hfq is not None:
            df["close_hfq"] = df_hfq["close"]

        return df

    # ------------------------------------------------------------------
    # Index data
    # ------------------------------------------------------------------

    def get_index_price(
        self,
        index_code: str,
        start_date: DateLike,
        end_date: DateLike,
        fields: Optional[list[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Index OHLCV data.

        Replaces: ak.stock_zh_index_hist_csindex, h5py reads from indexes.h5.
        """
        return rqdatac.get_price(
            index_code,
            start_date=start_date,
            end_date=end_date,
            frequency="1d",
            fields=fields,
            adjust_type="none",
            expect_df=True,
        )

    def get_index_weights(
        self, index_code: str, date: Optional[DateLike] = None
    ) -> Optional[pd.Series]:
        """Index constituent weights (monthly updated).

        Replaces: ak.index_stock_cons_weight_csindex.
        """
        return rqdatac.index_weights(index_code, date=date)

    def get_index_components(
        self, index_code: str, date: Optional[DateLike] = None
    ) -> Optional[list[str]]:
        """Index constituent list."""
        return rqdatac.index_components(index_code, date=date)

    def get_index_indicator(
        self,
        index_code: str,
        start_date: DateLike,
        end_date: DateLike,
        fields: Optional[list[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Index valuation indicators (PE, PB, dividend yield, market cap).

        Replaces: ak.stock_zh_index_value_csindex.
        Note: Not all indices are supported. Falls back to component aggregation.
        """
        result = rqdatac.index_indicator(
            index_code, start_date=start_date, end_date=end_date, fields=fields
        )
        if result is not None:
            return result

        # Fallback: aggregate PE from components via get_factor
        logger.warning(
            "index_indicator unavailable for %s, falling back to component aggregation",
            index_code,
        )
        components = self.get_index_weights(index_code, date=end_date)
        if components is None or components.empty:
            return None

        stock_ids = components.index.tolist()
        factors = self.get_factors(
            stock_ids, ["pe_ratio_ttm"], start_date, end_date
        )
        if factors is None:
            return None

        # Weighted average PE by date
        weights = components.reindex(factors.index.get_level_values("order_book_id"))
        weights.index = factors.index
        result = (
            factors.multiply(weights, axis=0)
            .groupby("date")
            .sum()
        )
        result.columns = ["pe_ttm"]
        return result

    # ------------------------------------------------------------------
    # Factor data
    # ------------------------------------------------------------------

    def get_factors(
        self,
        order_book_ids: Union[str, list[str]],
        factors: Union[str, list[str]],
        start_date: DateLike,
        end_date: DateLike,
    ) -> Optional[pd.DataFrame]:
        """Batch factor query (PE, PB, dividend yield, market cap, technicals, etc.).

        Replaces: ak.stock_a_indicator_lg (one-by-one) with single batch call.
        """
        return rqdatac.get_factor(
            order_book_ids,
            factors,
            start_date=start_date,
            end_date=end_date,
            expect_df=True,
        )

    def get_all_factor_names(self, type: Optional[str] = None) -> list[str]:
        """List all available factor names by category."""
        return rqdatac.get_all_factor_names(type=type)

    # ------------------------------------------------------------------
    # Bond yield
    # ------------------------------------------------------------------

    def get_bond_yield(
        self,
        start_date: DateLike,
        end_date: DateLike,
        tenor: str = "10Y",
    ) -> Optional[pd.DataFrame]:
        """China government bond yield curve.

        Replaces: ak.bond_zh_us_rate.
        """
        return rqdatac.get_yield_curve(
            start_date=start_date, end_date=end_date, tenor=tenor
        )

    # ------------------------------------------------------------------
    # Dividend / Split
    # ------------------------------------------------------------------

    def get_dividend(
        self,
        order_book_ids: Union[str, list[str]],
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
    ) -> Optional[pd.DataFrame]:
        """Cash dividend data.

        Replaces: ak.stock_fhps_detail_em.
        """
        return rqdatac.get_dividend(
            order_book_ids, start_date=start_date, end_date=end_date, expect_df=True
        )

    # ------------------------------------------------------------------
    # Reference data
    # ------------------------------------------------------------------

    def get_trading_dates(
        self, start_date: DateLike, end_date: DateLike
    ) -> list:
        """Trading calendar."""
        return rqdatac.get_trading_dates(start_date, end_date)

    def get_latest_trading_date(self) -> "pd.Timestamp":
        return rqdatac.get_latest_trading_date()

    def get_previous_trading_date(
        self, date: DateLike, n: int = 1
    ) -> "pd.Timestamp":
        return rqdatac.get_previous_trading_date(date, n=n)

    def get_instruments(
        self,
        type: str = "CS",
        date: Optional[DateLike] = None,
    ) -> Optional[pd.DataFrame]:
        """List all instruments.

        Replaces: pickle.load(instruments.pk).
        """
        return rqdatac.all_instruments(type=type, date=date)

    def get_instrument_info(
        self, order_book_ids: Union[str, list[str]]
    ):
        """Detailed instrument info."""
        return rqdatac.instruments(order_book_ids)

    def get_industry(
        self,
        order_book_ids: Union[str, list[str]],
        source: str = "citics_2019",
        level: int = 1,
        date: Optional[DateLike] = None,
    ) -> Optional[pd.DataFrame]:
        """Industry classification.

        Replaces: sector_code from instruments.pk.
        """
        return rqdatac.get_instrument_industry(
            order_book_ids, source=source, level=level, date=date
        )

    # ------------------------------------------------------------------
    # Suspension / ST
    # ------------------------------------------------------------------

    def is_suspended(
        self,
        order_book_ids: Union[str, list[str]],
        start_date: DateLike,
        end_date: DateLike,
    ) -> Optional[pd.DataFrame]:
        return rqdatac.is_suspended(
            order_book_ids, start_date=start_date, end_date=end_date
        )

    def is_st_stock(
        self,
        order_book_ids: Union[str, list[str]],
        start_date: DateLike,
        end_date: DateLike,
    ) -> Optional[pd.DataFrame]:
        return rqdatac.is_st_stock(
            order_book_ids, start_date=start_date, end_date=end_date
        )

    # ------------------------------------------------------------------
    # Turnover
    # ------------------------------------------------------------------

    def get_turnover_rate(
        self,
        order_book_ids: Union[str, list[str]],
        start_date: DateLike,
        end_date: DateLike,
        fields: Optional[list[str]] = None,
    ) -> Optional[pd.DataFrame]:
        return rqdatac.get_turnover_rate(
            order_book_ids,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
            expect_df=True,
        )

    # ------------------------------------------------------------------
    # Share capital
    # ------------------------------------------------------------------

    def get_shares(
        self,
        order_book_ids: Union[str, list[str]],
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        fields: Optional[list[str]] = None,
    ) -> Optional[pd.DataFrame]:
        return rqdatac.get_shares(
            order_book_ids,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
            expect_df=True,
        )

    # ------------------------------------------------------------------
    # Northbound (fallback to AKShare)
    # ------------------------------------------------------------------

    def get_northbound_flow(
        self, start_date: DateLike, end_date: DateLike
    ) -> Optional[pd.DataFrame]:
        """Northbound aggregate net flow.

        rqdatac.get_stock_connect may not work for aggregate flows.
        Falls back to AKShare.
        """
        try:
            from skyeye.data.compat import get_northbound_flow_akshare

            return get_northbound_flow_akshare(start_date, end_date)
        except Exception:
            logger.warning("Northbound flow unavailable from any source")
            return None
