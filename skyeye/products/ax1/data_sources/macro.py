"""AX1 macro data source — PMI, bond yield and aggregate northbound flow."""

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


class MacroDataSource(AX1DataSource):
    """Macro data source for PMI, bond yield and aggregate northbound flow."""

    source_family = "macro"

    def capabilities(self) -> list[DataSourceCapability]:
        return [
            DataSourceCapability(
                name="macro.pmi",
                source_family="macro",
                asset_type="both",
                point_in_time=True,
                observable_lag_days=1,
                requires_as_of_date=True,
                status="implemented",
                reason_code="connected_to_akshare_macro_pmi",
                description="Official manufacturing PMI with daily forward-fill after release",
            ),
            DataSourceCapability(
                name="macro.bond_yield",
                source_family="macro",
                asset_type="both",
                point_in_time=True,
                observable_lag_days=0,
                requires_as_of_date=True,
                status="implemented",
                reason_code="connected_to_rqdatac_yield_curve",
                description="10Y government bond yield from get_yield_curve",
            ),
            DataSourceCapability(
                name="macro.northbound_aggregate",
                source_family="flow",
                asset_type="both",
                point_in_time=True,
                observable_lag_days=1,
                requires_as_of_date=True,
                status="implemented",
                reason_code="connected_to_akshare_northbound",
                description="Aggregate northbound net flow (market-level signal)",
            ),
        ]

    def load_panel(
        self,
        order_book_ids: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        data_facade: Optional["DataFacade"] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Load macro data panel with PMI, bond yield and northbound aggregate flow."""
        if data_facade is None:
            from skyeye.data.facade import DataFacade
            data_facade = DataFacade()

        result_parts = []

        # PMI (monthly; forward-filled after release)
        pmi_df = self._load_pmi(start_date, end_date, data_facade)
        if pmi_df is not None and not pmi_df.empty:
            result_parts.append(pmi_df)

        # Bond yield
        bond_df = self._load_bond_yield(start_date, end_date, data_facade)
        if bond_df is not None and not bond_df.empty:
            result_parts.append(bond_df)

        # Northbound aggregate flow
        northbound_df = self._load_northbound_aggregate(start_date, end_date, data_facade)
        if northbound_df is not None and not northbound_df.empty:
            result_parts.append(northbound_df)

        if not result_parts:
            return pd.DataFrame(
                columns=[
                    "date",
                    "feature_bond_yield_10y",
                    "feature_northbound_aggregate_flow",
                    "feature_macro_pmi",
                ]
            )

        # Merge on date (all market-level, no order_book_id needed in merge)
        result = reduce(
            lambda left, right: pd.merge(left, right, on="date", how="outer"),
            result_parts,
        )

        result["date"] = pd.to_datetime(result["date"])

        # Drop order_book_id column if present (market-level data broadcasted later)
        for col in ["order_book_id_x", "order_book_id_y"]:
            if col in result.columns:
                result = result.drop(columns=[col])

        feature_cols = [column for column in result.columns if column.startswith("feature_")]
        result = result.sort_values("date").drop_duplicates(subset=["date"], keep="last").set_index("date")
        reindex_start = min(pd.Timestamp(start_date).normalize(), pd.Timestamp(result.index.min()).normalize())
        full_dates = pd.date_range(reindex_start, pd.Timestamp(end_date).normalize(), freq="D")
        result = result.reindex(full_dates)
        if feature_cols:
            result[feature_cols] = result[feature_cols].ffill()
        result.index.name = "date"
        result = result.reset_index()
        ordered_cols = [
            "date",
            "feature_bond_yield_10y",
            "feature_northbound_aggregate_flow",
            "feature_macro_pmi",
        ]
        existing_cols = [column for column in ordered_cols if column in result.columns]
        result = result.loc[
            (result["date"] >= pd.Timestamp(start_date).normalize())
            & (result["date"] <= pd.Timestamp(end_date).normalize())
        ]
        return result[existing_cols].sort_values("date").reset_index(drop=True)

    def _load_pmi(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        data_facade: "DataFacade",
    ) -> pd.DataFrame | None:
        """Load official manufacturing PMI and keep the latest known observation."""
        try:
            lookback_start = pd.Timestamp(start_date) - pd.Timedelta(days=45)
            raw = data_facade.get_macro_pmi(lookback_start, end_date)
            if raw is None or raw.empty:
                logger.warning("No PMI data returned")
                return None

            result = raw.copy()
            result["date"] = pd.to_datetime(result["date"])
            pmi_col = "pmi" if "pmi" in result.columns else result.select_dtypes(include=[np.number]).columns.tolist()[0]
            result = result[["date", pmi_col]].copy()
            result = result.rename(columns={pmi_col: "feature_macro_pmi"})
            result["feature_macro_pmi"] = pd.to_numeric(result["feature_macro_pmi"], errors="coerce")
            return result
        except Exception as exc:
            logger.error(f"Failed to load PMI: {exc}")
            return None

    def _load_bond_yield(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        data_facade: "DataFacade",
    ) -> pd.DataFrame | None:
        """Load 10Y bond yield."""
        try:
            raw = data_facade.get_bond_yield(start_date, end_date, tenor="10Y")
            if raw is None or raw.empty:
                logger.warning("No bond yield data returned")
                return None

            result = raw.copy()
            result["date"] = pd.to_datetime(result["date"])

            # Rename to feature column
            yield_col = "bond_yield_10Y" if "bond_yield_10Y" in result.columns else result.columns[-1]
            result = result[["date", yield_col]].copy()
            result = result.rename(columns={yield_col: "feature_bond_yield_10y"})

            # Normalize: express as decimal (e.g., 2.5% → 0.025)
            result["feature_bond_yield_10y"] = pd.to_numeric(result["feature_bond_yield_10y"], errors="coerce")
            if result["feature_bond_yield_10y"].median() > 1.0:
                result["feature_bond_yield_10y"] = result["feature_bond_yield_10y"] / 100.0

            return result

        except Exception as exc:
            logger.error(f"Failed to load bond yield: {exc}")
            return None

    def _load_northbound_aggregate(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        data_facade: "DataFacade",
    ) -> pd.DataFrame | None:
        """Load aggregate northbound flow (market-level)."""
        try:
            raw = data_facade.get_northbound_flow(start_date, end_date)
            if raw is None or raw.empty:
                logger.warning("No northbound aggregate flow data returned")
                return None

            result = raw.copy()
            result["date"] = pd.to_datetime(result["date"])

            # Find the flow column
            flow_col = None
            for col in ["northbound_net_flow", "north_net_flow", "net_flow", "flow"]:
                if col in result.columns:
                    flow_col = col
                    break

            if flow_col is None:
                # Use the last numeric column
                numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    flow_col = numeric_cols[0]

            if flow_col is None:
                logger.warning("No northbound flow column found")
                return None

            result = result[["date", flow_col]].copy()
            result = result.rename(columns={flow_col: "feature_northbound_aggregate_flow"})

            # Normalize to billions
            result["feature_northbound_aggregate_flow"] = pd.to_numeric(
                result["feature_northbound_aggregate_flow"], errors="coerce"
            )
            median_val = result["feature_northbound_aggregate_flow"].median()
            if pd.notna(median_val) and abs(median_val) > 1e9:
                result["feature_northbound_aggregate_flow"] = result["feature_northbound_aggregate_flow"] / 1e8

            return result

        except Exception as exc:
            logger.error(f"Failed to load northbound aggregate flow: {exc}")
            return None
