"""AX1 training data builder backed by the shared SkyEye data facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from skyeye.products.ax1.etf_metadata import resolve_ax1_etf_industry_map


DEFAULT_AX1_BAR_FIELDS = ("open", "high", "low", "close", "volume", "total_turnover")


@dataclass(frozen=True)
class AX1TrainingDataRequest:
    profile_config: Mapping[str, Any]
    start_date: str | pd.Timestamp
    end_date: str | pd.Timestamp
    data_source: str = "auto"
    adjust_type: str = "pre"
    fields: Sequence[str] = DEFAULT_AX1_BAR_FIELDS


class AX1TrainingDataBuilder:
    def __init__(self, data_facade=None) -> None:
        if data_facade is None:
            from skyeye.data import DataFacade

            data_facade = DataFacade()
        self.data_facade = data_facade

    def build(self, request: AX1TrainingDataRequest) -> pd.DataFrame:
        specs = _flatten_enabled_layer_specs(request.profile_config)
        order_book_ids = [spec["order_book_id"] for spec in specs]
        if not order_book_ids:
            raise ValueError("AX1 training data request resolved an empty universe")

        raw = self.data_facade.get_daily_bars(
            order_book_ids,
            request.start_date,
            request.end_date,
            fields=list(request.fields),
            adjust_type=str(request.adjust_type),
        )
        frame = _normalize_daily_bars(raw, requested_fields=list(request.fields))
        if frame.empty:
            raise ValueError("AX1 training data builder loaded no daily bars")

        frame = _attach_layer_metadata(frame, specs)
        frame = _attach_listing_dates(
            frame,
            self.data_facade,
            end_date=request.end_date,
        )
        frame = _attach_provider_industry(
            frame,
            self.data_facade,
            end_date=request.end_date,
        )
        frame = _attach_status_flags(
            frame,
            self.data_facade,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        frame = _attach_adjusted_price_contract(frame, adjust_type=str(request.adjust_type))
        return frame.sort_values(["date", "order_book_id"]).reset_index(drop=True)


def _flatten_enabled_layer_specs(profile_config: Mapping[str, Any]) -> list[dict[str, str]]:
    universe = profile_config.get("universe") or {}
    layers = universe.get("layers") or {}
    specs: list[dict[str, str]] = []
    seen: set[str] = set()
    for layer_name, payload in layers.items():
        layer = dict(payload or {})
        if not bool(layer.get("enabled", True)):
            continue
        asset_type = str(layer.get("asset_type", "") or "").strip().lower()
        exposure_group = str(layer.get("exposure_group") or layer_name)
        for raw_order_book_id in layer.get("include") or []:
            order_book_id = str(raw_order_book_id).strip()
            if not order_book_id:
                continue
            if order_book_id in seen:
                raise ValueError(f"AX1 training universe contains duplicate order_book_id: {order_book_id}")
            seen.add(order_book_id)
            specs.append(
                {
                    "order_book_id": order_book_id,
                    "asset_type": asset_type,
                    "universe_layer": str(layer_name),
                    "exposure_group": exposure_group,
                }
            )
    return specs


def _normalize_daily_bars(raw: Any, *, requested_fields: list[str]) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame(columns=["date", "order_book_id", *requested_fields])
    frame = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    if frame.empty:
        return pd.DataFrame(columns=["date", "order_book_id", *requested_fields])
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    elif "date" not in frame.columns:
        frame = frame.reset_index()
    if "datetime" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"datetime": "date"})
    if "index" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"index": "date"})
    if "date" not in frame.columns or "order_book_id" not in frame.columns:
        raise ValueError("DataFacade daily bars must include date and order_book_id")
    keep = ["date", "order_book_id", *[field for field in requested_fields if field in frame.columns]]
    frame = frame.loc[:, list(dict.fromkeys(keep))].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["order_book_id"] = frame["order_book_id"].astype(str)
    for column in requested_fields:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _attach_layer_metadata(frame: pd.DataFrame, specs: list[dict[str, str]]) -> pd.DataFrame:
    metadata = pd.DataFrame(specs).drop_duplicates("order_book_id", keep="last")
    result = frame.merge(metadata, on="order_book_id", how="left")
    result["asset_type"] = result["asset_type"].fillna("unknown").astype(str)
    result["universe_layer"] = result["universe_layer"].fillna(result["asset_type"]).astype(str)
    result["exposure_group"] = result["exposure_group"].fillna(result["universe_layer"]).astype(str)
    return result


def _attach_listing_dates(frame: pd.DataFrame, data_facade, *, end_date) -> pd.DataFrame:
    result = frame.copy()
    result["listed_date"] = pd.NaT
    method = getattr(data_facade, "all_instruments", None)
    if method is None:
        return result
    try:
        instruments = method(type="ETF", date=end_date)
    except TypeError:
        try:
            instruments = method(type="ETF")
        except Exception:
            return result
    except Exception:
        return result
    payload = instruments if isinstance(instruments, pd.DataFrame) else pd.DataFrame(instruments or [])
    if payload.empty or "order_book_id" not in payload.columns:
        return result
    listing_column = next(
        (column for column in ("listed_date", "list_date", "listing_date", "ipo_date") if column in payload.columns),
        None,
    )
    if listing_column is None:
        return result
    listing_map = payload.dropna(subset=["order_book_id"]).drop_duplicates("order_book_id", keep="last")
    result["listed_date"] = result["order_book_id"].map(
        listing_map.set_index("order_book_id")[listing_column]
    )
    result["listed_date"] = pd.to_datetime(result["listed_date"], errors="coerce")
    return result


def _attach_provider_industry(frame: pd.DataFrame, data_facade, *, end_date) -> pd.DataFrame:
    result = frame.copy()
    result["industry"] = "Unknown"
    method = getattr(data_facade, "get_industry", None)
    order_book_ids = sorted(result["order_book_id"].dropna().astype(str).unique())
    payload = None
    if method is not None:
        try:
            payload = method(order_book_ids, date=end_date)
        except TypeError:
            try:
                payload = method(order_book_ids)
            except Exception:
                payload = None
        except Exception:
            payload = None
    industry_map = resolve_ax1_etf_industry_map(order_book_ids, _industry_map_from_payload(payload))
    if industry_map:
        result["industry"] = result["order_book_id"].astype(str).map(industry_map).fillna("Unknown")
    return result


def _industry_map_from_payload(payload: Any) -> dict[str, str]:
    frame = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload or [])
    if frame.empty:
        return {}
    if isinstance(frame.index, pd.Index) and "order_book_id" not in frame.columns:
        frame = frame.reset_index()
    if "order_book_id" not in frame.columns:
        return {}
    industry_column = next(
        (
            column
            for column in ("industry", "industry_name", "sector", "sector_name", "sector_code_name", "industry_code")
            if column in frame.columns
        ),
        None,
    )
    if industry_column is None:
        value_columns = [column for column in frame.columns if column != "order_book_id"]
        industry_column = value_columns[0] if value_columns else None
    if industry_column is None:
        return {}
    payload = frame.dropna(subset=["order_book_id"]).drop_duplicates("order_book_id", keep="last")
    return {
        str(row["order_book_id"]): str(row[industry_column])
        for row in payload.to_dict("records")
        if pd.notna(row.get(industry_column))
    }


def _attach_status_flags(frame: pd.DataFrame, data_facade, *, start_date, end_date) -> pd.DataFrame:
    result = frame.copy()
    result["is_st"] = False
    result["is_suspended"] = False
    order_book_ids = sorted(result["order_book_id"].dropna().astype(str).unique())
    st_map = _status_map_from_method(
        getattr(data_facade, "is_st_stock", None),
        order_book_ids,
        start_date,
        end_date,
        value_columns=("is_st", "is_st_stock", "st"),
    )
    suspended_map = _status_map_from_method(
        getattr(data_facade, "is_suspended", None),
        order_book_ids,
        start_date,
        end_date,
        value_columns=("is_suspended", "suspended", "halted"),
    )
    if st_map:
        result["is_st"] = [
            bool(st_map.get((str(row.order_book_id), pd.Timestamp(row.date).normalize()), False))
            for row in result.itertuples(index=False)
        ]
    if suspended_map:
        result["is_suspended"] = [
            bool(suspended_map.get((str(row.order_book_id), pd.Timestamp(row.date).normalize()), False))
            for row in result.itertuples(index=False)
        ]
    return result


def _status_map_from_method(method, order_book_ids: list[str], start_date, end_date, *, value_columns: tuple[str, ...]) -> dict:
    if method is None:
        return {}
    try:
        payload = method(order_book_ids, start_date, end_date)
    except TypeError:
        try:
            payload = method(order_book_ids, start_date=start_date, end_date=end_date)
        except Exception:
            return {}
    except Exception:
        return {}
    frame = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload or [])
    if frame.empty:
        return {}
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    elif "date" not in frame.columns:
        frame = frame.reset_index()
    if "datetime" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"datetime": "date"})
    if "index" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"index": "date"})
    if "order_book_id" not in frame.columns:
        value_like = [column for column in frame.columns if column not in {"date", *value_columns}]
        if value_like and all(str(column) in set(order_book_ids) for column in value_like):
            value_name = next((column for column in value_columns if column in frame.columns), "value")
            frame = frame.melt(id_vars=["date"], var_name="order_book_id", value_name=value_name)
    if "date" not in frame.columns or "order_book_id" not in frame.columns:
        return {}
    value_column = next((column for column in value_columns if column in frame.columns), None)
    if value_column is None:
        value_candidates = [column for column in frame.columns if column not in {"date", "order_book_id"}]
        value_column = value_candidates[0] if value_candidates else None
    if value_column is None:
        return {}
    frame = frame.dropna(subset=["date", "order_book_id"]).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"])
    return {
        (str(row["order_book_id"]), pd.Timestamp(row["date"]).normalize()): bool(row[value_column])
        for row in frame.to_dict("records")
    }


def _attach_adjusted_price_contract(frame: pd.DataFrame, *, adjust_type: str) -> pd.DataFrame:
    result = frame.copy()
    if "adjusted_close" not in result.columns:
        if str(adjust_type).lower() != "pre":
            raise ValueError("AX1 training data requires adjusted_close unless adjust_type='pre'")
        result["adjusted_close"] = pd.to_numeric(result["close"], errors="coerce")
        result["price_adjustment_status"] = "pre_adjusted_via_data_facade"
    else:
        result["adjusted_close"] = pd.to_numeric(result["adjusted_close"], errors="coerce")
        if "price_adjustment_status" not in result.columns:
            result["price_adjustment_status"] = "adjusted_price_column"
    return result
