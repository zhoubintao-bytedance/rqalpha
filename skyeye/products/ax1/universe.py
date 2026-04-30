"""AX1 dynamic universe builder."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd

from skyeye.products.ax1.layers import LayerRegistry


class DynamicUniverseBuilder:
    """构建 AX1 每个 as-of date 可用证券池。"""

    def build(
        self,
        raw_df: pd.DataFrame,
        as_of_date=None,
        config: Mapping[str, Any] | None = None,
        data_provider=None,
    ) -> list[str]:
        metadata = self.build_with_metadata(
            raw_df,
            as_of_date=as_of_date,
            config=config,
            data_provider=data_provider,
        )
        if metadata.empty:
            return []
        return sorted(metadata["order_book_id"].drop_duplicates().astype(str).tolist())

    def build_with_metadata(
        self,
        raw_df: pd.DataFrame,
        as_of_date=None,
        config: Mapping[str, Any] | None = None,
        data_provider=None,
        purpose: str = "research",
    ) -> pd.DataFrame:
        cfg = _universe_config(config or {})
        if raw_df is None or raw_df.empty:
            return _annotate_metadata_audit(
                _empty_metadata(),
                _pit_audit(
                    as_of_date=as_of_date,
                    has_date_column=False,
                    source_status={"raw_frame": "empty"},
                    hard_blocks=[],
                    warnings=[],
                ),
            )
        if "order_book_id" not in raw_df.columns:
            raise ValueError("raw_df must contain order_book_id")

        frame = raw_df.copy()
        frame = frame[frame["order_book_id"].notna()]
        frame["order_book_id"] = frame["order_book_id"].astype(str).str.strip()
        frame = frame[frame["order_book_id"] != ""]

        cutoff: pd.Timestamp | None = None
        if as_of_date is not None:
            if "date" not in frame.columns:
                raise ValueError("raw_df must contain date when as_of_date is provided")
            cutoff = pd.Timestamp(as_of_date)
            frame = frame[pd.to_datetime(frame["date"]) <= cutoff]
        elif "date" in frame.columns and not frame.empty:
            cutoff = pd.to_datetime(frame["date"], errors="coerce").max()

        # PIT验证：确保universe只包含as_of_date实际存在的证券（防止幸存者偏差）
        pit_validated_ids: set[str] | None = None
        if cfg.get("validate_pit_universe", True) and cutoff is not None and data_provider is not None:
            pit_validated_ids = _get_pit_validated_ids(data_provider, cutoff, frame["order_book_id"].drop_duplicates().astype(str).tolist())
            if pit_validated_ids is not None:
                frame = frame[frame["order_book_id"].astype(str).isin(pit_validated_ids)]

        audit = _build_pit_audit(frame, as_of_date=as_of_date, cutoff=cutoff, config=cfg, data_provider=data_provider, pit_validated_ids=pit_validated_ids, purpose=purpose)
        registry = LayerRegistry.from_universe_config(cfg)
        min_listing_days = cfg.get("min_listing_days")
        if min_listing_days is not None and cutoff is not None:
            frame = self._filter_listing_age(
                frame,
                cutoff=cutoff,
                min_listing_days=int(min_listing_days),
                data_provider=data_provider,
            )

        if cfg.get("exclude_st") and cutoff is not None:
            frame = self._filter_st_status(frame, cutoff=cutoff, data_provider=data_provider)

        if cfg.get("exclude_suspended") and cutoff is not None:
            frame = self._filter_suspended(frame, cutoff=cutoff, data_provider=data_provider)

        min_aum = cfg.get("min_aum")
        if min_aum is not None and cutoff is not None:
            frame = self._filter_aum(
                frame,
                cutoff=cutoff,
                min_aum=float(min_aum),
                data_provider=data_provider,
            )

        min_daily_dollar_volume = cfg.get("min_daily_dollar_volume")
        if min_daily_dollar_volume is not None and cutoff is not None:
            frame = self._filter_liquidity(
                frame,
                cutoff=cutoff,
                min_daily_dollar_volume=float(min_daily_dollar_volume),
                lookback_days=int(cfg.get("liquidity_lookback_days", 20)),
                data_provider=data_provider,
            )

        metadata = _metadata_from_frame(frame, cutoff=cutoff, registry=registry)
        metadata = _apply_layer_limits(metadata, registry)
        if metadata.empty:
            return _annotate_metadata_audit(_empty_metadata(), audit)
        metadata = metadata.sort_values("order_book_id").reset_index(drop=True)
        return _annotate_metadata_audit(metadata, audit)

    def _filter_listing_age(
        self,
        raw_df: pd.DataFrame,
        *,
        cutoff: pd.Timestamp,
        min_listing_days: int,
        data_provider=None,
    ) -> pd.DataFrame:
        listing_map = _listing_date_map_from_frame(raw_df)
        if not listing_map and data_provider is not None:
            listing_map = _listing_date_map_from_provider(data_provider, raw_df["order_book_id"].drop_duplicates(), cutoff)
        if not listing_map:
            return raw_df
        min_date = cutoff.normalize() - pd.Timedelta(days=int(min_listing_days))
        keep_ids = {
            order_book_id
            for order_book_id in raw_df["order_book_id"].drop_duplicates().astype(str)
            if order_book_id not in listing_map or pd.Timestamp(listing_map[order_book_id]).normalize() <= min_date
        }
        return raw_df[raw_df["order_book_id"].astype(str).isin(keep_ids)].copy()

    def _filter_st_status(self, raw_df: pd.DataFrame, *, cutoff: pd.Timestamp, data_provider=None) -> pd.DataFrame:
        flagged_ids = _flagged_ids_from_frame(raw_df, cutoff, ("is_st", "is_st_stock", "st"))
        if not flagged_ids and data_provider is not None:
            flagged_ids = _flagged_ids_from_provider(
                data_provider,
                "is_st_stock",
                raw_df["order_book_id"].drop_duplicates().astype(str).tolist(),
                cutoff,
                ("is_st", "is_st_stock", "st"),
            )
        if not flagged_ids:
            return raw_df
        return raw_df[~raw_df["order_book_id"].astype(str).isin(flagged_ids)].copy()

    def _filter_suspended(self, raw_df: pd.DataFrame, *, cutoff: pd.Timestamp, data_provider=None) -> pd.DataFrame:
        flagged_ids = _flagged_ids_from_frame(raw_df, cutoff, ("is_suspended", "suspended", "halted"))
        if not flagged_ids and data_provider is not None:
            flagged_ids = _flagged_ids_from_provider(
                data_provider,
                "is_suspended",
                raw_df["order_book_id"].drop_duplicates().astype(str).tolist(),
                cutoff,
                ("is_suspended", "suspended", "halted"),
            )
        if not flagged_ids:
            return raw_df
        return raw_df[~raw_df["order_book_id"].astype(str).isin(flagged_ids)].copy()

    def _filter_aum(
        self,
        raw_df: pd.DataFrame,
        *,
        cutoff: pd.Timestamp,
        min_aum: float,
        data_provider=None,
    ) -> pd.DataFrame:
        """Filter out ETFs with AUM below the minimum threshold.

        Args:
            raw_df: DataFrame with order_book_id and potentially AUM data
            cutoff: The as-of date for PIT filtering
            min_aum: Minimum AUM in local currency (e.g., 50_000_000 for 5000万)
            data_provider: Optional data provider to fetch AUM data

        Returns:
            DataFrame filtered by AUM threshold
        """
        aum_map = _aum_map_from_frame(raw_df, cutoff)
        if not aum_map and data_provider is not None:
            aum_map = _aum_map_from_provider(
                data_provider,
                raw_df["order_book_id"].drop_duplicates().astype(str).tolist(),
                cutoff,
            )
        if not aum_map:
            # No AUM data available - return unchanged with warning in audit
            return raw_df

        keep_ids = {
            order_book_id
            for order_book_id in raw_df["order_book_id"].drop_duplicates().astype(str)
            if order_book_id not in aum_map or aum_map[order_book_id] >= min_aum
        }
        return raw_df[raw_df["order_book_id"].astype(str).isin(keep_ids)].copy()

    def _filter_liquidity(
        self,
        raw_df: pd.DataFrame,
        *,
        cutoff: pd.Timestamp,
        min_daily_dollar_volume: float,
        lookback_days: int,
        data_provider=None,
    ) -> pd.DataFrame:
        """Filter out ETFs with average daily dollar volume below threshold.

        Args:
            raw_df: DataFrame with OHLCV data
            cutoff: The as-of date for PIT filtering
            min_daily_dollar_volume: Minimum avg daily dollar volume (e.g., 10_000_000 for 1000万)
            lookback_days: Number of days to average over (default 20)
            data_provider: Optional data provider (unused, kept for API consistency)

        Returns:
            DataFrame filtered by liquidity threshold
        """
        dollar_volume_map = _dollar_volume_map_from_frame(raw_df, cutoff, lookback_days)
        if not dollar_volume_map:
            # No volume data available - return unchanged
            return raw_df

        keep_ids = {
            order_book_id
            for order_book_id, avg_dollar_volume in dollar_volume_map.items()
            if avg_dollar_volume >= min_daily_dollar_volume
        }
        return raw_df[raw_df["order_book_id"].astype(str).isin(keep_ids)].copy()


def _latest_rows(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    if "date" not in frame.columns:
        return frame.drop_duplicates("order_book_id", keep="last")
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working[working["date"].notna() & (working["date"] <= cutoff)]
    if working.empty:
        return working
    return working.sort_values(["order_book_id", "date"]).drop_duplicates("order_book_id", keep="last")


def _listing_date_map_from_frame(frame: pd.DataFrame) -> dict[str, pd.Timestamp]:
    listing_column = next(
        (
            column
            for column in ("listed_date", "list_date", "listing_date", "ipo_date")
            if column in frame.columns
        ),
        None,
    )
    if listing_column is None:
        return {}
    working = frame.dropna(subset=["order_book_id", listing_column]).copy()
    if working.empty:
        return {}
    working["order_book_id"] = working["order_book_id"].astype(str)
    working[listing_column] = pd.to_datetime(working[listing_column], errors="coerce")
    working = working.dropna(subset=[listing_column])
    if working.empty:
        return {}
    return working.groupby("order_book_id")[listing_column].min().to_dict()


def _listing_date_map_from_provider(data_provider, order_book_ids: Iterable[str], cutoff: pd.Timestamp) -> dict[str, pd.Timestamp]:
    provider_method = getattr(data_provider, "all_instruments", None) or getattr(data_provider, "get_instruments", None)
    if provider_method is None:
        return {}
    try:
        instruments = provider_method(type="CS", date=cutoff)
    except TypeError:
        try:
            instruments = provider_method("CS", cutoff)
        except Exception:
            return {}
    except Exception:
        return {}
    frame = pd.DataFrame(instruments) if instruments is not None else pd.DataFrame()
    if frame.empty or "order_book_id" not in frame.columns:
        return {}
    allowed_ids = {str(order_book_id) for order_book_id in order_book_ids}
    frame = frame[frame["order_book_id"].astype(str).isin(allowed_ids)]
    return _listing_date_map_from_frame(frame)


def _flagged_ids_from_frame(frame: pd.DataFrame, cutoff: pd.Timestamp, status_columns: tuple[str, ...]) -> set[str]:
    status_column = next((column for column in status_columns if column in frame.columns), None)
    if status_column is None:
        return set()
    latest = _latest_rows(frame, cutoff)
    if latest.empty:
        return set()
    flags = latest[status_column].map(_to_bool)
    return set(latest.loc[flags, "order_book_id"].astype(str))


def _flagged_ids_from_provider(
    data_provider,
    method_name: str,
    order_book_ids: list[str],
    cutoff: pd.Timestamp,
    status_columns: tuple[str, ...],
) -> set[str]:
    provider_method = getattr(data_provider, method_name, None)
    if provider_method is None or not order_book_ids:
        return set()
    try:
        payload = provider_method(order_book_ids, cutoff, cutoff)
    except TypeError:
        try:
            payload = provider_method(order_book_ids, start_date=cutoff, end_date=cutoff)
        except TypeError:
            try:
                payload = {order_book_id: provider_method(order_book_id, cutoff) for order_book_id in order_book_ids}
            except Exception:
                return set()
        except Exception:
            return set()
    except Exception:
        return set()
    return _flagged_ids_from_status_payload(payload, order_book_ids, cutoff, status_columns)


def _flagged_ids_from_status_payload(payload, order_book_ids: list[str], cutoff: pd.Timestamp, status_columns: tuple[str, ...]) -> set[str]:
    if payload is None:
        return set()
    if isinstance(payload, dict):
        return {str(order_book_id) for order_book_id, value in payload.items() if _to_bool(value)}
    if isinstance(payload, pd.Series):
        return {str(order_book_id) for order_book_id, value in payload.items() if _to_bool(value)}
    frame = pd.DataFrame(payload)
    if frame.empty:
        return set()
    if isinstance(frame.index, pd.DatetimeIndex):
        matched = frame.loc[frame.index <= cutoff]
        if matched.empty:
            return set()
        row = matched.iloc[-1]
        return {str(order_book_id) for order_book_id in order_book_ids if order_book_id in row.index and _to_bool(row[order_book_id])}
    if "order_book_id" not in frame.columns and len(order_book_ids) == 1:
        frame.insert(0, "order_book_id", order_book_ids[0])
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[frame["date"].notna() & (frame["date"] <= cutoff)]
    if "order_book_id" not in frame.columns:
        value_columns = [column for column in frame.columns if str(column) in set(order_book_ids)]
        flagged = set()
        for column in value_columns:
            values = frame[column].dropna()
            if not values.empty and _to_bool(values.iloc[-1]):
                flagged.add(str(column))
        return flagged
    value_column = next((column for column in status_columns if column in frame.columns), None)
    if value_column is None:
        candidates = [column for column in frame.columns if column not in {"date", "order_book_id"}]
        value_column = candidates[0] if candidates else None
    if value_column is None:
        return set()
    latest = _latest_rows(frame, cutoff)
    flags = latest[value_column].map(_to_bool)
    return set(latest.loc[flags, "order_book_id"].astype(str))


def _to_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "st", "suspended", "halted"}
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _aum_map_from_frame(frame: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float]:
    """Extract AUM (fund size) data from frame.

    Supports columns: aum, fund_size, fund_aum, total_assets
    Returns PIT value as of cutoff date.
    """
    aum_column = next(
        (column for column in ("aum", "fund_size", "fund_aum", "total_assets") if column in frame.columns),
        None,
    )
    if aum_column is None:
        return {}
    working = frame.dropna(subset=["order_book_id", aum_column]).copy()
    if working.empty:
        return {}
    working["order_book_id"] = working["order_book_id"].astype(str)
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        working = working[working["date"].notna() & (working["date"] <= cutoff)]
        if working.empty:
            return {}
        working = working.sort_values(["order_book_id", "date"]).drop_duplicates("order_book_id", keep="last")
    working[aum_column] = pd.to_numeric(working[aum_column], errors="coerce")
    working = working.dropna(subset=[aum_column])
    if working.empty:
        return {}
    return working.set_index("order_book_id")[aum_column].to_dict()


def _aum_map_from_provider(data_provider, order_book_ids: list[str], cutoff: pd.Timestamp) -> dict[str, float]:
    """Fetch AUM data from data provider.

    Tries common method names: get_fund_info, get_etf_info, get_aum, etc.
    """
    provider_method = (
        getattr(data_provider, "get_fund_info", None)
        or getattr(data_provider, "get_etf_info", None)
        or getattr(data_provider, "get_aum", None)
    )
    if provider_method is None:
        return {}
    try:
        # Try calling with list of IDs and date
        payload = provider_method(order_book_ids, cutoff)
    except TypeError:
        try:
            # Try calling with date and list
            payload = provider_method(cutoff, order_book_ids)
        except Exception:
            return {}
    except Exception:
        return {}

    if payload is None:
        return {}
    if isinstance(payload, dict):
        return {str(k): float(v) for k, v in payload.items() if pd.notna(v)}
    if isinstance(payload, pd.DataFrame):
        if "order_book_id" not in payload.columns:
            return {}
        aum_column = next(
            (column for column in ("aum", "fund_size", "fund_aum", "total_assets") if column in payload.columns),
            None,
        )
        if aum_column is None:
            return {}
        payload = payload.dropna(subset=["order_book_id", aum_column])
        if payload.empty:
            return {}
        payload["order_book_id"] = payload["order_book_id"].astype(str)
        payload[aum_column] = pd.to_numeric(payload[aum_column], errors="coerce")
        payload = payload.dropna(subset=[aum_column])
        if payload.empty:
            return {}
        return payload.set_index("order_book_id")[aum_column].to_dict()
    return {}


def _dollar_volume_map_from_frame(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    lookback_days: int,
) -> dict[str, float]:
    """Calculate average daily dollar volume for each ETF over lookback period.

    Dollar volume = close * volume
    Returns average over the last N trading days.
    """
    required = ["order_book_id", "date", "close", "volume"]
    if not all(column in frame.columns for column in required):
        return {}

    working = frame[required].copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working["order_book_id"] = working["order_book_id"].astype(str)
    working = working.dropna(subset=["date", "order_book_id", "close", "volume"])

    if working.empty:
        return {}

    # Filter to lookback window
    lookback_start = cutoff.normalize() - pd.Timedelta(days=lookback_days * 2)  # Extra buffer for non-trading days
    working = working[(working["date"] >= lookback_start) & (working["date"] <= cutoff)]

    if working.empty:
        return {}

    # Calculate dollar volume
    working["close"] = pd.to_numeric(working["close"], errors="coerce")
    working["volume"] = pd.to_numeric(working["volume"], errors="coerce")
    working["dollar_volume"] = working["close"] * working["volume"]
    working = working.dropna(subset=["dollar_volume"])

    if working.empty:
        return {}

    # For each ETF, take the last N trading days and average
    result = {}
    for order_book_id, group in working.groupby("order_book_id"):
        group = group.sort_values("date").tail(lookback_days)
        if len(group) >= max(1, lookback_days // 2):  # Require at least half the lookback period
            result[order_book_id] = group["dollar_volume"].mean()

    return result


def _get_pit_validated_ids(
    data_provider,
    cutoff: pd.Timestamp,
    order_book_ids: list[str],
) -> set[str] | None:
    """获取历史日期实际存在的证券列表，防止幸存者偏差。

    通过 data_provider.all_instruments(date=cutoff) 获取该日期实际存在的证券，
    确保回测中包含当时存在但后来退市的证券。

    Args:
        data_provider: 数据提供者，需支持 all_instruments 方法
        cutoff: PIT日期
        order_book_ids: 需要验证的证券列表

    Returns:
        该日期实际存在的证券ID集合，如果无法获取则返回None
    """
    provider_method = getattr(data_provider, "all_instruments", None) or getattr(data_provider, "get_instruments", None)
    if provider_method is None:
        return None

    try:
        # 尝试获取该日期的所有ETF
        instruments = provider_method(type="ETF", date=cutoff)
    except TypeError:
        try:
            instruments = provider_method("ETF", cutoff)
        except Exception:
            try:
                # 尝试获取所有类型
                instruments = provider_method(date=cutoff)
            except Exception:
                return None
    except Exception:
        return None

    if instruments is None:
        return None

    frame = pd.DataFrame(instruments) if not isinstance(instruments, pd.DataFrame) else instruments
    if frame.empty or "order_book_id" not in frame.columns:
        return None

    # 返回该日期实际存在的证券ID
    return set(frame["order_book_id"].astype(str).tolist())


def _build_pit_audit(
    frame: pd.DataFrame,
    *,
    as_of_date,
    cutoff: pd.Timestamp | None,
    config: Mapping[str, Any],
    data_provider=None,
    pit_validated_ids: set[str] | None = None,
    purpose: str = "research",
) -> dict[str, Any]:
    has_date_column = "date" in frame.columns
    hard_blocks: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    source_status: dict[str, str] = {}
    promotable = str(purpose) in {"promotable_training", "promotion", "live_training"}

    # PIT universe验证状态
    if pit_validated_ids is not None:
        source_status["pit_universe"] = "validated_via_provider"
    elif config.get("validate_pit_universe", True) and cutoff is not None:
        if data_provider is None:
            source_status["pit_universe"] = "validation_skipped_no_provider"
            issue = _audit_issue(
                "pit_universe_no_provider",
                "PIT universe validation skipped: no data_provider provided (potential survivorship bias)",
            )
            if promotable:
                hard_blocks.append(issue)
            else:
                warnings.append(issue)
        else:
            source_status["pit_universe"] = "validation_failed"
            warnings.append(
                _audit_issue(
                    "pit_universe_validation_failed",
                    "PIT universe validation failed: could not fetch historical instruments (potential survivorship bias)",
                )
            )
    else:
        source_status["pit_universe"] = "disabled"

    if as_of_date is not None and has_date_column:
        source_status["raw_frame"] = "point_in_time"
    elif has_date_column and cutoff is not None:
        source_status["raw_frame"] = "inferred_latest_from_frame"
        warnings.append(
            _audit_issue(
                "universe_as_of_date_inferred",
                "universe as-of date was inferred from raw frame max date",
            )
        )
    else:
        source_status["raw_frame"] = "latest_snapshot"
        hard_blocks.append(
            _audit_issue(
                "universe_latest_snapshot_not_point_in_time",
                "universe requires date column and explicit as-of date for promotable training",
            )
        )

    source_status["listing"] = _audit_filter_source(
        config,
        required_key="min_listing_days",
        frame=frame,
        frame_columns=("listed_date", "list_date", "listing_date", "ipo_date"),
        data_provider=data_provider,
    )
    if source_status["listing"] == "not_available":
        issue = _audit_issue(
            "listing_age_filter_no_data_source",
            "min_listing_days is enabled but no listing date columns found and no data_provider provided; "
            "listing-age filter degrades to pass-through (potential survivorship/quality issue)",
        )
        if bool(config.get("listing_age_requires_data_source", False)):
            hard_blocks.append(issue)
        else:
            warnings.append(issue)
    source_status["st_status"] = _audit_filter_source(
        config,
        required_key="exclude_st",
        frame=frame,
        frame_columns=("is_st", "is_st_stock", "st"),
        data_provider=data_provider,
    )
    source_status["suspension"] = _audit_filter_source(
        config,
        required_key="exclude_suspended",
        frame=frame,
        frame_columns=("is_suspended", "suspended", "halted"),
        data_provider=data_provider,
    )
    source_status["aum"] = _audit_filter_source(
        config,
        required_key="min_aum",
        frame=frame,
        frame_columns=("aum", "fund_size", "fund_aum", "total_assets"),
        data_provider=data_provider,
    )
    if source_status["aum"] == "not_available":
        issue = _audit_issue(
            "aum_filter_no_data_source",
            "min_aum is enabled but no AUM columns found and no data_provider provided; "
            "AUM filter degrades to pass-through (small/illiquid ETFs may enter universe)",
        )
        if bool(config.get("aum_requires_data_source", False)):
            hard_blocks.append(issue)
        else:
            warnings.append(issue)
    source_status["liquidity"] = _audit_liquidity_source(
        config,
        frame=frame,
        data_provider=data_provider,
    )
    if source_status["liquidity"] == "not_available":
        issue = _audit_issue(
            "liquidity_filter_no_data_source",
            "min_daily_dollar_volume is enabled but no close/volume columns found; "
            "liquidity filter degrades to pass-through (illiquid ETFs may enter universe)",
        )
        if bool(config.get("liquidity_requires_data_source", False)):
            hard_blocks.append(issue)
        else:
            warnings.append(issue)
    return _pit_audit(
        as_of_date=cutoff,
        has_date_column=has_date_column,
        source_status=source_status,
        hard_blocks=hard_blocks,
        warnings=warnings,
    )


def _audit_filter_source(
    config: Mapping[str, Any],
    *,
    required_key: str,
    frame: pd.DataFrame,
    frame_columns: tuple[str, ...],
    data_provider=None,
) -> str:
    required_value = config.get(required_key)
    if required_key != "min_listing_days" and not bool(required_value):
        return "not_required"
    if required_key == "min_listing_days" and required_value is None:
        return "not_required"
    if any(column in frame.columns for column in frame_columns):
        return "frame_point_in_time"
    if data_provider is not None:
        return "provider_as_of_requested"
    return "not_available"


def _audit_liquidity_source(
    config: Mapping[str, Any],
    *,
    frame: pd.DataFrame,
    data_provider=None,
) -> str:
    """Audit liquidity filter data source availability.

    Liquidity is calculated from close * volume, so it needs
    date, order_book_id, close, and volume columns.
    """
    min_daily_dollar_volume = config.get("min_daily_dollar_volume")
    if min_daily_dollar_volume is None:
        return "not_required"

    required = ["date", "order_book_id", "close", "volume"]
    if all(column in frame.columns for column in required):
        return "frame_point_in_time"

    # Liquidity cannot be fetched from provider - must be in frame
    return "not_available"


def _pit_audit(
    *,
    as_of_date,
    has_date_column: bool,
    source_status: dict[str, str],
    hard_blocks: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    as_of_ts = pd.Timestamp(as_of_date) if as_of_date is not None and not pd.isna(as_of_date) else None
    return {
        "schema_version": 1,
        "passed": not hard_blocks,
        "as_of_date": str(as_of_ts.date()) if as_of_ts is not None else None,
        "has_date_column": bool(has_date_column),
        "source_status": dict(source_status),
        "hard_block_count": int(len(hard_blocks)),
        "warning_count": int(len(warnings)),
        "hard_blocks": list(hard_blocks),
        "warnings": list(warnings),
    }


def _audit_issue(reason_code: str, message: str) -> dict[str, str]:
    return {"reason_code": str(reason_code), "message": str(message)}


def _annotate_metadata_audit(metadata: pd.DataFrame, audit: dict[str, Any]) -> pd.DataFrame:
    result = metadata.copy()
    result.attrs["pit_audit"] = dict(audit)
    if "universe_pit_status" not in result.columns:
        result["universe_pit_status"] = "point_in_time" if audit.get("passed") else "latest_snapshot"
    if "universe_as_of_date" not in result.columns:
        result["universe_as_of_date"] = audit.get("as_of_date")
    return result


def _empty_metadata() -> pd.DataFrame:
    return pd.DataFrame(columns=["order_book_id", "asset_type", "universe_layer"])


def _universe_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "layers" in config:
        return config
    universe = config.get("universe")
    if isinstance(universe, Mapping):
        return universe
    return config


def _metadata_from_frame(frame: pd.DataFrame, *, cutoff: pd.Timestamp | None, registry: LayerRegistry) -> pd.DataFrame:
    if frame.empty:
        return _empty_metadata()
    latest = _latest_rows(frame, cutoff) if cutoff is not None else frame.drop_duplicates("order_book_id", keep="last")
    if latest.empty:
        return _empty_metadata()
    enabled_layers = set(registry.enabled_layer_names())
    rows = []
    for _, row in latest.iterrows():
        order_book_id = str(row["order_book_id"])
        explicit_layer = registry.layer_name_for_id(order_book_id)
        asset_type = _normalize_asset_type(_first_present(row, ("asset_type", "instrument_type", "type", "fund_type")))
        if explicit_layer is not None:
            layer = explicit_layer
            asset_type = _normalize_asset_type(registry.spec(layer).asset_type)
        else:
            row_layer = _normalize_configured_layer(
                _first_present(row, ("universe_layer", "asset_category", "category")),
                enabled_layers,
            )
            if row_layer is not None:
                layer = row_layer
                asset_type = asset_type or _normalize_asset_type(registry.spec(layer).asset_type)
            elif asset_type is not None:
                layer = registry.default_layer_for_asset_type(asset_type)
            else:
                layer = None
        if layer is None or layer not in enabled_layers or asset_type is None:
            continue
        rows.append(
            {
                "order_book_id": order_book_id,
                "asset_type": asset_type,
                "universe_layer": layer,
            }
        )
    if not rows:
        return _empty_metadata()
    return pd.DataFrame(rows).drop_duplicates("order_book_id", keep="last")


def _apply_layer_limits(metadata: pd.DataFrame, registry: LayerRegistry) -> pd.DataFrame:
    if metadata.empty:
        return metadata
    frames = []
    for layer_name in registry.enabled_layer_names():
        spec = registry.spec(layer_name)
        layer_frame = metadata[metadata["universe_layer"].eq(layer_name)].copy()
        if layer_frame.empty:
            continue
        if spec.max_count is not None:
            layer_frame = layer_frame.sort_values("order_book_id").head(spec.max_count)
        frames.append(layer_frame)
    if not frames:
        return _empty_metadata()
    return pd.concat(frames, ignore_index=True)


def _first_present(row: pd.Series, columns: tuple[str, ...]):
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            return row[column]
    return None


def _normalize_asset_type(value) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"etf", "fund", "index_fund"}:
        return "etf"
    if normalized in {"stock", "cs", "equity", "common_stock"}:
        return "stock"
    return normalized or None


def _normalize_configured_layer(value, enabled_layers: set[str]) -> str | None:
    if value is not None:
        normalized = str(value).strip().lower()
        if normalized in enabled_layers:
            return normalized
    return None
