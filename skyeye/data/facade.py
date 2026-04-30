from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import pandas as pd

from skyeye.data.bundle_reader import BundleDataReader
from skyeye.data.cache_store import LocalDataCacheStore

DEFAULT_BAR_FIELDS = ("open", "high", "low", "close", "volume", "total_turnover", "prev_close")


class DataFacade:
    """统一串联 bundle、SQLite 缓存与 rqdatac 的数据访问门面。"""

    def __init__(self):
        self.provider = None
        self.bundle_reader = BundleDataReader()
        source = os.environ.get("SKYEYE_DATA_SOURCE", "rqdatac").strip().lower()
        self.source = source
        self.cache_store = None if source == "bundle" else LocalDataCacheStore()
        if source not in ("bundle", "local"):
            try:
                from skyeye.data.provider import RQDataProvider

                self.provider = RQDataProvider()
            except Exception:
                self.provider = None

    @staticmethod
    def _raise_if_quota_exceeded(exc: Exception) -> None:
        """遇到流量配额耗尽时直接抛错，禁止静默降级。"""
        try:
            from rqdatac.share.errors import QuotaExceeded
        except Exception:
            QuotaExceeded = tuple()

        if isinstance(exc, QuotaExceeded):
            raise exc

    @staticmethod
    def _normalize_date_key(value) -> str:
        """统一把日期值转成 `YYYY-MM-DD` 文本。"""
        return pd.Timestamp(value).normalize().strftime("%Y-%m-%d")

    @staticmethod
    def _as_order_book_ids(order_book_ids: Union[str, Sequence[str]]) -> list[str]:
        """把单证券 / 多证券入参统一转成列表。"""
        if isinstance(order_book_ids, str):
            return [order_book_ids]
        return [str(item) for item in order_book_ids]

    @staticmethod
    def _as_list(values: Union[str, Sequence[str], None], fallback: Sequence[str]) -> list[str]:
        """把字段入参统一转成列表。"""
        if values is None:
            return list(fallback)
        if isinstance(values, str):
            return [values]
        return [str(item) for item in values]

    @staticmethod
    def _merge_payload_maps(primary: dict[str, dict], secondary: dict[str, dict]) -> dict[str, dict]:
        """按字段合并两份 payload，`primary` 冲突时优先。"""
        merged = {date_key: dict(payload) for date_key, payload in secondary.items()}
        for date_key, payload in primary.items():
            base = dict(merged.get(date_key, {}))
            base.update(payload)
            merged[date_key] = base
        return merged

    @classmethod
    def _payload_map_from_frame(
        cls,
        frame: Optional[pd.DataFrame],
        *,
        order_book_id: Optional[str] = None,
    ) -> dict[str, dict]:
        """把 DataFrame 统一转成 `date -> payload` 的中间结构。"""
        if frame is None or frame.empty:
            return {}
        working = frame.copy()
        if isinstance(working.index, pd.MultiIndex):
            working = working.reset_index()
        elif "date" not in working.columns and isinstance(working.index, pd.DatetimeIndex):
            working = working.reset_index()
        elif "date" not in working.columns and "datetime" not in working.columns:
            working = working.reset_index()

        if "date" not in working.columns:
            if "datetime" in working.columns:
                working = working.rename(columns={"datetime": "date"})
            else:
                for column in working.columns:
                    if pd.api.types.is_datetime64_any_dtype(working[column]):
                        working = working.rename(columns={column: "date"})
                        break

        if "order_book_id" not in working.columns and order_book_id is not None:
            insert_at = 1 if "date" in working.columns else 0
            working.insert(insert_at, "order_book_id", order_book_id)

        if "date" not in working.columns:
            return {}

        working["date"] = pd.to_datetime(working["date"]).dt.normalize()
        result = {}
        for row in working.to_dict("records"):
            date_key = cls._normalize_date_key(row["date"])
            payload = {}
            for key, value in row.items():
                if key in {"date", "order_book_id"}:
                    continue
                payload[key] = value.item() if hasattr(value, "item") else value
                if pd.isna(payload[key]):
                    payload[key] = None
            result[date_key] = payload
        return result

    @staticmethod
    def _group_missing_ranges(ordered_date_keys: list[str], missing_date_keys: set[str]) -> list[tuple[str, str]]:
        """把离散缺失日期按原始顺序压成连续区间。"""
        ranges = []
        range_start = None
        previous = None
        for date_key in ordered_date_keys:
            if date_key in missing_date_keys:
                if range_start is None:
                    range_start = date_key
                previous = date_key
                continue
            if range_start is not None:
                ranges.append((range_start, previous))
                range_start = None
                previous = None
        if range_start is not None:
            ranges.append((range_start, previous))
        return ranges

    @classmethod
    def _extract_row_date_keys(
        cls,
        frame: Optional[pd.DataFrame],
        order_book_id: str,
    ) -> set[str]:
        """提取某只证券在返回结果中实际出现过的日期集合。"""
        if frame is None or frame.empty:
            return set()
        if "order_book_id" not in frame.columns or "date" not in frame.columns:
            return set()
        matched = frame.loc[frame["order_book_id"].astype(str) == str(order_book_id), "date"]
        if matched.empty:
            return set()
        normalized = pd.to_datetime(matched, errors="coerce").dropna().dt.normalize()
        return {
            cls._normalize_date_key(date)
            for date in normalized
        }

    @staticmethod
    def _cleanup_snapshot_frame(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """移除快照缓存内部辅助列。"""
        if frame is None:
            return None
        working = frame.copy()
        drop_columns = [column for column in ("_item_key", "_item_order") if column in working.columns]
        if drop_columns:
            working = working.drop(columns=drop_columns)
        return working.reset_index(drop=True)

    def _bundle_path(self) -> str:
        """返回 bundle 根目录，优先支持环境变量临时覆盖。"""
        override = os.environ.get("RQALPHA_BUNDLE_PATH") or os.environ.get("SKYEYE_BUNDLE_PATH")
        if override:
            return os.path.expanduser(override)
        return os.path.expanduser("~/.rqalpha/bundle")

    def _read_bundle_daily(self, order_book_id: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """兼容旧调用方式，继续暴露 bundle 日线读取私有方法。"""
        return self.bundle_reader.get_daily_bars(
            order_book_id,
            start_date,
            end_date,
            bundle_path=self._bundle_path(),
        )

    def _resolve_requested_trading_dates(self, start_date, end_date) -> list[pd.Timestamp]:
        """解析日线/因子补数时应覆盖的交易日列表。"""
        dates = list(self.get_trading_dates(start_date, end_date) or [])
        if dates:
            return [pd.Timestamp(date).normalize() for date in dates]
        return list(pd.bdate_range(pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()))

    def _normalize_online_daily_bars(
        self,
        frame: Optional[pd.DataFrame],
        order_book_ids: list[str],
        requested_fields: list[str],
    ) -> pd.DataFrame:
        """把 rqdatac 返回的各种形态统一成长表结构。"""
        if frame is None or len(frame) == 0:
            return pd.DataFrame(columns=["date", "order_book_id", *requested_fields])

        working = frame.copy()
        single_order_book_id = order_book_ids[0] if len(order_book_ids) == 1 else None

        if isinstance(working.columns, pd.MultiIndex):
            merged = None
            for field in requested_fields:
                part = None
                for level in range(working.columns.nlevels):
                    if field not in set(working.columns.get_level_values(level)):
                        continue
                    field_frame = working.xs(field, axis=1, level=level, drop_level=True)
                    if isinstance(field_frame, pd.Series):
                        part = field_frame.rename(field).reset_index()
                        part = part.rename(columns={part.columns[0]: "date"})
                        part.insert(1, "order_book_id", single_order_book_id)
                    else:
                        part = field_frame.stack(dropna=False).rename(field).reset_index()
                        part = part.rename(columns={part.columns[0]: "date", part.columns[1]: "order_book_id"})
                    break
                if part is None:
                    continue
                merged = part if merged is None else merged.merge(part, on=["date", "order_book_id"], how="outer")
            if merged is None:
                return pd.DataFrame(columns=["date", "order_book_id", *requested_fields])
            working = merged
        elif isinstance(working.index, pd.MultiIndex):
            working = working.reset_index()
        elif "date" not in working.columns and "datetime" not in working.columns:
            working = working.reset_index()

        if "date" not in working.columns:
            if "datetime" in working.columns:
                working = working.rename(columns={"datetime": "date"})
            else:
                for column in working.columns:
                    if pd.api.types.is_datetime64_any_dtype(working[column]):
                        working = working.rename(columns={column: "date"})
                        break

        if "order_book_id" not in working.columns:
            if len(order_book_ids) == 1:
                insert_at = 1 if "date" in working.columns else 0
                working.insert(insert_at, "order_book_id", single_order_book_id)
            elif len(requested_fields) == 1:
                value_columns = [column for column in working.columns if column != "date"]
                if value_columns and all(str(column) in set(order_book_ids) for column in value_columns):
                    value_name = requested_fields[0]
                    working = working.melt(id_vars=["date"], var_name="order_book_id", value_name=value_name)

        if "date" not in working.columns or "order_book_id" not in working.columns:
            return pd.DataFrame(columns=["date", "order_book_id", *requested_fields])

        keep = ["date", "order_book_id", *[field for field in requested_fields if field in working.columns]]
        working = working.loc[:, keep].copy()
        working["date"] = pd.to_datetime(working["date"]).dt.normalize()
        return working.sort_values(["date", "order_book_id"]).reset_index(drop=True)

    def _normalize_online_factor_frame(
        self,
        frame: Optional[pd.DataFrame],
        order_book_ids: list[str],
        factors: list[str],
    ) -> pd.DataFrame:
        """把在线因子结果统一成长表。"""
        if frame is None or len(frame) == 0:
            return pd.DataFrame(columns=["date", "order_book_id", *factors])

        working = frame.copy()
        if isinstance(working.index, pd.MultiIndex):
            working = working.reset_index()
        elif "date" not in working.columns:
            working = working.reset_index()

        if "date" not in working.columns:
            for column in working.columns:
                if pd.api.types.is_datetime64_any_dtype(working[column]):
                    working = working.rename(columns={column: "date"})
                    break

        if "order_book_id" not in working.columns and len(order_book_ids) == 1:
            working.insert(0, "order_book_id", order_book_ids[0])

        if "date" not in working.columns or "order_book_id" not in working.columns:
            return pd.DataFrame(columns=["date", "order_book_id", *factors])

        keep = ["date", "order_book_id", *[factor for factor in factors if factor in working.columns]]
        working = working.loc[:, keep].copy()
        working["date"] = pd.to_datetime(working["date"]).dt.normalize()
        return working.sort_values(["order_book_id", "date"]).reset_index(drop=True)

    def _build_daily_bar_records(
        self,
        order_book_id: str,
        payloads: dict[str, dict],
        requested_fields: list[str],
    ) -> list[dict]:
        """把按日期组织的日线 payload 还原成返回给上层的记录。"""
        records = []
        for date_key, payload in sorted(payloads.items()):
            if not any(field in payload for field in requested_fields):
                continue
            record = {"date": pd.Timestamp(date_key), "order_book_id": order_book_id}
            for field in requested_fields:
                record[field] = payload.get(field)
            records.append(record)
        return records

    def _daily_bar_cache_key(self, order_book_id: str, adjust_type: str, field: str) -> str:
        """构造日线覆盖区间的缓存键。"""
        return "{}|{}|{}".format(order_book_id, adjust_type, field)

    def _factor_cache_key(self, order_book_id: str, factor: str) -> str:
        """构造因子覆盖区间的缓存键。"""
        return "{}|{}".format(order_book_id, factor)

    @staticmethod
    def _snapshot_cache_key(*parts) -> str:
        """构造快照型缓存键。"""
        return "|".join(str(part) for part in parts)

    def get_daily_bars(
        self,
        order_book_ids: Union[str, Sequence[str]],
        start_date,
        end_date,
        fields: Optional[Sequence[str]] = None,
        adjust_type: str = "pre",
    ) -> Optional[pd.DataFrame]:
        """按 `bundle -> SQLite -> rqdatac` 顺序读取日线，缺口仅在线补齐。"""
        ids = self._as_order_book_ids(order_book_ids)
        requested_fields = self._as_list(fields, DEFAULT_BAR_FIELDS)
        trading_dates = self._resolve_requested_trading_dates(start_date, end_date)
        requested_date_keys = [self._normalize_date_key(date) for date in trading_dates]
        payloads_by_id: dict[str, dict[str, dict]] = {}

        for order_book_id in ids:
            bundle_payloads = {}
            if str(adjust_type).lower() in ("none", "pre"):
                bundle_df = self._read_bundle_daily(order_book_id, start_date, end_date)
                bundle_payloads = self._payload_map_from_frame(bundle_df, order_book_id=order_book_id)
            cache_payloads = {}
            if self.cache_store is not None:
                cache_payloads = self.cache_store.load_daily_bar_payloads(order_book_id, start_date, end_date, adjust_type)
            payloads_by_id[order_book_id] = self._merge_payload_maps(bundle_payloads, cache_payloads)

        if self.provider is not None:
            grouped_missing_ids: dict[tuple[tuple[str, str], ...], list[str]] = {}
            for order_book_id in ids:
                missing_date_keys = set()
                local_payloads = payloads_by_id[order_book_id]
                for field in requested_fields:
                    covered = {
                        date_key
                        for date_key, payload in local_payloads.items()
                        if field in payload
                    }
                    if self.cache_store is not None and requested_date_keys:
                        covered.update(
                            self.cache_store.get_covered_dates(
                                "daily_bars",
                                self._daily_bar_cache_key(order_book_id, adjust_type, field),
                                requested_date_keys,
                            )
                        )
                    missing_date_keys.update(date_key for date_key in requested_date_keys if date_key not in covered)
                missing_ranges = tuple(self._group_missing_ranges(requested_date_keys, missing_date_keys))
                if missing_ranges:
                    grouped_missing_ids.setdefault(missing_ranges, []).append(order_book_id)

            for missing_ranges, missing_ids in grouped_missing_ids.items():
                for range_start, range_end in missing_ranges:
                    try:
                        raw_frame = self.provider.get_price(
                            missing_ids[0] if len(missing_ids) == 1 else missing_ids,
                            pd.Timestamp(range_start),
                            pd.Timestamp(range_end),
                            frequency="1d",
                            adjust_type=adjust_type,
                            fields=list(requested_fields),
                        )
                        normalized = self._normalize_online_daily_bars(raw_frame, missing_ids, requested_fields)
                        if self.cache_store is not None and normalized is not None and not normalized.empty:
                            self.cache_store.save_daily_bars(normalized, adjust_type, source="rqdatac")
                        for order_book_id in missing_ids:
                            delivered_date_keys = self._extract_row_date_keys(normalized, order_book_id)
                            row_count = 0
                            if delivered_date_keys:
                                row_count = len(delivered_date_keys)
                            if self.cache_store is not None:
                                for field in requested_fields:
                                    cache_key = self._daily_bar_cache_key(order_book_id, adjust_type, field)
                                    if delivered_date_keys and field in normalized.columns:
                                        delivered_ranges = self._group_missing_ranges(requested_date_keys, delivered_date_keys)
                                        for covered_start, covered_end in delivered_ranges:
                                            self.cache_store.mark_coverage(
                                                "daily_bars",
                                                cache_key,
                                                covered_start,
                                                covered_end,
                                                row_count=row_count,
                                                source="rqdatac",
                                            )
                                    self.cache_store.record_fetch_audit(
                                        "daily_bars",
                                        cache_key,
                                        start_date=range_start,
                                        end_date=range_end,
                                        source="rqdatac",
                                        row_count=row_count,
                                        status="success",
                                    )
                    except Exception as exc:
                        if self.cache_store is not None:
                            for order_book_id in missing_ids:
                                for field in requested_fields:
                                    self.cache_store.record_fetch_audit(
                                        "daily_bars",
                                        self._daily_bar_cache_key(order_book_id, adjust_type, field),
                                        start_date=range_start,
                                        end_date=range_end,
                                        source="rqdatac",
                                        row_count=0,
                                        status="error",
                                        error_type=exc.__class__.__name__,
                                        error_message=str(exc),
                                    )
                        self._raise_if_quota_exceeded(exc)

            if self.cache_store is not None:
                for order_book_id in ids:
                    cache_payloads = self.cache_store.load_daily_bar_payloads(order_book_id, start_date, end_date, adjust_type)
                    payloads_by_id[order_book_id] = self._merge_payload_maps(payloads_by_id[order_book_id], cache_payloads)

        records = []
        for order_book_id in ids:
            records.extend(self._build_daily_bar_records(order_book_id, payloads_by_id[order_book_id], requested_fields))
        if not records:
            return None
        result = pd.DataFrame(records).sort_values(["date", "order_book_id"]).reset_index(drop=True)
        result = result.set_index("date")
        result.index.name = "date"
        return result

    def get_trading_dates(self, start_date, end_date):
        """按 `bundle -> SQLite -> rqdatac` 顺序读取交易日历。"""
        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        bundle_dates = self.bundle_reader.get_trading_dates(
            start_ts,
            end_ts,
            bundle_path=self._bundle_path(),
        )
        bundle_date_keys = [self._normalize_date_key(date) for date in bundle_dates]
        bundle_covered_keys = set()
        if bundle_dates:
            bundle_start = min(bundle_date_keys)
            bundle_end = max(bundle_date_keys)
            requested_calendar_keys = [
                self._normalize_date_key(date)
                for date in pd.date_range(start_ts, end_ts, freq="D")
            ]
            bundle_covered_keys = {
                date_key
                for date_key in requested_calendar_keys
                if bundle_start <= date_key <= bundle_end
            }
        else:
            requested_calendar_keys = [
                self._normalize_date_key(date)
                for date in pd.date_range(start_ts, end_ts, freq="D")
            ]

        cache_dates = []
        if self.cache_store is not None:
            cache_dates = self.cache_store.load_trading_dates(start_ts, end_ts)

        if self.provider is not None and self.cache_store is not None:
            covered_keys = set(bundle_covered_keys)
            covered_keys.update(self.cache_store.get_covered_dates("trading_dates", "calendar", requested_calendar_keys))
            missing_keys = {date_key for date_key in requested_calendar_keys if date_key not in covered_keys}
            missing_ranges = self._group_missing_ranges(requested_calendar_keys, missing_keys)
            for range_start, range_end in missing_ranges:
                try:
                    online_dates = list(self.provider.get_trading_dates(pd.Timestamp(range_start), pd.Timestamp(range_end)) or [])
                    self.cache_store.save_trading_dates(online_dates, source="rqdatac")
                    self.cache_store.mark_coverage(
                        "trading_dates",
                        "calendar",
                        range_start,
                        range_end,
                        row_count=len(online_dates),
                        source="rqdatac",
                    )
                    self.cache_store.record_fetch_audit(
                        "trading_dates",
                        "calendar",
                        start_date=range_start,
                        end_date=range_end,
                        source="rqdatac",
                        row_count=len(online_dates),
                        status="success",
                    )
                except Exception as exc:
                    self.cache_store.record_fetch_audit(
                        "trading_dates",
                        "calendar",
                        start_date=range_start,
                        end_date=range_end,
                        source="rqdatac",
                        row_count=0,
                        status="error",
                        error_type=exc.__class__.__name__,
                        error_message=str(exc),
                    )
                    self._raise_if_quota_exceeded(exc)
            cache_dates = self.cache_store.load_trading_dates(start_ts, end_ts)

        unique_dates = sorted({pd.Timestamp(date).normalize() for date in [*bundle_dates, *cache_dates]})
        return [date for date in unique_dates if start_ts <= date <= end_ts]

    def all_instruments(self, type: Optional[str] = None, date: Optional[Union[str, int, pd.Timestamp]] = None) -> Optional[pd.DataFrame]:
        """按 `bundle -> SQLite -> rqdatac` 顺序读取合约快照。"""
        instrument_type = type or "CS"
        if date is None:
            bundle_frame = self.bundle_reader.get_instruments(type=instrument_type, bundle_path=self._bundle_path())
            if bundle_frame is not None:
                return bundle_frame.reset_index(drop=True)

        cache_key = self._snapshot_cache_key(instrument_type, "__latest__" if date is None else self._normalize_date_key(date))
        if self.cache_store is not None:
            cached_frame, covered = self.cache_store.load_snapshot_frame("all_instruments", cache_key)
            cleaned = self._cleanup_snapshot_frame(cached_frame)
            if cleaned is not None:
                return cleaned
            if covered:
                return pd.DataFrame()

        if self.provider is None:
            return None

        try:
            frame = self.provider.get_instruments(type=instrument_type, date=date)
            result = pd.DataFrame(frame) if frame is not None else pd.DataFrame()
            if self.cache_store is not None:
                self.cache_store.save_snapshot_frame(
                    "all_instruments",
                    cache_key,
                    result,
                    item_key_field="order_book_id",
                    source="rqdatac",
                )
                self.cache_store.mark_snapshot_coverage(
                    "all_instruments",
                    cache_key,
                    row_count=len(result),
                    source="rqdatac",
                )
                self.cache_store.record_fetch_audit(
                    "all_instruments",
                    cache_key,
                    start_date=cache_key,
                    end_date=cache_key,
                    source="rqdatac",
                    row_count=len(result),
                    status="success",
                )
            return result.reset_index(drop=True)
        except Exception as exc:
            if self.cache_store is not None:
                self.cache_store.record_fetch_audit(
                    "all_instruments",
                    cache_key,
                    start_date=cache_key,
                    end_date=cache_key,
                    source="rqdatac",
                    row_count=0,
                    status="error",
                    error_type=exc.__class__.__name__,
                    error_message=str(exc),
                )
            self._raise_if_quota_exceeded(exc)
            return None

    def index_components(self, index_code: str, date: Optional[Union[str, int, pd.Timestamp]] = None) -> Optional[list[str]]:
        """读取指数成分股快照，优先使用本地缓存。"""
        cache_key = self._snapshot_cache_key(index_code, "__latest__" if date is None else self._normalize_date_key(date))
        if self.cache_store is not None:
            cached_frame, covered = self.cache_store.load_snapshot_frame("index_components", cache_key)
            cleaned = self._cleanup_snapshot_frame(cached_frame)
            if cleaned is not None:
                if "position" in cleaned.columns:
                    cleaned = cleaned.sort_values("position")
                if "order_book_id" in cleaned.columns:
                    return cleaned["order_book_id"].astype(str).tolist()
                return cleaned.iloc[:, 0].astype(str).tolist()
            if covered:
                return []

        if self.provider is None:
            return None

        try:
            values = self.provider.get_index_components(index_code, date=date) or []
            if self.cache_store is not None:
                frame = pd.DataFrame(
                    [{"order_book_id": order_book_id, "position": idx} for idx, order_book_id in enumerate(values)]
                )
                self.cache_store.save_snapshot_frame(
                    "index_components",
                    cache_key,
                    frame,
                    item_key_field="order_book_id",
                    source="rqdatac",
                )
                self.cache_store.mark_snapshot_coverage(
                    "index_components",
                    cache_key,
                    row_count=len(values),
                    source="rqdatac",
                )
                self.cache_store.record_fetch_audit(
                    "index_components",
                    cache_key,
                    start_date=cache_key,
                    end_date=cache_key,
                    source="rqdatac",
                    row_count=len(values),
                    status="success",
                )
            return list(values)
        except Exception as exc:
            if self.cache_store is not None:
                self.cache_store.record_fetch_audit(
                    "index_components",
                    cache_key,
                    start_date=cache_key,
                    end_date=cache_key,
                    source="rqdatac",
                    row_count=0,
                    status="error",
                    error_type=exc.__class__.__name__,
                    error_message=str(exc),
                )
            self._raise_if_quota_exceeded(exc)
            return None

    def index_weights(self, index_code: str, date: Optional[Union[str, int, pd.Timestamp]] = None) -> Optional[pd.Series]:
        """读取指数权重快照，优先使用本地缓存。"""
        cache_key = self._snapshot_cache_key(index_code, "__latest__" if date is None else self._normalize_date_key(date))
        if self.cache_store is not None:
            cached_frame, covered = self.cache_store.load_snapshot_frame("index_weights", cache_key)
            cleaned = self._cleanup_snapshot_frame(cached_frame)
            if cleaned is not None and not cleaned.empty:
                if "weight" not in cleaned.columns or "order_book_id" not in cleaned.columns:
                    return pd.Series(dtype=float)
                series = pd.Series(cleaned["weight"].astype(float).values, index=cleaned["order_book_id"].astype(str))
                series.index.name = None
                return series
            if covered:
                return pd.Series(dtype=float)

        if self.provider is None:
            return None

        try:
            values = self.provider.get_index_weights(index_code, date=date)
            series = pd.Series(dtype=float) if values is None else values.astype(float)
            if self.cache_store is not None:
                frame = pd.DataFrame(
                    [{"order_book_id": str(order_book_id), "weight": float(weight)} for order_book_id, weight in series.items()]
                )
                self.cache_store.save_snapshot_frame(
                    "index_weights",
                    cache_key,
                    frame,
                    item_key_field="order_book_id",
                    source="rqdatac",
                )
                self.cache_store.mark_snapshot_coverage(
                    "index_weights",
                    cache_key,
                    row_count=len(series),
                    source="rqdatac",
                )
                self.cache_store.record_fetch_audit(
                    "index_weights",
                    cache_key,
                    start_date=cache_key,
                    end_date=cache_key,
                    source="rqdatac",
                    row_count=len(series),
                    status="success",
                )
            return series
        except Exception as exc:
            if self.cache_store is not None:
                self.cache_store.record_fetch_audit(
                    "index_weights",
                    cache_key,
                    start_date=cache_key,
                    end_date=cache_key,
                    source="rqdatac",
                    row_count=0,
                    status="error",
                    error_type=exc.__class__.__name__,
                    error_message=str(exc),
                )
            self._raise_if_quota_exceeded(exc)
            return None

    def get_factor(
        self,
        order_book_ids: Union[str, Sequence[str]],
        factors: Union[str, Sequence[str]],
        start_date,
        end_date,
    ) -> Optional[pd.DataFrame]:
        """按 `SQLite -> rqdatac` 顺序读取因子，并把在线补数落本地缓存。"""
        ids = self._as_order_book_ids(order_book_ids)
        factor_list = self._as_list(factors, ())
        if self.cache_store is None and self.provider is None:
            return None

        requested_trading_dates = self._resolve_requested_trading_dates(start_date, end_date)
        requested_date_keys = [self._normalize_date_key(date) for date in requested_trading_dates]
        local_payloads = {} if self.cache_store is None else self.cache_store.load_factor_payloads(ids, start_date, end_date)

        if self.provider is not None:
            grouped_missing_ids: dict[tuple[tuple[str, str], ...], list[str]] = {}
            for order_book_id in ids:
                missing_date_keys = set()
                for factor in factor_list:
                    covered = {
                        date_key
                        for (payload_order_book_id, date_key), payload in local_payloads.items()
                        if payload_order_book_id == order_book_id and factor in payload
                    }
                    if self.cache_store is not None and requested_date_keys:
                        covered.update(
                            self.cache_store.get_covered_dates(
                                "factor_values",
                                self._factor_cache_key(order_book_id, factor),
                                requested_date_keys,
                            )
                        )
                    missing_date_keys.update(date_key for date_key in requested_date_keys if date_key not in covered)
                missing_ranges = tuple(self._group_missing_ranges(requested_date_keys, missing_date_keys))
                if missing_ranges:
                    grouped_missing_ids.setdefault(missing_ranges, []).append(order_book_id)

            for missing_ranges, missing_ids in grouped_missing_ids.items():
                for range_start, range_end in missing_ranges:
                    try:
                        raw_frame = self.provider.get_factors(
                            missing_ids[0] if len(missing_ids) == 1 else missing_ids,
                            factor_list,
                            range_start,
                            range_end,
                        )
                        normalized = self._normalize_online_factor_frame(raw_frame, missing_ids, factor_list)
                        if self.cache_store is not None and normalized is not None and not normalized.empty:
                            self.cache_store.save_factor_values(normalized, source="rqdatac")
                        for order_book_id in missing_ids:
                            delivered_date_keys = self._extract_row_date_keys(normalized, order_book_id)
                            row_count = 0
                            if delivered_date_keys:
                                row_count = len(delivered_date_keys)
                            if self.cache_store is not None:
                                for factor in factor_list:
                                    cache_key = self._factor_cache_key(order_book_id, factor)
                                    if delivered_date_keys and factor in normalized.columns:
                                        delivered_ranges = self._group_missing_ranges(requested_date_keys, delivered_date_keys)
                                        for covered_start, covered_end in delivered_ranges:
                                            self.cache_store.mark_coverage(
                                                "factor_values",
                                                cache_key,
                                                covered_start,
                                                covered_end,
                                                row_count=row_count,
                                                source="rqdatac",
                                            )
                                    self.cache_store.record_fetch_audit(
                                        "factor_values",
                                        cache_key,
                                        start_date=range_start,
                                        end_date=range_end,
                                        source="rqdatac",
                                        row_count=row_count,
                                        status="success",
                                    )
                    except Exception as exc:
                        if self.cache_store is not None:
                            for order_book_id in missing_ids:
                                for factor in factor_list:
                                    self.cache_store.record_fetch_audit(
                                        "factor_values",
                                        self._factor_cache_key(order_book_id, factor),
                                        start_date=range_start,
                                        end_date=range_end,
                                        source="rqdatac",
                                        row_count=0,
                                        status="error",
                                        error_type=exc.__class__.__name__,
                                        error_message=str(exc),
                                    )
                        self._raise_if_quota_exceeded(exc)

            if self.cache_store is not None:
                local_payloads = self.cache_store.load_factor_payloads(ids, start_date, end_date)

        if self.cache_store is None:
            return None
        return self.cache_store.load_factor_frame(ids, factor_list, start_date, end_date)

    def get_industry(
        self,
        order_book_ids: Union[str, Sequence[str]],
        source: str = "citics_2019",
        level: int = 1,
        date=None,
    ) -> Optional[pd.DataFrame]:
        """行业分类轻包装；失败时保持和其他快照接口一致返回 None。"""
        if self.provider is None:
            return None
        try:
            return self.provider.get_industry(order_book_ids, source=source, level=level, date=date)
        except Exception as exc:
            self._raise_if_quota_exceeded(exc)
            return None

    def is_suspended(
        self,
        order_book_ids: Union[str, Sequence[str]],
        start_date,
        end_date,
    ) -> Optional[pd.DataFrame]:
        """停牌状态轻包装。"""
        if self.provider is None:
            return None
        try:
            return self.provider.is_suspended(order_book_ids, start_date=start_date, end_date=end_date)
        except Exception as exc:
            self._raise_if_quota_exceeded(exc)
            return None

    def get_securities_margin(
        self,
        order_book_ids: Union[str, Sequence[str]],
        start_date,
        end_date,
        fields: Optional[Union[str, Sequence[str]]] = None,
    ) -> Optional[pd.DataFrame]:
        """按 `SQLite -> rqdatac` 顺序读取融资融券数据，复用 factor_values 表缓存。

        Args:
            order_book_ids: 证券代码或代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段名或字段列表，如 'margin_balance' 或 ['margin_balance', 'short_balance']

        Returns:
            长表格式 DataFrame，包含 date, order_book_id 及请求的字段列
        """
        ids = self._as_order_book_ids(order_book_ids)
        field_list = self._as_list(fields, ["margin_balance"])
        if self.cache_store is None:
            return None

        # 复用 factor_values 表的缓存机制
        requested_trading_dates = self._resolve_requested_trading_dates(start_date, end_date)
        requested_date_keys = [self._normalize_date_key(date) for date in requested_trading_dates]
        local_payloads = {} if self.cache_store is None else self.cache_store.load_factor_payloads(ids, start_date, end_date)

        # Check for missing dates and fetch online
        grouped_missing_ids: dict[tuple[tuple[str, str], ...], list[str]] = {}
        for order_book_id in ids:
            missing_date_keys = set()
            for field in field_list:
                covered = {
                    date_key
                    for (payload_order_book_id, date_key), payload in local_payloads.items()
                    if payload_order_book_id == order_book_id and field in payload
                }
                if self.cache_store is not None and requested_date_keys:
                    covered.update(
                        self.cache_store.get_covered_dates(
                            "factor_values",
                            self._factor_cache_key(order_book_id, field),
                            requested_date_keys,
                        )
                    )
                missing_date_keys.update(date_key for date_key in requested_date_keys if date_key not in covered)
            missing_ranges = tuple(self._group_missing_ranges(requested_date_keys, missing_date_keys))
            if missing_ranges:
                grouped_missing_ids.setdefault(missing_ranges, []).append(order_book_id)

        for missing_ranges, missing_ids in grouped_missing_ids.items():
            for range_start, range_end in missing_ranges:
                    try:
                        # Direct call to rqalpha.apis (provider doesn't have this method)
                        from rqalpha.apis import get_securities_margin

                        raw_frame = get_securities_margin(
                            missing_ids[0] if len(missing_ids) == 1 else missing_ids,
                            start_date=range_start,
                            end_date=range_end,
                            fields=field_list[0] if len(field_list) == 1 else field_list,
                            expect_df=True,
                        )
                        normalized = self._normalize_online_factor_frame(raw_frame, missing_ids, field_list)
                        if self.cache_store is not None and normalized is not None and not normalized.empty:
                            self.cache_store.save_factor_values(normalized, source="rqdatac")
                        for order_book_id in missing_ids:
                            delivered_date_keys = self._extract_row_date_keys(normalized, order_book_id)
                            row_count = len(delivered_date_keys) if delivered_date_keys else 0
                            if self.cache_store is not None:
                                for field in field_list:
                                    cache_key = self._factor_cache_key(order_book_id, field)
                                    if delivered_date_keys and field in normalized.columns:
                                        delivered_ranges = self._group_missing_ranges(requested_date_keys, delivered_date_keys)
                                        for covered_start, covered_end in delivered_ranges:
                                            self.cache_store.mark_coverage(
                                                "factor_values",
                                                cache_key,
                                                covered_start,
                                                covered_end,
                                                row_count=row_count,
                                                source="rqdatac",
                                            )
                                    self.cache_store.record_fetch_audit(
                                        "factor_values",
                                        cache_key,
                                        start_date=range_start,
                                        end_date=range_end,
                                        source="rqdatac",
                                        row_count=row_count,
                                        status="success",
                                    )
                    except Exception as exc:
                        if self.cache_store is not None:
                            for order_book_id in missing_ids:
                                for field in field_list:
                                    self.cache_store.record_fetch_audit(
                                        "factor_values",
                                        self._factor_cache_key(order_book_id, field),
                                        start_date=range_start,
                                        end_date=range_end,
                                        source="rqdatac",
                                        row_count=0,
                                        status="error",
                                        error_type=exc.__class__.__name__,
                                        error_message=str(exc),
                                    )
                        self._raise_if_quota_exceeded(exc)

            if self.cache_store is not None:
                local_payloads = self.cache_store.load_factor_payloads(ids, start_date, end_date)

        if self.cache_store is None:
            return None
        return self.cache_store.load_factor_frame(ids, field_list, start_date, end_date)

    def get_stock_connect(
        self,
        order_book_ids: Union[str, Sequence[str]],
        start_date,
        end_date,
        fields: Optional[Union[str, Sequence[str]]] = None,
    ) -> Optional[pd.DataFrame]:
        """按 `SQLite -> rqdatac` 顺序读取北向资金数据，复用 factor_values 表缓存。

        Args:
            order_book_ids: 股票代码或代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段名或字段列表，如 'shares_holding' 或 ['shares_holding', 'holding_ratio']

        Returns:
            长表格式 DataFrame，包含 date, order_book_id 及请求的字段列
        """
        ids = self._as_order_book_ids(order_book_ids)
        field_list = self._as_list(fields, ["shares_holding", "holding_ratio"])
        if self.cache_store is None:
            return None

        # 复用 factor_values 表的缓存机制
        requested_trading_dates = self._resolve_requested_trading_dates(start_date, end_date)
        requested_date_keys = [self._normalize_date_key(date) for date in requested_trading_dates]
        local_payloads = {} if self.cache_store is None else self.cache_store.load_factor_payloads(ids, start_date, end_date)

        # Check for missing dates and fetch online
        grouped_missing_ids: dict[tuple[tuple[str, str], ...], list[str]] = {}
        for order_book_id in ids:
            missing_date_keys = set()
            for field in field_list:
                covered = {
                    date_key
                    for (payload_order_book_id, date_key), payload in local_payloads.items()
                    if payload_order_book_id == order_book_id and field in payload
                }
                if self.cache_store is not None and requested_date_keys:
                    covered.update(
                        self.cache_store.get_covered_dates(
                            "factor_values",
                            self._factor_cache_key(order_book_id, field),
                            requested_date_keys,
                        )
                    )
                missing_date_keys.update(date_key for date_key in requested_date_keys if date_key not in covered)
            missing_ranges = tuple(self._group_missing_ranges(requested_date_keys, missing_date_keys))
            if missing_ranges:
                grouped_missing_ids.setdefault(missing_ranges, []).append(order_book_id)

        for missing_ranges, missing_ids in grouped_missing_ids.items():
            for range_start, range_end in missing_ranges:
                    try:
                        # Direct call to rqalpha.apis (provider doesn't have this method)
                        from rqalpha.apis import get_stock_connect

                        raw_frame = get_stock_connect(
                            missing_ids[0] if len(missing_ids) == 1 else missing_ids,
                            start_date=range_start,
                            end_date=range_end,
                            fields=field_list,
                            expect_df=True,
                        )
                        normalized = self._normalize_online_factor_frame(raw_frame, missing_ids, field_list)
                        if self.cache_store is not None and normalized is not None and not normalized.empty:
                            self.cache_store.save_factor_values(normalized, source="rqdatac")
                        for order_book_id in missing_ids:
                            delivered_date_keys = self._extract_row_date_keys(normalized, order_book_id)
                            row_count = len(delivered_date_keys) if delivered_date_keys else 0
                            if self.cache_store is not None:
                                for field in field_list:
                                    cache_key = self._factor_cache_key(order_book_id, field)
                                    if delivered_date_keys and field in normalized.columns:
                                        delivered_ranges = self._group_missing_ranges(requested_date_keys, delivered_date_keys)
                                        for covered_start, covered_end in delivered_ranges:
                                            self.cache_store.mark_coverage(
                                                "factor_values",
                                                cache_key,
                                                covered_start,
                                                covered_end,
                                                row_count=row_count,
                                                source="rqdatac",
                                            )
                                    self.cache_store.record_fetch_audit(
                                        "factor_values",
                                        cache_key,
                                        start_date=range_start,
                                        end_date=range_end,
                                        source="rqdatac",
                                        row_count=row_count,
                                        status="success",
                                    )
                    except Exception as exc:
                        if self.cache_store is not None:
                            for order_book_id in missing_ids:
                                for field in field_list:
                                    self.cache_store.record_fetch_audit(
                                        "factor_values",
                                        self._factor_cache_key(order_book_id, field),
                                        start_date=range_start,
                                        end_date=range_end,
                                        source="rqdatac",
                                        row_count=0,
                                        status="error",
                                        error_type=exc.__class__.__name__,
                                        error_message=str(exc),
                                    )
                        self._raise_if_quota_exceeded(exc)

            if self.cache_store is not None:
                local_payloads = self.cache_store.load_factor_payloads(ids, start_date, end_date)

        if self.cache_store is None:
            return None
        return self.cache_store.load_factor_frame(ids, field_list, start_date, end_date)

    def get_pit_financials(
        self,
        order_book_ids: Union[str, Sequence[str]],
        fields: Union[str, Sequence[str]],
        count: int = 4,
        statements: str = "latest",
    ) -> Optional[pd.DataFrame]:
        """读取 Point-in-time 财务数据，使用 snapshot_cache 表缓存。

        Args:
            order_book_ids: 股票代码或代码列表
            fields: 财务字段名或字段列表，如 'net_profit' 或 ['net_profit', 'total_owner_equities']
            count: 获取最近几个季度的数据，默认4
            statements: 'latest' 使用最新发布的数据，'all' 返回所有版本

        Returns:
            DataFrame 包含财务数据
        """
        ids = self._as_order_book_ids(order_book_ids)
        field_list = self._as_list(fields, [])
        if not field_list:
            raise ValueError("fields parameter is required for get_pit_financials")

        # 使用 snapshot_cache 表存储 PIT 财务数据
        # cache_key 格式: {count}__{statements}__{field_list_hash}
        import hashlib
        field_hash = hashlib.md5("|".join(sorted(field_list)).encode()).hexdigest()[:8]
        cache_key = f"pit_fin_{count}_{statements}_{field_hash}"

        # 尝试从缓存加载
        if self.cache_store is not None:
            cached_frame, covered = self.cache_store.load_snapshot_frame("pit_financials", cache_key)
            if cached_frame is not None and not cached_frame.empty:
                # 过滤出请求的order_book_ids
                if "order_book_id" in cached_frame.columns:
                    filtered = cached_frame[cached_frame["order_book_id"].isin(ids)]
                    if not filtered.empty:
                        return filtered
            if covered:
                return pd.DataFrame()

        # 在线获取数据
        try:
            # Direct call to rqalpha.apis (provider doesn't have this method)
            from rqalpha.apis import get_pit_financials_ex

            result = get_pit_financials_ex(
                order_book_ids=ids,
                fields=field_list,
                count=count,
                statements=statements,
            )

            if result is not None and not result.empty:
                # 标准化格式
                if isinstance(result.index, pd.MultiIndex):
                    result = result.reset_index()

                # 保存到缓存
                if self.cache_store is not None:
                    try:
                        # Create composite item_key to ensure uniqueness
                        # PIT financials returns multiple rows per stock (one per quarter)
                        # Use "order_book_id__quarter" as unique identifier
                        cache_result = result.copy()
                        if 'quarter' in cache_result.columns:
                            cache_result['_cache_item_key'] = (
                                cache_result['order_book_id'].astype(str) + '__' +
                                cache_result['quarter'].astype(str)
                            )
                            item_key_field = '_cache_item_key'
                        else:
                            # Fallback to order_book_id if quarter is not available
                            item_key_field = 'order_book_id'

                        self.cache_store.save_snapshot_frame(
                            "pit_financials",
                            cache_key,
                            cache_result,
                            item_key_field=item_key_field,
                            source="rqdatac",
                        )
                        self.cache_store.mark_snapshot_coverage(
                            "pit_financials",
                            cache_key,
                            row_count=len(result),
                            source="rqdatac",
                        )
                        self.cache_store.record_fetch_audit(
                            "pit_financials",
                            cache_key,
                            start_date=None,
                            end_date=None,
                            source="rqdatac",
                            row_count=len(result),
                            status="success",
                        )
                    except Exception as cache_exc:
                        # Cache failure should not prevent returning data
                        import logging
                        logging.warning(f"Failed to cache PIT financials data: {cache_exc}")

                return result

        except Exception as exc:
            if self.cache_store is not None:
                self.cache_store.record_fetch_audit(
                    "pit_financials",
                    cache_key,
                    start_date=None,
                    end_date=None,
                    source="rqdatac",
                    row_count=0,
                    status="error",
                    error_type=exc.__class__.__name__,
                    error_message=str(exc),
                )
            self._raise_if_quota_exceeded(exc)
            # Re-raise non-quota exceptions so callers know what went wrong
            raise

        return None

    def get_turnover_rate(
        self,
        order_book_ids: Union[str, Sequence[str]],
        start_date,
        end_date,
        fields: Optional[Union[str, Sequence[str]]] = None,
    ) -> Optional[pd.DataFrame]:
        """按 `SQLite -> rqdatac` 顺序读取换手率数据，复用 factor_values 表缓存。"""
        ids = self._as_order_book_ids(order_book_ids)
        field_list = self._as_list(fields, ["today"])
        if self.provider is None and self.cache_store is None:
            return None

        # 复用 get_factor 缓存机制，用 turnover_rate 前缀的 cache_key
        factor_names = [f"turnover_rate_{field}" for field in field_list]
        result = self.get_factor(ids, factor_names, start_date, end_date)
        if result is not None and not result.empty:
            # Rename columns to simple field names
            rename_map = {f"turnover_rate_{field}": field for field in field_list if f"turnover_rate_{field}" in result.columns}
            result = result.rename(columns=rename_map)
        return result

    def get_bond_yield(
        self,
        start_date,
        end_date,
        tenor: str = "10Y",
    ) -> Optional[pd.DataFrame]:
        """读取国债收益率曲线，缓存到 factor_values 表。"""
        if self.provider is None:
            return None
        try:
            raw = self.provider.get_bond_yield(start_date, end_date, tenor=tenor)
            if raw is None or raw.empty:
                return None
            result = raw.copy()
            if isinstance(result.index, pd.DatetimeIndex):
                result = result.reset_index()
                result = result.rename(columns={"index": "date"})
            # Ensure date column
            for col in ["date", "DateTime"]:
                if col in result.columns:
                    result["date"] = pd.to_datetime(result[col])
                    break
            # Use the tenor column as the yield value
            yield_col = tenor if tenor in result.columns else result.columns[-1]
            result = result[["date", yield_col]].copy()
            result = result.rename(columns={yield_col: f"bond_yield_{tenor}"})
            result["order_book_id"] = "MARKET"
            return result
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load bond yield: {exc}")
            return None

    def get_northbound_flow(
        self,
        start_date,
        end_date,
    ) -> Optional[pd.DataFrame]:
        """读取北向资金整体净流入（市场级），优先 rqdatac，fallback AKShare。"""
        if self.provider is None:
            # Try AKShare fallback
            try:
                from skyeye.data.compat import get_northbound_flow_akshare
                raw = get_northbound_flow_akshare(start_date, end_date)
                if raw is not None and not raw.empty:
                    result = raw.copy()
                    if "north_net_flow" in result.columns:
                        result = result.rename(columns={"north_net_flow": "northbound_net_flow"})
                    result["order_book_id"] = "MARKET"
                    return result
            except Exception:
                pass
            return None
        try:
            raw = self.provider.get_northbound_flow(start_date, end_date)
            if raw is None or raw.empty:
                return None
            result = raw.copy()
            if isinstance(result.index, pd.DatetimeIndex):
                result = result.reset_index()
                result = result.rename(columns={"index": "date"})
            for col in ["date", "DateTime"]:
                if col in result.columns:
                    result["date"] = pd.to_datetime(result[col])
                    break
            result["order_book_id"] = "MARKET"
            return result
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load northbound flow from provider: {exc}")
            # Try AKShare fallback
            try:
                from skyeye.data.compat import get_northbound_flow_akshare
                raw = get_northbound_flow_akshare(start_date, end_date)
                if raw is not None and not raw.empty:
                    result = raw.copy()
                    if "north_net_flow" in result.columns:
                        result = result.rename(columns={"north_net_flow": "northbound_net_flow"})
                    result["order_book_id"] = "MARKET"
                    return result
            except Exception:
                pass
            return None

    def get_macro_pmi(
        self,
        start_date,
        end_date,
    ) -> Optional[pd.DataFrame]:
        """读取官方制造业 PMI，优先 provider 扩展，fallback AKShare。"""
        raw = None
        if self.provider is not None:
            provider_getter = getattr(self.provider, "get_macro_pmi", None)
            if callable(provider_getter):
                try:
                    raw = provider_getter(start_date, end_date)
                except Exception as exc:
                    import logging

                    logging.getLogger(__name__).warning(f"Failed to load macro PMI from provider: {exc}")

        if raw is None or raw.empty:
            try:
                from skyeye.data.compat import get_macro_pmi_akshare

                raw = get_macro_pmi_akshare(start_date, end_date)
            except Exception:
                raw = None

        if raw is None or raw.empty:
            return None

        result = raw.copy()
        if isinstance(result.index, pd.DatetimeIndex):
            result = result.reset_index().rename(columns={"index": "date"})
        for col in ["date", "DateTime"]:
            if col in result.columns:
                result["date"] = pd.to_datetime(result[col])
                break
        if "date" not in result.columns:
            return None

        pmi_col = None
        for candidate in ["pmi", "PMI", "manufacturing_pmi", "macro_pmi"]:
            if candidate in result.columns:
                pmi_col = candidate
                break
        if pmi_col is None:
            numeric_cols = result.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                return None
            pmi_col = numeric_cols[0]

        result = result[["date", pmi_col]].copy()
        result = result.rename(columns={pmi_col: "pmi"})
        result["pmi"] = pd.to_numeric(result["pmi"], errors="coerce")
        result["order_book_id"] = "MARKET"
        return result

    def is_st_stock(
        self,
        order_book_ids: Union[str, Sequence[str]],
        start_date,
        end_date,
    ) -> Optional[pd.DataFrame]:
        """ST 状态轻包装。"""
        if self.provider is None:
            return None
        try:
            return self.provider.is_st_stock(order_book_ids, start_date=start_date, end_date=end_date)
        except Exception as exc:
            self._raise_if_quota_exceeded(exc)
            return None
