# -*- coding: utf-8 -*-
"""Skyeye 本地 SQLite 数据缓存。"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd

SQLITE_CONNECT_TIMEOUT = 30
SQLITE_BUSY_TIMEOUT_MS = 30000
SCHEMA_VERSION = "1"

CREATE_TABLE_SQL = (
    """
    CREATE TABLE IF NOT EXISTS daily_bars (
        date TEXT NOT NULL,
        order_book_id TEXT NOT NULL,
        adjust_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (order_book_id, adjust_type, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS factor_values (
        date TEXT NOT NULL,
        order_book_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (order_book_id, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trading_dates (
        date TEXT PRIMARY KEY,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS snapshot_cache (
        data_kind TEXT NOT NULL,
        cache_key TEXT NOT NULL,
        item_key TEXT NOT NULL,
        item_order INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (data_kind, cache_key, item_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS coverage_checkpoint (
        data_kind TEXT NOT NULL,
        cache_key TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        source TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (data_kind, cache_key, start_date, end_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fetch_audit (
        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        data_kind TEXT NOT NULL,
        cache_key TEXT NOT NULL,
        start_date TEXT,
        end_date TEXT,
        source TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        status TEXT NOT NULL,
        error_type TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cache_meta (
        meta_key TEXT PRIMARY KEY,
        meta_value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
)


class LocalDataCacheStore:
    """负责本地 SQLite 缓存的读写、覆盖区间和审计记录。"""

    def __init__(self, db_path: Optional[str] = None) -> None:
        configured_path = db_path or os.environ.get("SKYEYE_DATA_CACHE_PATH")
        self.db_path = os.path.expanduser(configured_path or "~/.rqalpha/skyeye_data_cache.sqlite3")
        try:
            self._init_db()
        except sqlite3.OperationalError:
            # 默认路径在受限执行环境里可能不可写；未显式配置时回退到 /tmp 以保证运行。
            if configured_path is not None:
                raise
            self.db_path = "/tmp/skyeye_data_cache.sqlite3"
            self._init_db()

    @staticmethod
    def _normalize_date(value) -> str:
        """统一把日期值转成 `YYYY-MM-DD`。"""
        return pd.Timestamp(value).normalize().strftime("%Y-%m-%d")

    @staticmethod
    def _serialize_value(value):
        """把 pandas / numpy 值转换成 JSON 可持久化的 Python 基础类型。"""
        if value is None or pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except Exception:
                pass
        return value

    @classmethod
    def _row_payload(cls, row: dict, excluded_fields: set[str]) -> dict:
        """从行字典中抽取需要缓存的业务字段。"""
        payload = {}
        for key, value in row.items():
            if key in excluded_fields:
                continue
            payload[key] = cls._serialize_value(value)
        return payload

    @staticmethod
    def _json_dumps(payload: dict) -> str:
        """稳定地序列化 JSON。"""
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _json_loads(payload_json: str) -> dict:
        """反序列化 JSON。"""
        return json.loads(payload_json) if payload_json else {}

    @staticmethod
    def _merge_payload(existing: dict, incoming: dict) -> dict:
        """合并同一行的字段缓存，保证后续新增字段不会覆盖掉旧字段。"""
        merged = dict(existing)
        merged.update(incoming)
        return merged

    @staticmethod
    def _now() -> str:
        """返回当前时间戳字符串。"""
        return datetime.now().isoformat()

    def _connect(self) -> sqlite3.Connection:
        """创建 SQLite 连接并设置并发相关参数。"""
        conn = sqlite3.connect(self.db_path, timeout=SQLITE_CONNECT_TIMEOUT)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = {}".format(SQLITE_BUSY_TIMEOUT_MS))
        return conn

    def _init_db(self) -> None:
        """初始化缓存数据库。"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            for sql in CREATE_TABLE_SQL:
                conn.execute(sql)
            conn.execute(
                """
                INSERT INTO cache_meta (meta_key, meta_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(meta_key) DO UPDATE SET
                    meta_value = excluded.meta_value,
                    updated_at = excluded.updated_at
                """,
                ("schema_version", SCHEMA_VERSION, self._now()),
            )

    def load_trading_dates(self, start_date, end_date) -> list[pd.Timestamp]:
        """读取 SQLite 中已缓存的交易日历。"""
        start_key = self._normalize_date(start_date)
        end_key = self._normalize_date(end_date)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT date
                FROM trading_dates
                WHERE date >= ? AND date <= ?
                ORDER BY date
                """,
                (start_key, end_key),
            ).fetchall()
        return [pd.Timestamp(row["date"]) for row in rows]

    def save_trading_dates(self, dates: list, source: str) -> int:
        """批量写入交易日历。"""
        values = [self._normalize_date(date) for date in dates]
        if not values:
            return 0
        now = self._now()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO trading_dates (date, updated_at)
                VALUES (?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    updated_at = excluded.updated_at
                """,
                [(date, now) for date in values],
            )
        return len(values)

    def load_daily_bar_payloads(
        self,
        order_book_id: str,
        start_date,
        end_date,
        adjust_type: str,
    ) -> dict[str, dict]:
        """按证券和日期区间读取缓存中的日线 payload。"""
        start_key = self._normalize_date(start_date)
        end_key = self._normalize_date(end_date)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT date, payload_json
                FROM daily_bars
                WHERE order_book_id = ? AND adjust_type = ? AND date >= ? AND date <= ?
                ORDER BY date
                """,
                (order_book_id, adjust_type, start_key, end_key),
            ).fetchall()
        return {
            row["date"]: self._json_loads(row["payload_json"])
            for row in rows
        }

    def save_daily_bars(self, frame: pd.DataFrame, adjust_type: str, source: str) -> int:
        """写入在线补到的日线数据，并与旧字段做并集。"""
        if frame is None or frame.empty:
            return 0
        working = frame.copy()
        working["date"] = pd.to_datetime(working["date"]).dt.normalize()
        order_book_ids = sorted(set(working["order_book_id"].astype(str)))
        start_key = self._normalize_date(working["date"].min())
        end_key = self._normalize_date(working["date"].max())
        placeholders = ",".join("?" for _ in order_book_ids)
        with self._connect() as conn:
            existing_rows = conn.execute(
                """
                SELECT date, order_book_id, payload_json
                FROM daily_bars
                WHERE adjust_type = ? AND order_book_id IN ({}) AND date >= ? AND date <= ?
                """.format(placeholders),
                [adjust_type, *order_book_ids, start_key, end_key],
            ).fetchall()
            existing_map = {
                (row["order_book_id"], row["date"]): self._json_loads(row["payload_json"])
                for row in existing_rows
            }
            now = self._now()
            payload_rows = []
            for row in working.to_dict("records"):
                date_key = self._normalize_date(row["date"])
                order_book_id = str(row["order_book_id"])
                payload = self._row_payload(row, {"date", "order_book_id"})
                merged = self._merge_payload(existing_map.get((order_book_id, date_key), {}), payload)
                payload_rows.append(
                    (date_key, order_book_id, adjust_type, self._json_dumps(merged), now)
                )
            conn.executemany(
                """
                INSERT INTO daily_bars (date, order_book_id, adjust_type, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(order_book_id, adjust_type, date) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                payload_rows,
            )
        return len(payload_rows)

    def load_factor_payloads(
        self,
        order_book_ids: list[str],
        start_date,
        end_date,
    ) -> dict[tuple[str, str], dict]:
        """按证券和日期区间读取缓存中的因子 payload。"""
        if not order_book_ids:
            return {}
        start_key = self._normalize_date(start_date)
        end_key = self._normalize_date(end_date)
        placeholders = ",".join("?" for _ in order_book_ids)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT date, order_book_id, payload_json
                FROM factor_values
                WHERE order_book_id IN ({}) AND date >= ? AND date <= ?
                ORDER BY order_book_id, date
                """.format(placeholders),
                [*order_book_ids, start_key, end_key],
            ).fetchall()
        return {
            (row["order_book_id"], row["date"]): self._json_loads(row["payload_json"])
            for row in rows
        }

    def save_factor_values(self, frame: pd.DataFrame, source: str) -> int:
        """写入在线补到的因子数据，并保留旧字段。"""
        if frame is None or frame.empty:
            return 0
        working = frame.copy()
        working["date"] = pd.to_datetime(working["date"]).dt.normalize()
        order_book_ids = sorted(set(working["order_book_id"].astype(str)))
        start_key = self._normalize_date(working["date"].min())
        end_key = self._normalize_date(working["date"].max())
        placeholders = ",".join("?" for _ in order_book_ids)
        with self._connect() as conn:
            existing_rows = conn.execute(
                """
                SELECT date, order_book_id, payload_json
                FROM factor_values
                WHERE order_book_id IN ({}) AND date >= ? AND date <= ?
                """.format(placeholders),
                [*order_book_ids, start_key, end_key],
            ).fetchall()
            existing_map = {
                (row["order_book_id"], row["date"]): self._json_loads(row["payload_json"])
                for row in existing_rows
            }
            now = self._now()
            payload_rows = []
            for row in working.to_dict("records"):
                date_key = self._normalize_date(row["date"])
                order_book_id = str(row["order_book_id"])
                payload = self._row_payload(row, {"date", "order_book_id"})
                merged = self._merge_payload(existing_map.get((order_book_id, date_key), {}), payload)
                payload_rows.append((date_key, order_book_id, self._json_dumps(merged), now))
            conn.executemany(
                """
                INSERT INTO factor_values (date, order_book_id, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(order_book_id, date) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                payload_rows,
            )
        return len(payload_rows)

    def load_factor_frame(
        self,
        order_book_ids: list[str],
        factors: list[str],
        start_date,
        end_date,
    ) -> Optional[pd.DataFrame]:
        """把缓存里的因子 payload 还原成上层熟悉的多级索引 DataFrame。"""
        payloads = self.load_factor_payloads(order_book_ids, start_date, end_date)
        records = []
        for (order_book_id, date_key), payload in sorted(payloads.items()):
            record = {"order_book_id": order_book_id, "date": pd.Timestamp(date_key)}
            has_any_factor = False
            for factor in factors:
                if factor in payload:
                    has_any_factor = True
                record[factor] = payload.get(factor)
            if has_any_factor:
                records.append(record)
        if not records:
            return None
        frame = pd.DataFrame(records).sort_values(["order_book_id", "date"]).reset_index(drop=True)
        return frame.set_index(["order_book_id", "date"])

    def load_snapshot_frame(self, data_kind: str, cache_key: str) -> tuple[Optional[pd.DataFrame], bool]:
        """读取快照型缓存，并返回“是否存在覆盖记录”。"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT item_key, item_order, payload_json
                FROM snapshot_cache
                WHERE data_kind = ? AND cache_key = ?
                ORDER BY item_order, item_key
                """,
                (data_kind, cache_key),
            ).fetchall()
            coverage = conn.execute(
                """
                SELECT 1
                FROM coverage_checkpoint
                WHERE data_kind = ? AND cache_key = ? AND start_date = ? AND end_date = ?
                LIMIT 1
                """,
                (data_kind, cache_key, cache_key, cache_key),
            ).fetchone()
        if not rows:
            return None, coverage is not None
        records = []
        for row in rows:
            payload = self._json_loads(row["payload_json"])
            payload["_item_key"] = row["item_key"]
            payload["_item_order"] = row["item_order"]
            records.append(payload)
        frame = pd.DataFrame(records).sort_values(["_item_order", "_item_key"]).reset_index(drop=True)
        return frame, True

    def save_snapshot_frame(
        self,
        data_kind: str,
        cache_key: str,
        frame: pd.DataFrame,
        *,
        item_key_field: str,
        source: str,
    ) -> int:
        """写入按 key 命中的整表快照缓存。"""
        working = frame.copy() if frame is not None else pd.DataFrame()
        now = self._now()
        rows = []
        if not working.empty:
            for idx, row in enumerate(working.to_dict("records")):
                item_key = str(row.get(item_key_field))
                payload = self._row_payload(row, set())
                rows.append((data_kind, cache_key, item_key, idx, self._json_dumps(payload), now))
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM snapshot_cache WHERE data_kind = ? AND cache_key = ?",
                (data_kind, cache_key),
            )
            if rows:
                conn.executemany(
                    """
                    INSERT INTO snapshot_cache (data_kind, cache_key, item_key, item_order, payload_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        return len(rows)

    def get_covered_dates(
        self,
        data_kind: str,
        cache_key: str,
        date_keys: list[str],
    ) -> set[str]:
        """查询某个缓存键在覆盖区间表中已经声明过的日期。"""
        if not date_keys:
            return set()
        start_key = min(date_keys)
        end_key = max(date_keys)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT start_date, end_date
                FROM coverage_checkpoint
                WHERE data_kind = ? AND cache_key = ? AND NOT (end_date < ? OR start_date > ?)
                """,
                (data_kind, cache_key, start_key, end_key),
            ).fetchall()
        covered = set()
        for row in rows:
            covered.update(
                date_key
                for date_key in date_keys
                if row["start_date"] <= date_key <= row["end_date"]
            )
        return covered

    def mark_coverage(
        self,
        data_kind: str,
        cache_key: str,
        start_date,
        end_date,
        *,
        row_count: int,
        source: str,
    ) -> None:
        """记录某段请求区间已经被本地缓存确认处理过。"""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO coverage_checkpoint (data_kind, cache_key, start_date, end_date, row_count, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(data_kind, cache_key, start_date, end_date) DO UPDATE SET
                    row_count = excluded.row_count,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (
                    data_kind,
                    cache_key,
                    str(start_date),
                    str(end_date),
                    int(row_count),
                    source,
                    self._now(),
                ),
            )

    def mark_snapshot_coverage(
        self,
        data_kind: str,
        cache_key: str,
        *,
        row_count: int,
        source: str,
    ) -> None:
        """记录一次整表快照已经完成缓存。"""
        self.mark_coverage(
            data_kind,
            cache_key,
            cache_key,
            cache_key,
            row_count=row_count,
            source=source,
        )

    def record_fetch_audit(
        self,
        data_kind: str,
        cache_key: str,
        *,
        start_date=None,
        end_date=None,
        source: str,
        row_count: int,
        status: str,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """记录一次在线/本地补数的审计信息。"""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fetch_audit (
                    data_kind, cache_key, start_date, end_date, source, row_count, status, error_type, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data_kind,
                    cache_key,
                    None if start_date is None else str(start_date),
                    None if end_date is None else str(end_date),
                    source,
                    int(row_count),
                    status,
                    error_type,
                    error_message,
                    self._now(),
                ),
            )
