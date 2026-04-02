# -*- coding: utf-8 -*-

import os
import re
import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd

from rqalpha.data.data_proxy import DataProxy
from rqalpha.data.base_data_source.data_source import BaseDataSource
from rqalpha.utils import RqAttrDict

from skyeye.data import DataFacade
from skyeye.data.provider import RQDataProvider
from skyeye.products.dividend_low_vol.scorer.config import (
    CACHE_DB_PATH,
    CACHE_EXPIRED_DAYS,
    CACHE_STALE_DAYS,
    DATA_GAP_THRESHOLD,
    ETF_CODE,
    INDEX_CODE,
    SYNC_DEFAULT_START_DATE,
    SYNC_OVERLAP_TRADING_DAYS,
)

SQLITE_CONNECT_TIMEOUT = 30
SQLITE_BUSY_TIMEOUT_MS = 30000
SQLITE_INIT_RETRIES = 3
SYNC_SOURCE_NAMES = (
    "index_daily",
    "etf_daily",
    "etf_nav",
    "bond_yield",
    "index_weight",
    "stock_indicator",
)
TRADING_DAY_SYNC_SOURCE_NAMES = (
    "index_daily",
    "etf_daily",
    "etf_nav",
    "bond_yield",
    "stock_indicator",
)


CREATE_TABLE_SQL = (
    """
    CREATE TABLE IF NOT EXISTS index_daily (
        date TEXT NOT NULL CHECK (date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
        index_code TEXT NOT NULL,
        close REAL,
        pe_ttm REAL,
        volume REAL,
        amount REAL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (date, index_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS etf_daily (
        date TEXT NOT NULL CHECK (date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
        etf_code TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        close_hfq REAL,
        volume REAL,
        amount REAL,
        turnover_rate REAL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (date, etf_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS etf_nav (
        date TEXT NOT NULL CHECK (date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
        etf_code TEXT NOT NULL,
        nav REAL,
        acc_nav REAL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (date, etf_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bond_yield (
        date TEXT NOT NULL CHECK (date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
        china_10y REAL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stock_indicator (
        date TEXT NOT NULL CHECK (date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
        stock_code TEXT NOT NULL CHECK (stock_code GLOB '[0-9][0-9][0-9][0-9][0-9][0-9]'),
        dv_ttm REAL,
        pe_ttm REAL,
        pb REAL,
        total_mv REAL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (date, stock_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS index_weight (
        index_code TEXT NOT NULL,
        stock_code TEXT NOT NULL CHECK (stock_code GLOB '[0-9][0-9][0-9][0-9][0-9][0-9]'),
        stock_name TEXT,
        weight REAL,
        snapshot_date TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (index_code, stock_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_source_meta (
        source_name TEXT PRIMARY KEY,
        last_update_date TEXT,
        last_fetch_time TEXT,
        record_count INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sync_checkpoint (
        source_name TEXT PRIMARY KEY,
        sync_start_date TEXT,
        sync_end_date TEXT,
        updated_at TEXT NOT NULL
    )
    """,
)

UPSERT_INDEX_DAILY_SQL = """
INSERT INTO index_daily (date, index_code, close, pe_ttm, volume, amount, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(date, index_code) DO UPDATE SET
    close      = excluded.close,
    pe_ttm     = excluded.pe_ttm,
    volume     = excluded.volume,
    amount     = excluded.amount,
    updated_at = excluded.updated_at
"""

UPSERT_INDEX_PE_SQL = """
INSERT INTO index_daily (date, index_code, pe_ttm, updated_at)
VALUES (?, ?, ?, ?)
ON CONFLICT(date, index_code) DO UPDATE SET
    pe_ttm     = COALESCE(excluded.pe_ttm, index_daily.pe_ttm),
    updated_at = excluded.updated_at
"""

UPSERT_ETF_DAILY_SQL = """
INSERT INTO etf_daily (date, etf_code, open, high, low, close, close_hfq, volume, amount, turnover_rate, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(date, etf_code) DO UPDATE SET
    open          = excluded.open,
    high          = excluded.high,
    low           = excluded.low,
    close         = excluded.close,
    close_hfq     = excluded.close_hfq,
    volume        = excluded.volume,
    amount        = excluded.amount,
    turnover_rate = excluded.turnover_rate,
    updated_at    = excluded.updated_at
"""

UPSERT_ETF_NAV_SQL = """
INSERT INTO etf_nav (date, etf_code, nav, acc_nav, updated_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(date, etf_code) DO UPDATE SET
    nav        = excluded.nav,
    acc_nav    = excluded.acc_nav,
    updated_at = excluded.updated_at
"""

UPSERT_BOND_YIELD_SQL = """
INSERT INTO bond_yield (date, china_10y, updated_at)
VALUES (?, ?, ?)
ON CONFLICT(date) DO UPDATE SET
    china_10y  = excluded.china_10y,
    updated_at = excluded.updated_at
"""

UPSERT_STOCK_INDICATOR_SQL = """
INSERT INTO stock_indicator (date, stock_code, dv_ttm, pe_ttm, pb, total_mv, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(date, stock_code) DO UPDATE SET
    dv_ttm     = excluded.dv_ttm,
    pe_ttm     = excluded.pe_ttm,
    pb         = excluded.pb,
    total_mv   = excluded.total_mv,
    updated_at = excluded.updated_at
"""

UPSERT_INDEX_WEIGHT_SQL = """
INSERT INTO index_weight (index_code, stock_code, stock_name, weight, snapshot_date, updated_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(index_code, stock_code) DO UPDATE SET
    stock_name    = excluded.stock_name,
    weight        = excluded.weight,
    snapshot_date = excluded.snapshot_date,
    updated_at    = excluded.updated_at
"""

UPSERT_META_SQL = """
INSERT INTO data_source_meta (source_name, last_update_date, last_fetch_time, record_count)
VALUES (?, ?, ?, ?)
ON CONFLICT(source_name) DO UPDATE SET
    last_update_date = excluded.last_update_date,
    last_fetch_time  = excluded.last_fetch_time,
    record_count     = excluded.record_count
"""

UPSERT_SYNC_CHECKPOINT_SQL = """
INSERT INTO sync_checkpoint (source_name, sync_start_date, sync_end_date, updated_at)
VALUES (?, ?, ?, ?)
ON CONFLICT(source_name) DO UPDATE SET
    sync_start_date = excluded.sync_start_date,
    sync_end_date   = excluded.sync_end_date,
    updated_at      = excluded.updated_at
"""


class DataGapError(RuntimeError):
    pass


class DataFetcher(object):
    def __init__(
        self,
        db_path=None,
        etf_code=ETF_CODE,
        index_code=INDEX_CODE,
        data_proxy=None,
        bundle_path=None,
    ):
        self.db_path = os.path.expanduser(db_path or CACHE_DB_PATH)
        self.etf_code = etf_code
        self.index_code = index_code
        self.data_proxy = data_proxy
        self.bundle_path = os.path.expanduser(bundle_path) if bundle_path else None
        self._bundle_proxy = None
        self.facade = DataFacade()
        self.provider = RQDataProvider()
        self._etf_order_book_id = self._to_order_book_id(etf_code, "XSHG")
        self._index_order_book_id = self._to_order_book_id(index_code, "XSHG")
        self._init_db()

    @staticmethod
    def _to_order_book_id(code, default_exchange):
        if "." in code:
            return code
        return "{}.{}".format(code, default_exchange)

    def sync_all(self, start_date=None, end_date=None, progress=None):
        start_date = self._normalize_date(start_date) if start_date is not None else None
        end_date = self._normalize_date(end_date or datetime.now().date())
        progress = progress or NullSyncProgress()
        progress.start(total_steps=6)
        try:
            with self._connect() as conn:
                index_start, index_end = self._resolve_source_sync_range(conn, "index_daily", start_date, end_date)
                self._run_sync_stage(
                    conn=conn,
                    progress=progress,
                    source_name="index_daily",
                    label="指数估值",
                    start_date=index_start,
                    end_date=index_end,
                    runner=lambda: self._sync_index_daily(conn, index_start, index_end),
                    detail_formatter=lambda rows: self._format_done_detail(conn, "index_daily", rows),
                )

                etf_daily_start, etf_daily_end = self._resolve_source_sync_range(conn, "etf_daily", start_date, end_date)
                self._run_sync_stage(
                    conn=conn,
                    progress=progress,
                    source_name="etf_daily",
                    label="ETF行情",
                    start_date=etf_daily_start,
                    end_date=etf_daily_end,
                    runner=lambda: self._sync_etf_daily(conn, etf_daily_start, etf_daily_end),
                    detail_formatter=lambda rows: self._format_done_detail(conn, "etf_daily", rows),
                )

                etf_nav_start, etf_nav_end = self._resolve_source_sync_range(conn, "etf_nav", start_date, end_date)
                self._run_sync_stage(
                    conn=conn,
                    progress=progress,
                    source_name="etf_nav",
                    label="ETF净值",
                    start_date=etf_nav_start,
                    end_date=etf_nav_end,
                    runner=lambda: self._sync_etf_nav(conn, etf_nav_start, etf_nav_end),
                    detail_formatter=lambda rows: self._format_done_detail(conn, "etf_nav", rows),
                )

                bond_start, bond_end = self._resolve_source_sync_range(conn, "bond_yield", start_date, end_date)
                self._run_sync_stage(
                    conn=conn,
                    progress=progress,
                    source_name="bond_yield",
                    label="国债利率",
                    start_date=bond_start,
                    end_date=bond_end,
                    runner=lambda: self._sync_bond_yield(conn, bond_start, bond_end),
                    detail_formatter=lambda rows: self._format_done_detail(conn, "bond_yield", rows),
                )

                weight_start, weight_end = self._resolve_source_sync_range(conn, "index_weight", start_date, end_date)
                stock_codes = self._run_sync_stage(
                    conn=conn,
                    progress=progress,
                    source_name="index_weight",
                    label="指数权重",
                    start_date=weight_start,
                    end_date=weight_end,
                    runner=lambda: self._sync_index_weights(conn),
                    detail_formatter=lambda codes: self._format_done_detail(conn, "index_weight", len(codes)),
                    skip_value_factory=lambda: self._load_cached_stock_codes(conn),
                )

                stock_start, stock_end = self._resolve_source_sync_range(conn, "stock_indicator", start_date, end_date)
                self._run_sync_stage(
                    conn=conn,
                    progress=progress,
                    source_name="stock_indicator",
                    label="成分股指标",
                    start_date=stock_start,
                    end_date=stock_end,
                    runner=lambda: self._sync_stock_indicators(
                        conn, stock_codes, stock_start, stock_end,
                        name_map=self._load_stock_name_map(conn),
                        progress=progress,
                    ),
                    detail_formatter=lambda summary: self._format_stock_indicator_done_detail(conn, summary),
                )
        finally:
            progress.close()

    def _run_sync_stage(
        self,
        conn,
        progress,
        source_name,
        label,
        start_date,
        end_date,
        runner,
        detail_formatter,
        skip_value_factory=None,
    ):
        progress.start_step(source_name, label)
        try:
            if self._should_skip_sync(conn, source_name, start_date, end_date):
                progress.finish_step("skip", self._format_skip_detail(conn, source_name))
                return skip_value_factory() if skip_value_factory is not None else None
            result = runner()
            self._update_sync_checkpoint(conn, source_name, start_date, end_date)
            progress.finish_step("done", detail_formatter(result))
            return result
        except Exception as exc:
            progress.finish_step("fail", self._format_error_detail(exc))
            raise

    def load_history(self, start_date, end_date):
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)
        with self._connect() as conn:
            etf_daily = self._read_sql(
                conn,
                """
                SELECT date, open, high, low, close, close_hfq, volume, amount, turnover_rate
                FROM etf_daily
                WHERE etf_code = ? AND date BETWEEN ? AND ?
                ORDER BY date
                """,
                (self.etf_code, start_date, end_date),
            )
            if etf_daily.empty:
                raise RuntimeError("etf_daily cache is empty, run sync_all first")

            nav = self._read_sql(
                conn,
                """
                SELECT date, nav, acc_nav
                FROM etf_nav
                WHERE etf_code = ? AND date BETWEEN ? AND ?
                ORDER BY date
                """,
                (self.etf_code, start_date, end_date),
            )
            index_daily = self._read_sql(
                conn,
                """
                SELECT date, close, pe_ttm
                FROM index_daily
                WHERE index_code = ? AND date BETWEEN ? AND ?
                ORDER BY date
                """,
                (self.index_code, start_date, end_date),
            )
            bond_yield = self._read_sql(
                conn,
                """
                SELECT date, china_10y
                FROM bond_yield
                WHERE date BETWEEN ? AND ?
                ORDER BY date
                """,
                (start_date, end_date),
            )

            base = etf_daily.rename(
                columns={
                    "close": "etf_close",
                    "close_hfq": "etf_close_hfq",
                    "volume": "etf_volume",
                    "amount": "etf_amount",
                    "turnover_rate": "etf_turnover_rate",
                }
            )
            base = base.merge(nav.rename(columns={"nav": "etf_nav", "acc_nav": "etf_acc_nav"}), on="date", how="left")
            base = base.merge(index_daily.rename(columns={"pe_ttm": "pe_ttm", "close": "index_close"}), on="date", how="left")
            base = base.merge(bond_yield.rename(columns={"china_10y": "bond_10y"}), on="date", how="left")
            base["dividend_yield"] = self._load_dividend_yield_series(conn, start_date, end_date).reindex(base["date"]).values
            base["bond_10y"] = self._percent_series_to_decimal(base["bond_10y"])
            base["premium_rate"] = np.where(
                base["etf_nav"].notna() & (base["etf_nav"] != 0),
                base["etf_close"] / base["etf_nav"] - 1.0,
                np.nan,
            )
            base["date"] = pd.to_datetime(base["date"])
            base = base.set_index("date").sort_index()
            calendar_proxy = self.data_proxy or self._bundle_data_proxy()
            self.validate_trading_day_coverage(base, start_date, end_date, data_proxy=calendar_proxy)
            return base

    def load_latest(self, date):
        history = self.load_history(date, date)
        if history.empty:
            raise RuntimeError("no cached data for {}".format(date))
        latest = history.iloc[-1].to_dict()
        latest["date"] = history.index[-1].strftime("%Y-%m-%d")
        return latest

    def get_data_freshness(self, reference_date=None):
        reference_date = self._normalize_date(reference_date or datetime.now().date())
        result = {}
        with self._connect() as conn:
            meta = self._read_sql(conn, "SELECT * FROM data_source_meta", ())
        for row in meta.to_dict("records"):
            last_update = row.get("last_update_date")
            stale_days = self._count_trading_days(last_update, reference_date)
            if last_update is None:
                status = "missing"
            elif stale_days > CACHE_EXPIRED_DAYS:
                status = "expired"
            elif stale_days > CACHE_STALE_DAYS:
                status = "stale"
            else:
                status = "fresh"
            result[row["source_name"]] = {
                "last_update_date": last_update,
                "last_fetch_time": row.get("last_fetch_time"),
                "record_count": row.get("record_count"),
                "stale_trading_days": stale_days,
                "status": status,
            }
        return result

    def get_available_range(self):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM etf_daily WHERE etf_code = ?",
                (self.etf_code,),
            ).fetchone()
        if row is None or row["min_date"] is None or row["max_date"] is None:
            raise RuntimeError("etf_daily cache is empty, run sync_all first")
        return row["min_date"], row["max_date"]

    def suggest_sync_range(self, start_date=None, end_date=None):
        end_date = self._latest_trading_day_on_or_before(end_date or datetime.now().date())
        if start_date is not None:
            return self._normalize_date(start_date), end_date
        with self._connect() as conn:
            source_starts = [
                self._resolve_source_sync_range(conn, source_name, None, end_date)[0]
                for source_name in SYNC_SOURCE_NAMES
            ]
        return min(source_starts) if source_starts else SYNC_DEFAULT_START_DATE, end_date

    @staticmethod
    def validate_trading_day_coverage(df, start_date, end_date, gap_threshold=DATA_GAP_THRESHOLD, data_proxy=None):
        if df.empty:
            raise DataGapError("history dataframe is empty")
        if data_proxy is None:
            return
        expected_dates = DataFetcher._get_trading_dates(
            start_date=start_date,
            end_date=end_date,
            data_proxy=data_proxy,
        )
        if len(expected_dates) == 0:
            return
        observed_dates = pd.DatetimeIndex(df.index).normalize().unique()
        missing_dates = expected_dates.difference(observed_dates)
        gap_rate = float(len(missing_dates)) / float(len(expected_dates))
        if gap_rate > gap_threshold:
            raise DataGapError(
                "missing {} / {} trading days ({:.2%}): {}".format(
                    len(missing_dates),
                    len(expected_dates),
                    gap_rate,
                    ", ".join(d.strftime("%Y-%m-%d") for d in missing_dates[:20]),
                )
            )
        if len(missing_dates) > 0:
            warnings = list(df.attrs.get("warnings", []))
            warnings.append(
                "missing_trading_days: {} / {} ({:.2%})".format(
                    len(missing_dates), len(expected_dates), gap_rate
                )
            )
            df.attrs["warnings"] = warnings

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=SQLITE_CONNECT_TIMEOUT)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = {}".format(SQLITE_BUSY_TIMEOUT_MS))
        return conn

    def _resolve_source_sync_range(self, conn, source_name, start_date, end_date):
        end_date = self._resolve_source_sync_end_date(source_name, end_date or datetime.now().date())
        if start_date is not None:
            return self._normalize_date(start_date), end_date
        if not self._has_cached_source_data(conn, source_name):
            return SYNC_DEFAULT_START_DATE, end_date
        actual_start, actual_end = self._derive_source_coverage_range(conn, source_name, None, None)
        checkpoint = self._read_sync_checkpoint(conn, source_name)
        coverage_end = actual_end
        if coverage_end is None and checkpoint is not None:
            coverage_end = checkpoint.get("sync_end_date")
        if coverage_end is None:
            return SYNC_DEFAULT_START_DATE, end_date
        overlap_start = self._shift_trading_days(coverage_end, -SYNC_OVERLAP_TRADING_DAYS)
        return max(overlap_start, SYNC_DEFAULT_START_DATE), end_date

    def _resolve_source_sync_end_date(self, source_name, end_date):
        end_date = self._normalize_date(end_date)
        if source_name not in TRADING_DAY_SYNC_SOURCE_NAMES:
            return end_date
        return self._latest_trading_day_on_or_before(end_date)

    def _latest_trading_day_on_or_before(self, date_value):
        end_ts = pd.Timestamp(self._normalize_date(date_value))
        data_proxy = self.data_proxy or self._bundle_data_proxy()
        window_start = end_ts - pd.Timedelta(days=32)
        trading_dates = self._get_trading_dates(window_start, end_ts, data_proxy=data_proxy)
        if len(trading_dates) == 0:
            if end_ts.weekday() >= 5:
                return pd.Timestamp(end_ts + pd.tseries.offsets.BDay(-1)).strftime("%Y-%m-%d")
            return end_ts.strftime("%Y-%m-%d")
        return pd.Timestamp(trading_dates[-1]).strftime("%Y-%m-%d")

    def _shift_trading_days(self, date_value, offset):
        base = pd.Timestamp(self._normalize_date(date_value))
        if offset == 0:
            return base.strftime("%Y-%m-%d")

        data_proxy = self.data_proxy or self._bundle_data_proxy()
        if data_proxy is not None:
            lookback_days = max(abs(offset) * 5, 30)
            if offset < 0:
                window_start = base - pd.Timedelta(days=lookback_days)
                trading_dates = self._get_trading_dates(window_start, base, data_proxy=data_proxy)
                if len(trading_dates) > abs(offset):
                    return pd.Timestamp(trading_dates[-(abs(offset) + 1)]).strftime("%Y-%m-%d")
                if len(trading_dates) > 0:
                    return pd.Timestamp(trading_dates[0]).strftime("%Y-%m-%d")
            else:
                window_end = base + pd.Timedelta(days=lookback_days)
                trading_dates = self._get_trading_dates(base, window_end, data_proxy=data_proxy)
                if len(trading_dates) > offset:
                    return pd.Timestamp(trading_dates[offset]).strftime("%Y-%m-%d")
                if len(trading_dates) > 0:
                    return pd.Timestamp(trading_dates[-1]).strftime("%Y-%m-%d")

        shifted = base + pd.tseries.offsets.BDay(offset)
        return pd.Timestamp(shifted).strftime("%Y-%m-%d")

    def _init_db(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        # Remove stale WAL/SHM files: empty WAL + non-empty SHM = crashed previous session
        wal_path = self.db_path + "-wal"
        shm_path = self.db_path + "-shm"
        wal_stale = os.path.exists(wal_path) and os.path.getsize(wal_path) == 0
        shm_exists = os.path.exists(shm_path)
        if wal_stale and shm_exists:
            os.remove(wal_path)
            os.remove(shm_path)
        last_exc = None
        for attempt in range(1, SQLITE_INIT_RETRIES + 1):
            try:
                with self._connect() as conn:
                    conn.execute("PRAGMA journal_mode = WAL")
                    for sql in CREATE_TABLE_SQL:
                        conn.execute(sql)
                return
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if "locked" not in str(exc).lower() or attempt >= SQLITE_INIT_RETRIES:
                    break
                time.sleep(float(attempt))
        if last_exc is not None and "locked" in str(last_exc).lower():
            raise RuntimeError(
                "cache db is locked: {}. another dividend_scorer sync process is probably still running; stop the old process and retry".format(
                    self.db_path
                )
            ) from last_exc
        if last_exc is not None:
            raise last_exc

    def _sync_index_daily(self, conn, start_date, end_date):
        inserted_rows = 0
        now = datetime.now().isoformat()

        # Index OHLCV from rqdatac
        df = self.provider.get_index_price(
            self._index_order_book_id, start_date, end_date,
            fields=["close", "volume", "total_turnover"],
        )
        if df is not None and not df.empty:
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            else:
                df = df.reset_index().rename(columns={"index": "date"})
            if "date" not in df.columns:
                for col in df.columns:
                    if "date" in str(col).lower() or "trading_date" in str(col).lower():
                        df = df.rename(columns={col: "date"})
                        break
            rows = [
                (
                    self._normalize_date(row.get("date")),
                    self.index_code,
                    self._maybe_float(row.get("close")),
                    None,  # pe_ttm filled below
                    self._maybe_float(row.get("volume")),
                    self._maybe_float(row.get("total_turnover")),
                    now,
                )
                for _, row in df.iterrows()
            ]
            if rows:
                conn.executemany(UPSERT_INDEX_DAILY_SQL, rows)
                inserted_rows += len(rows)

        # Index valuation (PE) from rqdatac
        valuation_df = self.provider.get_index_indicator(
            self._index_order_book_id, start_date, end_date,
        )
        if valuation_df is not None and not valuation_df.empty:
            if isinstance(valuation_df.index, pd.MultiIndex):
                valuation_df = valuation_df.reset_index()
            else:
                valuation_df = valuation_df.reset_index().rename(columns={"index": "date"})
            if "date" not in valuation_df.columns:
                for col in valuation_df.columns:
                    if "date" in str(col).lower():
                        valuation_df = valuation_df.rename(columns={col: "date"})
                        break
            pe_col = self._find_column(valuation_df, ("pe_ttm", "pe_ratio_ttm", "pe"))
            if pe_col is not None:
                pe_rows = [
                    (
                        self._normalize_date(row.get("date")),
                        self.index_code,
                        self._maybe_float(row.get(pe_col)),
                        now,
                    )
                    for _, row in valuation_df.iterrows()
                ]
                if pe_rows:
                    conn.executemany(UPSERT_INDEX_PE_SQL, pe_rows)
                    inserted_rows += len(pe_rows)

        if inserted_rows == 0:
            return 0
        self._update_meta(conn, "index_daily", "index_daily", self.index_code)
        return inserted_rows

    def _sync_etf_daily(self, conn, start_date, end_date):
        # Raw (unadjusted) ETF price
        raw_df = self.provider.get_price(
            self._etf_order_book_id, start_date, end_date,
            frequency="1d", adjust_type="none",
        )
        if raw_df is None or raw_df.empty:
            return 0

        # Post-adjusted close for close_hfq
        hfq_df = self.provider.get_price(
            self._etf_order_book_id, start_date, end_date,
            frequency="1d", adjust_type="post", fields=["close"],
        )
        hfq_close_map = {}
        if hfq_df is not None and not hfq_df.empty:
            hfq_reset = hfq_df.reset_index()
            if "date" not in hfq_reset.columns:
                for col in hfq_reset.columns:
                    if "date" in str(col).lower():
                        hfq_reset = hfq_reset.rename(columns={col: "date"})
                        break
            for _, row in hfq_reset.iterrows():
                hfq_close_map[self._normalize_date(row.get("date"))] = self._maybe_float(row.get("close"))

        raw_reset = raw_df.reset_index()
        if "date" not in raw_reset.columns:
            for col in raw_reset.columns:
                if "date" in str(col).lower():
                    raw_reset = raw_reset.rename(columns={col: "date"})
                    break

        now = datetime.now().isoformat()
        rows = []
        for _, row in raw_reset.iterrows():
            date_value = self._normalize_date(row.get("date"))
            rows.append((
                date_value,
                self.etf_code,
                self._maybe_float(row.get("open")),
                self._maybe_float(row.get("high")),
                self._maybe_float(row.get("low")),
                self._maybe_float(row.get("close")),
                hfq_close_map.get(date_value),
                self._maybe_float(row.get("volume")),
                self._maybe_float(row.get("total_turnover")),
                None,  # turnover_rate not in get_price; will be filled if needed
                now,
            ))
        conn.executemany(UPSERT_ETF_DAILY_SQL, rows)
        self._update_meta(conn, "etf_daily", "etf_daily", self.etf_code)
        return len(rows)

    def _sync_etf_nav(self, conn, start_date, end_date, progress=None):
        # rqdatac ETF price includes 'iopv' field as NAV proxy
        df = self.provider.get_price(
            self._etf_order_book_id, start_date, end_date,
            frequency="1d", adjust_type="none", fields=["iopv", "close"],
        )
        if df is None or df.empty:
            return 0
        df = df.reset_index()
        if "date" not in df.columns:
            for col in df.columns:
                if "date" in str(col).lower():
                    df = df.rename(columns={col: "date"})
                    break
        now = datetime.now().isoformat()
        rows = [
            (
                self._normalize_date(row.get("date")),
                self.etf_code,
                self._maybe_float(row.get("iopv")),
                None,  # acc_nav not available from rqdatac
                now,
            )
            for _, row in df.iterrows()
        ]
        conn.executemany(UPSERT_ETF_NAV_SQL, rows)
        self._update_meta(conn, "etf_nav", "etf_nav", self.etf_code)
        return len(rows)

    def _sync_bond_yield(self, conn, start_date, end_date):
        df = self.provider.get_bond_yield(start_date, end_date, tenor="10Y")
        if df is None or df.empty:
            return 0
        df = df.reset_index()
        if "date" not in df.columns:
            for col in df.columns:
                if "date" in str(col).lower() or "trading" in str(col).lower():
                    df = df.rename(columns={col: "date"})
                    break
        yield_col = self._find_column(df, ("10Y", "china_10y"))
        if yield_col is None:
            return 0
        now = datetime.now().isoformat()
        rows = [
            (
                self._normalize_date(row.get("date")),
                # rqdatac returns decimal (0.025 = 2.5%), convert to percentage for SQLite schema
                self._maybe_float(row.get(yield_col)) * 100.0 if self._maybe_float(row.get(yield_col)) is not np.nan else np.nan,
                now,
            )
            for _, row in df.iterrows()
        ]
        conn.executemany(UPSERT_BOND_YIELD_SQL, rows)
        self._update_meta(conn, "bond_yield", "bond_yield", None)
        return len(rows)

    def _sync_index_weights(self, conn):
        weights = self.provider.get_index_weights(self._index_order_book_id)
        if weights is None or weights.empty:
            return []
        # weights is a Series: index=order_book_id, value=weight
        snapshot_date = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().isoformat()
        rows = []
        stock_codes = []
        # Get instrument names in batch
        instruments = self.provider.get_instruments(type="CS")
        name_map = {}
        if instruments is not None and not instruments.empty:
            if "order_book_id" in instruments.columns and "symbol" in instruments.columns:
                name_map = dict(zip(instruments["order_book_id"], instruments["symbol"]))

        for order_book_id, weight in weights.items():
            stock_code = self._normalize_stock_code(order_book_id)
            if stock_code is None:
                continue
            stock_codes.append(stock_code)
            stock_name = name_map.get(order_book_id, "")
            rows.append((
                self.index_code,
                stock_code,
                str(stock_name),
                self._maybe_float(weight),
                snapshot_date,
                now,
            ))
        conn.executemany(UPSERT_INDEX_WEIGHT_SQL, rows)
        self._update_meta(conn, "index_weight", "index_weight", self.index_code)
        return stock_codes

    def _sync_stock_indicators(self, conn, stock_codes, start_date, end_date, name_map=None, progress=None):
        if not stock_codes:
            return {
                "total_stocks": 0,
                "synced_stocks": 0,
                "empty_fetches": 0,
                "out_of_range": 0,
                "invalid_rows": 0,
                "rows": 0,
            }
        # Convert 6-digit codes to order_book_id format for rqdatac
        code_to_obid = {}
        for code in stock_codes:
            if code.startswith(("6", "9", "5")):
                code_to_obid[code] = "{}.XSHG".format(code)
            else:
                code_to_obid[code] = "{}.XSHE".format(code)
        order_book_ids = list(code_to_obid.values())

        if progress is not None:
            progress.update_step(current=0, total=len(stock_codes), detail="批量获取因子数据...")

        # Single batch call replaces 300+ serial AKShare calls
        factors_df = self.provider.get_factors(
            order_book_ids,
            ["pe_ratio_ttm", "pb_ratio", "dividend_yield_ttm", "market_cap"],
            start_date, end_date,
        )
        if factors_df is None or factors_df.empty:
            raise RuntimeError("stock_indicator sync produced no rows from rqdatac get_factor")

        now = datetime.now().isoformat()
        savepoint_name = "stock_indicator_sync"
        conn.execute("SAVEPOINT {}".format(savepoint_name))
        try:
            # factors_df has MultiIndex (order_book_id, date)
            factors_df = factors_df.reset_index()
            inserted_rows = 0
            synced_stocks = set()
            total = len(stock_codes)

            for _, row in factors_df.iterrows():
                obid = row.get("order_book_id", "")
                stock_code = self._normalize_stock_code(obid)
                if stock_code is None or stock_code not in code_to_obid:
                    continue
                date_value = self._normalize_date(row.get("date"))
                if date_value is None:
                    continue
                conn.execute(
                    UPSERT_STOCK_INDICATOR_SQL,
                    (
                        date_value,
                        stock_code,
                        self._maybe_float(row.get("dividend_yield_ttm")),
                        self._maybe_float(row.get("pe_ratio_ttm")),
                        self._maybe_float(row.get("pb_ratio")),
                        self._maybe_float(row.get("market_cap")),
                        now,
                    ),
                )
                inserted_rows += 1
                synced_stocks.add(stock_code)

            conn.execute("RELEASE SAVEPOINT {}".format(savepoint_name))
        except Exception:
            conn.execute("ROLLBACK TO SAVEPOINT {}".format(savepoint_name))
            conn.execute("RELEASE SAVEPOINT {}".format(savepoint_name))
            raise

        if progress is not None:
            progress.update_step(current=total, total=total, detail="批量写入完成 +{}行".format(inserted_rows))

        if inserted_rows == 0:
            raise RuntimeError("stock_indicator sync produced no rows")
        self._update_meta(conn, "stock_indicator", "stock_indicator", None)
        return {
            "total_stocks": total,
            "synced_stocks": len(synced_stocks),
            "empty_fetches": total - len(synced_stocks),
            "out_of_range": 0,
            "invalid_rows": 0,
            "rows": inserted_rows,
        }

    def _load_dividend_yield_series(self, conn, start_date, end_date):
        weights = self._read_sql(
            conn,
            """
            SELECT stock_code, weight
            FROM index_weight
            WHERE index_code = ?
            """,
            (self.index_code,),
        )
        indicators = self._read_sql(
            conn,
            """
            SELECT date, stock_code, dv_ttm
            FROM stock_indicator
            WHERE date BETWEEN ? AND ?
            """,
            (start_date, end_date),
        )
        if weights.empty or indicators.empty:
            return pd.Series(dtype="float64")
        weights["stock_code"] = weights["stock_code"].map(self._normalize_stock_code)
        indicators["stock_code"] = indicators["stock_code"].map(self._normalize_stock_code)
        indicators["dv_ttm"] = self._percent_series_to_decimal(indicators["dv_ttm"])
        merged = indicators.merge(weights, on="stock_code", how="inner")
        merged = merged.dropna(subset=["dv_ttm", "weight"])
        if merged.empty:
            return pd.Series(dtype="float64")
        merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce")
        merged["weighted_dividend"] = merged["dv_ttm"] * merged["weight"]
        grouped = merged.groupby("date")
        dividend_yield = grouped["weighted_dividend"].sum() / grouped["weight"].sum().replace(0, np.nan)
        dividend_yield.index = dividend_yield.index.map(self._normalize_date)
        dividend_yield.name = "dividend_yield"
        return dividend_yield

    def _update_meta(self, conn, source_name, table_name, code_filter):
        now = datetime.now().isoformat()
        if table_name == "index_daily":
            row = conn.execute(
                "SELECT MAX(date) AS last_update_date, COUNT(*) AS record_count FROM index_daily WHERE index_code = ?",
                (code_filter,),
            ).fetchone()
        elif table_name == "etf_daily":
            row = conn.execute(
                "SELECT MAX(date) AS last_update_date, COUNT(*) AS record_count FROM etf_daily WHERE etf_code = ?",
                (code_filter,),
            ).fetchone()
        elif table_name == "etf_nav":
            row = conn.execute(
                "SELECT MAX(date) AS last_update_date, COUNT(*) AS record_count FROM etf_nav WHERE etf_code = ?",
                (code_filter,),
            ).fetchone()
        elif table_name == "index_weight":
            row = conn.execute(
                "SELECT MAX(snapshot_date) AS last_update_date, COUNT(*) AS record_count FROM index_weight WHERE index_code = ?",
                (code_filter,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT MAX(date) AS last_update_date, COUNT(*) AS record_count FROM {}".format(table_name)
            ).fetchone()
        conn.execute(
            UPSERT_META_SQL,
            (source_name, row["last_update_date"], now, row["record_count"]),
        )

    def _bundle_data_proxy(self):
        if self._bundle_proxy is not None:
            return self._bundle_proxy
        bundle_path = self.bundle_path or os.path.expanduser("~/.rqalpha/bundle")
        if not os.path.exists(bundle_path):
            return None
        base_config = RqAttrDict({
            "data_bundle_path": bundle_path,
            "future_info": {},
        })
        self._bundle_proxy = DataProxy(BaseDataSource(base_config), None)
        return self._bundle_proxy

    @classmethod
    def _get_trading_dates(cls, start_date, end_date, data_proxy=None):
        try:
            dates = DataFacade().get_trading_dates(start_date, end_date)
            if dates is not None and len(dates) > 0:
                return pd.DatetimeIndex(dates).normalize()
        except Exception:
            pass
        if data_proxy is not None:
            return data_proxy.get_trading_dates(start_date, end_date)
        return pd.bdate_range(start=start_date, end=end_date)

    def _count_trading_days(self, start_date, end_date):
        if start_date is None:
            return None
        if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
            return 0
        trading_dates = self._get_trading_dates(start_date, end_date, data_proxy=self.data_proxy or self._bundle_data_proxy())
        if len(trading_dates) == 0:
            return 0
        return max(len(trading_dates) - 1, 0)

    @staticmethod
    def _read_sql(conn, sql, params):
        return pd.read_sql_query(sql, conn, params=params)

    @staticmethod
    def _normalize_date(value):
        if value is None:
            return None
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    @staticmethod
    def _normalize_stock_code(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        matched = re.search(r"(\d{6})", str(value))
        return matched.group(1) if matched else None

    @staticmethod
    def _find_column(df, candidates):
        if df is None or df.empty:
            return None
        normalized = {}
        for col in df.columns:
            key = re.sub(r"[\s_]+", "", str(col)).lower()
            normalized[key] = col
        for candidate in candidates:
            key = re.sub(r"[\s_]+", "", str(candidate)).lower()
            if key in normalized:
                return normalized[key]
        for candidate in candidates:
            for col in df.columns:
                if str(candidate).lower() in str(col).lower():
                    return col
        return None

    @staticmethod
    def _filter_date_range(df, start_date, end_date):
        if df is None or df.empty:
            return pd.DataFrame()
        date_col = DataFetcher._find_column(df, ("date", "日期", "trade_date", "净值日期", "调整日期"))
        if date_col is None:
            return df
        normalized_dates = pd.to_datetime(df[date_col], errors="coerce")
        mask = (normalized_dates >= pd.Timestamp(start_date)) & (normalized_dates <= pd.Timestamp(end_date))
        return df.loc[mask].copy()

    @staticmethod
    def _maybe_float(value):
        if value is None:
            return np.nan
        try:
            if pd.isna(value):
                return np.nan
        except TypeError:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    @staticmethod
    def _percent_series_to_decimal(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.dropna().empty:
            return numeric
        if numeric.dropna().abs().median() > 1.0:
            return numeric / 100.0
        return numeric

    def _should_skip_sync(self, conn, source_name, start_date, end_date):
        if not self._has_cached_source_data(conn, source_name):
            return False
        return self._checkpoint_covers_range(conn, source_name, start_date, end_date)

    def _has_cached_source_data(self, conn, source_name):
        if source_name == "index_daily":
            row = conn.execute(
                "SELECT 1 FROM index_daily WHERE index_code = ? LIMIT 1",
                (self.index_code,),
            ).fetchone()
        elif source_name == "etf_daily":
            row = conn.execute(
                "SELECT 1 FROM etf_daily WHERE etf_code = ? LIMIT 1",
                (self.etf_code,),
            ).fetchone()
        elif source_name == "etf_nav":
            row = conn.execute(
                "SELECT 1 FROM etf_nav WHERE etf_code = ? LIMIT 1",
                (self.etf_code,),
            ).fetchone()
        elif source_name == "bond_yield":
            row = conn.execute("SELECT 1 FROM bond_yield LIMIT 1").fetchone()
        elif source_name == "index_weight":
            row = conn.execute(
                "SELECT 1 FROM index_weight WHERE index_code = ? LIMIT 1",
                (self.index_code,),
            ).fetchone()
        elif source_name == "stock_indicator":
            row = conn.execute("SELECT 1 FROM stock_indicator LIMIT 1").fetchone()
        else:
            row = None
        return row is not None

    def _checkpoint_covers_range(self, conn, source_name, start_date, end_date):
        row = self._read_sync_checkpoint(conn, source_name)
        if row is None or row["sync_start_date"] is None or row["sync_end_date"] is None:
            return False
        if not (row["sync_start_date"] <= start_date and row["sync_end_date"] >= end_date):
            return False
        actual_start, actual_end = self._derive_source_coverage_range(conn, source_name, None, None)
        if actual_start is None or actual_end is None:
            return False
        return actual_start <= start_date and actual_end >= end_date

    def _update_sync_checkpoint(self, conn, source_name, start_date, end_date):
        actual_start, actual_end = self._derive_source_coverage_range(conn, source_name, start_date, end_date)
        conn.execute(
            UPSERT_SYNC_CHECKPOINT_SQL,
            (source_name, actual_start, actual_end, datetime.now().isoformat()),
        )

    def _read_sync_checkpoint(self, conn, source_name):
        row = conn.execute(
            """
            SELECT sync_start_date, sync_end_date
            FROM sync_checkpoint
            WHERE source_name = ?
            """,
            (source_name,),
        ).fetchone()
        return dict(row) if row is not None else None

    def _derive_source_coverage_range(self, conn, source_name, start_date, end_date):
        query = None
        params = ()
        if source_name == "index_daily":
            query = "SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM index_daily WHERE index_code = ?"
            params = (self.index_code,)
        elif source_name == "etf_daily":
            query = "SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM etf_daily WHERE etf_code = ?"
            params = (self.etf_code,)
        elif source_name == "etf_nav":
            query = "SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM etf_nav WHERE etf_code = ?"
            params = (self.etf_code,)
        elif source_name == "bond_yield":
            query = "SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM bond_yield"
        elif source_name == "stock_indicator":
            query = "SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM stock_indicator"
        elif source_name == "index_weight":
            query = "SELECT MIN(snapshot_date) AS min_date, MAX(snapshot_date) AS max_date FROM index_weight WHERE index_code = ?"
            params = (self.index_code,)
        if query is None:
            return start_date, end_date
        row = conn.execute(query, params).fetchone()
        if row is None or row["min_date"] is None or row["max_date"] is None:
            return None, None
        return row["min_date"], row["max_date"]

    def _load_cached_stock_codes(self, conn):
        rows = conn.execute(
            """
            SELECT stock_code
            FROM index_weight
            WHERE index_code = ?
            ORDER BY stock_code
            """,
            (self.index_code,),
        ).fetchall()
        return [row["stock_code"] for row in rows]

    def _load_stock_name_map(self, conn):
        rows = conn.execute(
            """
            SELECT stock_code, stock_name
            FROM index_weight
            WHERE index_code = ?
            """,
            (self.index_code,),
        ).fetchall()
        return {row["stock_code"]: row["stock_name"] for row in rows}

    def _format_done_detail(self, conn, source_name, rows_inserted):
        meta = self._read_source_meta(conn, source_name)
        if meta is None:
            return "new={:,}".format(int(rows_inserted))
        cache_size = meta.get("record_count", "-")
        cache_repr = "{:,}".format(int(cache_size)) if isinstance(cache_size, (int, float)) else cache_size
        return "new={} | cache={} | last={}".format(
            "{:,}".format(int(rows_inserted)),
            cache_repr,
            meta.get("last_update_date") or "-",
        )

    def _format_stock_indicator_done_detail(self, conn, summary):
        meta = self._read_source_meta(conn, "stock_indicator")
        cache_size = meta.get("record_count", "-") if meta is not None else "-"
        cache_repr = "{:,}".format(int(cache_size)) if isinstance(cache_size, (int, float)) else cache_size
        parts = [
            "stocks={}/{}".format(summary.get("synced_stocks", 0), summary.get("total_stocks", 0)),
        ]
        empty = summary.get("empty_fetches", 0)
        out = summary.get("out_of_range", 0) + summary.get("invalid_rows", 0)
        if empty > 0:
            parts.append("empty={}".format(empty))
        if out > 0:
            parts.append("out={}".format(out))
        parts.extend([
            "new={}".format("{:,}".format(int(summary.get("rows", 0)))),
            "cache={}".format(cache_repr),
            "last={}".format(meta.get("last_update_date") if meta else "-"),
        ])
        return " | ".join(parts)

    def _format_skip_detail(self, conn, source_name):
        checkpoint = conn.execute(
            """
            SELECT sync_start_date, sync_end_date
            FROM sync_checkpoint
            WHERE source_name = ?
            """,
            (source_name,),
        ).fetchone()
        meta = self._read_source_meta(conn, source_name)
        if checkpoint is None:
            range_repr = "- -> -"
        else:
            range_repr = "{} -> {}".format(
                checkpoint["sync_start_date"] or "-",
                checkpoint["sync_end_date"] or "-",
            )
        return "covered {} | last={}".format(range_repr, meta.get("last_update_date") if meta else "-")

    @staticmethod
    def _format_error_detail(exc, limit=120):
        text = "{}: {}".format(type(exc).__name__, exc)
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _read_source_meta(self, conn, source_name):
        row = conn.execute(
            """
            SELECT source_name, last_update_date, last_fetch_time, record_count
            FROM data_source_meta
            WHERE source_name = ?
            """,
            (source_name,),
        ).fetchone()
        return dict(row) if row is not None else None


class NullSyncProgress(object):
    def banner(self, title, start_date, end_date, db_path):
        pass

    def start(self, total_steps):
        pass

    def start_step(self, source_name, label):
        pass

    def update_step(self, current=None, total=None, detail=None, stats=None):
        pass

    def finish_step(self, status, detail=None):
        pass

    def close(self):
        pass
