# -*- coding: utf-8 -*-
"""RQAlpha bundle 的只读访问封装。"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import h5py
import numpy as np
import pandas as pd


class BundleDataReader:
    """读取本地 RQAlpha bundle，作为数据访问链路的第一层基线缓存。"""

    @staticmethod
    def resolve_bundle_path() -> str:
        """解析 bundle 根目录，优先允许环境变量覆盖。"""
        override = os.environ.get("RQALPHA_BUNDLE_PATH") or os.environ.get("SKYEYE_BUNDLE_PATH")
        if override:
            return os.path.expanduser(override)
        return os.path.expanduser("~/.rqalpha/bundle")

    def get_trading_dates(
        self,
        start_date,
        end_date,
        *,
        bundle_path: Optional[str] = None,
    ) -> list[pd.Timestamp]:
        """读取 bundle 交易日历。"""
        path = os.path.join(bundle_path or self.resolve_bundle_path(), "trading_dates.npy")
        if not os.path.exists(path):
            return []
        arr = np.load(path)
        dates = [pd.Timestamp(str(int(value))) for value in arr]
        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        return [date for date in dates if start_ts <= date <= end_ts]

    def get_instruments(
        self,
        *,
        type: Optional[str] = None,
        bundle_path: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """读取 bundle 中的 instruments 快照。"""
        path = os.path.join(bundle_path or self.resolve_bundle_path(), "instruments.pk")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as handle:
            rows = pickle.load(handle)
        frame = pd.DataFrame(rows)
        if type:
            frame = frame[frame.get("type") == type]
        return frame.reset_index(drop=True)

    def get_daily_bars(
        self,
        order_book_id: str,
        start_date,
        end_date,
        *,
        bundle_path: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """读取 bundle 日线，并统一成按日期索引的表结构。"""
        base = bundle_path or self.resolve_bundle_path()
        for name in ("stocks.h5", "indexes.h5", "funds.h5"):
            path = os.path.join(base, name)
            if not os.path.exists(path):
                continue
            with h5py.File(path, "r") as h5:
                if order_book_id not in h5:
                    continue
                frame = pd.DataFrame(h5[order_book_id][:])
            if "datetime" in frame.columns:
                if np.issubdtype(frame["datetime"].dtype, np.integer):
                    date_series = frame["datetime"].astype(str).str[:8]
                    frame["date"] = pd.to_datetime(date_series, format="%Y%m%d")
                else:
                    frame["date"] = pd.to_datetime(frame["datetime"]).dt.normalize()
            elif "date" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            else:
                return None
            frame = frame.set_index("date").sort_index()
            start_ts = pd.Timestamp(start_date).normalize()
            end_ts = pd.Timestamp(end_date).normalize()
            return frame.loc[str(start_ts):str(end_ts)]
        return None
