# -*- coding: utf-8 -*-
"""TX1 live advisor 的 runtime universe fast path。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from skyeye.data.bundle_reader import BundleDataReader


DEFAULT_TX1_UNIVERSE_CACHE_ROOT = Path(os.path.expanduser("~/.rqalpha/tx1_live_advisor/universe_cache"))
RUNTIME_UNIVERSE_STRATEGY_ID = "tx1.bundle_volume_median.v1"
BUNDLE_DATA_END_PROBE_ID = "000300.XSHG"


def resolve_runtime_liquid_universe(
    *,
    trade_date,
    universe_size: int = 300,
    cache_root: str | Path | None = None,
    bundle_path: str | Path | None = None,
    min_history_days: int = 500,
) -> dict:
    """解析 live runtime 使用的 liquid top universe，优先命中本地 cache。"""
    trade_ts = pd.Timestamp(trade_date).normalize()
    data_end = _resolve_bundle_data_end(trade_ts, bundle_path=bundle_path)
    cached = load_runtime_liquid_universe_snapshot(
        data_end=data_end,
        universe_size=universe_size,
        cache_root=cache_root,
    )
    if cached is not None:
        return cached

    payload = _build_runtime_liquid_universe_snapshot(
        data_end=data_end,
        universe_size=universe_size,
        bundle_path=bundle_path,
        min_history_days=min_history_days,
    )
    save_runtime_liquid_universe_snapshot(payload, cache_root=cache_root)
    return payload


def load_runtime_liquid_universe_snapshot(
    *,
    data_end,
    universe_size: int,
    cache_root: str | Path | None = None,
) -> dict | None:
    """按 data_end 与 universe_size 读取 universe cache 快照。"""
    cache_path = runtime_liquid_universe_cache_path(
        data_end=data_end,
        universe_size=universe_size,
        cache_root=cache_root,
    )
    if not cache_path.is_file():
        return None
    with cache_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("universe_size", 0)) != int(universe_size):
        return None
    return payload


def save_runtime_liquid_universe_snapshot(
    payload: dict,
    *,
    cache_root: str | Path | None = None,
) -> Path:
    """把 universe cache 快照写入本地磁盘，供后续 runtime 直接复用。"""
    data_end = pd.Timestamp(payload["data_end_date"]).normalize()
    universe_size = int(payload["universe_size"])
    cache_path = runtime_liquid_universe_cache_path(
        data_end=data_end,
        universe_size=universe_size,
        cache_root=cache_root,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    return cache_path


def runtime_liquid_universe_cache_path(
    *,
    data_end,
    universe_size: int,
    cache_root: str | Path | None = None,
) -> Path:
    """根据 data_end 与 universe_size 计算 universe cache 文件路径。"""
    data_end_ts = pd.Timestamp(data_end).normalize()
    root = resolve_runtime_liquid_universe_cache_root(cache_root)
    filename = "liquid_top_{size}_{date}.json".format(
        size=int(universe_size),
        date=data_end_ts.strftime("%Y%m%d"),
    )
    return root / filename


def resolve_runtime_liquid_universe_cache_root(cache_root: str | Path | None = None) -> Path:
    """解析 runtime universe cache 根目录，优先允许环境变量覆盖。"""
    override = cache_root or os.environ.get("TX1_LIVE_UNIVERSE_CACHE_ROOT")
    if override:
        return Path(os.path.expanduser(str(override)))
    return DEFAULT_TX1_UNIVERSE_CACHE_ROOT


def _resolve_bundle_data_end(
    trade_date: pd.Timestamp,
    *,
    bundle_path: str | Path | None = None,
) -> pd.Timestamp:
    """找到不晚于请求日的最新 bundle 行情日，避免把未来交易日历当成实际数据截止日。"""
    bundle_root = Path(bundle_path or BundleDataReader.resolve_bundle_path())
    probe_date = _load_h5_dataset_latest_date(
        bundle_root / "indexes.h5",
        BUNDLE_DATA_END_PROBE_ID,
        trade_date,
    )
    if probe_date is not None:
        return probe_date

    probe_date = _load_h5_dataset_latest_date(
        bundle_root / "stocks.h5",
        "000001.XSHE",
        trade_date,
    )
    if probe_date is not None:
        return probe_date
    raise ValueError("bundle daily bars unavailable for runtime universe")


def _load_h5_dataset_latest_date(
    h5_path: Path,
    dataset_key: str,
    trade_date,
) -> pd.Timestamp | None:
    """读取单个 HDF5 dataset 的最新实际数据日，作为 bundle 数据边界探针。"""
    if not h5_path.is_file():
        return None
    cutoff = int(pd.Timestamp(trade_date).strftime("%Y%m%d"))
    with h5py.File(h5_path, "r") as h5:
        if dataset_key not in h5:
            return None
        dataset = h5[dataset_key][:]
    dtype_names = set(dataset.dtype.names or [])
    if "datetime" not in dtype_names:
        return None
    trade_dates = dataset["datetime"] // 1_000_000
    valid_dates = trade_dates[trade_dates <= cutoff]
    if len(valid_dates) == 0:
        return None
    return pd.Timestamp(str(int(valid_dates.max()))).normalize()


def _build_runtime_liquid_universe_snapshot(
    *,
    data_end: pd.Timestamp,
    universe_size: int,
    bundle_path: str | Path | None = None,
    min_history_days: int = 500,
) -> dict:
    """直接从 bundle 计算 liquid top universe，避免走研究侧重路径。"""
    bundle_root = Path(bundle_path or BundleDataReader.resolve_bundle_path())
    reader = BundleDataReader()
    instruments = reader.get_instruments(type="CS", bundle_path=str(bundle_root))
    if instruments is None or instruments.empty:
        raise ValueError("bundle instruments unavailable for runtime universe")

    active = instruments.copy()
    if "status" in active.columns:
        active = active[active["status"] == "Active"]
    if "exchange" in active.columns:
        active = active[active["exchange"].isin(["XSHG", "XSHE"])]
    order_book_ids = active["order_book_id"].astype(str).tolist()
    cutoff = int(pd.Timestamp(data_end).strftime("%Y%m%d"))
    medians = {}

    stocks_path = bundle_root / "stocks.h5"
    if not stocks_path.is_file():
        raise ValueError("bundle stocks.h5 unavailable for runtime universe")

    with h5py.File(stocks_path, "r") as h5:
        for order_book_id in order_book_ids:
            if order_book_id not in h5:
                continue
            dataset = h5[order_book_id][:]
            dtype_names = set(dataset.dtype.names or [])
            if "datetime" not in dtype_names or "volume" not in dtype_names:
                continue
            trade_dates = dataset["datetime"] // 1_000_000
            valid_mask = trade_dates <= cutoff
            if int(valid_mask.sum()) < int(min_history_days):
                continue
            medians[order_book_id] = float(np.median(dataset["volume"][valid_mask]))

    ranked = sorted(
        medians.items(),
        key=lambda item: item[1],
        reverse=True,
    )[: int(universe_size)]
    return {
        "version": 1,
        "strategy_id": RUNTIME_UNIVERSE_STRATEGY_ID,
        "source": "bundle_fast_path",
        "data_end_date": pd.Timestamp(data_end).strftime("%Y-%m-%d"),
        "universe_size": int(universe_size),
        "min_history_days": int(min_history_days),
        "order_book_ids": [order_book_id for order_book_id, _ in ranked],
    }
