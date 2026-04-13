import pickle

import h5py
import numpy as np
import pandas as pd

from skyeye.products.tx1.live_advisor import universe as runtime_universe


def _write_bundle_daily_dataset(group, order_book_id, dates, volumes):
    """向临时 bundle 写入最小日线数据集，只保留 fast path 需要的字段。"""
    dtype = np.dtype(
        [
            ("datetime", "<i8"),
            ("volume", "<f8"),
        ]
    )
    values = np.zeros(len(dates), dtype=dtype)
    values["datetime"] = [int(pd.Timestamp(date).strftime("%Y%m%d")) * 1_000_000 for date in dates]
    values["volume"] = volumes
    group.create_dataset(order_book_id, data=values)


def test_resolve_runtime_liquid_universe_builds_and_hits_cache(tmp_path, monkeypatch):
    """验证 runtime fast path 会从 bundle 构建候选池，并在二次调用时命中 cache。"""
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    cache_root = tmp_path / "cache"
    dates = np.array([20260105, 20260106, 20260107, 20260108, 20260109], dtype=np.int64)
    np.save(bundle_root / "trading_dates.npy", dates)
    instruments = [
        {"order_book_id": "A.XSHE", "type": "CS", "status": "Active", "exchange": "XSHE"},
        {"order_book_id": "B.XSHE", "type": "CS", "status": "Active", "exchange": "XSHE"},
        {"order_book_id": "C.XSHE", "type": "CS", "status": "Active", "exchange": "XSHE"},
        {"order_book_id": "D.XSHE", "type": "CS", "status": "Delisted", "exchange": "XSHE"},
    ]
    with open(bundle_root / "instruments.pk", "wb") as handle:
        pickle.dump(instruments, handle)

    with h5py.File(bundle_root / "stocks.h5", "w") as h5:
        bundle_dates = [np.datetime64("2026-01-05"), np.datetime64("2026-01-06"), np.datetime64("2026-01-07")]
        _write_bundle_daily_dataset(h5, "A.XSHE", bundle_dates, [100.0, 100.0, 100.0])
        _write_bundle_daily_dataset(h5, "B.XSHE", bundle_dates, [300.0, 300.0, 300.0])
        _write_bundle_daily_dataset(h5, "C.XSHE", bundle_dates, [200.0, 200.0, 200.0])
    with h5py.File(bundle_root / "indexes.h5", "w") as h5:
        _write_bundle_daily_dataset(h5, "000300.XSHG", bundle_dates, [1.0, 1.0, 1.0])

    payload = runtime_universe.resolve_runtime_liquid_universe(
        trade_date="2026-01-10",
        universe_size=2,
        cache_root=cache_root,
        bundle_path=bundle_root,
        min_history_days=3,
    )

    assert payload["order_book_ids"] == ["B.XSHE", "C.XSHE"]
    assert payload["data_end_date"] == "2026-01-07"
    assert payload["source"] == "bundle_fast_path"

    monkeypatch.setattr(
        runtime_universe,
        "_build_runtime_liquid_universe_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should hit cache")),
    )

    cached = runtime_universe.resolve_runtime_liquid_universe(
        trade_date="2026-01-10",
        universe_size=2,
        cache_root=cache_root,
        bundle_path=bundle_root,
        min_history_days=3,
    )

    assert cached["order_book_ids"] == ["B.XSHE", "C.XSHE"]
    assert cached["data_end_date"] == "2026-01-07"
