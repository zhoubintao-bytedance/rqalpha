# -*- coding: utf-8 -*-
"""TX1 live advisor 的实时快照构建。"""

from __future__ import annotations

import inspect

import pandas as pd

from skyeye.data import DataFacade
from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.evaluator import FEATURE_COLUMNS
from skyeye.products.tx1.run_baseline_experiment import build_live_raw_df


DATA_FACADE = DataFacade()


def build_live_snapshot(
    *,
    trade_date,
    raw_df=None,
    universe=None,
    required_features=None,
    universe_size=300,
    market_cap_floor_quantile=None,
    market_cap_column=None,
    universe_source="runtime_fast",
    universe_cache_root=None,
) -> dict:
    """构建指定日期的 live snapshot。"""
    requested_trade_date = pd.Timestamp(trade_date).normalize()
    source_summary = {}
    if raw_df is None:
        raw_df = _call_with_supported_kwargs(
            build_live_raw_df,
            trade_date=requested_trade_date,
            universe=universe,
            universe_size=universe_size,
            market_cap_floor_quantile=market_cap_floor_quantile,
            market_cap_column=market_cap_column,
            universe_source=universe_source,
            universe_cache_root=universe_cache_root,
            required_features=required_features,
        )
    if raw_df is None or len(raw_df) == 0:
        raise ValueError("raw_df must not be empty")
    source_summary = dict(raw_df.attrs.get("data_source_summary", {}))

    frame = raw_df.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    raw_data_end_date = pd.Timestamp(frame["date"].max()).normalize()
    dataset = DatasetBuilder(input_window=60).build(frame)
    available_dates = sorted(pd.to_datetime(dataset["date"]).dt.normalize().unique())
    if not available_dates:
        raise ValueError("no snapshot dates available after dataset build")
    resolved_trade_date = _resolve_trade_date(available_dates, requested_trade_date)
    latest_available_trade_date = pd.Timestamp(max(available_dates)).normalize()
    requested_vs_available_trading_gap = _count_requested_gap(
        requested_trade_date=requested_trade_date,
        latest_available_trade_date=latest_available_trade_date,
    )

    snapshot_df = dataset[dataset["date"] == resolved_trade_date].copy()
    if snapshot_df.empty:
        raise ValueError("no snapshot rows found for trade_date={}".format(resolved_trade_date))

    required_features = list(required_features or [column for column in FEATURE_COLUMNS if column in dataset.columns])
    dropped_reasons = {}
    eligible_mask = []
    for row in snapshot_df.itertuples(index=False):
        reasons = []
        for feature_name in required_features:
            if feature_name not in snapshot_df.columns:
                reasons.append("missing_feature_column:{}".format(feature_name))
                continue
            if pd.isna(getattr(row, feature_name)):
                reasons.append("missing_feature_value:{}".format(feature_name))
        order_book_id = getattr(row, "order_book_id")
        if reasons:
            dropped_reasons[str(order_book_id)] = reasons
            eligible_mask.append(False)
        else:
            eligible_mask.append(True)

    eligible_df = snapshot_df.loc[eligible_mask].copy().reset_index(drop=True)
    history_counts = _build_history_counts(dataset, required_features)
    feature_coverage = {}
    for feature_name in required_features:
        if feature_name not in snapshot_df.columns:
            feature_coverage[feature_name] = 0.0
        else:
            feature_coverage[feature_name] = float(snapshot_df[feature_name].notna().mean())

    return {
        "requested_trade_date": requested_trade_date.strftime("%Y-%m-%d"),
        "latest_available_trade_date": latest_available_trade_date.strftime("%Y-%m-%d"),
        "requested_vs_available_trading_gap": int(requested_vs_available_trading_gap),
        "trade_date": pd.Timestamp(resolved_trade_date).strftime("%Y-%m-%d"),
        "raw_data_end_date": raw_data_end_date.strftime("%Y-%m-%d"),
        "required_features": required_features,
        "eligible_universe": eligible_df["order_book_id"].astype(str).tolist(),
        "dropped_reasons": dropped_reasons,
        "feature_coverage_summary": {
            "total_candidates": int(len(snapshot_df)),
            "eligible_count": int(len(eligible_df)),
            "per_feature": feature_coverage,
        },
        "data_source_summary": source_summary,
        "history_counts": history_counts,
        "snapshot_features": eligible_df,
    }


def _resolve_trade_date(available_dates, requested_trade_date):
    """把请求日期归约到不晚于它的最近可用交易日。"""
    requested = pd.Timestamp(requested_trade_date).normalize()
    candidates = [pd.Timestamp(value).normalize() for value in available_dates if pd.Timestamp(value).normalize() <= requested]
    if not candidates:
        raise ValueError("requested trade_date {} is earlier than available dataset dates".format(requested.date()))
    return max(candidates)


def _build_history_counts(dataset, required_features, lookback_days=60):
    """统计最近窗口内每天满足特征完备性的股票数。"""
    counts = []
    for date, day_df in dataset.groupby("date", sort=True):
        eligible = day_df.copy()
        for feature_name in required_features:
            if feature_name not in eligible.columns:
                eligible = eligible.iloc[0:0]
                break
            eligible = eligible[eligible[feature_name].notna()]
        counts.append(
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "eligible_count": int(len(eligible)),
            }
        )
    return counts[-lookback_days:]


def _count_requested_gap(
    *,
    requested_trade_date: pd.Timestamp,
    latest_available_trade_date: pd.Timestamp,
) -> int:
    """统计请求日相对最新可用快照日缺了多少个交易日。"""
    requested = pd.Timestamp(requested_trade_date).normalize()
    latest = pd.Timestamp(latest_available_trade_date).normalize()
    if requested <= latest:
        return 0

    trading_dates = []
    try:
        trading_dates = list(DATA_FACADE.get_trading_dates(latest, requested) or [])
    except Exception:
        trading_dates = []
    if trading_dates:
        return max(len(trading_dates) - 1, 0)

    fallback_dates = pd.bdate_range(latest, requested)
    return max(len(fallback_dates) - 1, 0)


def _call_with_supported_kwargs(func, **kwargs):
    """兼容测试替身和旧签名，只传目标函数声明过的关键字参数。"""
    parameters = inspect.signature(func).parameters
    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in parameters
    }
    return func(**supported_kwargs)
