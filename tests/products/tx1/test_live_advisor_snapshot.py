import pandas as pd

from skyeye.products.tx1.evaluator import FEATURE_COLUMNS
from skyeye.products.tx1.live_advisor.snapshot import build_live_snapshot


def test_build_live_snapshot_extracts_single_trade_date(make_raw_panel):
    """验证 snapshot 会只保留指定交易日的候选股票与历史覆盖摘要。"""
    raw_df = make_raw_panel(periods=180, extended=True)
    trade_date = pd.Timestamp(raw_df["date"].max())
    required_features = [feature for feature in FEATURE_COLUMNS if feature in raw_df.columns or feature != "amihud_20d"]

    snapshot = build_live_snapshot(
        trade_date=trade_date,
        raw_df=raw_df,
        required_features=["mom_40d", "volatility_20d", "reversal_5d", "amihud_20d"],
    )

    assert snapshot["trade_date"] == trade_date.strftime("%Y-%m-%d")
    assert snapshot["raw_data_end_date"] == trade_date.strftime("%Y-%m-%d")
    assert snapshot["snapshot_features"]["date"].nunique() == 1
    assert len(snapshot["eligible_universe"]) == len(snapshot["snapshot_features"])
    assert len(snapshot["history_counts"]) > 0
    assert snapshot["feature_coverage_summary"]["eligible_count"] == len(snapshot["eligible_universe"])


def test_build_live_snapshot_tracks_requested_and_latest_available_trade_dates(make_raw_panel):
    """验证请求日超过最新可用快照日时，snapshot 会显式记录二者差异。"""
    raw_df = make_raw_panel(periods=180, extended=True)
    latest_trade_date = pd.Timestamp(raw_df["date"].max())
    requested_trade_date = latest_trade_date + pd.offsets.BDay(5)

    snapshot = build_live_snapshot(
        trade_date=requested_trade_date,
        raw_df=raw_df,
        required_features=["mom_40d", "volatility_20d", "reversal_5d", "amihud_20d"],
    )

    assert snapshot["requested_trade_date"] == requested_trade_date.strftime("%Y-%m-%d")
    assert snapshot["latest_available_trade_date"] == latest_trade_date.strftime("%Y-%m-%d")
    assert snapshot["trade_date"] == latest_trade_date.strftime("%Y-%m-%d")
    assert snapshot["requested_vs_available_trading_gap"] >= 5
