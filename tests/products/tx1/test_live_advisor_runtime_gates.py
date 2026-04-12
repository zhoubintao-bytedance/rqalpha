import pandas as pd

from skyeye.products.tx1.live_advisor.runtime_gates import (
    evaluate_score_runtime_gates,
    evaluate_snapshot_runtime_gates,
)


def test_snapshot_runtime_gate_blocks_stale_snapshot():
    """验证当日快照滞后过多时，系统会拒绝出分。"""
    snapshot = {
        "requested_trade_date": "2026-04-12",
        "trade_date": "2026-03-31",
        "latest_available_trade_date": "2026-03-31",
        "requested_vs_available_trading_gap": 8,
        "raw_data_end_date": "2026-03-31",
        "eligible_universe": ["A"] * 350,
        "history_counts": [380] * 60,
        "snapshot_features": pd.DataFrame({"mom_40d": [0.1], "volatility_20d": [0.2]}),
    }

    gate = evaluate_snapshot_runtime_gates(
        snapshot,
        required_features=["mom_40d", "volatility_20d"],
    )

    assert gate["passed"] is False
    assert any("requested_trade_date" in reason for reason in gate["reasons"])
    assert gate["warnings"][0]["level"] == "critical"
    assert "2026-04-12" in gate["warnings"][0]["message"]
    assert "2026-03-31" in gate["warnings"][0]["message"]


def test_score_runtime_gate_blocks_distribution_collapse():
    """验证预测分布塌缩时，系统会触发 stop-serve。"""
    scored = pd.DataFrame(
        {
            "order_book_id": ["A", "B", "C", "D"],
            "prediction": [0.1, 0.1, 0.1, 0.1],
        }
    )
    calibration_bundle = {
        "score_sanity_reference": {
            "prediction_std_p01": 0.02,
            "prediction_std_p05": 0.03,
            "top_spread_p01": 0.01,
            "top_spread_p05": 0.02,
            "n_days": 20,
        }
    }

    gate = evaluate_score_runtime_gates(scored, calibration_bundle)

    assert gate["passed"] is False
    assert any("score_distribution" in reason for reason in gate["reasons"])


def test_snapshot_runtime_gate_blocks_when_total_candidates_below_base_threshold():
    """候选池总量本身低于 coverage 下限时，也必须 stop-serve。"""
    snapshot = {
        "requested_trade_date": "2026-04-12",
        "trade_date": "2026-04-12",
        "latest_available_trade_date": "2026-04-12",
        "requested_vs_available_trading_gap": 0,
        "raw_data_end_date": "2026-04-12",
        "eligible_universe": ["A"] * 100,
        "history_counts": [400] * 60,
        "feature_coverage_summary": {"total_candidates": 100},
        "snapshot_features": pd.DataFrame({"mom_40d": [0.1] * 100, "volatility_20d": [0.2] * 100}),
    }

    gate = evaluate_snapshot_runtime_gates(
        snapshot,
        required_features=["mom_40d", "volatility_20d"],
    )

    assert gate["passed"] is False
    assert any("universe_coverage_below_threshold" in reason for reason in gate["reasons"])
    assert gate["thresholds"]["min_eligible_count"] == 320
