import pandas as pd

from skyeye.products.tx1.baseline_models import create_model, dump_model_bundle
from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.evaluator import FEATURE_COLUMNS
from skyeye.products.tx1.label_builder import LabelBuilder
from skyeye.products.tx1.live_advisor.calibration import build_calibration_bundle
from skyeye.products.tx1.live_advisor.package_io import build_live_package_payload, save_live_package
from skyeye.products.tx1.live_advisor import service as live_advisor_service_module
from skyeye.products.tx1.live_advisor.service import LiveAdvisorService
from skyeye.products.tx1.preprocessor import FeaturePreprocessor


def test_live_advisor_service_returns_ranked_recommendations(make_raw_panel, tmp_path, monkeypatch):
    """验证 live advisor 能输出隐藏原始分数的推荐卡片。"""
    monkeypatch.setattr(
        live_advisor_service_module,
        "evaluate_snapshot_runtime_gates",
        lambda snapshot, required_features: {"passed": True, "reasons": [], "warnings": []},
    )
    raw_df = make_raw_panel(periods=220, extended=True)
    dataset = DatasetBuilder(input_window=60).build(raw_df)
    labeled = LabelBuilder(horizon=20, transform="rank").build(dataset)
    feature_columns = [column for column in FEATURE_COLUMNS if column in labeled.columns]
    preprocessor = FeaturePreprocessor(neutralize=False, winsorize_scale=None, standardize=False)
    transformed = preprocessor.transform(labeled, feature_columns)

    model = create_model("linear")
    model.fit(transformed[feature_columns], transformed["target_label"])

    predictions_df = transformed[
        [
            "date",
            "order_book_id",
            "label_return_raw",
            "label_volatility_horizon",
            "label_max_drawdown_horizon",
        ]
    ].copy()
    predictions_df["prediction"] = model.predict(transformed[feature_columns])
    calibration_bundle = build_calibration_bundle(
        {"fold_results": [{"predictions_df": predictions_df}]},
        bucket_count=5,
    )
    payload = build_live_package_payload(
        manifest={
            "package_id": "tx1_live_service_demo",
            "package_type": "canary_live",
            "source_experiment": "synthetic_demo",
            "horizon": 20,
            "fit_end_date": str(pd.Timestamp(labeled["date"].max()).date()),
            "data_end_date": str(pd.Timestamp(raw_df["date"].max()).date()),
            "created_at": "2026-04-12T00:00:00",
            "model_kind": "linear",
            "required_features": feature_columns,
            "hashes": {
                "feature_schema": "sha256:feature",
                "preprocessor_bundle": "sha256:preproc",
                "model_bundle": "sha256:model",
                "calibration_bundle": "sha256:calibration",
                "portfolio_policy": "sha256:policy",
            },
            "gate_summary": {
                "gate_level": "canary_live",
                "passed": True,
            },
        },
        feature_schema={"feature_columns": feature_columns, "label_horizon": 20},
        preprocessor_bundle=preprocessor.to_bundle(feature_columns),
        model_bundle=dump_model_bundle(model, model_kind="linear", feature_columns=feature_columns),
        calibration_bundle=calibration_bundle,
        portfolio_policy={"buy_top_k": 25, "hold_top_k": 45},
    )
    save_live_package(payload, packages_root=tmp_path)

    service = LiveAdvisorService(packages_root=tmp_path)
    result = service.get_recommendations(
        "tx1_live_service_demo",
        trade_date=str(pd.Timestamp(raw_df["date"].max()).date()),
        top_k=2,
        raw_df=raw_df,
    )

    assert result["status"] == "ok"
    assert result["package_id"] == "tx1_live_service_demo"
    assert result["gate_level"] == "canary_live"
    assert len(result["recommendations"]) == 2
    assert "prediction" not in result["recommendations"][0]
    assert "rank" in result["recommendations"][0]
    assert "win_rate" in result["recommendations"][0]
    assert "median_return" in result["recommendations"][0]


def test_live_advisor_service_returns_recent_canary_evidence_and_portfolio_advice(make_raw_panel, tmp_path, monkeypatch):
    """验证 service 会同时返回 recent canary 证据和组合调仓建议。"""
    monkeypatch.setattr(
        live_advisor_service_module,
        "evaluate_snapshot_runtime_gates",
        lambda snapshot, required_features: {"passed": True, "reasons": [], "warnings": []},
    )
    raw_df = make_raw_panel(periods=220, extended=True)
    dataset = DatasetBuilder(input_window=60).build(raw_df)
    labeled = LabelBuilder(horizon=20, transform="rank").build(dataset)
    feature_columns = [column for column in FEATURE_COLUMNS if column in labeled.columns]
    preprocessor = FeaturePreprocessor(neutralize=False, winsorize_scale=None, standardize=False)
    transformed = preprocessor.transform(labeled, feature_columns)

    model = create_model("linear")
    model.fit(transformed[feature_columns], transformed["target_label"])

    predictions_df = transformed[
        [
            "date",
            "order_book_id",
            "label_return_raw",
            "label_volatility_horizon",
            "label_max_drawdown_horizon",
        ]
    ].copy()
    predictions_df["prediction"] = model.predict(transformed[feature_columns])
    calibration_bundle = build_calibration_bundle(
        {"fold_results": [{"predictions_df": predictions_df}]},
        bucket_count=5,
    )
    payload = build_live_package_payload(
        manifest={
            "package_id": "tx1_live_service_portfolio_demo",
            "package_type": "canary_live",
            "source_experiment": "synthetic_demo",
            "horizon": 20,
            "fit_end_date": str(pd.Timestamp(labeled["date"].max()).date()),
            "data_end_date": str(pd.Timestamp(raw_df["date"].max()).date()),
            "created_at": "2026-04-12T00:00:00",
            "model_kind": "linear",
            "required_features": feature_columns,
            "hashes": {
                "feature_schema": "sha256:feature",
                "preprocessor_bundle": "sha256:preproc",
                "model_bundle": "sha256:model",
                "calibration_bundle": "sha256:calibration",
                "portfolio_policy": "sha256:policy",
                "recent_canary_bundle": "sha256:recent",
            },
            "gate_summary": {
                "gate_level": "canary_live",
                "passed": True,
            },
        },
        feature_schema={"feature_columns": feature_columns, "label_horizon": 20},
        preprocessor_bundle=preprocessor.to_bundle(feature_columns),
        model_bundle=dump_model_bundle(model, model_kind="linear", feature_columns=feature_columns),
        calibration_bundle=calibration_bundle,
        portfolio_policy={"buy_top_k": 2, "hold_top_k": 3, "rebalance_interval": 20, "holding_bonus": 0.5},
        recent_canary_bundle={
            "window": {
                "start_date": str(pd.Timestamp(labeled["date"].max() - pd.offsets.BDay(19)).date()),
                "end_date": str(pd.Timestamp(labeled["date"].max()).date()),
            },
            "bucket_edges": [0.0, 0.5, 1.0],
            "bucket_stats": [
                {"bucket_id": "b00", "sample_count": 120, "win_rate": 0.51, "mean_return": 0.003, "median_return": 0.001, "return_quantiles": {"p25": -0.01, "p75": 0.02}},
                {"bucket_id": "b01", "sample_count": 120, "win_rate": 0.62, "mean_return": 0.011, "median_return": 0.006, "return_quantiles": {"p25": -0.005, "p75": 0.03}},
            ],
        },
    )
    save_live_package(payload, packages_root=tmp_path)

    service = LiveAdvisorService(packages_root=tmp_path)
    result = service.get_recommendations(
        "tx1_live_service_portfolio_demo",
        trade_date=str(pd.Timestamp(raw_df["date"].max()).date()),
        top_k=2,
        raw_df=raw_df,
        current_holdings={"000001.XSHE": 0.5, "000002.XSHE": 0.5},
        last_rebalance_date=str(pd.Timestamp(raw_df["date"].max() - pd.offsets.BDay(25)).date()),
    )

    assert result["status"] == "ok"
    assert result["recommendations"][0]["recent_canary_evidence"]["window"]["end_date"] == str(pd.Timestamp(labeled["date"].max()).date())
    assert result["portfolio_advice"]["estimated_turnover"] >= 0.0
    assert result["portfolio_advice"]["target_weights"]
    assert result["portfolio_advice"]["actions"]


def test_live_advisor_service_keeps_current_weights_when_rebalance_not_due():
    """非调仓日应沿用当前持仓权重，不应强制改成等权。"""
    scored = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-04-10")] * 3,
            "order_book_id": ["A", "B", "C"],
            "prediction": [3.0, 2.0, 1.0],
        }
    )

    advice = LiveAdvisorService._build_portfolio_advice(
        scored,
        {"buy_top_k": 2, "hold_top_k": 3, "rebalance_interval": 20, "holding_bonus": 0.5},
        current_holdings={"A": 0.9, "B": 0.1},
        last_rebalance_date="2026-04-09",
        trade_date="2026-04-10",
    )

    assert advice["rebalance_due"] is False
    assert advice["target_weights"] == {"A": 0.9, "B": 0.1}
    assert advice["estimated_turnover"] == 0.0
    assert [item["action"] for item in advice["actions"]] == ["keep", "keep"]


def test_live_advisor_service_surfaces_freshness_metadata_and_gate_diagnostics(make_raw_panel, tmp_path, monkeypatch):
    """验证 service 会把 manifest freshness 元数据和 gate diagnostics 一并返回。"""
    raw_df = make_raw_panel(periods=220, extended=True)
    dataset = DatasetBuilder(input_window=60).build(raw_df)
    labeled = LabelBuilder(horizon=20, transform="rank").build(dataset)
    feature_columns = [column for column in FEATURE_COLUMNS if column in labeled.columns]
    preprocessor = FeaturePreprocessor(neutralize=False, winsorize_scale=None, standardize=False)
    transformed = preprocessor.transform(labeled, feature_columns)

    model = create_model("linear")
    model.fit(transformed[feature_columns], transformed["target_label"])

    predictions_df = transformed[
        [
            "date",
            "order_book_id",
            "label_return_raw",
            "label_volatility_horizon",
            "label_max_drawdown_horizon",
        ]
    ].copy()
    predictions_df["prediction"] = model.predict(transformed[feature_columns])
    calibration_bundle = build_calibration_bundle(
        {"fold_results": [{"predictions_df": predictions_df}]},
        bucket_count=5,
    )
    payload = build_live_package_payload(
        manifest={
            "package_id": "tx1_live_service_freshness_demo",
            "package_type": "canary_live",
            "source_experiment": "synthetic_demo",
            "horizon": 20,
            "fit_end_date": str(pd.Timestamp(labeled["date"].max()).date()),
            "label_end_date": str(pd.Timestamp(labeled["date"].max()).date()),
            "data_end_date": str(pd.Timestamp(raw_df["date"].max()).date()),
            "evidence_end_date": str(pd.Timestamp(labeled["date"].max()).date()),
            "created_at": "2026-04-12T00:00:00",
            "model_kind": "linear",
            "required_features": feature_columns,
            "hashes": {
                "feature_schema": "sha256:feature",
                "preprocessor_bundle": "sha256:preproc",
                "model_bundle": "sha256:model",
                "calibration_bundle": "sha256:calibration",
                "portfolio_policy": "sha256:policy",
                "recent_canary_bundle": "sha256:recent",
            },
            "gate_summary": {
                "gate_level": "canary_live",
                "passed": True,
            },
            "canary_reason": ["stability_score", "cv"],
            "data_dependency_summary": {
                "bundle_price": True,
                "bundle_volume": True,
                "northbound_flow": False,
                "fundamental_factors": [],
            },
            "freshness_policy": {
                "snapshot_max_delay_days": 1,
                "model_warning_days": 20,
                "model_stop_days": 40,
                "evidence_warning_days": 20,
                "evidence_stop_days": 40,
            },
        },
        feature_schema={"feature_columns": feature_columns, "label_horizon": 20},
        preprocessor_bundle=preprocessor.to_bundle(feature_columns),
        model_bundle=dump_model_bundle(model, model_kind="linear", feature_columns=feature_columns),
        calibration_bundle=calibration_bundle,
        portfolio_policy={"buy_top_k": 25, "hold_top_k": 45},
        recent_canary_bundle={
            "window": {
                "start_date": str(pd.Timestamp(labeled["date"].max() - pd.offsets.BDay(19)).date()),
                "end_date": str(pd.Timestamp(labeled["date"].max()).date()),
            },
            "bucket_edges": [0.0, 1.0],
            "bucket_stats": [{"bucket_id": "b0", "sample_count": 300, "win_rate": 0.58, "mean_return": 0.01, "median_return": 0.005, "return_quantiles": {"p25": -0.01, "p75": 0.02}}],
        },
    )
    save_live_package(payload, packages_root=tmp_path)

    monkeypatch.setattr(
        live_advisor_service_module,
        "evaluate_snapshot_runtime_gates",
        lambda snapshot, required_features, **kwargs: {
            "passed": True,
            "reasons": [],
            "warnings": [
                {
                    "level": "warning",
                    "code": "model_freshness_warning",
                    "message": "model is getting stale",
                }
            ],
            "metrics": {
                "model_freshness_gap_days": 12,
                "evidence_freshness_gap_days": 12,
            },
            "diagnostics": {
                "model_freshness": {"status": "warning"},
                "evidence_freshness": {"status": "warning"},
            },
        },
    )

    service = LiveAdvisorService(packages_root=tmp_path)
    result = service.get_recommendations(
        "tx1_live_service_freshness_demo",
        trade_date=str(pd.Timestamp(raw_df["date"].max()).date()),
        top_k=1,
        raw_df=raw_df,
    )

    assert result["label_end_date"] == str(pd.Timestamp(labeled["date"].max()).date())
    assert result["evidence_end_date"] == str(pd.Timestamp(labeled["date"].max()).date())
    assert any(item["code"] == "model_freshness_warning" for item in result["warnings"])
    assert result["gate_diagnostics"]["model_freshness"]["status"] == "warning"
    assert result["data_dependency_summary"]["bundle_price"] is True


def test_live_advisor_service_marks_portfolio_advice_blocked_when_target_weights_too_fragmented():
    """验证 target weight 过碎时，portfolio advice 会显式标记 blocked。"""
    scored = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-04-10")] * 100,
            "order_book_id": ["S{:03d}".format(idx) for idx in range(100)],
            "prediction": list(range(100, 0, -1)),
        }
    )

    advice = LiveAdvisorService._build_portfolio_advice(
        scored,
        {
            "buy_top_k": 100,
            "hold_top_k": 100,
            "rebalance_interval": 20,
            "holding_bonus": 0.5,
            "cash_buffer": 0.1,
            "min_weight": 0.02,
            "max_turnover": 0.8,
        },
        current_holdings=None,
        last_rebalance_date=None,
        trade_date="2026-04-10",
    )

    assert advice["advice_level"] == "blocked"
    assert "min_weight_below_threshold" in advice["execution_blockers"]
    assert advice["preflight_checks"]["min_weight_ok"]["passed"] is False
    assert abs(sum(advice["target_weights"].values()) - 0.9) < 1e-9


def test_live_advisor_service_applies_single_stock_cap():
    """验证 single_stock_cap 会限制个股权重上限（EMA产生不等权后cap生效）。"""
    scored = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-04-10")] * 3,
            "order_book_id": ["A", "B", "C"],
            "prediction": [3.0, 2.0, 1.0],
        }
    )

    # previous_target has A very heavy, EMA will blend and produce unequal weights
    # where A may exceed the cap
    previous_target = {"A": 0.80, "B": 0.10, "C": 0.10}

    advice = LiveAdvisorService._build_portfolio_advice(
        scored,
        {
            "buy_top_k": 3,
            "hold_top_k": 3,
            "rebalance_interval": 20,
            "holding_bonus": 0.0,
            "ema_halflife": 5,
            "ema_min_weight": 0.005,
            "single_stock_cap": 0.40,
        },
        current_holdings=None,
        last_rebalance_date=None,
        trade_date="2026-04-10",
        previous_target_weights=previous_target,
    )

    # After EMA, A's weight should be pulled toward 0.80, then capped at 0.40
    for weight in advice["target_weights"].values():
        assert weight <= 0.40 + 1e-9
    assert abs(sum(advice["target_weights"].values()) - 1.0) < 1e-9


def test_live_advisor_service_skips_rebalance_below_turnover_threshold():
    """验证调仓日换手率低于阈值时跳过调仓。"""
    scored = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-04-10")] * 3,
            "order_book_id": ["A", "B", "C"],
            "prediction": [3.0, 2.0, 1.0],
        }
    )

    # First rebalance day (no last_rebalance_date), current holdings almost match target
    advice = LiveAdvisorService._build_portfolio_advice(
        scored,
        {
            "buy_top_k": 3,
            "hold_top_k": 3,
            "rebalance_interval": 20,
            "holding_bonus": 0.0,
            "turnover_threshold": 0.50,
        },
        current_holdings={"A": 0.33, "B": 0.34, "C": 0.33},
        last_rebalance_date=None,
        trade_date="2026-04-10",
    )

    # rebalance_due=True (first rebalance), but turnover is low
    assert advice["rebalance_due"] is True
    assert advice["skip_rebalance"] is True
    assert advice["advice_level"] == "skipped"
    assert "turnover_below_threshold" in advice["execution_blockers"]


def test_live_advisor_service_applies_ema_smoothing():
    """验证 EMA 平滑会渐变目标权重。"""
    scored = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-04-10")] * 3,
            "order_book_id": ["A", "B", "C"],
            "prediction": [3.0, 2.0, 1.0],
        }
    )

    previous_target = {"A": 0.1, "B": 0.1, "C": 0.8}

    advice = LiveAdvisorService._build_portfolio_advice(
        scored,
        {
            "buy_top_k": 3,
            "hold_top_k": 3,
            "rebalance_interval": 1,
            "holding_bonus": 0.0,
            "ema_halflife": 5,
            "ema_min_weight": 0.005,
        },
        current_holdings=None,
        last_rebalance_date=None,
        trade_date="2026-04-10",
        previous_target_weights=previous_target,
    )

    # EMA should blend raw equal-weight with previous state
    assert advice["ema_state"] is not None
    # C's weight should be pulled toward 0.8 by EMA (not stay at 1/3)
    assert advice["target_weights"]["C"] > 1.0 / 3.0
    assert abs(sum(advice["target_weights"].values()) - 1.0) < 1e-9


def test_live_advisor_service_estimates_trading_costs():
    """验证交易成本估算正确。"""
    scored = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-04-10")] * 3,
            "order_book_id": ["A", "B", "C"],
            "prediction": [3.0, 2.0, 1.0],
        }
    )

    advice = LiveAdvisorService._build_portfolio_advice(
        scored,
        {
            "buy_top_k": 3,
            "hold_top_k": 3,
            "rebalance_interval": 20,
            "holding_bonus": 0.0,
            "commission_rate": 0.0008,
            "stamp_tax_rate": 0.0005,
            "slippage_bps": 5.0,
        },
        current_holdings={"A": 0.5, "B": 0.5},
        last_rebalance_date=None,
        trade_date="2026-04-10",
    )

    assert advice["estimated_costs"]["total_estimated_cost"] > 0
    assert advice["estimated_costs"]["commission_cost"] > 0
