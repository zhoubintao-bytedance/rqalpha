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
