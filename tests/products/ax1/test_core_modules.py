import numpy as np
import pandas as pd
import pytest

from skyeye.products.ax1.execution.smoother import ExecutionSmoother
from skyeye.products.ax1.evaluation.metrics import evaluate_portfolio_layer, evaluate_signal_layer
from skyeye.products.ax1.features import AX1FeatureViewBuilder
from skyeye.products.ax1.labels import MultiHorizonLabelBuilder
from skyeye.products.ax1.models.registry import ModelRegistry, Predictor
from skyeye.products.ax1.optimizer.allocation import OpportunityPoolOptimizer
from skyeye.products.ax1.risk.models import FactorRiskModel, HistoricalCovarianceRiskModel
from skyeye.products.ax1.layers import LegacyLayerConfigError
from skyeye.products.ax1.universe import DynamicUniverseBuilder
from skyeye.products.ax1.view_fusion.black_litterman import BlackLittermanViewFusion, NoOpViewFusion


def make_raw_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    rows = []
    for asset_index, order_book_id in enumerate(["000001.XSHE", "000002.XSHE", "600000.XSHG"]):
        for step, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "asset_type": "stock",
                    "close": 10.0 + asset_index + step * (0.1 + asset_index * 0.02),
                }
            )
    rows.append({"date": dates[-1], "order_book_id": None, "close": 1.0})
    return pd.DataFrame(rows)


def make_prediction_panel() -> pd.DataFrame:
    raw_df = make_raw_panel().dropna(subset=["order_book_id"])
    raw_df = raw_df[raw_df["date"].eq(pd.Timestamp("2024-01-01"))].reset_index(drop=True)
    return pd.DataFrame(
        {
            "date": raw_df["date"],
            "order_book_id": raw_df["order_book_id"],
            "expected_relative_net_return_5d": [0.01, 0.00, -0.01],
            "expected_relative_net_return_10d": [0.02, 0.01, 0.00],
            "expected_relative_net_return_20d": [0.03, 0.02, 0.01],
            "risk_forecast": [0.10, 0.11, 0.12],
            "confidence": [0.80, 0.70, 0.60],
            "liquidity_score": [1.00, 0.90, 0.80],
            "cost_forecast": [0.001, 0.001, 0.001],
        }
    )


class DummyMultiTargetPredictor(Predictor):
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame(index=features.index)
        predictions[["date", "order_book_id"]] = features[["date", "order_book_id"]].reset_index(drop=True)
        predictions["expected_relative_net_return_5d"] = 0.01
        predictions["expected_relative_net_return_10d"] = 0.02
        predictions["expected_relative_net_return_20d"] = 0.03
        predictions["risk_forecast"] = 0.10
        predictions["confidence"] = 0.80
        predictions["liquidity_score"] = 1.00
        predictions["cost_forecast"] = 0.001
        return predictions


def _stock_universe_config() -> dict:
    return {"layers": {"stock_satellite": {"asset_type": "stock", "enabled": True}}}


def test_dynamic_universe_builder_filters_cutoff_and_sorts_unique_ids():
    raw_df = make_raw_panel()

    universe = DynamicUniverseBuilder().build(
        raw_df,
        as_of_date="2024-01-04",
        config=_stock_universe_config(),
    )

    assert universe == ["000001.XSHE", "000002.XSHE", "600000.XSHG"]


def test_dynamic_universe_builder_applies_listing_st_and_suspension_filters():
    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-10"] * 4),
            "order_book_id": ["KEEP", "STOCK_ST", "HALTED", "NEW"],
            "listed_date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01", "2024-01-01"]),
            "close": [10.0, 11.0, 12.0, 13.0],
            "asset_type": ["stock"] * 4,
            "is_st": [False, True, False, False],
            "is_suspended": [False, False, True, False],
        }
    )

    universe = DynamicUniverseBuilder().build(
        raw_df,
        as_of_date="2024-01-10",
        config={
            "layers": {"stock_satellite": {"asset_type": "stock", "enabled": True}},
            "min_listing_days": 120,
            "exclude_st": True,
            "exclude_suspended": True,
        },
    )

    assert universe == ["KEEP"]


def test_dynamic_universe_builder_emits_point_in_time_audit_metadata():
    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-08", "2024-01-10", "2024-01-10"]),
            "order_book_id": ["KEEP", "KEEP", "DROP_ST"],
            "listed_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
            "close": [9.8, 10.0, 11.0],
            "asset_type": ["stock"] * 3,
            "is_st": [False, False, True],
            "is_suspended": [False, False, False],
        }
    )

    metadata = DynamicUniverseBuilder().build_with_metadata(
        raw_df,
        as_of_date="2024-01-10",
        config={
            "layers": {"stock_satellite": {"asset_type": "stock", "enabled": True}},
            "min_listing_days": 120,
            "exclude_st": True,
            "exclude_suspended": True,
        },
    )

    assert metadata["order_book_id"].tolist() == ["KEEP"]
    assert metadata["universe_pit_status"].eq("point_in_time").all()
    audit = metadata.attrs["pit_audit"]
    assert audit["passed"] is True
    assert audit["as_of_date"] == "2024-01-10"
    assert audit["source_status"]["raw_frame"] == "point_in_time"
    assert audit["hard_blocks"] == []


def test_dynamic_universe_builder_consumes_registry_layers_and_optional_stock_satellite():
    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-10"] * 5),
            "order_book_id": ["510300.XSHG", "512000.XSHG", "159915.XSHE", "000001.XSHE", "000002.XSHE"],
            "asset_type": ["ETF", "ETF", "ETF", "CS", "CS"],
            "universe_layer": ["core", "industry", "style", "stock_satellite", "stock_satellite"],
            "close": [4.0, 1.2, 2.5, 10.0, 9.0],
            "listed_date": pd.to_datetime(["2020-01-01"] * 5),
            "is_st": [False] * 5,
            "is_suspended": [False] * 5,
        }
    )

    etf_only = DynamicUniverseBuilder().build(
        raw_df,
        as_of_date="2024-01-10",
        config={
            "min_listing_days": 120,
            "exclude_st": True,
            "exclude_suspended": True,
            "layers": {
                "core": {"asset_type": "etf", "include": ["510300.XSHG"]},
                "industry": {"asset_type": "etf", "include": ["512000.XSHG"]},
                "style": {"asset_type": "etf", "include": ["159915.XSHE"]},
                "stock_satellite": {
                    "asset_type": "stock",
                    "enabled": False,
                    "include": ["000001.XSHE"],
                    "max_count": 1,
                },
            },
        },
    )

    with_satellite_metadata = DynamicUniverseBuilder().build_with_metadata(
        raw_df,
        as_of_date="2024-01-10",
        config={
            "layers": {
                "core": {"asset_type": "etf", "include": ["510300.XSHG"]},
                "industry": {"asset_type": "etf", "include": ["512000.XSHG"]},
                "style": {"asset_type": "etf", "include": ["159915.XSHE"]},
                "stock_satellite": {
                    "asset_type": "stock",
                    "enabled": True,
                    "include": ["000001.XSHE"],
                    "max_count": 1,
                },
            },
        },
    )

    assert etf_only == ["159915.XSHE", "510300.XSHG", "512000.XSHG"]
    assert with_satellite_metadata["order_book_id"].tolist() == [
        "000001.XSHE",
        "159915.XSHE",
        "510300.XSHG",
        "512000.XSHG",
    ]
    assert with_satellite_metadata.set_index("order_book_id").loc["000001.XSHE", "universe_layer"] == "stock_satellite"
    assert with_satellite_metadata.set_index("order_book_id").loc["000001.XSHE", "asset_type"] == "stock"


def test_dynamic_universe_builder_rejects_legacy_layer_keys_without_fallback_bridge():
    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-10"]),
            "order_book_id": ["510300.XSHG"],
            "asset_type": ["ETF"],
            "close": [4.0],
        }
    )

    with pytest.raises(LegacyLayerConfigError, match="core_etfs.*stock_satellite"):
        DynamicUniverseBuilder().build(
            raw_df,
            as_of_date="2024-01-10",
            config={
                "core_etfs": ["510300.XSHG"],
                "stock_satellite": {"enabled": True, "max_count": 1},
            },
        )


def test_feature_view_builder_computes_unified_common_features():
    raw_df = make_raw_panel().dropna(subset=["order_book_id"]).reset_index(drop=True)
    raw_df["asset_type"] = "stock"
    raw_df["universe_layer"] = "stock_satellite"
    raw_df["volume"] = 1_000_000
    metadata = raw_df.drop_duplicates("order_book_id")[["order_book_id", "asset_type", "universe_layer"]]

    feature_view = AX1FeatureViewBuilder({"include_scopes": ["common"]}).build(raw_df, universe_metadata=metadata)
    feature_df = feature_view.frame

    assert "feature_momentum_5d" in feature_view.columns_by_scope["common"]
    assert "feature_dollar_volume" in feature_view.columns_by_scope["common"]
    assert "feature_dollar_volume" in feature_df.columns
    assert feature_df["asset_type"].eq("stock").all()


def test_multi_horizon_label_builder_adds_forward_return_columns():
    raw_df = make_raw_panel().dropna(subset=["order_book_id"]).reset_index(drop=True)

    labeled = MultiHorizonLabelBuilder(horizons=(1, 3)).build(raw_df)

    first_asset = labeled[labeled["order_book_id"] == "000001.XSHE"].sort_values("date")
    assert {"label_return_1d", "label_return_3d"}.issubset(labeled.columns)
    assert first_asset.iloc[0]["label_return_1d"] == pytest.approx(10.1 / 10.0 - 1.0)
    assert first_asset.iloc[0]["label_return_3d"] == pytest.approx(10.3 / 10.0 - 1.0)
    assert np.isnan(first_asset.iloc[-1]["label_return_1d"])


def test_model_registry_and_dummy_multi_target_predictor_emit_required_columns():
    raw_df = make_raw_panel().dropna(subset=["order_book_id"]).reset_index(drop=True)
    registry = ModelRegistry()
    registry.register("dummy", DummyMultiTargetPredictor())

    predictions = registry.get("dummy").predict(raw_df)

    assert registry.list_names() == ["dummy"]
    assert {
        "expected_relative_net_return_5d",
        "expected_relative_net_return_10d",
        "expected_relative_net_return_20d",
        "risk_forecast",
        "confidence",
        "liquidity_score",
        "cost_forecast",
    }.issubset(predictions.columns)
    assert predictions["confidence"].between(0.0, 1.0).all()
    assert predictions[["date", "order_book_id"]].equals(raw_df[["date", "order_book_id"]])


def test_historical_covariance_risk_model_builds_shrunk_covariance_matrix():
    raw_df = make_raw_panel().dropna(subset=["order_book_id"]).reset_index(drop=True)

    risk_model = HistoricalCovarianceRiskModel(shrinkage=0.2).fit(raw_df)
    covariance = risk_model.get_covariance_matrix()

    assert list(covariance.index) == ["000001.XSHE", "000002.XSHE", "600000.XSHG"]
    assert list(covariance.columns) == list(covariance.index)
    assert np.allclose(covariance.to_numpy(), covariance.to_numpy().T)
    assert (np.diag(covariance.to_numpy()) >= 0.0).all()


def test_factor_risk_model_builds_low_rank_statistical_covariance_matrix():
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    rows = []
    for asset_index, order_book_id in enumerate(["ETF_A", "ETF_B", "ETF_C", "ETF_D"]):
        close = 10.0 + asset_index
        for step, date in enumerate(dates):
            market = 0.001 + 0.0004 * np.sin(step / 5.0)
            style = 0.0007 * np.cos(step / 7.0)
            asset_return = market + (asset_index - 1.5) * style + 0.0001 * asset_index
            close *= 1.0 + asset_return
            rows.append({"date": date, "order_book_id": order_book_id, "close": close})
    raw_df = pd.DataFrame(rows)

    risk_model = FactorRiskModel(n_factors=2, shrinkage=0.10, lookback_days=60, idiosyncratic_floor=1e-8).fit(raw_df)
    covariance = risk_model.get_covariance_matrix()

    assert list(covariance.index) == ["ETF_A", "ETF_B", "ETF_C", "ETF_D"]
    assert list(covariance.columns) == list(covariance.index)
    assert np.allclose(covariance.to_numpy(), covariance.to_numpy().T)
    assert (np.diag(covariance.to_numpy()) >= 1e-8).all()
    assert np.linalg.eigvalsh(covariance.to_numpy()).min() >= -1e-10


def test_opportunity_pool_optimizer_generates_alpha_ranked_component():
    predictions = make_prediction_panel()

    targets = OpportunityPoolOptimizer().optimize(
        predictions,
        constraints={"target_gross_exposure": 1.0, "cash_buffer": 0.0, "max_single_weight": 0.80},
        allocation_config={
            "kind": "opportunity_pool_optimizer",
            "exposure_groups": {"stock": {"max_weight": 1.0}},
        },
    )

    assert {"date", "order_book_id", "target_weight", "component"}.issubset(targets.columns)
    assert set(targets["component"]) == {"opportunity_pool"}
    assert targets.groupby("date")["target_weight"].sum().iloc[0] == pytest.approx(1.0)
    weights = targets.set_index("order_book_id")["target_weight"]
    assert weights["000001.XSHE"] > weights["000002.XSHE"]


def test_execution_smoother_aggregates_filters_and_normalizes_target_weights():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "order_book_id": ["000001.XSHE", "000002.XSHE", "600000.XSHG"],
            "target_weight": [0.6, 0.35, 0.05],
            "component": ["base", "base", "overlay"],
        }
    )

    smoothed = ExecutionSmoother(min_weight=0.1).smooth(target_weights)

    assert list(smoothed["order_book_id"]) == ["000001.XSHE", "000002.XSHE"]
    assert smoothed["target_weight"].sum() == pytest.approx(1.0)
    assert smoothed["component"].eq("smoothed").all()


def test_view_fusion_noop_adjusts_expected_return_and_bl_is_explicit_skeleton():
    predictions = make_prediction_panel().head(2)

    fused = NoOpViewFusion().fuse(predictions)

    assert "adjusted_expected_return" in fused.columns
    assert fused["adjusted_expected_return"].eq(fused["expected_relative_net_return_10d"]).all()
    with pytest.raises(NotImplementedError):
        BlackLittermanViewFusion().fuse(predictions)


def test_evaluation_metrics_return_signal_and_portfolio_sections():
    predictions = make_prediction_panel()
    targets = OpportunityPoolOptimizer().optimize(
        predictions,
        constraints={"target_gross_exposure": 1.0, "cash_buffer": 0.0, "max_single_weight": 0.80},
        allocation_config={"kind": "opportunity_pool_optimizer"},
    )

    signal_metrics = evaluate_signal_layer(predictions)
    portfolio_metrics = evaluate_portfolio_layer(targets)

    assert set(signal_metrics) == {"signal", "portfolio"}
    assert signal_metrics["signal"]["row_count"] == 3
    assert set(portfolio_metrics) == {"signal", "portfolio"}
    assert portfolio_metrics["portfolio"]["normalized"] is True
