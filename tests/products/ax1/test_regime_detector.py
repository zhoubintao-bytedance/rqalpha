import warnings

import numpy as np
import pandas as pd
import pytest

from skyeye.market_regime_layer import MarketRegimeConfig, required_market_regime_history_days
from skyeye.products.ax1.regime import RegimeDetector


def make_regime_inputs(periods: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if periods is None:
        periods = required_market_regime_history_days(MarketRegimeConfig()) + 60
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    rows = []
    metadata_rows = [
        {"order_book_id": "510300.XSHG", "universe_layer": "core", "asset_type": "ETF"},
        {"order_book_id": "159919.XSHE", "universe_layer": "core", "asset_type": "ETF"},
    ]
    industry_ids = [f"5100{i}.XSHG" for i in range(10, 15)]
    metadata_rows.extend(
        {"order_book_id": order_book_id, "universe_layer": "industry", "asset_type": "ETF"}
        for order_book_id in industry_ids
    )

    for step, date in enumerate(dates):
        cycle = np.sin(step / 11.0)
        for asset_index, order_book_id in enumerate(["510300.XSHG", "159919.XSHE"]):
            close = 3.0 + asset_index * 0.2 + step * (0.012 + asset_index * 0.001) + cycle * 0.05
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "close": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "volume": 1_000_000 + step * 1000 + asset_index * 20_000,
                }
            )
        for industry_index, order_book_id in enumerate(industry_ids):
            close = 1.5 + step * (0.004 + industry_index * 0.001) + np.sin(step / (7 + industry_index)) * 0.03
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "close": close,
                    "high": close * 1.008,
                    "low": close * 0.992,
                    "volume": 500_000 + step * 500 + industry_index * 10_000,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(metadata_rows)


def test_regime_detector_emits_required_fields_and_uses_raw_core_proxy_without_fallback():
    raw_df, universe_metadata = make_regime_inputs()
    as_of_date = raw_df["date"].max()

    result = RegimeDetector().detect(
        raw_df,
        as_of_date=as_of_date,
        universe_metadata=universe_metadata,
    )

    assert {
        "as_of_date",
        "market_regime",
        "strength",
        "risk_state",
        "style_state",
        "volatility_state",
        "rotation_state",
        "source",
        "benchmark_proxy_ids",
        "diagnostics",
    }.issubset(result)
    assert result["as_of_date"] == as_of_date
    assert result["source"] != "fallback"
    assert result["market_regime"] in {
        "bull_co_move",
        "bull_rotation",
        "range_co_move",
        "range_rotation",
        "bear_co_move",
        "bear_rotation",
    }
    assert 0.0 <= result["strength"] <= 1.0
    assert result["benchmark_proxy_ids"] == ["510300.XSHG"]


def test_regime_detector_falls_back_when_history_is_insufficient():
    raw_df, universe_metadata = make_regime_inputs(periods=20)

    result = RegimeDetector({"fallback_regime": "bear_rotation"}).detect(
        raw_df,
        as_of_date="2024-01-20",
        universe_metadata=universe_metadata,
    )

    assert result["market_regime"] == "bear_rotation"
    assert result["strength"] == 0.0
    assert result["risk_state"] == "risk_off"
    assert result["style_state"] == "balanced"
    assert result["volatility_state"] == "unknown"
    assert result["rotation_state"] == "rotation"
    assert result["source"] == "fallback"
    assert result["diagnostics"]["reason"] == "insufficient_history"
    assert result["diagnostics"]["history_days"] == 20
    assert result["diagnostics"]["required_history_days"] > result["diagnostics"]["history_days"]
    assert result["diagnostics"]["configured_lookback_days"] == 0


def test_regime_detector_default_config_responds_after_required_history_days():
    required_days = required_market_regime_history_days(MarketRegimeConfig())
    raw_df, universe_metadata = make_regime_inputs(periods=required_days + 15)

    result = RegimeDetector().detect(
        raw_df,
        as_of_date=raw_df["date"].max(),
        universe_metadata=universe_metadata,
    )

    assert result["source"] != "fallback"
    assert result["diagnostics"]["required_history_days"] == required_days
    assert result["diagnostics"]["history_days"] == required_days + 15
    assert result["diagnostics"]["configured_lookback_days"] == 0


def test_regime_detector_coerces_market_regime_config_dict():
    raw_df, universe_metadata = make_regime_inputs(periods=80)

    result = RegimeDetector(
        {
            "lookback_days": 0,
            "market_regime_config": {
                "ma_long": 12,
                "price_position_window": 12,
                "boll_percentile_window": 20,
                "hurst_window": 30,
                "hurst_max_lag": 8,
                "atr_ma_window": 12,
                "rsrs_m": 20,
                "trendiness_range_threshold": 0.0,
            },
        }
    ).detect(
        raw_df,
        as_of_date="2024-03-20",
        universe_metadata=universe_metadata,
    )

    assert result["source"] != "fallback"
    assert result["diagnostics"]["required_history_days"] < 80
    assert result["diagnostics"]["configured_lookback_days"] == 0


def test_regime_detector_ignores_rows_after_as_of_date():
    raw_df, universe_metadata = make_regime_inputs()
    as_of_date = raw_df["date"].max()
    baseline = RegimeDetector().detect(raw_df, as_of_date=as_of_date, universe_metadata=universe_metadata)

    future = raw_df[raw_df["date"] == raw_df["date"].max()].copy()
    future["date"] = pd.Timestamp(as_of_date) + pd.Timedelta(days=30)
    future["close"] = future["close"] * 1000.0
    future["high"] = future["close"] * 1.2
    future["low"] = future["close"] * 0.8
    future["volume"] = future["volume"] * 1000.0
    with_future = pd.concat([raw_df, future], ignore_index=True)

    outlier_result = RegimeDetector().detect(with_future, as_of_date=as_of_date, universe_metadata=universe_metadata)

    assert outlier_result == baseline


def test_regime_detector_reports_industry_diagnostics_when_industry_etfs_are_available():
    raw_df, universe_metadata = make_regime_inputs()

    result = RegimeDetector().detect(
        raw_df,
        as_of_date=raw_df["date"].max(),
        universe_metadata=universe_metadata,
    )

    structure_diagnostics = result["diagnostics"].get("structure_diagnostics", {})
    assert result["source"] != "fallback"
    assert structure_diagnostics.get("industry_count", 0) >= 5


def test_regime_detector_handles_constant_volume_without_runtime_warning():
    raw_df, universe_metadata = make_regime_inputs()
    raw_df["volume"] = 1_000_000

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = RegimeDetector().detect(
            raw_df,
            as_of_date=raw_df["date"].max(),
            universe_metadata=universe_metadata,
        )

    assert result["source"] != "fallback"
    assert result["diagnostics"]["direction_diagnostics"]["trendiness_components"]["volume_trend_score"] is None


def test_regime_detector_honors_equal_weight_core_proxy_config():
    raw_df, universe_metadata = make_regime_inputs()

    result = RegimeDetector(
        {
            "preferred_benchmark_ids": ["MISSING.XSHG"],
            "core_proxy_method": "equal_weight_core",
        }
    ).detect(
        raw_df,
        as_of_date=raw_df["date"].max(),
        universe_metadata=universe_metadata,
    )

    assert result["source"] == "raw_core_equal_weight"
    assert result["benchmark_proxy_ids"] == ["159919.XSHE", "510300.XSHG"]


def test_regime_detector_rejects_explicit_data_provider_source_until_implemented():
    raw_df, universe_metadata = make_regime_inputs()

    with pytest.raises(NotImplementedError, match="benchmark_source=data_provider"):
        RegimeDetector({"benchmark_source": "data_provider"}).detect(
            raw_df,
            as_of_date="2024-09-10",
            universe_metadata=universe_metadata,
            data_provider=object(),
        )


def test_regime_detector_builds_point_in_time_states_without_future_leakage():
    raw_df, universe_metadata = make_regime_inputs(periods=90)
    detector = RegimeDetector({"fallback_regime": "range_rotation"})
    target_date = pd.Timestamp("2024-02-15")

    baseline = detector.detect_by_date(raw_df, universe_metadata=universe_metadata)
    future = raw_df[raw_df["date"] == raw_df["date"].max()].copy()
    future["date"] = target_date + pd.Timedelta(days=45)
    future["close"] = future["close"] * 100.0
    expanded = pd.concat([raw_df, future], ignore_index=True)
    with_future = detector.detect_by_date(expanded, universe_metadata=universe_metadata)

    assert target_date in baseline
    assert with_future[target_date] == baseline[target_date]
    early = baseline[pd.Timestamp("2024-01-05")]
    assert early["market_regime"] == "range_rotation"
    assert early["strength"] == 0.0


def test_regime_detector_warmup_fallback_reports_configured_lookback_separately():
    raw_df, universe_metadata = make_regime_inputs(periods=80)
    detector = RegimeDetector(
        {
            "lookback_days": 60,
            "fallback_regime": "range_rotation",
            "market_regime_config": {
                "ma_long": 12,
                "price_position_window": 12,
                "boll_percentile_window": 20,
                "hurst_window": 30,
                "hurst_max_lag": 8,
                "atr_ma_window": 12,
                "rsrs_m": 20,
            },
        }
    )

    states = detector.detect_by_date(
        raw_df,
        universe_metadata=universe_metadata,
        as_of_dates=[pd.Timestamp("2024-02-15"), pd.Timestamp("2024-03-05")],
    )

    early = states[pd.Timestamp("2024-02-15")]
    later = states[pd.Timestamp("2024-03-05")]
    assert early["source"] == "fallback"
    assert early["diagnostics"]["reason"] == "warmup_lookback_days"
    assert early["diagnostics"]["configured_lookback_days"] == 60
    assert early["diagnostics"]["required_history_days"] < 60
    assert later["source"] != "fallback"


def test_regime_detector_can_respond_to_synthetic_turn_before_260_days():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    values = np.r_[np.linspace(100.0, 170.0, 70), np.linspace(168.0, 90.0, 50)]
    rows = []
    metadata_rows = [
        {"order_book_id": "510300.XSHG", "universe_layer": "core", "asset_type": "ETF"},
        *[
            {"order_book_id": f"5100{i}.XSHG", "universe_layer": "industry", "asset_type": "ETF"}
            for i in range(10, 15)
        ],
    ]
    synthetic_ids = [
        "510300.XSHG",
        "510010.XSHG",
        "510011.XSHG",
        "510012.XSHG",
        "510013.XSHG",
        "510014.XSHG",
    ]
    for step, (date, value) in enumerate(zip(dates, values, strict=True)):
        for asset_index, order_book_id in enumerate(synthetic_ids):
            close = value * (1.0 + asset_index * 0.01)
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "close": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "volume": 1_000_000 + step * 1000 + asset_index * 10_000,
                }
            )
    raw_df = pd.DataFrame(rows)
    universe_metadata = pd.DataFrame(metadata_rows)
    detector = RegimeDetector(
        {
            "lookback_days": 0,
            "market_regime_config": {
                "ma_short": 3,
                "ma_mid": 8,
                "ma_long": 15,
                "return_window": 10,
                "price_position_window": 15,
                "boll_window": 10,
                "boll_percentile_window": 20,
                "atr_window": 5,
                "atr_ma_window": 10,
                "vol_trend_window": 10,
                "hurst_window": 30,
                "hurst_max_lag": 8,
                "rsrs_n": 8,
                "rsrs_m": 20,
                "industry_return_window": 5,
                "dispersion_percentile_window": 20,
                "trendiness_range_threshold": 0.0,
                "direction_bull_threshold": 0.05,
                "direction_bear_threshold": -0.05,
            },
        }
    )

    before_turn = detector.detect(raw_df, as_of_date=dates[69], universe_metadata=universe_metadata)
    after_turn = detector.detect(raw_df, as_of_date=dates[119], universe_metadata=universe_metadata)

    assert before_turn["source"] != "fallback"
    assert after_turn["source"] != "fallback"
    assert before_turn["diagnostics"]["required_history_days"] < 260
    assert after_turn["diagnostics"]["history_days"] == 120
    assert before_turn["risk_state"] == "risk_on"
    assert after_turn["risk_state"] == "risk_off"
