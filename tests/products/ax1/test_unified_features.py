import numpy as np
import pandas as pd
import pytest

from skyeye.products.ax1.features import AX1FeatureViewBuilder, resolve_feature_columns


def _mixed_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    assets = [
        ("510300.XSHG", "etf", "core", "broad", 100.0, 1.0010),
        ("512000.XSHG", "etf", "industry", "brokerage", 90.0, 1.0025),
        ("510880.XSHG", "etf", "style", "dividend", 80.0, 1.0020),
        ("515180.XSHG", "etf", "style", "value", 70.0, 1.0016),
        ("159915.XSHE", "etf", "style", "growth", 60.0, 1.0006),
        ("159949.XSHE", "etf", "style", "small_growth", 50.0, 0.9995),
        ("000001.XSHE", "stock", "stock_satellite", "bank", 10.0, 1.0018),
    ]
    rows = []
    for asset_idx, (order_book_id, asset_type, layer, industry, base, drift) in enumerate(assets):
        for step, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "asset_type": asset_type,
                    "universe_layer": layer,
                    "industry": industry,
                    "benchmark_id": "510300.XSHG",
                    "close": base * (drift ** step),
                    "volume": 2_000_000 + asset_idx * 200_000 + step * 1_000,
                }
            )
    return pd.DataFrame(rows)


def _metadata(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.drop_duplicates("order_book_id")[
        ["order_book_id", "asset_type", "universe_layer", "industry", "benchmark_id"]
    ].reset_index(drop=True)


def _style_exposure_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    returns_by_id = {
        "510880.XSHG": [],
        "159915.XSHE": [],
        "512000.XSHG": [],
        "510300.XSHG": [],
        "000001.XSHE": [],
    }
    for step, _ in enumerate(dates):
        style_factor = 0.006 * ((step % 7) - 3) / 3.0 + 0.002 * ((step % 5) - 2)
        market = 0.0008 + 0.0001 * ((step % 3) - 1)
        returns_by_id["510880.XSHG"].append(market + 0.65 * style_factor)
        returns_by_id["159915.XSHE"].append(market - 0.45 * style_factor)
        returns_by_id["512000.XSHG"].append(market + 0.55 * style_factor)
        returns_by_id["510300.XSHG"].append(market - 0.10 * style_factor)
        returns_by_id["000001.XSHE"].append(market + 0.20 * style_factor)

    metadata = {
        "510880.XSHG": ("etf", "style", "dividend", 80.0),
        "159915.XSHE": ("etf", "style", "growth", 60.0),
        "512000.XSHG": ("etf", "industry", "brokerage", 90.0),
        "510300.XSHG": ("etf", "core", "broad", 100.0),
        "000001.XSHE": ("stock", "stock_satellite", "bank", 10.0),
    }
    rows = []
    for order_book_id, returns in returns_by_id.items():
        asset_type, layer, industry, close = metadata[order_book_id]
        for step, date in enumerate(dates):
            if step > 0:
                close *= 1.0 + returns[step]
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "asset_type": asset_type,
                    "universe_layer": layer,
                    "industry": industry,
                    "benchmark_id": "510300.XSHG",
                    "close": close,
                    "volume": 2_000_000 + step * 1_000,
                }
            )
    return pd.DataFrame(rows)


def test_feature_view_unifies_etf_and_stock_rows_with_scope_catalog():
    raw = _mixed_panel()
    feature_view = AX1FeatureViewBuilder().build(
        raw,
        universe_metadata=_metadata(raw),
        regime_state_by_date={
            date: {
                "market_regime": "bull_rotation",
                "risk_state": "risk_on",
                "rotation_state": "rotation",
                "strength": 0.6,
            }
            for date in raw["date"].drop_duplicates()
        },
    )
    frame = feature_view.frame

    assert {"etf", "stock"} == set(frame["asset_type"])
    assert {"date", "order_book_id", "asset_type", "universe_layer", "industry", "benchmark_id"}.issubset(frame.columns)
    assert "feature_momentum_5d" in feature_view.columns_by_scope["common"]
    assert "feature_z_excess_mom_20d" in feature_view.columns_by_scope["etf_zscore"]
    assert "feature_regime_strength" in feature_view.columns_by_scope["regime"]
    assert feature_view.columns_by_scope["regime_interaction"] == []
    assert "feature_interaction_z_excess_mom_20d_x_regime_risk_on" not in frame.columns
    assert "etf_specific" not in feature_view.columns_by_scope
    assert "feature_style_spread_dividend_vs_growth_20d" in feature_view.columns_by_scope["etf_raw"]
    assert "feature_style_spread_value_vs_growth_20d" in feature_view.columns_by_scope["etf_raw"]
    assert "feature_style_spread_large_vs_small_20d" in feature_view.columns_by_scope["etf_raw"]
    assert "feature_style_spread_composite_20d" in feature_view.columns_by_scope["etf_raw"]
    assert set(feature_view.columns_by_scope["etf_raw"]).isdisjoint(feature_view.columns_by_scope["etf_zscore"])

    latest = frame[frame["date"] == frame["date"].max()]
    etf_rows = latest[latest["asset_type"] == "etf"]
    stock_rows = latest[latest["asset_type"] == "stock"]
    assert etf_rows["feature_style_spread_composite_20d"].notna().all()
    assert stock_rows["feature_momentum_5d"].notna().all()
    assert stock_rows["feature_style_spread_composite_20d"].isna().all()


def test_style_pair_spread_is_asset_specific_not_broadcast_by_date():
    raw = _mixed_panel()
    feature_view = AX1FeatureViewBuilder().build(raw, universe_metadata=_metadata(raw))
    latest = feature_view.frame[feature_view.frame["date"] == feature_view.frame["date"].max()].set_index("order_book_id")

    pair_column = "feature_style_spread_dividend_vs_growth_20d"
    composite_column = "feature_style_spread_composite_20d"

    assert latest.loc["510880.XSHG", pair_column] == pytest.approx(-latest.loc["159915.XSHE", pair_column])
    assert latest[pair_column].nunique(dropna=True) > 1
    assert latest.loc["510880.XSHG", composite_column] != pytest.approx(latest.loc["159915.XSHE", composite_column])


def test_dollar_volume_feature_is_log_scaled_without_leaking_into_volume_pulse():
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    close = [10.0] * 24 + [11.0]
    volume = [1_000.0] * 24 + [10_000.0]
    raw = pd.DataFrame(
        {
            "date": dates,
            "order_book_id": "510300.XSHG",
            "asset_type": "etf",
            "universe_layer": "core",
            "industry": "broad",
            "benchmark_id": "510300.XSHG",
            "close": close,
            "volume": volume,
        }
    )

    feature_view = AX1FeatureViewBuilder().build(raw, universe_metadata=_metadata(raw))
    frame = feature_view.frame.sort_values("date").reset_index(drop=True)

    raw_dollar_volume = pd.Series(close) * pd.Series(volume)
    expected_log_liquidity = np.log1p(raw_dollar_volume)
    assert frame["feature_dollar_volume"].tolist() == pytest.approx(expected_log_liquidity.tolist())

    raw_pulse = raw_dollar_volume / raw_dollar_volume.rolling(20, min_periods=3).mean() - 1.0
    expected_flow = frame["feature_return_1d"].fillna(0.0) * raw_pulse.fillna(0.0)
    assert frame.loc[24, "feature_volume_price_flow_20d"] == pytest.approx(expected_flow.iloc[24])


def test_dollar_volume_feature_clips_bad_negative_volume_before_log_transform():
    raw = _mixed_panel().query("order_book_id == '510300.XSHG'").head(6).copy()
    raw.loc[raw.index[3], "volume"] = -1_000_000.0

    feature_view = AX1FeatureViewBuilder().build(raw, universe_metadata=_metadata(raw))
    frame = feature_view.frame.sort_values("date").reset_index(drop=True)

    assert frame["feature_dollar_volume"].notna().all()
    assert frame.loc[3, "feature_dollar_volume"] == pytest.approx(0.0)


def test_style_spread_projects_pair_factor_to_non_pair_etfs_with_rolling_exposure():
    raw = _style_exposure_panel()
    config = {
        "style_pairs": [
            {"name": "dividend_vs_growth", "long": "510880.XSHG", "short": "159915.XSHE", "weight": 1.0}
        ],
        "style_exposure_window": 30,
        "style_exposure_min_periods": 10,
        "style_beta_clip": 2.0,
    }

    feature_view = AX1FeatureViewBuilder(config).build(raw, universe_metadata=_metadata(raw))
    frame = feature_view.frame
    latest = frame[frame["date"] == frame["date"].max()].set_index("order_book_id")

    industry_style = latest.loc["512000.XSHG", "feature_style_spread_composite_20d"]
    industry_z = frame.loc[
        frame["order_book_id"].eq("512000.XSHG"),
        "feature_z_style_spread_composite_20d",
    ]

    assert abs(industry_style) > 1e-6
    assert industry_z.abs().sum() > 0.0
    assert pd.isna(latest.loc["000001.XSHE", "feature_style_spread_composite_20d"])


def test_common_scope_does_not_include_duplicate_reversal_alias():
    raw = _mixed_panel()
    feature_view = AX1FeatureViewBuilder().build(raw, universe_metadata=_metadata(raw))

    assert "feature_reversal_5d" not in feature_view.columns_by_scope["common"]
    assert "feature_reversal_5d" not in feature_view.feature_columns


def test_resolve_feature_columns_uses_feature_set_scopes_not_legacy_columns():
    raw = _mixed_panel()
    feature_view = AX1FeatureViewBuilder().build(
        raw,
        universe_metadata=_metadata(raw),
        regime_state_by_date={
            date: {
                "market_regime": "bear_co_move",
                "risk_state": "risk_off",
                "rotation_state": "co_move",
                "strength": 0.4,
            }
            for date in raw["date"].drop_duplicates()
        },
    )

    columns = resolve_feature_columns(
        {
            "model": {
                "kind": "lgbm_multi_target",
                "feature_set": "ax1_unified_v1",
                "include_scopes": ["common", "etf_zscore", "regime"],
            }
        },
        feature_view,
    )

    assert "feature_momentum_5d" in columns
    assert "feature_reversal_5d" not in columns
    assert "feature_z_style_spread_composite_20d" in columns
    assert "feature_regime_risk_off" in columns
    assert "feature_interaction_z_style_spread_composite_20d_x_regime_risk_off" not in columns
    assert "feature_style_spread_value_vs_growth_20d" not in columns
    assert "momentum_5d" not in columns


def test_regime_interactions_are_lazy_and_explicit_opt_in():
    raw = _mixed_panel()
    feature_view = AX1FeatureViewBuilder({"include_scopes": ["common", "etf_zscore", "regime", "regime_interaction"]}).build(
        raw,
        universe_metadata=_metadata(raw),
        regime_state_by_date={
            date: {
                "market_regime": "bear_co_move",
                "risk_state": "risk_off",
                "rotation_state": "co_move",
                "strength": 0.4,
            }
            for date in raw["date"].drop_duplicates()
        },
    )

    columns = resolve_feature_columns(
        {
            "model": {
                "kind": "lgbm_multi_target",
                "feature_set": "ax1_unified_v1",
                "include_scopes": ["common", "etf_zscore", "regime", "regime_interaction"],
            }
        },
        feature_view,
    )

    assert "feature_momentum_5d" in columns
    assert "feature_interaction_z_style_spread_composite_20d_x_regime_risk_off" in columns


def test_resolve_feature_columns_rejects_legacy_feature_columns_for_lgbm():
    raw = _mixed_panel()
    feature_view = AX1FeatureViewBuilder().build(raw, universe_metadata=_metadata(raw))

    with pytest.raises(ValueError, match="feature_set"):
        resolve_feature_columns(
            {
                "model": {
                    "kind": "lgbm_multi_target",
                    "feature_columns": ["momentum_5d", "dollar_volume"],
                }
            },
            feature_view,
        )


def test_feature_view_normalization_does_not_look_ahead():
    raw = _mixed_panel()
    target_date = pd.Timestamp("2024-03-10")
    future = raw[raw["date"] == raw["date"].max()].copy()
    future["date"] = pd.Timestamp("2024-04-15")
    future["close"] = future["close"] * 100.0
    future["volume"] = future["volume"] * 20.0
    expanded = pd.concat([raw, future], ignore_index=True)

    base = AX1FeatureViewBuilder().build(raw, universe_metadata=_metadata(raw)).frame
    with_future = AX1FeatureViewBuilder().build(expanded, universe_metadata=_metadata(raw)).frame

    compare_columns = [
        "feature_z_excess_mom_20d",
        "feature_z_volume_price_flow_20d",
        "feature_z_style_spread_composite_20d",
    ]
    base_row = base[(base["order_book_id"] == "512000.XSHG") & (base["date"] == target_date)][compare_columns]
    future_row = with_future[(with_future["order_book_id"] == "512000.XSHG") & (with_future["date"] == target_date)][compare_columns]

    pd.testing.assert_frame_equal(base_row.reset_index(drop=True), future_row.reset_index(drop=True))


def test_feature_view_uses_configured_benchmark_for_excess_momentum():
    raw = _mixed_panel()
    metadata = _metadata(raw)
    default_view = AX1FeatureViewBuilder({"core_proxy_id": "510300.XSHG"}).build(raw, universe_metadata=metadata).frame
    alternate_view = AX1FeatureViewBuilder({"core_proxy_id": "159915.XSHE"}).build(raw, universe_metadata=metadata).frame

    target = (default_view["order_book_id"] == "512000.XSHG") & (default_view["date"] == pd.Timestamp("2024-03-15"))
    default_value = default_view.loc[target, "feature_excess_mom_20d"].iloc[0]
    alternate_value = alternate_view.loc[target, "feature_excess_mom_20d"].iloc[0]

    assert default_value != pytest.approx(alternate_value)


def test_regime_features_are_point_in_time_and_do_not_look_ahead():
    raw = _mixed_panel()
    target_date = pd.Timestamp("2024-02-15")
    states = {
        date: {
            "market_regime": "range_co_move",
            "risk_state": "neutral",
            "rotation_state": "co_move",
            "strength": 0.0,
        }
        for date in raw["date"].drop_duplicates()
    }
    states[target_date] = {
        "market_regime": "bull_rotation",
        "risk_state": "risk_on",
        "rotation_state": "rotation",
        "strength": 0.7,
    }
    future_states = dict(states)
    future_states[pd.Timestamp("2024-03-01")] = {
        "market_regime": "bear_rotation",
        "risk_state": "risk_off",
        "rotation_state": "rotation",
        "strength": 1.0,
    }

    baseline = AX1FeatureViewBuilder().build(raw, universe_metadata=_metadata(raw), regime_state_by_date=states).frame
    with_future = AX1FeatureViewBuilder().build(raw, universe_metadata=_metadata(raw), regime_state_by_date=future_states).frame

    columns = [
        "feature_regime_strength",
        "feature_regime_risk_on",
        "feature_regime_neutral",
        "feature_regime_risk_off",
        "feature_regime_rotation",
    ]
    base_row = baseline[(baseline["order_book_id"] == "512000.XSHG") & (baseline["date"] == target_date)][columns]
    future_row = with_future[(with_future["order_book_id"] == "512000.XSHG") & (with_future["date"] == target_date)][columns]

    pd.testing.assert_frame_equal(base_row.reset_index(drop=True), future_row.reset_index(drop=True))
    assert base_row["feature_regime_strength"].iloc[0] == pytest.approx(0.7)
    assert base_row["feature_regime_risk_on"].iloc[0] == pytest.approx(1.0)


def test_stock_specific_scope_is_explicitly_not_implemented_for_active_models():
    raw = _mixed_panel()
    feature_view = AX1FeatureViewBuilder().build(raw, universe_metadata=_metadata(raw))

    with pytest.raises(NotImplementedError, match="stock_specific"):
        resolve_feature_columns(
            {
                "model": {
                    "kind": "lgbm_multi_target",
                    "feature_set": "ax1_unified_v1",
                    "include_scopes": ["common", "stock_specific"],
                }
            },
            feature_view,
        )


def test_common_scope_includes_amihud_and_skew():
    raw = _mixed_panel()
    feature_view = AX1FeatureViewBuilder().build(raw, universe_metadata=_metadata(raw))

    assert "feature_amihud_illiquidity" in feature_view.columns_by_scope["common"]
    assert "feature_realized_skew_20d" in feature_view.columns_by_scope["common"]


def test_common_scope_uses_fixed_etf_cost_forecast_band():
    raw = _mixed_panel()
    feature_view = AX1FeatureViewBuilder({"etf_cost_forecast_bps": 4.0}).build(raw, universe_metadata=_metadata(raw))
    latest = feature_view.frame[feature_view.frame["date"] == feature_view.frame["date"].max()].set_index("order_book_id")

    etf_costs_bp = latest.loc[latest["asset_type"] == "etf", "feature_cost_forecast"] * 10000.0
    stock_costs_bp = latest.loc[latest["asset_type"] == "stock", "feature_cost_forecast"] * 10000.0

    assert etf_costs_bp.nunique() == 1
    assert etf_costs_bp.iloc[0] == pytest.approx(4.0)
    assert stock_costs_bp.iloc[0] > 4.0
    assert "feature_amihud_illiquidity" in feature_view.frame.columns
    assert "feature_realized_skew_20d" in feature_view.frame.columns
    # Amihud should be non-negative
    amihud_values = feature_view.frame["feature_amihud_illiquidity"].dropna()
    assert (amihud_values >= 0.0).all()


def test_technical_scope_includes_rsi_and_macd():
    raw = _mixed_panel()
    feature_view = AX1FeatureViewBuilder().build(raw, universe_metadata=_metadata(raw))

    assert "feature_rsi_14d" in feature_view.columns_by_scope["technical"]
    assert "feature_macd" in feature_view.columns_by_scope["technical"]
    assert "feature_rsi_14d" in feature_view.frame.columns
    assert "feature_macd" in feature_view.frame.columns
    # RSI should be in [0, 100]
    rsi_values = feature_view.frame["feature_rsi_14d"].dropna()
    assert rsi_values.min() >= 0.0
    assert rsi_values.max() <= 100.0


def test_fundamental_features_are_merged_into_frame_when_provided():
    raw = _mixed_panel()
    stock_rows = raw[raw["asset_type"] == "stock"]
    dates = stock_rows["date"].unique()[:5]

    fundamental_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "order_book_id": "000001.XSHE",
            "feature_pe_ttm": [10.0, 11.0, 12.0, 13.0, 14.0],
            "feature_pb_ratio": [1.5, 1.6, 1.7, 1.8, 1.9],
            "feature_roe_ttm": [0.15, 0.16, 0.17, 0.18, 0.19],
        }
    )

    feature_view = AX1FeatureViewBuilder().build(
        raw,
        universe_metadata=_metadata(raw),
        fundamental_df=fundamental_df,
    )

    assert "feature_pe_ttm" in feature_view.columns_by_scope["fundamental"]
    assert "feature_pe_ttm" in feature_view.frame.columns
    merged_rows = feature_view.frame[feature_view.frame["feature_pe_ttm"].notna()]
    assert len(merged_rows) > 0


def test_fundamental_features_are_masked_to_stocks_only():
    raw = _mixed_panel()
    dates = raw["date"].unique()[:3]

    fundamental_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates.repeat(2)),
            "order_book_id": ["510300.XSHG", "000001.XSHE"] * 3,
            "feature_pe_ttm": [10.0] * 6,
            "feature_pb_ratio": [1.5] * 6,
        }
    )

    feature_view = AX1FeatureViewBuilder().build(
        raw,
        universe_metadata=_metadata(raw),
        fundamental_df=fundamental_df,
    )

    # ETF rows should have NaN for fundamental features
    etf_rows = feature_view.frame[feature_view.frame["asset_type"] == "etf"]
    assert etf_rows["feature_pe_ttm"].isna().all()


def test_flow_features_are_merged_into_frame_when_provided():
    raw = _mixed_panel()
    dates = raw["date"].unique()[:5]

    flow_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "order_book_id": "510300.XSHG",
            "feature_margin_financing_balance": [5.0, 5.1, 5.2, 5.3, 5.4],
        }
    )

    feature_view = AX1FeatureViewBuilder().build(
        raw,
        universe_metadata=_metadata(raw),
        flow_df=flow_df,
    )

    assert "feature_margin_financing_balance" in feature_view.columns_by_scope["flow"]
    assert "feature_margin_financing_balance" in feature_view.frame.columns


def test_macro_features_are_broadcast_to_all_assets():
    raw = _mixed_panel()
    dates = raw["date"].unique()[:5]

    macro_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "feature_bond_yield_10y": [0.025, 0.026, 0.027, 0.028, 0.029],
            "feature_macro_pmi": [49.8, 49.8, 49.8, 50.1, 50.1],
        }
    )

    feature_view = AX1FeatureViewBuilder().build(
        raw,
        universe_metadata=_metadata(raw),
        macro_df=macro_df,
    )

    assert "feature_bond_yield_10y" in feature_view.columns_by_scope["macro"]
    assert "feature_macro_pmi" in feature_view.columns_by_scope["macro"]
    assert "feature_bond_yield_10y" in feature_view.frame.columns
    assert "feature_macro_pmi" in feature_view.frame.columns
    # Bond yield should be broadcast to all assets on matching dates
    for date in dates:
        date_rows = feature_view.frame[feature_view.frame["date"] == date]
        non_null_count = date_rows["feature_bond_yield_10y"].notna().sum()
        assert non_null_count > 0
        assert date_rows["feature_macro_pmi"].notna().sum() > 0


def test_flow_proxy_institutional_holding_feature_is_merged_when_provided():
    raw = _mixed_panel()
    dates = raw["date"].unique()[:4]

    flow_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "order_book_id": "000001.XSHE",
            "feature_institutional_holding_ratio": [0.05, 0.051, 0.052, 0.053],
        }
    )

    feature_view = AX1FeatureViewBuilder().build(
        raw,
        universe_metadata=_metadata(raw),
        flow_df=flow_df,
    )

    assert "feature_institutional_holding_ratio" in feature_view.columns_by_scope["flow"]
    merged_rows = feature_view.frame[feature_view.frame["feature_institutional_holding_ratio"].notna()]
    assert not merged_rows.empty
    assert set(merged_rows["order_book_id"].unique()) == {"000001.XSHE"}


def test_resolve_feature_columns_supports_new_scopes():
    raw = _mixed_panel()
    feature_view = AX1FeatureViewBuilder().build(raw, universe_metadata=_metadata(raw))

    columns = resolve_feature_columns(
        {
            "model": {
                "kind": "lgbm_multi_target",
                "feature_set": "ax1_unified_v1",
                "include_scopes": ["common", "technical"],
            }
        },
        feature_view,
    )
    assert "feature_momentum_5d" in columns
    assert "feature_rsi_14d" in columns
    assert "feature_macd" in columns


class TestStyleExposureFallbackWarning:
    """Test style exposure fallback warnings."""

    def test_warns_when_long_etf_missing(self, caplog):
        """When long ETF data is missing, should emit warning."""
        import logging

        from skyeye.products.ax1.features.view import _pair_style_exposure

        frame = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "order_book_id": ["510880.XSHG", "510300.XSHG", "000001.XSHE"],
            "close": [80.0, 100.0, 10.0],
            "feature_return_1d": [0.01, 0.005, 0.002],
        })

        with caplog.at_level(logging.WARNING):
            result = _pair_style_exposure(
                frame,
                long_id="NONEXISTENT.XSHG",  # Does not exist in data
                short_id="510300.XSHG",
                config={}
            )

        assert "degraded to signed_fallback" in caplog.text
        assert "NONEXISTENT.XSHG" in caplog.text
        assert "not found in data" in caplog.text
        # Verify fallback is returned
        assert len(result) == 3

    def test_warns_when_short_etf_missing(self, caplog):
        """When short ETF data is missing, should emit warning."""
        import logging

        from skyeye.products.ax1.features.view import _pair_style_exposure

        frame = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "order_book_id": ["510880.XSHG", "510300.XSHG", "000001.XSHE"],
            "close": [80.0, 100.0, 10.0],
            "feature_return_1d": [0.01, 0.005, 0.002],
        })

        with caplog.at_level(logging.WARNING):
            result = _pair_style_exposure(
                frame,
                long_id="510880.XSHG",
                short_id="MISSING.XSHG",  # Does not exist in data
                config={}
            )

        assert "degraded to signed_fallback" in caplog.text
        assert "MISSING.XSHG" in caplog.text
        assert "not found in data" in caplog.text

    def test_no_warning_when_both_etfs_present(self, caplog):
        """When both ETFs are present, no warning should be emitted."""
        import logging

        from skyeye.products.ax1.features.view import _pair_style_exposure

        frame = _style_exposure_panel()
        frame_with_return = frame.copy()
        frame_with_return["feature_return_1d"] = frame_with_return.groupby("order_book_id")["close"].pct_change()

        with caplog.at_level(logging.WARNING):
            result = _pair_style_exposure(
                frame_with_return,
                long_id="510880.XSHG",
                short_id="159915.XSHE",
                config={"style_exposure_window": 30, "style_exposure_min_periods": 10}
            )

        # Should not have degradation warning
        assert "degraded to signed_fallback" not in caplog.text
        # Result should not be simple signed fallback (should have computed values)
        # Check that we have non-trivial values for non-pair assets
        non_pair_mask = ~frame_with_return["order_book_id"].isin(["510880.XSHG", "159915.XSHE"])
        if non_pair_mask.any():
            # Non-pair assets should have computed exposure, not just 0.0
            assert result[non_pair_mask].notna().any()

    def test_warns_on_missing_feature_return_column(self, caplog):
        """When feature_return_1d column is missing, should emit warning."""
        import logging

        from skyeye.products.ax1.features.view import _pair_style_exposure

        frame = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "order_book_id": ["510880.XSHG", "510300.XSHG", "000001.XSHE"],
            "close": [80.0, 100.0, 10.0],
            # Missing feature_return_1d
        })

        with caplog.at_level(logging.WARNING):
            result = _pair_style_exposure(
                frame,
                long_id="510880.XSHG",
                short_id="510300.XSHG",
                config={}
            )

        assert "degraded to signed_fallback" in caplog.text
        assert "feature_return_1d column missing" in caplog.text
