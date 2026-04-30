import pytest


def test_default_feature_catalog_describes_current_etf_and_regime_features():
    from skyeye.market_regime_layer import MarketRegimeConfig, required_market_regime_history_days
    from skyeye.products.ax1.features.catalog import build_default_feature_catalog

    catalog = build_default_feature_catalog()
    expected_regime_lookback = required_market_regime_history_days(MarketRegimeConfig())

    momentum = catalog.get("feature_momentum_5d")
    assert momentum.scope == "common"
    assert momentum.asset_type == "both"
    assert momentum.source_family == "price_volume"
    assert momentum.observable_lag_days == 0
    assert momentum.to_dict()["decision_time"] == "after_close"
    assert momentum.to_dict()["tradable_lag_days"] == 1
    assert momentum.status == "implemented"

    style_z = catalog.get("feature_z_style_spread_composite_20d")
    assert style_z.scope == "etf_zscore"
    assert style_z.asset_type == "etf"
    assert style_z.depends_on == ("feature_style_spread_composite_20d",)

    interaction = catalog.get("feature_interaction_z_style_spread_composite_20d_x_regime_risk_on")
    assert interaction.scope == "regime_interaction"
    assert interaction.asset_type == "etf"
    assert "feature_regime_risk_on" in interaction.depends_on
    assert interaction.lookback_days == expected_regime_lookback

    regime_strength = catalog.get("feature_regime_strength")
    assert regime_strength.lookback_days == expected_regime_lookback

    active = catalog.names_for_scopes(["common", "etf_zscore", "regime"])
    assert "feature_momentum_5d" in active
    assert "feature_z_excess_mom_20d" in active
    assert "feature_regime_strength" in active


def test_feature_catalog_rejects_unknown_or_unimplemented_active_features():
    from skyeye.products.ax1.features.catalog import build_default_feature_catalog

    catalog = build_default_feature_catalog()

    with pytest.raises(ValueError, match="unknown AX1 feature.*does_not_exist"):
        catalog.require_active(["does_not_exist"])


def test_feature_catalog_marks_macro_and_institutional_proxy_features_as_implemented():
    from skyeye.products.ax1.features.catalog import build_default_feature_catalog

    catalog = build_default_feature_catalog()

    macro_pmi = catalog.get("feature_macro_pmi")
    assert macro_pmi.scope == "macro"
    assert macro_pmi.status == "implemented"
    assert macro_pmi.to_dict()["data_source_status"] == "implemented"
    assert macro_pmi.observable_lag_days == 1

    institutional = catalog.get("feature_institutional_holding_ratio")
    assert institutional.scope == "flow"
    assert institutional.asset_type == "stock"
    assert institutional.status == "implemented"
    assert institutional.to_dict()["data_source_status"] == "implemented"
    assert institutional.observable_lag_days == 1

    catalog.require_active(["feature_macro_pmi", "feature_institutional_holding_ratio"])


def test_feature_catalog_has_no_remaining_not_implemented_features():
    from skyeye.products.ax1.features.catalog import build_default_feature_catalog

    catalog = build_default_feature_catalog()

    not_implemented = [
        name
        for name in catalog.names_for_scopes(["common", "technical", "fundamental", "flow", "macro"])
        if catalog.get(name).status == "not_implemented"
        or catalog.get(name).to_dict()["data_source_status"] == "not_implemented"
    ]

    assert not_implemented == []

def test_feature_catalog_declares_fundamental_and_flow_as_implemented():
    from skyeye.products.ax1.features.catalog import build_default_feature_catalog

    catalog = build_default_feature_catalog()

    # Test implemented fundamental and flow features
    implemented_features = {
        "feature_pe_ttm": "fundamental",
        "feature_pb_ratio": "fundamental",
        "feature_roe_ttm": "fundamental",
        "feature_margin_financing_balance": "flow",
        "feature_northbound_net_flow": "flow",
        "feature_institutional_holding_ratio": "flow",
    }
    for feature_name, source_family in implemented_features.items():
        definition = catalog.get(feature_name)
        assert definition.source_family == source_family
        assert definition.status == "implemented"
        assert definition.to_dict()["data_source_status"] == "implemented"
        assert definition.observable_lag_days == 1
