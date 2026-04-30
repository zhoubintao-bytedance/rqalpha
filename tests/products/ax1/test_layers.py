import pytest

from skyeye.products.ax1.layers import (
    LayerRegistry,
    LegacyLayerConfigError,
    UnknownAllocationLayerError,
    validate_no_legacy_layer_keys,
)


def make_layer_config():
    return {
        "universe": {
            "layers": {
                "core": {
                    "asset_type": "ETF",
                    "include": ["510300.XSHG", "159919.XSHE"],
                    "max_count": 5,
                    "exposure_group": "broad_beta",
                },
                "industry": {
                    "asset_type": "ETF",
                    "include": ["512000.XSHG"],
                    "enabled": True,
                    "exposure_group": "sector",
                },
                "stock_satellite": {
                    "asset_type": "stock",
                    "include": ["000001.XSHE"],
                    "enabled": True,
                    "max_count": 20,
                    "exposure_group": "stock_alpha",
                },
            },
        },
        "allocation": {
            "kind": "opportunity_pool_optimizer",
            "exposure_groups": {
                "broad_beta": {"max_weight": 0.65},
                "sector": {"max_weight": 0.55},
                "stock_alpha": {"max_weight": 0.10},
            },
        },
    }


def test_layer_registry_accepts_new_arbitrary_layers_and_exposes_exposure_metadata():
    config = make_layer_config()
    config["universe"]["layers"]["quality_income"] = {
        "asset_type": "ETF",
        "include": ["515000.XSHG"],
        "enabled": True,
        "max_count": 3,
        "exposure_group": "style_factor",
    }

    registry = LayerRegistry.from_config(config)

    assert registry.layer_name_for_id("515000.XSHG") == "quality_income"
    assert registry.spec("quality_income").max_count == 3
    assert registry.exposure_group_for_layer("quality_income") == "style_factor"
    assert registry.enabled_layer_names() == [
        "core",
        "industry",
        "stock_satellite",
        "quality_income",
    ]


def test_stock_satellite_is_canonical_layer_not_collapsed_to_stock_asset_type():
    registry = LayerRegistry.from_config(make_layer_config())

    assert registry.layer_name_for_id("000001.XSHE") == "stock_satellite"
    assert registry.infer_layer_name("000001.XSHE", asset_type="stock") == "stock_satellite"
    assert registry.default_layer_for_asset_type("stock") == "stock_satellite"


def test_registry_maps_explicit_ids_before_asset_type_defaults():
    registry = LayerRegistry.from_config(make_layer_config())

    assert registry.layer_name_for_id("512000.XSHG") == "industry"
    assert registry.infer_layer_name("512000.XSHG", asset_type="ETF") == "industry"
    assert registry.infer_layer_name("UNKNOWN_ETF", asset_type="ETF") == "core"


def test_legacy_layer_keys_are_rejected_without_compatibility_bridge():
    legacy_config = {
        "universe": {
            "core_etfs": ["510300.XSHG"],
            "industry_etfs": ["512000.XSHG"],
            "stock_satellite": {"enabled": True, "include": ["000001.XSHE"]},
        },
    }

    with pytest.raises(LegacyLayerConfigError, match="core_etfs.*industry_etfs.*stock_satellite"):
        validate_no_legacy_layer_keys(legacy_config)

    with pytest.raises(LegacyLayerConfigError):
        LayerRegistry.from_config(legacy_config)


def test_allocation_no_longer_accepts_legacy_layer_budgets():
    config = make_layer_config()
    config["allocation"]["layers"] = {"core": {"budget": 0.10}}

    with pytest.raises(UnknownAllocationLayerError, match="allocation.layers"):
        LayerRegistry.from_config(config)
