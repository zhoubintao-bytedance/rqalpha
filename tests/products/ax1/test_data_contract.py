def test_data_audit_allows_default_point_in_time_price_features():
    from skyeye.products.ax1.data_contract import audit_feature_set
    from skyeye.products.ax1.features.catalog import build_default_feature_catalog

    catalog = build_default_feature_catalog()

    report = audit_feature_set(
        ["feature_momentum_5d", "feature_z_excess_mom_20d", "feature_regime_strength"],
        catalog=catalog,
        purpose="promotable_training",
    )

    assert report["passed"] is True
    assert report["hard_block_count"] == 0
    assert report["warning_count"] == 0


def test_data_audit_blocks_after_close_price_features_without_execution_lag():
    from skyeye.products.ax1.data_contract import audit_feature_set
    from skyeye.products.ax1.features.catalog import FeatureCatalog, FeatureDefinition

    catalog = FeatureCatalog(
        [
            FeatureDefinition(
                name="feature_momentum_5d",
                scope="common",
                asset_type="both",
                source_family="price_volume",
                required_columns=("close",),
                observable_lag_days=0,
                decision_time="after_close",
                tradable_lag_days=0,
                status="implemented",
            )
        ]
    )

    report = audit_feature_set(
        ["feature_momentum_5d"],
        catalog=catalog,
        purpose="promotable_training",
    )

    assert report["passed"] is False
    assert report["hard_blocks"][0]["reason_code"] == "after_close_price_feature_requires_execution_lag"


def test_data_audit_blocks_latest_index_snapshot_and_unlagged_fundamental():
    from skyeye.products.ax1.data_contract import audit_feature_set
    from skyeye.products.ax1.features.catalog import FeatureCatalog, FeatureDefinition

    catalog = FeatureCatalog(
        [
            FeatureDefinition(
                name="feature_index_weight_latest",
                scope="index_membership",
                asset_type="stock",
                source_family="index_snapshot",
                observable_lag_days=0,
                requires_as_of_date=True,
                uses_latest_snapshot=True,
                status="implemented",
            ),
            FeatureDefinition(
                name="feature_pe_ttm",
                scope="fundamental",
                asset_type="stock",
                source_family="fundamental",
                observable_lag_days=0,
                status="implemented",
            ),
        ]
    )

    report = audit_feature_set(
        ["feature_index_weight_latest", "feature_pe_ttm"],
        catalog=catalog,
        purpose="promotable_training",
    )

    assert report["passed"] is False
    reason_codes = {item["reason_code"] for item in report["hard_blocks"]}
    assert "latest_snapshot_not_point_in_time" in reason_codes
    assert "fundamental_observable_lag_too_short" in reason_codes


def test_universe_audit_blocks_latest_snapshot_metadata_for_promotion():
    import pandas as pd

    from skyeye.products.ax1.data_contract import audit_universe_metadata

    metadata = pd.DataFrame(
        {
            "order_book_id": ["510300.XSHG"],
            "asset_type": ["etf"],
            "universe_layer": ["core"],
            "universe_pit_status": ["latest_snapshot"],
        }
    )
    metadata.attrs["pit_audit"] = {
        "passed": False,
        "hard_blocks": [
            {
                "reason_code": "universe_latest_snapshot_not_point_in_time",
                "message": "universe built without explicit as-of date",
            }
        ],
        "warnings": [],
    }

    report = audit_universe_metadata(metadata, purpose="promotable_training")

    assert report["passed"] is False
    assert report["hard_blocks"][0]["reason_code"] == "universe_latest_snapshot_not_point_in_time"


def test_data_audit_marks_akshare_only_northbound_as_experimental():
    from skyeye.products.ax1.data_contract import audit_feature_set
    from skyeye.products.ax1.features.catalog import FeatureCatalog, FeatureDefinition

    catalog = FeatureCatalog(
        [
            FeatureDefinition(
                name="feature_northbound_net_flow",
                scope="northbound",
                asset_type="both",
                source_family="northbound_akshare",
                observable_lag_days=1,
                status="experimental",
            )
        ]
    )

    research_report = audit_feature_set(
        ["feature_northbound_net_flow"],
        catalog=catalog,
        purpose="research",
    )
    assert research_report["passed"] is True
    assert research_report["warnings"][0]["reason_code"] == "experimental_unstable_source"

    promotion_report = audit_feature_set(
        ["feature_northbound_net_flow"],
        catalog=catalog,
        purpose="promotable_training",
    )
    assert promotion_report["passed"] is False
    assert promotion_report["hard_blocks"][0]["reason_code"] == "experimental_source_not_promotable"


def test_macro_pmi_and_technical_features_pass_training_audit_once_implemented():
    from skyeye.products.ax1.data_contract import audit_feature_set
    from skyeye.products.ax1.features.catalog import build_default_feature_catalog

    report = audit_feature_set(
        ["feature_pe_ttm", "feature_macro_pmi", "feature_rsi_14d"],
        catalog=build_default_feature_catalog(),
        purpose="promotable_training",
    )

    assert report["passed"] is True
    assert report["hard_blocks"] == []
