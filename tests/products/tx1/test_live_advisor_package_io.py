from pathlib import Path

import pytest

from skyeye.products.tx1.artifacts import resolve_live_package
from skyeye.products.tx1.live_advisor.package_io import (
    build_live_package_payload,
    load_live_package,
    package_component_path,
    save_live_package,
)


def test_save_and_load_live_package_round_trip(tmp_path):
    """验证 promoted package 六件套可以完整往返读写。"""
    payload = build_live_package_payload(
        manifest={
            "package_id": "tx1_live_canary_demo",
            "package_type": "canary_live",
            "source_experiment": "tx1_refresh_demo",
            "horizon": 20,
            "fit_end_date": "2026-03-03",
            "data_end_date": "2026-03-31",
            "created_at": "2026-04-12T00:00:00",
            "model_kind": "linear",
            "required_features": ["mom_40d", "volatility_20d"],
            "hashes": {
                "feature_schema": "sha256:feature",
                "preprocessor_bundle": "sha256:preproc",
                "model_bundle": "sha256:model",
                "calibration_bundle": "sha256:calibration",
                "portfolio_policy": "sha256:policy",
            },
            "gate_summary": {"gate_level": "canary_live", "passed": True},
        },
        feature_schema={
            "feature_columns": ["mom_40d", "volatility_20d"],
            "label_horizon": 20,
        },
        preprocessor_bundle={"enabled": False},
        model_bundle={"kind": "linear", "state": {"coef": [0.1, 0.2, 0.3]}},
        calibration_bundle={
            "bucket_edges": [0.0, 0.5, 1.0],
            "bucket_stats": [
                {"bucket_id": "b0", "sample_count": 320, "win_rate": 0.55},
                {"bucket_id": "b1", "sample_count": 330, "win_rate": 0.62},
            ],
        },
        portfolio_policy={"buy_top_k": 25, "hold_top_k": 45},
    )

    package_root = save_live_package(payload, packages_root=tmp_path)
    loaded = load_live_package(package_root)
    resolved = resolve_live_package("tx1_live_canary_demo", packages_root=tmp_path)

    assert package_root == tmp_path / "tx1_live_canary_demo"
    assert loaded["manifest"]["package_id"] == "tx1_live_canary_demo"
    assert loaded["feature_schema"]["feature_columns"] == ["mom_40d", "volatility_20d"]
    assert loaded["calibration_bundle"]["bucket_stats"][1]["win_rate"] == 0.62
    assert resolved.package_id == "tx1_live_canary_demo"
    assert resolved.package_root == package_root
    assert resolved.manifest["package_type"] == "canary_live"


def test_resolve_live_package_rejects_missing_required_component(tmp_path):
    """验证 live package 缺关键文件时 resolver 会直接失败。"""
    package_root = tmp_path / "tx1_live_missing_calibration"
    package_root.mkdir(parents=True)

    payload = build_live_package_payload(
        manifest={
            "package_id": "tx1_live_missing_calibration",
            "package_type": "canary_live",
            "source_experiment": "tx1_refresh_demo",
            "horizon": 20,
            "fit_end_date": "2026-03-03",
            "data_end_date": "2026-03-31",
            "created_at": "2026-04-12T00:00:00",
            "model_kind": "linear",
            "required_features": ["mom_40d"],
            "hashes": {
                "feature_schema": "sha256:feature",
                "preprocessor_bundle": "sha256:preproc",
                "model_bundle": "sha256:model",
                "calibration_bundle": "sha256:calibration",
                "portfolio_policy": "sha256:policy",
            },
            "gate_summary": {"gate_level": "canary_live", "passed": True},
        },
        feature_schema={"feature_columns": ["mom_40d"], "label_horizon": 20},
        preprocessor_bundle={"enabled": False},
        model_bundle={"kind": "linear", "state": {"coef": [0.1, 0.2]}},
        calibration_bundle={"bucket_edges": [0.0, 1.0], "bucket_stats": []},
        portfolio_policy={"buy_top_k": 25, "hold_top_k": 45},
    )

    for component_name in ("manifest", "feature_schema", "preprocessor_bundle", "model_bundle", "portfolio_policy"):
        component_path = package_component_path(package_root, component_name)
        component_path.write_text(
            __import__("json").dumps(payload[component_name], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    with pytest.raises(FileNotFoundError):
        resolve_live_package("tx1_live_missing_calibration", packages_root=tmp_path)


def test_load_live_package_accepts_absolute_package_path(tmp_path):
    """验证 package 既可以按 package_id 解析，也可以直接按绝对路径加载。"""
    payload = build_live_package_payload(
        manifest={
            "package_id": "tx1_live_path_demo",
            "package_type": "default_live",
            "source_experiment": "tx1_refresh_demo",
            "horizon": 20,
            "fit_end_date": "2026-03-03",
            "data_end_date": "2026-03-31",
            "created_at": "2026-04-12T00:00:00",
            "model_kind": "tree",
            "required_features": ["mom_40d"],
            "hashes": {
                "feature_schema": "sha256:feature",
                "preprocessor_bundle": "sha256:preproc",
                "model_bundle": "sha256:model",
                "calibration_bundle": "sha256:calibration",
                "portfolio_policy": "sha256:policy",
            },
            "gate_summary": {"gate_level": "default_live", "passed": True},
        },
        feature_schema={"feature_columns": ["mom_40d"], "label_horizon": 20},
        preprocessor_bundle={"enabled": False},
        model_bundle={"kind": "tree", "state": {"threshold": 0.1}},
        calibration_bundle={"bucket_edges": [0.0, 1.0], "bucket_stats": []},
        portfolio_policy={"buy_top_k": 25, "hold_top_k": 45},
    )

    package_root = save_live_package(payload, packages_root=tmp_path)
    resolved = resolve_live_package(str(package_root), packages_root=tmp_path)
    loaded = load_live_package(Path(package_root))

    assert resolved.package_root == package_root
    assert loaded["manifest"]["package_id"] == "tx1_live_path_demo"


def test_save_and_load_live_package_with_recent_canary_bundle(tmp_path):
    """验证 live package 可以额外携带 recent canary 证据。"""
    payload = build_live_package_payload(
        manifest={
            "package_id": "tx1_live_recent_canary_demo",
            "package_type": "canary_live",
            "source_experiment": "tx1_refresh_demo",
            "horizon": 20,
            "fit_end_date": "2026-03-03",
            "data_end_date": "2026-03-31",
            "created_at": "2026-04-12T00:00:00",
            "model_kind": "linear",
            "required_features": ["mom_40d"],
            "hashes": {
                "feature_schema": "sha256:feature",
                "preprocessor_bundle": "sha256:preproc",
                "model_bundle": "sha256:model",
                "calibration_bundle": "sha256:calibration",
                "portfolio_policy": "sha256:policy",
                "recent_canary_bundle": "sha256:recent",
            },
            "gate_summary": {"gate_level": "canary_live", "passed": True},
        },
        feature_schema={"feature_columns": ["mom_40d"], "label_horizon": 20},
        preprocessor_bundle={"enabled": False},
        model_bundle={"kind": "linear", "state": {"coef": [0.1]}},
        calibration_bundle={"bucket_edges": [0.0, 1.0], "bucket_stats": []},
        portfolio_policy={"buy_top_k": 25, "hold_top_k": 45},
        recent_canary_bundle={
            "window": {"start_date": "2026-02-01", "end_date": "2026-03-03"},
            "bucket_edges": [0.0, 1.0],
            "bucket_stats": [{"bucket_id": "b0", "sample_count": 300, "win_rate": 0.58}],
        },
    )

    package_root = save_live_package(payload, packages_root=tmp_path)
    loaded = load_live_package(package_root)

    assert loaded["recent_canary_bundle"]["window"]["end_date"] == "2026-03-03"
