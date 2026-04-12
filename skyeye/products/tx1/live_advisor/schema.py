# -*- coding: utf-8 -*-
"""TX1 live advisor 产包协议定义。"""

from __future__ import annotations


LIVE_PACKAGE_REQUIRED_FILENAMES = {
    "manifest": "manifest.json",
    "feature_schema": "feature_schema.json",
    "preprocessor_bundle": "preprocessor_bundle.json",
    "model_bundle": "model_bundle.json",
    "calibration_bundle": "calibration_bundle.json",
    "portfolio_policy": "portfolio_policy.json",
}

LIVE_PACKAGE_OPTIONAL_FILENAMES = {
    "recent_canary_bundle": "recent_canary_bundle.json",
}

LIVE_PACKAGE_FILENAMES = {
    **LIVE_PACKAGE_REQUIRED_FILENAMES,
    **LIVE_PACKAGE_OPTIONAL_FILENAMES,
}

LIVE_PACKAGE_REQUIRED_COMPONENTS = tuple(LIVE_PACKAGE_REQUIRED_FILENAMES.keys())
LIVE_PACKAGE_OPTIONAL_COMPONENTS = tuple(LIVE_PACKAGE_OPTIONAL_FILENAMES.keys())

LIVE_PACKAGE_REQUIRED_MANIFEST_FIELDS = (
    "package_id",
    "package_type",
    "source_experiment",
    "horizon",
    "fit_end_date",
    "data_end_date",
    "created_at",
    "model_kind",
    "required_features",
    "hashes",
    "gate_summary",
)


def validate_live_package_manifest(manifest: dict) -> None:
    """校验 manifest 的关键字段是否齐全。"""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dict")
    missing_fields = [
        field_name
        for field_name in LIVE_PACKAGE_REQUIRED_MANIFEST_FIELDS
        if field_name not in manifest
    ]
    if missing_fields:
        raise ValueError(
            "live package manifest missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )
    if not manifest.get("package_id"):
        raise ValueError("live package manifest package_id must not be empty")
    if not isinstance(manifest.get("required_features"), list):
        raise ValueError("live package manifest required_features must be a list")
    if not isinstance(manifest.get("hashes"), dict):
        raise ValueError("live package manifest hashes must be a dict")
    if not isinstance(manifest.get("gate_summary"), dict):
        raise ValueError("live package manifest gate_summary must be a dict")


def validate_live_package_payload(payload: dict) -> None:
    """校验整包六件套是否齐全，避免运行期静默缺件。"""
    if not isinstance(payload, dict):
        raise ValueError("live package payload must be a dict")
    missing_components = [
        component_name
        for component_name in LIVE_PACKAGE_REQUIRED_COMPONENTS
        if component_name not in payload
    ]
    if missing_components:
        raise ValueError(
            "live package payload missing required components: {}".format(
                ", ".join(missing_components)
            )
        )
    validate_live_package_manifest(payload["manifest"])
    if "recent_canary_bundle" in payload and not isinstance(payload["recent_canary_bundle"], dict):
        raise ValueError("live package payload recent_canary_bundle must be a dict")
