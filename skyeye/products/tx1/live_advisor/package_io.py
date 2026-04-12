# -*- coding: utf-8 -*-
"""TX1 live advisor promoted package 的读写工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skyeye.products.tx1.live_advisor.schema import (
    LIVE_PACKAGE_FILENAMES,
    LIVE_PACKAGE_OPTIONAL_COMPONENTS,
    validate_live_package_payload,
)


DEFAULT_TX1_PACKAGES_ROOT = (
    Path(__file__).resolve().parents[4] / "artifacts" / "packages" / "tx1"
)


def resolve_packages_root(packages_root: str | Path | None = None) -> Path:
    """解析 TX1 live package 根目录。"""
    if packages_root is None:
        return DEFAULT_TX1_PACKAGES_ROOT
    return Path(packages_root)


def package_root_for_id(
    package_id: str,
    packages_root: str | Path | None = None,
) -> Path:
    """根据 package_id 计算产包目录。"""
    if not package_id:
        raise ValueError("package_id must not be empty")
    return resolve_packages_root(packages_root) / package_id


def package_component_path(
    package_ref: str | Path,
    component_name: str,
    packages_root: str | Path | None = None,
) -> Path:
    """根据组件名返回组件文件路径。"""
    if component_name not in LIVE_PACKAGE_FILENAMES:
        raise KeyError("unknown live package component: {}".format(component_name))
    package_root = _resolve_package_root(package_ref, packages_root=packages_root)
    return package_root / LIVE_PACKAGE_FILENAMES[component_name]


def build_live_package_payload(
    *,
    manifest: dict,
    feature_schema: dict,
    preprocessor_bundle: dict,
    model_bundle: dict,
    calibration_bundle: dict,
    portfolio_policy: dict,
    recent_canary_bundle: dict | None = None,
) -> dict[str, Any]:
    """构造并校验 live package 六件套。"""
    payload = {
        "manifest": manifest,
        "feature_schema": feature_schema,
        "preprocessor_bundle": preprocessor_bundle,
        "model_bundle": model_bundle,
        "calibration_bundle": calibration_bundle,
        "portfolio_policy": portfolio_policy,
    }
    if recent_canary_bundle is not None:
        payload["recent_canary_bundle"] = recent_canary_bundle
    validate_live_package_payload(payload)
    return payload


def save_live_package(
    payload: dict,
    packages_root: str | Path | None = None,
) -> Path:
    """将 live package 六件套落盘到标准目录。"""
    validate_live_package_payload(payload)
    package_id = payload["manifest"]["package_id"]
    package_root = package_root_for_id(package_id, packages_root=packages_root)
    package_root.mkdir(parents=True, exist_ok=True)

    for component_name, filename in LIVE_PACKAGE_FILENAMES.items():
        if component_name in LIVE_PACKAGE_OPTIONAL_COMPONENTS and component_name not in payload:
            continue
        component_path = package_root / filename
        with component_path.open("w", encoding="utf-8") as handle:
            json.dump(payload[component_name], handle, indent=2, ensure_ascii=False, default=str)

    return package_root


def load_live_package(
    package_ref: str | Path,
    packages_root: str | Path | None = None,
) -> dict[str, Any]:
    """从标准目录加载 live package 六件套。"""
    package_root = _resolve_package_root(package_ref, packages_root=packages_root)
    payload = {}
    for component_name in LIVE_PACKAGE_FILENAMES:
        component_path = package_component_path(
            package_root,
            component_name,
            packages_root=packages_root,
        )
        if component_name in LIVE_PACKAGE_OPTIONAL_COMPONENTS and not component_path.is_file():
            continue
        if not component_path.is_file():
            raise FileNotFoundError("missing live package component: {}".format(component_path))
        with component_path.open("r", encoding="utf-8") as handle:
            payload[component_name] = json.load(handle)
    validate_live_package_payload(payload)
    return payload


def _resolve_package_root(
    package_ref: str | Path,
    packages_root: str | Path | None = None,
) -> Path:
    """兼容 package_id 和绝对目录路径两种引用方式。"""
    package_path = Path(package_ref)
    if package_path.is_absolute():
        return package_path
    package_text = str(package_ref)
    if "/" in package_text or package_text.startswith("."):
        return package_path.resolve()
    return package_root_for_id(package_text, packages_root=packages_root)
