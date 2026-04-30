# -*- coding: utf-8 -*-
"""AX1 research package manifest helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from skyeye.products.ax1.persistence import make_json_compatible


MANIFEST_FILENAME = "manifest.json"
EXPERIMENT_REF_FILENAME = "experiment_ref.json"
CALIBRATION_BUNDLE_FILENAME = "calibration_bundle.json"
SCHEMA_VERSION = "1.0"


def build_package_manifest(
    experiment: dict,
    package_id: str | None = None,
) -> dict:
    """从实验结果构造 AX1 research package manifest。"""
    if not isinstance(experiment, dict):
        raise TypeError("experiment must be a dict")

    experiment_payload = make_json_compatible(experiment)
    resolved_package_id = package_id or experiment_payload.get("package_id")
    if not resolved_package_id:
        resolved_package_id = _default_package_id(experiment_payload)

    component_manifest = experiment_payload.get("component_manifest", {})
    return {
        "package_id": str(resolved_package_id),
        "product": "ax1",
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "source_experiment": _source_experiment(experiment_payload),
        "config": experiment_payload.get("config", {}),
        "feature_schema": _component_value(experiment_payload, component_manifest, "feature_schema"),
        "factor_schema": _component_value(experiment_payload, component_manifest, "factor_schema"),
        "model_schema": _component_value(experiment_payload, component_manifest, "model_schema"),
        "regime": _component_value(experiment_payload, component_manifest, "regime"),
        "signals": _component_value(experiment_payload, component_manifest, "signals"),
        "preprocessor": _component_value(experiment_payload, component_manifest, "preprocessor"),
        "splitter": _component_value(experiment_payload, component_manifest, "splitter"),
        "view_fusion": _component_value(experiment_payload, component_manifest, "view_fusion"),
        "risk_model": _component_value(experiment_payload, component_manifest, "risk_model"),
        "optimizer": _component_value(experiment_payload, component_manifest, "optimizer"),
        "allocation": _component_value(experiment_payload, component_manifest, "allocation"),
        "constraints": _component_value(experiment_payload, component_manifest, "constraints"),
        "costs": _component_value(experiment_payload, component_manifest, "costs"),
        "execution": _component_value(experiment_payload, component_manifest, "execution"),
        "implementation_status": _component_value(experiment_payload, component_manifest, "implementation_status"),
        "artifact_schema_version": experiment_payload.get("artifact_schema_version"),
        "fold_artifacts": _fold_artifacts_manifest(experiment_payload),
        "model_bundle_refs": _collect_fold_refs(experiment_payload, "model_bundle_ref"),
        "preprocessor_bundle_refs": _collect_fold_refs(experiment_payload, "preprocessor_bundle_ref"),
        "calibration_bundle": _calibration_manifest(experiment_payload),
        "gate_summary": experiment_payload.get("gate_summary", {}),
        "tradable_outcome": experiment_payload.get("tradable_outcome", {}),
        "alpha_transfer_ledger": experiment_payload.get("alpha_transfer_ledger", {}),
        "confidence_diagnostic": experiment_payload.get("confidence_diagnostic", {}),
        "gate_required": bool(experiment_payload.get("gate_required", False)),
        "evaluation": experiment_payload.get("evaluation", {}),
        "data_range": experiment_payload.get("data_range", {}),
        "raw_data_quality": experiment_payload.get("raw_data_quality", {}),
        "feature_data_quality": experiment_payload.get("feature_data_quality", {}),
        "data_version": experiment_payload.get("data_version", {}),
        "price_adjustment": experiment_payload.get("price_adjustment", {}),
        "parameter_validation_summary": experiment_payload.get("parameter_validation_summary", {}),
        "status": "research_package",
    }


def save_package(
    experiment: dict,
    packages_root: str | Path,
    package_id: str | None = None,
) -> Path:
    """将 AX1 research package 写入 packages_root/package_id。"""
    experiment_payload = make_json_compatible(experiment)
    manifest = build_package_manifest(experiment_payload, package_id=package_id)
    package_root = Path(packages_root) / manifest["package_id"]
    package_root.mkdir(parents=True, exist_ok=True)

    _write_json(package_root / MANIFEST_FILENAME, manifest)
    _write_json(
        package_root / EXPERIMENT_REF_FILENAME,
        _build_experiment_ref(experiment_payload, manifest),
    )
    calibration_bundle = experiment_payload.get("calibration_bundle")
    if isinstance(calibration_bundle, dict) and calibration_bundle:
        _write_json(package_root / CALIBRATION_BUNDLE_FILENAME, calibration_bundle)

    return package_root


def load_package(
    package_ref: str | Path,
    packages_root: str | Path | None = None,
    validate_artifacts: bool = False,
) -> dict:
    """按直接路径或 packages_root/package_id 加载 AX1 package。"""
    package_root = _resolve_package_root(package_ref, packages_root=packages_root)
    payload = {
        "manifest": _read_json(package_root / MANIFEST_FILENAME),
        "experiment_ref": _read_json(package_root / EXPERIMENT_REF_FILENAME),
    }
    calibration_path = package_root / CALIBRATION_BUNDLE_FILENAME
    if calibration_path.is_file():
        payload["calibration_bundle"] = _read_json(calibration_path)
    if validate_artifacts:
        payload["artifact_validation"] = _validate_package_artifacts(payload, package_root)
    return payload


def _build_experiment_ref(experiment: dict, manifest: dict) -> dict:
    return {
        "package_id": manifest["package_id"],
        "source_experiment": manifest["source_experiment"],
        "experiment_name": experiment.get("experiment_name"),
        "experiment_id": experiment.get("experiment_id"),
        "output_dir": experiment.get("output_dir") or experiment.get("experiment_dir"),
        "created_at": experiment.get("created_at"),
    }


def _resolve_package_root(
    package_ref: str | Path,
    packages_root: str | Path | None = None,
) -> Path:
    package_path = Path(package_ref)
    if package_path.is_absolute() or _looks_like_path(package_ref):
        return package_path
    if packages_root is not None:
        return Path(packages_root) / str(package_ref)
    return package_path


def _looks_like_path(package_ref: str | Path) -> bool:
    package_text = str(package_ref)
    return "/" in package_text or "\\" in package_text or package_text.startswith(".")


def _source_experiment(experiment: dict) -> str:
    for key in ("source_experiment", "experiment_name", "experiment_id", "name", "output_dir", "experiment_dir"):
        value = experiment.get(key)
        if value:
            return str(value)
    return "unknown"


def _component_value(experiment: dict, component_manifest: dict, key: str) -> dict:
    value = experiment.get(key)
    if isinstance(value, dict) and value:
        return value
    value = component_manifest.get(key)
    return value if isinstance(value, dict) else {}


def _calibration_manifest(experiment: dict) -> dict:
    bundle = experiment.get("calibration_bundle")
    if not isinstance(bundle, dict) or not bundle:
        return {}
    summary = bundle.get("summary")
    return {
        "path": CALIBRATION_BUNDLE_FILENAME,
        "score_column": bundle.get("score_column"),
        "summary": summary if isinstance(summary, dict) else {},
    }


def _fold_artifacts_manifest(experiment: dict) -> list[dict[str, Any]]:
    folds = experiment.get("folds")
    if isinstance(folds, list):
        return [item for item in folds if isinstance(item, dict)]
    fold_results = (experiment.get("training_summary") or {}).get("fold_results")
    if isinstance(fold_results, list):
        return [
            {
                key: value
                for key, value in item.items()
                if key.endswith("_ref") or key in {"fold_id", "path", "date_range", "row_counts"}
            }
            for item in fold_results
            if isinstance(item, dict)
        ]
    return []


def _collect_fold_refs(experiment: dict, ref_key: str) -> list[str]:
    refs = []
    for item in _fold_artifacts_manifest(experiment):
        value = item.get(ref_key)
        if value:
            refs.append(str(value))
    return refs


def _validate_package_artifacts(payload: dict[str, Any], package_root: Path) -> dict[str, Any]:
    manifest = payload.get("manifest") or {}
    experiment_ref = payload.get("experiment_ref") or {}
    roots = [package_root]
    output_dir = experiment_ref.get("output_dir")
    if output_dir:
        roots.append(Path(output_dir))

    checked_folds = []
    for fold in manifest.get("fold_artifacts") or []:
        if not isinstance(fold, dict):
            continue
        model_ref = fold.get("model_bundle_ref")
        preprocessor_ref = fold.get("preprocessor_bundle_ref")
        if not model_ref or not preprocessor_ref:
            continue
        model_bundle = _read_json(_resolve_artifact_ref(str(model_ref), roots))
        preprocessor_bundle = _read_json(_resolve_artifact_ref(str(preprocessor_ref), roots))
        model_features = _feature_columns_from_model_bundle(model_bundle)
        preprocessor_features = [str(item) for item in preprocessor_bundle.get("feature_columns") or []]
        if model_features != preprocessor_features:
            raise ValueError(
                "feature_columns mismatch between model bundle and preprocessor bundle: "
                f"model={model_features} preprocessor={preprocessor_features}"
            )
        checked_folds.append(
            {
                "fold_id": fold.get("fold_id"),
                "model_bundle_ref": str(model_ref),
                "preprocessor_bundle_ref": str(preprocessor_ref),
                "feature_columns": model_features,
            }
        )
    return {"schema_version": 1, "passed": True, "checked_folds": checked_folds}


def _resolve_artifact_ref(ref: str, roots: list[Path]) -> Path:
    path = Path(ref)
    if path.is_absolute() and path.is_file():
        return path
    for root in roots:
        candidate = root / ref
        if candidate.is_file():
            return candidate
    raise ValueError(f"package artifact ref not found: {ref}")


def _feature_columns_from_model_bundle(bundle: dict[str, Any]) -> list[str]:
    columns = bundle.get("feature_columns")
    if columns:
        return [str(item) for item in columns]
    model_bundle = bundle.get("model_bundle")
    if isinstance(model_bundle, dict):
        nested_columns = model_bundle.get("feature_columns")
        if nested_columns:
            return [str(item) for item in nested_columns]
    return []


def _default_package_id(experiment: dict) -> str:
    source = _slugify(_source_experiment(experiment))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"ax1_pkg_{source}_{timestamp}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug[:64] or "experiment"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
