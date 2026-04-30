"""Point-in-time data audit for AX1 feature sets."""

from __future__ import annotations

from typing import Any, Iterable

from skyeye.products.ax1.features.catalog import FeatureCatalog, build_default_feature_catalog


PROMOTABLE_PURPOSES = {"promotable_training", "promotion", "live_training"}


def audit_feature_set(
    feature_names: Iterable[str],
    *,
    catalog: FeatureCatalog | None = None,
    purpose: str = "research",
) -> dict:
    catalog = catalog or build_default_feature_catalog()
    hard_blocks: list[dict] = []
    warnings: list[dict] = []
    features: dict[str, dict] = {}
    promotable = str(purpose) in PROMOTABLE_PURPOSES

    for raw_name in feature_names:
        name = str(raw_name)
        try:
            definition = catalog.get(name)
        except KeyError:
            hard_blocks.append(_issue(name, "unknown_feature", "feature is not registered in AX1 FeatureCatalog"))
            continue

        feature_issues: list[dict] = []
        if definition.status == "not_implemented":
            issue = _issue(name, "feature_not_implemented", "feature is registered but not implemented")
            hard_blocks.append(issue)
            feature_issues.append(issue)
        elif definition.status == "experimental":
            issue = _issue(name, "experimental_unstable_source", "feature source is experimental or unstable")
            if promotable:
                blocked = _issue(name, "experimental_source_not_promotable", "experimental features cannot enter promotion")
                hard_blocks.append(blocked)
                feature_issues.append(blocked)
            else:
                warnings.append(issue)
                feature_issues.append(issue)

        data_source_status = str(getattr(definition, "data_source_status", definition.status))
        if data_source_status == "not_implemented":
            issue = _issue(name, "data_source_not_implemented", "feature data source is not implemented")
            hard_blocks.append(issue)
            feature_issues.append(issue)
        elif data_source_status == "experimental":
            issue = _issue(name, "data_source_experimental", "feature data source is experimental")
            if promotable:
                blocked = _issue(name, "experimental_data_source_not_promotable", "experimental data sources cannot enter promotion")
                hard_blocks.append(blocked)
                feature_issues.append(blocked)
            else:
                warnings.append(issue)
                feature_issues.append(issue)

        if definition.observable_lag_days is None:
            issue = _issue(name, "missing_observable_lag", "feature must declare observable_lag_days")
            hard_blocks.append(issue)
            feature_issues.append(issue)
        elif definition.source_family == "fundamental" and int(definition.observable_lag_days) < 1:
            issue = _issue(name, "fundamental_observable_lag_too_short", "fundamental features must be lagged")
            hard_blocks.append(issue)
            feature_issues.append(issue)

        if (
            promotable
            and definition.source_family in {"price_volume", "regime_price_volume"}
            and str(getattr(definition, "decision_time", "")) == "after_close"
            and int(getattr(definition, "tradable_lag_days", 0) or 0) < 1
        ):
            issue = _issue(
                name,
                "after_close_price_feature_requires_execution_lag",
                "after-close price/volume features require next-session execution",
            )
            hard_blocks.append(issue)
            feature_issues.append(issue)

        if definition.source_family == "index_snapshot" and definition.uses_latest_snapshot:
            issue = _issue(name, "latest_snapshot_not_point_in_time", "index snapshot must use explicit historical date")
            hard_blocks.append(issue)
            feature_issues.append(issue)
        elif definition.source_family == "index_snapshot" and definition.requires_as_of_date:
            issue = _issue(name, "index_snapshot_requires_as_of_date", "index snapshot requires explicit as-of date")
            warnings.append(issue)
            feature_issues.append(issue)

        features[name] = {
            "status": definition.status,
            "source_family": definition.source_family,
            "observable_lag_days": definition.observable_lag_days,
            "data_source_status": data_source_status,
            "decision_time": getattr(definition, "decision_time", None),
            "tradable_lag_days": getattr(definition, "tradable_lag_days", None),
            "issues": feature_issues,
        }

    return {
        "schema_version": 1,
        "purpose": str(purpose),
        "passed": not hard_blocks,
        "hard_block_count": len(hard_blocks),
        "warning_count": len(warnings),
        "hard_blocks": hard_blocks,
        "warnings": warnings,
        "features": features,
    }


def audit_universe_metadata(universe_metadata: Any, *, purpose: str = "research") -> dict:
    promotable = str(purpose) in PROMOTABLE_PURPOSES
    hard_blocks: list[dict] = []
    warnings: list[dict] = []
    audit = dict(getattr(universe_metadata, "attrs", {}).get("pit_audit") or {})

    if not audit:
        issue = _universe_issue("missing_universe_pit_audit", "universe metadata must include PIT audit")
        if promotable:
            hard_blocks.append(issue)
        else:
            warnings.append(issue)
    else:
        for item in audit.get("hard_blocks", []) or []:
            hard_blocks.append(_normalize_universe_issue(item))
        for item in audit.get("warnings", []) or []:
            warnings.append(_normalize_universe_issue(item))

    try:
        statuses = set(universe_metadata.get("universe_pit_status", []).astype(str).str.lower())
    except Exception:
        statuses = set()
    if "latest_snapshot" in statuses:
        issue = _universe_issue(
            "universe_latest_snapshot_not_point_in_time",
            "universe metadata contains latest snapshot rows",
        )
        if issue["reason_code"] not in {item["reason_code"] for item in hard_blocks}:
            hard_blocks.append(issue)

    return {
        "schema_version": 1,
        "purpose": str(purpose),
        "passed": not hard_blocks,
        "hard_block_count": int(len(hard_blocks)),
        "warning_count": int(len(warnings)),
        "hard_blocks": hard_blocks,
        "warnings": warnings,
        "pit_audit": audit,
    }


def _issue(feature: str, reason_code: str, message: str) -> dict:
    return {
        "feature": str(feature),
        "reason_code": str(reason_code),
        "message": str(message),
    }


def _normalize_universe_issue(item: dict) -> dict:
    return _universe_issue(
        str(item.get("reason_code", "universe_pit_audit_issue")),
        str(item.get("message", "universe PIT audit issue")),
    )


def _universe_issue(reason_code: str, message: str) -> dict:
    return {
        "scope": "universe",
        "reason_code": str(reason_code),
        "message": str(message),
    }
