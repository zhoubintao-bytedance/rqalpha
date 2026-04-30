"""AX1 raw and feature matrix data quality gates."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROMOTABLE_PURPOSES = {"promotable_training", "promotion", "live_training"}


def build_raw_data_quality_report(raw_df: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or {})
    data_cfg = dict(config.get("data") or {})
    quality_cfg = dict(data_cfg.get("quality") or {})
    price_cfg = dict(data_cfg.get("price_adjustment") or {})
    required_columns = [str(column) for column in quality_cfg.get("required_raw_columns", ["date", "order_book_id", "close", "volume"])]
    hard_blocks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if raw_df is None:
        raw_df = pd.DataFrame()
    frame = raw_df.copy()
    missing_required = [column for column in required_columns if column not in frame.columns]
    for column in missing_required:
        hard_blocks.append(_issue("raw_panel", "missing_required_raw_column", f"raw panel missing required column: {column}", column=column))

    price_adjustment = _resolve_price_adjustment(frame, price_cfg)
    hard_blocks.extend(price_adjustment.pop("_hard_blocks", []))

    optional_ohlc = [str(column) for column in quality_cfg.get("optional_ohlc_columns", ["open", "high", "low"])]
    for column in optional_ohlc:
        if column not in frame.columns:
            warnings.append(_issue("raw_panel", "missing_optional_ohlc_column", f"raw panel missing optional OHLC column: {column}", column=column))

    if {"date", "order_book_id"}.issubset(frame.columns):
        key_counts = frame.groupby(["date", "order_book_id"], dropna=False).size()
        duplicates = key_counts[key_counts > 1]
        if not duplicates.empty:
            hard_blocks.append(
                _issue(
                    "raw_panel",
                    "duplicate_date_order_book_id",
                    "raw panel contains duplicate (date, order_book_id) keys",
                    count=int(duplicates.sum() - len(duplicates)),
                )
            )

    label_price = _label_price_for_quality(frame, price_adjustment)
    if label_price is not None:
        missing_count = int(label_price.isna().sum())
        if missing_count:
            hard_blocks.append(_issue("raw_panel", "missing_label_price", "label price contains missing values", count=missing_count))
        non_positive_count = int((label_price.dropna() <= 0.0).sum())
        if non_positive_count:
            hard_blocks.append(_issue("raw_panel", "non_positive_price", "label price contains non-positive values", count=non_positive_count))
        jump_warning = _suspicious_jump_warning(frame, label_price, threshold=float(quality_cfg.get("suspicious_jump_abs_return", 0.35)))
        if jump_warning is not None:
            warnings.append(jump_warning)

    coverage = _coverage_summary(frame)
    min_asset_coverage = float(quality_cfg.get("min_asset_coverage_ratio", 0.50))
    low_coverage_assets = [
        asset
        for asset, payload in coverage.get("assets", {}).items()
        if float(payload.get("coverage_ratio", 0.0)) < min_asset_coverage
    ]
    if low_coverage_assets:
        warnings.append(
            _issue(
                "raw_panel",
                "low_asset_coverage",
                "some assets have low date coverage",
                assets=low_coverage_assets[:20],
                count=len(low_coverage_assets),
            )
        )

    data_version = build_data_version(frame)
    return {
        "schema_version": 1,
        "passed": not hard_blocks,
        "hard_block_count": int(len(hard_blocks)),
        "warning_count": int(len(warnings)),
        "hard_blocks": hard_blocks,
        "warnings": warnings,
        "row_count": int(len(frame)),
        "date_range": _date_range(frame),
        "asset_count": int(frame["order_book_id"].nunique()) if "order_book_id" in frame.columns else 0,
        "coverage": coverage,
        "price_adjustment": price_adjustment,
        "data_version": data_version,
    }


def build_feature_matrix_quality_report(
    feature_frame: pd.DataFrame,
    feature_columns: Iterable[str],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(config or {})
    quality_cfg = dict((config.get("data") or {}).get("quality") or {})
    warning_ratio = float(quality_cfg.get("feature_missing_warning_ratio", 0.50))
    frame = feature_frame if feature_frame is not None else pd.DataFrame()
    hard_blocks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    features: dict[str, dict[str, Any]] = {}

    for raw_name in feature_columns or []:
        name = str(raw_name)
        if name not in frame.columns:
            hard_blocks.append(_issue(name, "missing_feature_column", "active feature column is missing"))
            features[name] = {"missing_ratio": 1.0, "non_null_count": 0, "row_count": int(len(frame))}
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        row_count = int(len(values))
        missing_count = int(values.isna().sum())
        non_null_count = row_count - missing_count
        missing_ratio = float(missing_count / row_count) if row_count else 1.0
        features[name] = {
            "row_count": row_count,
            "missing_count": missing_count,
            "non_null_count": non_null_count,
            "missing_ratio": missing_ratio,
        }
        if non_null_count == 0:
            hard_blocks.append(_issue(name, "all_empty_feature_column", "active feature column is entirely missing"))
        elif missing_ratio > warning_ratio:
            warnings.append(
                _issue(
                    name,
                    "high_feature_missingness",
                    "active feature column has high missingness before LGBM fillna",
                    missing_ratio=missing_ratio,
                )
            )

    return {
        "schema_version": 1,
        "passed": not hard_blocks,
        "hard_block_count": int(len(hard_blocks)),
        "warning_count": int(len(warnings)),
        "hard_blocks": hard_blocks,
        "warnings": warnings,
        "feature_count": int(len(features)),
        "features": features,
    }


def enforce_data_quality(report: dict[str, Any], *, context: str) -> None:
    if report.get("passed", False):
        return
    reason_codes = [str(item.get("reason_code")) for item in report.get("hard_blocks", [])]
    raise ValueError(f"AX1 {context} failed: {reason_codes}")


def build_data_version(frame: pd.DataFrame, *, feature_columns: Iterable[str] | None = None) -> dict[str, Any]:
    if frame is None:
        frame = pd.DataFrame()
    columns = [
        column
        for column in (
            "date",
            "order_book_id",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "close_adj",
            "adjust_factor",
            "volume",
            "price_adjustment_status",
        )
        if column in frame.columns
    ]
    normalized = frame.loc[:, columns].copy() if columns else pd.DataFrame()
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "order_book_id" in normalized.columns:
        normalized["order_book_id"] = normalized["order_book_id"].astype(str)
    sort_columns = [column for column in ("date", "order_book_id") if column in normalized.columns]
    if sort_columns:
        normalized = normalized.sort_values(sort_columns).reset_index(drop=True)
    feature_names = [str(column) for column in dict.fromkeys(feature_columns or [])]
    payload = json.dumps(
        {
            "rows": _json_records(normalized),
            "feature_columns": feature_names,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    schema_payload = json.dumps({"columns": columns, "feature_columns": feature_names}, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "row_count": int(len(frame)),
        "date_range": _date_range(frame),
        "asset_count": int(frame["order_book_id"].nunique()) if "order_book_id" in frame.columns else 0,
        "columns": columns,
        "feature_columns": feature_names,
        "feature_column_count": int(len(feature_names)),
        "feature_schema_hash": hashlib.sha256(json.dumps(feature_names, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "data_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "schema_hash": hashlib.sha256(schema_payload.encode("utf-8")).hexdigest(),
    }


def _resolve_price_adjustment(frame: pd.DataFrame, price_cfg: dict[str, Any]) -> dict[str, Any]:
    required = bool(price_cfg.get("required", True))
    price_column = str(price_cfg.get("price_column", "close") or "close")
    adjusted_price_column = price_cfg.get("adjusted_price_column")
    adjusted_price_column = str(adjusted_price_column) if adjusted_price_column else None
    factor_column = price_cfg.get("adjustment_factor_column")
    factor_column = str(factor_column) if factor_column else None
    status_column = str(price_cfg.get("adjustment_status_column", "price_adjustment_status") or "price_adjustment_status")
    hard_blocks: list[dict[str, Any]] = []

    if adjusted_price_column and adjusted_price_column in frame.columns:
        return {
            "required": required,
            "method": "adjusted_price_column",
            "price_column": price_column,
            "adjusted_price_column": adjusted_price_column,
            "adjustment_factor_column": factor_column,
            "_hard_blocks": hard_blocks,
        }
    if factor_column and factor_column in frame.columns and price_column in frame.columns:
        return {
            "required": required,
            "method": "price_times_adjustment_factor",
            "price_column": price_column,
            "adjusted_price_column": None,
            "adjustment_factor_column": factor_column,
            "_hard_blocks": hard_blocks,
        }
    if _declares_adjusted_close(frame, price_cfg, status_column):
        return {
            "required": required,
            "method": "declared_adjusted_price_column",
            "price_column": price_column,
            "adjusted_price_column": None,
            "adjustment_factor_column": None,
            "status_column": status_column,
            "_hard_blocks": hard_blocks,
        }
    if required:
        hard_blocks.append(
            _issue(
                "raw_panel",
                "missing_price_adjustment_contract",
                "training requires adjusted price column, adjustment factor, or explicit adjusted close declaration",
            )
        )
    return {
        "required": required,
        "method": "missing" if required else "not_required",
        "price_column": price_column,
        "adjusted_price_column": adjusted_price_column,
        "adjustment_factor_column": factor_column,
        "_hard_blocks": hard_blocks,
    }


def _declares_adjusted_close(frame: pd.DataFrame, price_cfg: dict[str, Any], status_column: str) -> bool:
    if not bool(price_cfg.get("allow_declared_adjusted_close", False)):
        return False
    if status_column not in frame.columns:
        return False
    accepted = {
        str(item).strip().lower()
        for item in price_cfg.get("accepted_adjusted_statuses", ["adjusted", "forward_adjusted", "backward_adjusted"])
    }
    statuses = set(frame[status_column].dropna().astype(str).str.strip().str.lower())
    return bool(statuses) and statuses.issubset(accepted)


def _label_price_for_quality(frame: pd.DataFrame, price_adjustment: dict[str, Any]) -> pd.Series | None:
    method = price_adjustment.get("method")
    if method == "adjusted_price_column":
        column = price_adjustment.get("adjusted_price_column")
        return pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else None
    if method == "price_times_adjustment_factor":
        price_column = price_adjustment.get("price_column")
        factor_column = price_adjustment.get("adjustment_factor_column")
        if price_column in frame.columns and factor_column in frame.columns:
            return pd.to_numeric(frame[price_column], errors="coerce") * pd.to_numeric(frame[factor_column], errors="coerce")
        return None
    price_column = price_adjustment.get("price_column", "close")
    if price_column in frame.columns:
        return pd.to_numeric(frame[price_column], errors="coerce")
    return None


def _suspicious_jump_warning(frame: pd.DataFrame, label_price: pd.Series, *, threshold: float) -> dict[str, Any] | None:
    if "order_book_id" not in frame.columns:
        return None
    returns = label_price.groupby(frame["order_book_id"]).pct_change(fill_method=None)
    count = int((returns.abs() > float(threshold)).sum())
    if count <= 0:
        return None
    return _issue(
        "raw_panel",
        "suspicious_price_jump",
        "adjusted label price contains large one-day moves",
        count=count,
        threshold=float(threshold),
    )


def _coverage_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or not {"date", "order_book_id"}.issubset(frame.columns):
        return {"assets": {}, "dates": {}}
    working = frame[["date", "order_book_id"]].copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    unique_dates = int(working["date"].nunique())
    assets = {}
    for order_book_id, group in working.groupby("order_book_id", sort=True):
        observed = int(group["date"].nunique())
        assets[str(order_book_id)] = {
            "date_count": observed,
            "coverage_ratio": float(observed / unique_dates) if unique_dates else 0.0,
        }
    dates = {
        str(pd.Timestamp(date).date()): int(count)
        for date, count in working.groupby("date")["order_book_id"].nunique().items()
        if pd.notna(date)
    }
    return {"assets": assets, "dates": dates}


def _date_range(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty or "date" not in frame.columns:
        return {"start": None, "end": None}
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return {"start": None, "end": None}
    return {"start": str(dates.min().date()), "end": str(dates.max().date())}


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    safe = frame.replace([np.inf, -np.inf], np.nan)
    for raw in safe.to_dict(orient="records"):
        record = {}
        for key, value in raw.items():
            if pd.isna(value):
                record[str(key)] = None
            elif isinstance(value, float):
                record[str(key)] = round(float(value), 12)
            else:
                record[str(key)] = value
        records.append(record)
    return records


def _issue(scope: str, reason_code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"scope": str(scope), "reason_code": str(reason_code), "message": str(message)}
    payload.update(extra)
    return payload


def build_survivorship_bias_report(
    raw_df: pd.DataFrame,
    *,
    universe_metadata: pd.DataFrame | None = None,
    data_provider=None,
    purpose: str = "research",
) -> dict[str, Any]:
    """检查幸存者偏差风险。

    对于 promotable 用途，如果 data_provider 不可用且无法验证 PIT universe，
    则产生 hard block。

    Args:
        raw_df: 原始数据面板
        universe_metadata: universe 元数据（包含 PIT audit 信息）
        data_provider: 数据提供者
        purpose: 用途类型

    Returns:
        包含 hard_blocks 和 warnings 的报告
    """
    hard_blocks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    promotable = str(purpose) in PROMOTABLE_PURPOSES

    # 从 universe metadata 获取 PIT audit 信息
    pit_audit = {}
    if universe_metadata is not None and not universe_metadata.empty:
        pit_audit = dict(getattr(universe_metadata, "attrs", {}).get("pit_audit") or {})

    # 检查 PIT universe 验证状态
    pit_universe_status = pit_audit.get("source_status", {}).get("pit_universe", "")

    if pit_universe_status == "validated_via_provider":
        # PIT 验证成功，无问题
        pass
    elif pit_universe_status == "validation_skipped_no_provider":
        issue = _issue(
            "survivorship_bias",
            "pit_universe_validation_skipped_no_provider",
            "PIT universe validation skipped: data_provider is required for promotable training to prevent survivorship bias",
        )
        if promotable:
            hard_blocks.append(issue)
        else:
            warnings.append(issue)
    elif pit_universe_status == "validation_failed":
        issue = _issue(
            "survivorship_bias",
            "pit_universe_validation_failed",
            "PIT universe validation failed: could not fetch historical instruments (potential survivorship bias)",
        )
        if promotable:
            hard_blocks.append(issue)
        else:
            warnings.append(issue)
    elif pit_universe_status == "disabled":
        issue = _issue(
            "survivorship_bias",
            "pit_universe_validation_disabled",
            "PIT universe validation is disabled (potential survivorship bias)",
        )
        if promotable:
            hard_blocks.append(issue)
        else:
            warnings.append(issue)

    # 检查 universe audit 的 hard blocks
    for item in pit_audit.get("hard_blocks", []) or []:
        if item.get("reason_code") not in {b.get("reason_code") for b in hard_blocks}:
            hard_blocks.append(
                _issue(
                    "survivorship_bias",
                    str(item.get("reason_code", "universe_pit_issue")),
                    str(item.get("message", "universe PIT audit issue")),
                )
            )

    return {
        "schema_version": 1,
        "passed": not hard_blocks,
        "hard_block_count": int(len(hard_blocks)),
        "warning_count": int(len(warnings)),
        "hard_blocks": hard_blocks,
        "warnings": warnings,
        "pit_universe_status": pit_universe_status,
        "promotable": promotable,
    }
