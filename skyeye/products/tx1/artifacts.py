# -*- coding: utf-8 -*-
"""TX1 Artifact Line and Resolver.

Provides artifact identity parsing, resolution against the experiment
artifact tree, and frozen benchmark enforcement per the TX1 RFC.
"""

from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from skyeye.products.tx1.live_advisor.package_io import (
    DEFAULT_TX1_PACKAGES_ROOT,
    package_component_path,
)
from skyeye.products.tx1.live_advisor.schema import LIVE_PACKAGE_REQUIRED_COMPONENTS


# Benchmark is frozen at strategy_id + artifact_line level per RFC.
FROZEN_BENCHMARK = "000300.XSHG"


@dataclass(frozen=True)
class ArtifactLine:
    """Parsed artifact line reference."""

    strategy_id: str
    artifact_line_id: str


@dataclass(frozen=True)
class ResolvedArtifact:
    """Fully resolved artifact with paths and metadata."""

    strategy_id: str
    artifact_line_id: str
    benchmark_id: str
    artifact_root: Path
    metadata_path: Path
    train_cutoff: str
    metadata: dict


@dataclass(frozen=True)
class ReplaySignal:
    """Frozen out-of-sample target portfolio for a signal date."""

    signal_date: str
    fold_index: int
    test_start: str
    test_end: str
    target_weights: dict[str, float]
    predictions: dict[str, float]


@dataclass(frozen=True)
class ResolvedLivePackage:
    """已解析的 TX1 live promoted package。"""

    package_id: str
    package_root: Path
    manifest_path: Path
    component_paths: dict[str, Path]
    manifest: dict


def parse_artifact_line(ref: str) -> ArtifactLine:
    """Parse 'strategy_id@artifact_line_id' into an ArtifactLine.

    Raises:
        ValueError: If format is invalid or parts are empty.
    """
    if not isinstance(ref, str) or "@" not in ref:
        raise ValueError(
            "artifact reference must be 'strategy_id@artifact_line_id', got: {!r}".format(ref)
        )
    parts = ref.split("@", 1)
    strategy_id = parts[0].strip()
    artifact_line_id = parts[1].strip()
    if not strategy_id:
        raise ValueError("strategy_id must not be empty in artifact reference: {!r}".format(ref))
    if not artifact_line_id:
        raise ValueError("artifact_line_id must not be empty in artifact reference: {!r}".format(ref))
    return ArtifactLine(strategy_id=strategy_id, artifact_line_id=artifact_line_id)


def resolve_artifact(line: ArtifactLine, artifacts_root: Path) -> ResolvedArtifact:
    """Resolve an artifact line to concrete paths and metadata.

    Looks for directory ``artifacts_root / "tx1_<artifact_line_id>"``,
    loads ``experiment.json``, and extracts train_cutoff from the last
    fold's date_range.

    Args:
        line: Parsed artifact line reference.
        artifacts_root: Root directory containing TX1 experiment artifacts.

    Returns:
        ResolvedArtifact with paths and metadata.

    Raises:
        FileNotFoundError: If artifact directory or metadata file is missing,
            or if strategy_id does not match TX1 product.
    """
    if not line.strategy_id.startswith("tx1."):
        raise FileNotFoundError(
            "artifact strategy_id {!r} does not belong to TX1 product".format(line.strategy_id)
        )

    artifact_dir = artifacts_root / "tx1_{}".format(line.artifact_line_id)
    if not artifact_dir.is_dir():
        raise FileNotFoundError(
            "artifact directory not found: {} (strategy={}, line={})".format(
                artifact_dir, line.strategy_id, line.artifact_line_id
            )
        )

    metadata_path = artifact_dir / "experiment.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "artifact metadata not found: {} (strategy={}, line={})".format(
                metadata_path, line.strategy_id, line.artifact_line_id
            )
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_cutoff = _extract_train_cutoff(metadata)

    return ResolvedArtifact(
        strategy_id=line.strategy_id,
        artifact_line_id=line.artifact_line_id,
        benchmark_id=FROZEN_BENCHMARK,
        artifact_root=artifact_dir,
        metadata_path=metadata_path,
        train_cutoff=train_cutoff,
        metadata=metadata,
    )


def load_replay_signal_book(resolved_artifact: ResolvedArtifact) -> dict[str, ReplaySignal]:
    """Load replayable out-of-sample weights keyed by signal_date.

    Overlapping dates across adjacent folds are resolved by preferring the fold
    with the later ``test_start``. This matches the intended semantics that the
    newer model line becomes authoritative from its own test start onward.
    """
    return _load_replay_signal_book_cached(
        str(resolved_artifact.artifact_root),
        str(resolved_artifact.metadata_path),
    )


def resolve_live_package(
    ref: str | Path,
    packages_root: Path | None = None,
) -> ResolvedLivePackage:
    """解析 live package，并校验六件套是否齐全。"""
    package_root = _resolve_live_package_root(ref, packages_root=packages_root)
    if not package_root.is_dir():
        raise FileNotFoundError("live package directory not found: {}".format(package_root))

    component_paths = {}
    for component_name in LIVE_PACKAGE_REQUIRED_COMPONENTS:
        component_path = package_component_path(package_root, component_name)
        if not component_path.is_file():
            raise FileNotFoundError(
                "live package missing required component {}: {}".format(
                    component_name,
                    component_path,
                )
            )
        component_paths[component_name] = component_path

    manifest_path = component_paths["manifest"]
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    package_id = str(manifest.get("package_id") or package_root.name)
    return ResolvedLivePackage(
        package_id=package_id,
        package_root=package_root,
        manifest_path=manifest_path,
        component_paths=component_paths,
        manifest=manifest,
    )


def _extract_train_cutoff(metadata: dict) -> str:
    """Extract train_cutoff from the last fold's date_range.train_end."""
    folds = metadata.get("folds", [])
    if not folds:
        return ""
    last_fold = folds[-1]
    date_range = last_fold.get("date_range", {})
    raw = date_range.get("train_end", "")
    if raw and " " in raw:
        return raw.split(" ")[0]
    return raw or ""


@lru_cache(maxsize=16)
def _load_replay_signal_book_cached(
    artifact_root_str: str,
    metadata_path_str: str,
) -> dict[str, ReplaySignal]:
    artifact_root = Path(artifact_root_str)
    metadata_path = Path(metadata_path_str)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    signal_book: dict[str, ReplaySignal] = {}
    for fold_meta in sorted(metadata.get("folds", []), key=_fold_priority):
        fold_dir = artifact_root / fold_meta["path"]
        weights_path = fold_dir / "weights.parquet"
        if not weights_path.is_file():
            raise FileNotFoundError("missing weights artifact: {}".format(weights_path))

        weights_frame = pd.read_parquet(weights_path)
        required_columns = {"date", "order_book_id", "weight"}
        missing_columns = required_columns - set(weights_frame.columns)
        if missing_columns:
            raise ValueError(
                "weights artifact missing required columns {}: {}".format(
                    sorted(missing_columns),
                    weights_path,
                )
            )

        signal_dates = pd.to_datetime(weights_frame["date"]).dt.strftime("%Y-%m-%d")
        weights_frame = weights_frame.assign(_signal_date=signal_dates)

        fold_index = int(fold_meta.get("index", 0))
        date_range = fold_meta.get("date_range", {})
        test_start = _normalize_date_text(date_range.get("test_start"))
        test_end = _normalize_date_text(date_range.get("test_end"))

        for signal_date, group in weights_frame.groupby("_signal_date", sort=True):
            target_weights = _extract_numeric_mapping(group, "weight")
            if not target_weights:
                continue
            predictions = (
                _extract_numeric_mapping(group, "prediction")
                if "prediction" in group.columns
                else {}
            )
            signal_book[signal_date] = ReplaySignal(
                signal_date=signal_date,
                fold_index=fold_index,
                test_start=test_start,
                test_end=test_end,
                target_weights=target_weights,
                predictions=predictions,
            )

    return signal_book


def _fold_priority(fold_meta: dict) -> tuple[str, int]:
    date_range = fold_meta.get("date_range", {})
    return (
        _normalize_date_text(date_range.get("test_start")),
        int(fold_meta.get("index", 0)),
    )


def _normalize_date_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if " " in text:
        return text.split(" ", 1)[0]
    return text


def _resolve_live_package_root(
    ref: str | Path,
    packages_root: Path | None = None,
) -> Path:
    """兼容 package_id 与显式目录路径的 live package 引用。"""
    ref_path = Path(ref)
    if ref_path.is_absolute():
        return ref_path
    ref_text = str(ref)
    if "/" in ref_text or ref_text.startswith("."):
        return ref_path.resolve()
    root = packages_root or DEFAULT_TX1_PACKAGES_ROOT
    return Path(root) / ref_text


def _extract_numeric_mapping(frame: pd.DataFrame, value_column: str) -> dict[str, float]:
    mapping = {}
    ordered = frame.sort_values(["order_book_id"], kind="stable")
    for row in ordered.itertuples(index=False):
        order_book_id = getattr(row, "order_book_id", None)
        value = getattr(row, value_column, None)
        if not order_book_id or pd.isna(value):
            continue
        numeric_value = float(value)
        if not pd.notna(numeric_value):
            continue
        if value_column == "weight" and numeric_value <= 0:
            continue
        mapping[str(order_book_id)] = numeric_value
    return mapping
