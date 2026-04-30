"""AX1 frozen research artifact resolver."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ArtifactLine:
    strategy_id: str
    artifact_line_id: str


@dataclass(frozen=True)
class ResolvedArtifact:
    strategy_id: str
    artifact_line_id: str
    artifact_root: Path
    metadata_path: Path
    train_cutoff: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReplaySignal:
    signal_date: str
    fold_id: int
    test_start: str
    test_end: str
    target_weights: dict[str, float]
    predictions: dict[str, float]


def parse_artifact_line(ref: str) -> ArtifactLine:
    if not isinstance(ref, str) or "@" not in ref:
        raise ValueError(f"artifact reference must be 'strategy_id@artifact_line_id', got: {ref!r}")
    strategy_id, artifact_line_id = (part.strip() for part in ref.split("@", 1))
    if not strategy_id:
        raise ValueError(f"strategy_id must not be empty in artifact reference: {ref!r}")
    if not artifact_line_id:
        raise ValueError(f"artifact_line_id must not be empty in artifact reference: {ref!r}")
    return ArtifactLine(strategy_id=strategy_id, artifact_line_id=artifact_line_id)


def resolve_artifact(line: ArtifactLine, artifacts_root: str | Path) -> ResolvedArtifact:
    if not line.strategy_id.startswith("ax1."):
        raise FileNotFoundError(f"artifact strategy_id {line.strategy_id!r} does not belong to AX1 product")
    artifact_root = Path(artifacts_root) / f"ax1_{line.artifact_line_id}"
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"artifact directory not found: {artifact_root}")
    metadata_path = artifact_root / "experiment.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"artifact metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return ResolvedArtifact(
        strategy_id=line.strategy_id,
        artifact_line_id=line.artifact_line_id,
        artifact_root=artifact_root,
        metadata_path=metadata_path,
        train_cutoff=_extract_train_cutoff(metadata),
        metadata=metadata,
    )


def load_replay_signal_book(resolved_artifact: ResolvedArtifact) -> dict[str, ReplaySignal]:
    return _load_replay_signal_book_cached(
        str(resolved_artifact.artifact_root),
        str(resolved_artifact.metadata_path),
    )


@lru_cache(maxsize=16)
def _load_replay_signal_book_cached(
    artifact_root_text: str,
    metadata_path_text: str,
) -> dict[str, ReplaySignal]:
    artifact_root = Path(artifact_root_text)
    with Path(metadata_path_text).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    signal_book: dict[str, ReplaySignal] = {}
    for fold_meta in sorted(metadata.get("folds", []), key=_fold_priority):
        weights_path = _fold_weights_path(artifact_root, fold_meta)
        if not weights_path.is_file():
            raise FileNotFoundError(f"missing weights artifact: {weights_path}")
        weights = pd.read_parquet(weights_path)
        required = {"date", "order_book_id"}
        missing = required - set(weights.columns)
        if missing:
            raise ValueError(f"weights artifact missing required columns {sorted(missing)}: {weights_path}")
        weight_column = _weight_column(weights)
        if weight_column is None:
            raise ValueError(f"weights artifact missing target_weight/weight column: {weights_path}")

        frame = weights.copy()
        frame["_signal_date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        fold_id = int(fold_meta.get("fold_id", fold_meta.get("index", 0)))
        test_start = _fold_date(fold_meta, "test_start")
        test_end = _fold_date(fold_meta, "test_end")
        for signal_date, group in frame.groupby("_signal_date", sort=True):
            target_weights = _numeric_mapping(group, weight_column, positive_only=True)
            if not target_weights:
                continue
            signal_book[signal_date] = ReplaySignal(
                signal_date=str(signal_date),
                fold_id=fold_id,
                test_start=test_start,
                test_end=test_end,
                target_weights=target_weights,
                predictions=_prediction_mapping(group),
            )
    return signal_book


def _fold_weights_path(artifact_root: Path, fold_meta: dict[str, Any]) -> Path:
    weights_ref = fold_meta.get("weights_ref")
    if weights_ref:
        return artifact_root / str(weights_ref)
    fold_path = fold_meta.get("path")
    if fold_path:
        return artifact_root / str(fold_path) / "weights.parquet"
    fold_id = int(fold_meta.get("fold_id", fold_meta.get("index", 0)))
    return artifact_root / "folds" / f"fold_{fold_id:03d}" / "weights.parquet"


def _fold_priority(fold_meta: dict[str, Any]) -> tuple[str, int]:
    return (
        _fold_date(fold_meta, "test_start"),
        int(fold_meta.get("fold_id", fold_meta.get("index", 0))),
    )


def _fold_date(fold_meta: dict[str, Any], key: str) -> str:
    if key in fold_meta:
        return _normalize_date_text(fold_meta.get(key))
    date_range = fold_meta.get("date_range")
    if isinstance(date_range, dict):
        return _normalize_date_text(date_range.get(key))
    return ""


def _extract_train_cutoff(metadata: dict[str, Any]) -> str:
    folds = metadata.get("folds") or []
    if not folds:
        return ""
    last_fold = sorted(folds, key=_fold_priority)[-1]
    if "train_end" in last_fold:
        return _normalize_date_text(last_fold.get("train_end"))
    date_range = last_fold.get("date_range")
    if isinstance(date_range, dict):
        return _normalize_date_text(date_range.get("train_end"))
    return ""


def _normalize_date_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if "T" in text:
        text = text.split("T", 1)[0]
    if " " in text:
        text = text.split(" ", 1)[0]
    return text


def _weight_column(frame: pd.DataFrame) -> str | None:
    if "target_weight" in frame.columns:
        return "target_weight"
    if "weight" in frame.columns:
        return "weight"
    return None


def _prediction_mapping(frame: pd.DataFrame) -> dict[str, float]:
    for column in ("prediction", "expected_relative_net_return_10d", "adjusted_expected_return"):
        if column in frame.columns:
            return _numeric_mapping(frame, column, positive_only=False)
    return {}


def _numeric_mapping(frame: pd.DataFrame, value_column: str, *, positive_only: bool) -> dict[str, float]:
    mapping: dict[str, float] = {}
    ordered = frame.sort_values(["order_book_id"], kind="stable")
    for row in ordered.itertuples(index=False):
        order_book_id = getattr(row, "order_book_id", None)
        value = getattr(row, value_column, None)
        if not order_book_id or pd.isna(value):
            continue
        numeric_value = float(value)
        if not pd.notna(numeric_value):
            continue
        if positive_only and numeric_value <= 0:
            continue
        mapping[str(order_book_id)] = numeric_value
    return mapping
