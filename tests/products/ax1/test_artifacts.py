import json
from pathlib import Path

import pandas as pd
import pytest

from skyeye.products.ax1.artifacts import (
    ArtifactLine,
    ReplaySignal,
    ResolvedArtifact,
    load_replay_signal_book,
    parse_artifact_line,
    resolve_artifact,
)


def test_parse_artifact_line_accepts_ax1_strategy_and_line():
    line = parse_artifact_line("ax1.personal_etf_core@demo")

    assert line == ArtifactLine(strategy_id="ax1.personal_etf_core", artifact_line_id="demo")


def test_parse_artifact_line_rejects_invalid_refs():
    with pytest.raises(ValueError):
        parse_artifact_line("ax1.personal_etf_core")
    with pytest.raises(ValueError):
        parse_artifact_line("@demo")
    with pytest.raises(ValueError):
        parse_artifact_line("ax1.personal_etf_core@")


def test_resolve_artifact_loads_ax1_experiment_metadata(tmp_path):
    artifact_root = _write_demo_artifact(tmp_path)

    resolved = resolve_artifact(
        parse_artifact_line("ax1.personal_etf_core@demo"),
        tmp_path,
    )

    assert isinstance(resolved, ResolvedArtifact)
    assert resolved.strategy_id == "ax1.personal_etf_core"
    assert resolved.artifact_line_id == "demo"
    assert resolved.artifact_root == artifact_root
    assert resolved.metadata_path == artifact_root / "experiment.json"
    assert resolved.train_cutoff == "2024-04-30"
    assert resolved.metadata["artifact_schema_version"] == 1


def test_resolve_artifact_rejects_cross_product_strategy(tmp_path):
    _write_demo_artifact(tmp_path)

    with pytest.raises(FileNotFoundError):
        resolve_artifact(
            ArtifactLine(strategy_id="tx1.rolling_score", artifact_line_id="demo"),
            tmp_path,
        )


def test_load_replay_signal_book_prefers_later_fold_for_overlap_dates(tmp_path):
    _write_demo_artifact(tmp_path)
    resolved = resolve_artifact(
        parse_artifact_line("ax1.personal_etf_core@demo"),
        tmp_path,
    )

    signal_book = load_replay_signal_book(resolved)
    overlap_signal = signal_book["2024-05-08"]

    assert isinstance(overlap_signal, ReplaySignal)
    assert overlap_signal.signal_date == "2024-05-08"
    assert overlap_signal.fold_id == 1
    assert overlap_signal.test_start == "2024-05-08"
    assert overlap_signal.target_weights == {"510300.XSHG": 0.25, "510500.XSHG": 0.50}
    assert overlap_signal.predictions == {"510300.XSHG": 0.03, "510500.XSHG": 0.01}
    assert sorted(signal_book) == ["2024-05-06", "2024-05-08", "2024-05-09"]


def _write_demo_artifact(tmp_path: Path) -> Path:
    artifact_root = tmp_path / "ax1_demo"
    fold_0 = artifact_root / "folds" / "fold_000"
    fold_1 = artifact_root / "folds" / "fold_001"
    fold_0.mkdir(parents=True)
    fold_1.mkdir(parents=True)

    metadata = {
        "product": "ax1",
        "artifact_schema_version": 1,
        "folds": [
            {
                "fold_id": 0,
                "path": "folds/fold_000",
                "train_end": "2024-03-29",
                "test_start": "2024-05-01",
                "test_end": "2024-05-08",
                "weights_ref": "folds/fold_000/weights.parquet",
            },
            {
                "fold_id": 1,
                "path": "folds/fold_001",
                "train_end": "2024-04-30",
                "test_start": "2024-05-08",
                "test_end": "2024-05-31",
                "weights_ref": "folds/fold_001/weights.parquet",
            },
        ],
    }
    (artifact_root / "experiment.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-06", "2024-05-08"]),
            "order_book_id": ["510300.XSHG", "510300.XSHG"],
            "target_weight": [0.60, 0.70],
            "prediction": [0.02, 0.02],
        }
    ).to_parquet(fold_0 / "weights.parquet", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-08", "2024-05-08", "2024-05-09"]),
            "order_book_id": ["510300.XSHG", "510500.XSHG", "510300.XSHG"],
            "target_weight": [0.25, 0.50, 0.30],
            "prediction": [0.03, 0.01, 0.04],
        }
    ).to_parquet(fold_1 / "weights.parquet", index=False)
    return artifact_root
