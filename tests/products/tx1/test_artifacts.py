from pathlib import Path

import pytest

from skyeye.products.tx1.artifacts import (
    ArtifactLine,
    ReplaySignal,
    ResolvedArtifact,
    load_replay_signal_book,
    parse_artifact_line,
    resolve_artifact,
)


ARTIFACTS_ROOT = Path("/home/tiger/rqalpha/skyeye/artifacts/experiments/tx1")


def test_parse_artifact_line_accepts_strategy_id_and_line():
    line = parse_artifact_line("tx1.rolling_score@baseline_linear")

    assert line.strategy_id == "tx1.rolling_score"
    assert line.artifact_line_id == "baseline_linear"


def test_parse_artifact_line_rejects_missing_line():
    with pytest.raises(ValueError):
        parse_artifact_line("tx1.rolling_score")


def test_parse_artifact_line_rejects_empty_parts():
    with pytest.raises(ValueError):
        parse_artifact_line("@baseline_linear")
    with pytest.raises(ValueError):
        parse_artifact_line("tx1.rolling_score@")


def test_resolve_artifact_returns_known_linear_experiment():
    resolved = resolve_artifact(
        parse_artifact_line("tx1.rolling_score@baseline_linear"),
        ARTIFACTS_ROOT,
    )

    assert isinstance(resolved, ResolvedArtifact)
    assert resolved.strategy_id == "tx1.rolling_score"
    assert resolved.artifact_line_id == "baseline_linear"
    assert resolved.benchmark_id == "000300.XSHG"
    assert resolved.artifact_root == ARTIFACTS_ROOT / "tx1_baseline_linear"
    assert resolved.metadata_path == ARTIFACTS_ROOT / "tx1_baseline_linear" / "experiment.json"
    assert resolved.train_cutoff == "2024-10-06"


def test_resolve_artifact_rejects_unknown_artifact_line():
    with pytest.raises(FileNotFoundError):
        resolve_artifact(
            parse_artifact_line("tx1.rolling_score@missing_line"),
            ARTIFACTS_ROOT,
        )


def test_resolve_artifact_rejects_cross_strategy_artifact_match():
    with pytest.raises(FileNotFoundError):
        resolve_artifact(
            ArtifactLine(strategy_id="other.strategy", artifact_line_id="baseline_linear"),
            ARTIFACTS_ROOT,
        )


def test_load_replay_signal_book_uses_later_fold_for_overlap_dates():
    resolved = resolve_artifact(
        parse_artifact_line("tx1.rolling_score@baseline_linear"),
        ARTIFACTS_ROOT,
    )

    signal_book = load_replay_signal_book(resolved)
    overlap_signal = signal_book["2019-06-03"]

    assert isinstance(overlap_signal, ReplaySignal)
    assert overlap_signal.signal_date == "2019-06-03"
    assert overlap_signal.fold_index == 2
    assert min(signal_book) == "2018-12-07"
    assert max(signal_book) == "2025-12-03"
    assert pytest.approx(sum(overlap_signal.target_weights.values()), rel=1e-6) == 1.0
    assert 20 <= len(overlap_signal.target_weights) <= 30
