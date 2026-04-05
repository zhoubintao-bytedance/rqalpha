import pytest

from skyeye.products.tx1.strategies.rolling_score import runtime as tx1_runtime
from skyeye.products.tx1.strategies.rolling_score.runtime import build_runtime


def test_runtime_builds_from_strategy_id_and_artifact_line():
    runtime = build_runtime(
        strategy_id="tx1.rolling_score",
        artifact_line="baseline_linear",
    )

    assert runtime["strategy_id"] == "tx1.rolling_score"
    assert runtime["artifact_line_id"] == "baseline_linear"
    assert runtime["benchmark"] == "000300.XSHG"
    assert runtime["profile"]["profile"] == "smooth"


def test_runtime_uses_spec_default_artifact_line_when_not_overridden():
    runtime = build_runtime(strategy_id="tx1.rolling_score")

    assert runtime["artifact_line_id"] == "combo_b25_h45"
    assert runtime["artifact"]["model_kind"] == "lgbm"


def test_runtime_allows_env_override_for_artifact_line(monkeypatch):
    monkeypatch.setenv("SKYEYE_TX1_ARTIFACT_LINE", "baseline_tree")

    runtime = build_runtime(strategy_id="tx1.rolling_score")

    assert runtime["artifact_line_id"] == "baseline_tree"
    assert runtime["artifact"]["model_kind"] == "tree"


def test_runtime_allows_extra_config_override_for_artifact_line(monkeypatch):
    class _Extra:
        tx1_artifact_line = "baseline_tree"

    class _Config:
        extra = _Extra()

    class _Env:
        config = _Config()

    monkeypatch.delenv("SKYEYE_TX1_ARTIFACT_LINE", raising=False)
    monkeypatch.setattr(tx1_runtime.Environment, "get_instance", classmethod(lambda cls: _Env()))

    runtime = build_runtime(strategy_id="tx1.rolling_score")

    assert runtime["artifact_line_id"] == "baseline_tree"
    assert runtime["artifact"]["model_kind"] == "tree"


def test_runtime_can_preload_signal_book():
    runtime = build_runtime(
        strategy_id="tx1.rolling_score",
        artifact_line="baseline_linear",
        load_signal_book=True,
    )

    assert "signal_book" in runtime
    assert runtime["signal_book"]["2019-06-03"].fold_index == 2


def test_runtime_rejects_profile_benchmark_override():
    with pytest.raises(ValueError):
        build_runtime(
            strategy_id="tx1.rolling_score",
            artifact_line="baseline_linear",
            profile_overrides={"benchmark": "000905.XSHG"},
        )


def test_runtime_rejects_profile_artifact_line_override():
    with pytest.raises(ValueError):
        build_runtime(
            strategy_id="tx1.rolling_score",
            artifact_line="baseline_linear",
            profile_overrides={"artifact_line_id": "baseline_tree"},
        )
