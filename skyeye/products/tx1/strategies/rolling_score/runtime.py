"""TX1 rolling-score runtime assembly.

Builds a validated runtime context from strategy spec, artifact resolver,
and profile loader. Enforces frozen fields (benchmark, artifact_line)
cannot be overridden by profiles.
"""

from __future__ import annotations

import os
from pathlib import Path

from rqalpha.environment import Environment

from skyeye.products.tx1.artifacts import (
    load_replay_signal_book,
    parse_artifact_line,
    resolve_artifact,
)
from skyeye.products.tx1.registry import get_strategy_spec
from skyeye.products.tx1.strategies.rolling_score.params import FROZEN_FIELDS, load_profile


ARTIFACTS_ROOT = Path(__file__).resolve().parents[4] / "artifacts" / "experiments" / "tx1"


def build_runtime(
    strategy_id: str,
    artifact_line: str | None = None,
    profile_name: str | None = None,
    profile_overrides: dict | None = None,
    artifacts_root: Path | None = None,
    load_signal_book: bool = False,
) -> dict:
    """Build a validated TX1 runtime context.

    Args:
        strategy_id: TX1 strategy identifier (e.g. "tx1.rolling_score")
        artifact_line: Artifact line identifier (e.g. "baseline_linear"). If omitted,
            resolves from env/extra/spec defaults.
        profile_name: Optional profile name override
        profile_overrides: Optional dict of profile field overrides
        artifacts_root: Optional custom artifacts root path
        load_signal_book: Whether to preload replayable OOS weights

    Returns:
        Validated runtime context dict

    Raises:
        ValueError: If profile_overrides contains frozen fields
        FileNotFoundError: If artifact cannot be resolved
        KeyError: If strategy_id is unknown
    """
    # Validate frozen field overrides
    if profile_overrides:
        forbidden = FROZEN_FIELDS & set(profile_overrides.keys())
        if forbidden:
            raise ValueError(
                "profile cannot override frozen fields: {}. "
                "Change benchmark or artifact_line requires new strategy_id "
                "or new artifact_line.".format(", ".join(sorted(forbidden)))
            )

    # Load strategy spec
    spec = get_strategy_spec(strategy_id)
    resolved_artifact_line = _resolve_artifact_line_id(spec, artifact_line)

    # Resolve artifact
    ref = "{}@{}".format(strategy_id, resolved_artifact_line)
    line = parse_artifact_line(ref)
    root = artifacts_root or ARTIFACTS_ROOT
    resolved = resolve_artifact(line, root)

    # Load and merge profile
    profile = load_profile(profile_name)
    if profile_overrides:
        safe_overrides = {k: v for k, v in profile_overrides.items() if k not in FROZEN_FIELDS}
        profile = {**profile, **safe_overrides}

    _validate_frozen_runtime_fields(
        strategy_id=strategy_id,
        spec=spec.raw,
        profile=profile,
        benchmark_id=resolved.benchmark_id,
        artifact_line_id=resolved.artifact_line_id,
    )

    runtime = {
        "strategy_id": strategy_id,
        "artifact_line_id": resolved_artifact_line,
        "benchmark": resolved.benchmark_id,
        "spec": spec.raw,
        "profile": profile,
        "artifact": {
            "artifact_root": str(resolved.artifact_root),
            "metadata_path": str(resolved.metadata_path),
            "train_cutoff": resolved.train_cutoff,
            "model_kind": resolved.metadata.get("model_kind"),
            "experiment_name": resolved.metadata.get("experiment_name"),
            "num_folds": resolved.metadata.get("num_folds"),
        },
    }
    if load_signal_book:
        runtime["signal_book"] = load_replay_signal_book(resolved)
    return runtime


def _resolve_artifact_line_id(spec, explicit_line: str | None) -> str:
    if explicit_line:
        return explicit_line

    env_line = os.environ.get("SKYEYE_TX1_ARTIFACT_LINE")
    if env_line:
        return env_line

    extra_line = _extra_runtime_value("tx1_artifact_line")
    if extra_line:
        return extra_line

    spec_line = spec.raw.get("artifact_line_id")
    if spec_line:
        return spec_line

    raise RuntimeError("missing artifact_line_id for strategy {}".format(spec.strategy_id))


def _extra_runtime_value(field_name: str):
    try:
        env = Environment.get_instance()
    except Exception:
        return None
    extra = getattr(getattr(env, "config", None), "extra", None)
    if extra is None:
        return None
    return getattr(extra, field_name, None)


def _validate_frozen_runtime_fields(
    strategy_id: str,
    spec: dict,
    profile: dict,
    benchmark_id: str,
    artifact_line_id: str,
) -> None:
    spec_benchmark = spec.get("benchmark")
    if spec_benchmark and spec_benchmark != benchmark_id:
        raise ValueError(
            "strategy {} benchmark mismatch between spec ({}) and artifact ({})".format(
                strategy_id,
                spec_benchmark,
                benchmark_id,
            )
        )

    profile_benchmark = profile.get("benchmark")
    if profile_benchmark and profile_benchmark != benchmark_id:
        raise ValueError(
            "profile benchmark {} does not match frozen benchmark {}".format(
                profile_benchmark,
                benchmark_id,
            )
        )

    profile_artifact_line = profile.get("artifact_line_id")
    if profile_artifact_line and profile_artifact_line != artifact_line_id:
        raise ValueError(
            "profile artifact_line_id {} does not match runtime artifact_line_id {}".format(
                profile_artifact_line,
                artifact_line_id,
            )
        )
