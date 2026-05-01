"""AX1 feature-selection experiment runner."""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skyeye.products.ax1.autoresearch.phase0 import build_phase0_feature_audit
from skyeye.products.ax1.config import load_profile
from skyeye.products.ax1.data_builder import AX1TrainingDataBuilder, AX1TrainingDataRequest
from skyeye.products.ax1.run_experiment import run_experiment
from skyeye.products.ax1.train import DEFAULT_OUTPUT_ROOT, resolve_profile_path


DEFAULT_SCOPE_CANDIDATES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("scope_common", ("common",), "common-only baseline"),
    ("scope_common_etf_zscore", ("common", "etf_zscore"), "common plus ETF z-score signals"),
    (
        "scope_common_etf_zscore_regime",
        ("common", "etf_zscore", "regime"),
        "add regime state without interactions",
    ),
    (
        "scope_full",
        ("common", "etf_zscore", "regime", "regime_interaction"),
        "current full scope baseline",
    ),
)


@dataclass(frozen=True)
class FeatureSelectionCandidateSpec:
    candidate_id: str
    rationale: str
    include_scopes: tuple[str, ...] = ()
    feature_allowlist: tuple[str, ...] = ()
    feature_blocklist: tuple[str, ...] = ()
    phase: str = "feature_selection"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "phase": self.phase,
            "rationale": self.rationale,
            "include_scopes": list(self.include_scopes),
            "feature_allowlist": list(self.feature_allowlist),
            "feature_blocklist": list(self.feature_blocklist),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureSelectionCandidateSpec":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            rationale=str(payload.get("rationale", "")),
            include_scopes=tuple(str(item) for item in payload.get("include_scopes", []) or []),
            feature_allowlist=tuple(str(item) for item in payload.get("feature_allowlist", []) or []),
            feature_blocklist=tuple(str(item) for item in payload.get("feature_blocklist", []) or []),
            phase=str(payload.get("phase", "feature_selection")),
        )

    def to_config_override(self) -> dict[str, Any]:
        model: dict[str, Any] = {}
        features: dict[str, Any] = {}
        if self.include_scopes:
            scopes = list(self.include_scopes)
            model["include_scopes"] = scopes
            features["include_scopes"] = scopes
        if self.feature_allowlist:
            model["feature_allowlist"] = list(self.feature_allowlist)
        if self.feature_blocklist:
            model["feature_blocklist"] = list(self.feature_blocklist)
        override: dict[str, Any] = {"model": model}
        if features:
            override["features"] = features
        return override


def build_feature_selection_candidate_specs(
    *,
    baseline_features: list[str] | tuple[str, ...],
    phase0_audit: dict[str, Any],
    max_candidates: int | None = None,
) -> list[FeatureSelectionCandidateSpec]:
    ordered_features = tuple(str(feature) for feature in baseline_features)
    phase0_features = {
        str(feature): dict(payload or {})
        for feature, payload in (phase0_audit.get("features") or {}).items()
    }
    specs: list[FeatureSelectionCandidateSpec] = [
        FeatureSelectionCandidateSpec(
            candidate_id=candidate_id,
            include_scopes=scopes,
            rationale=rationale,
            phase="scope_ablation",
        )
        for candidate_id, scopes, rationale in DEFAULT_SCOPE_CANDIDATES
    ]

    hard_excluded = {
        feature
        for feature, payload in phase0_features.items()
        if str(payload.get("decision", "")) == "hard_exclude"
    }
    keep_features = tuple(
        feature
        for feature in ordered_features
        if str(phase0_features.get(feature, {}).get("decision", "")) == "keep_candidate"
    )
    no_hard_exclude = tuple(feature for feature in ordered_features if feature not in hard_excluded)

    if no_hard_exclude:
        specs.append(
            FeatureSelectionCandidateSpec(
                candidate_id="phase0_no_hard_exclude",
                feature_allowlist=no_hard_exclude,
                rationale="drop Phase 0 hard exclusions and keep watch-list features",
                phase="phase0_filter",
            )
        )
    if keep_features:
        specs.append(
            FeatureSelectionCandidateSpec(
                candidate_id="phase0_keep_only",
                feature_allowlist=keep_features,
                rationale="keep only Phase 0 statistically usable candidates",
                phase="phase0_filter",
            )
        )

    reason_candidates = (
        ("drop_low_variance", "low_cross_sectional_variance", "drop low cross-sectional variance features"),
        ("drop_stable_negative_ic", "stable_negative_ic", "drop features with stable negative IC"),
        ("drop_redundant", "redundant_non_representative", "drop non-representative correlated features"),
        ("drop_low_coverage", "low_coverage", "drop low-coverage features"),
        ("drop_weak_statistical_evidence", "weak_statistical_evidence", "drop weak-significance watch-list features"),
    )
    for candidate_id, reason, rationale in reason_candidates:
        removed = _features_with_reason(phase0_features, reason)
        if not removed:
            continue
        allowlist = tuple(feature for feature in ordered_features if feature not in removed)
        if allowlist:
            specs.append(
                FeatureSelectionCandidateSpec(
                    candidate_id=candidate_id,
                    feature_allowlist=allowlist,
                    rationale=rationale,
                    phase="reason_ablation",
                )
            )

    specs.extend(_build_family_ablation_specs(ordered_features))
    specs.extend(_build_top_stat_specs(ordered_features, phase0_features))
    return _limit_specs(_dedupe_specs(specs), max_candidates)


def summarize_experiment_for_leaderboard(
    spec: FeatureSelectionCandidateSpec,
    result: dict[str, Any],
) -> dict[str, Any]:
    training_summary = dict(result.get("training_summary") or {})
    evaluation = dict(result.get("evaluation") or {})
    signal = dict(evaluation.get("signal") or {})
    portfolio = dict(evaluation.get("portfolio") or {})
    gate_summary = dict(result.get("gate_summary") or {})
    positive_ratio = dict(training_summary.get("positive_ratio") or {})
    stability = dict(training_summary.get("stability") or {})
    robustness = dict(training_summary.get("robustness") or {})
    bootstrap_ci = dict(robustness.get("bootstrap_ci") or {})
    sample_decay = dict(robustness.get("sample_decay") or {})
    overfit_flags = dict(training_summary.get("overfit_flags") or robustness.get("overfit_flags") or {})
    feature_review_summary = dict(training_summary.get("feature_review_summary") or {})
    feature_columns = result.get("training_summary", {}).get("feature_columns") or list(spec.feature_allowlist)

    return {
        "candidate_id": spec.candidate_id,
        "status": str(result.get("status", "")),
        "phase": spec.phase,
        "rationale": spec.rationale,
        "output_dir": str(result.get("output_dir", "")),
        "include_scopes": list(spec.include_scopes),
        "feature_count": int(len(feature_columns)),
        "feature_warning_count": int(feature_review_summary.get("warning_count", 0) or 0),
        "gate_passed": bool(gate_summary.get("passed", False)),
        "failed_checks": list(gate_summary.get("failed_checks") or []),
        "rank_ic_mean": _num(signal.get("rank_ic_mean")),
        "rank_ic_p_value": _num((signal.get("rank_ic_significance") or {}).get("p_value"), default=1.0),
        "top_bucket_spread_mean": _num(signal.get("top_bucket_spread_mean")),
        "positive_ratio": _num(positive_ratio.get("positive_ratio")),
        "cv": _num(stability.get("cv")),
        "stability_score": _num(stability.get("stability_score")),
        "bootstrap_ci_low": _num(bootstrap_ci.get("ci_low")),
        "bootstrap_ci_high": _num(bootstrap_ci.get("ci_high")),
        "bootstrap_ci_crosses_zero": bool(bootstrap_ci.get("ci_crosses_zero", False)),
        "sample_decay_late_minus_early": _num(sample_decay.get("late_minus_early")),
        "flag_late_decay": bool(sample_decay.get("flag_late_decay", False)),
        "flag_ic_decay": bool(overfit_flags.get("flag_ic_decay", False)),
        "flag_spread_decay": bool(overfit_flags.get("flag_spread_decay", False)),
        "flag_val_dominant": bool(overfit_flags.get("flag_val_dominant", False)),
        "net_mean_return": _num(portfolio.get("net_mean_return")),
        "max_drawdown": _num(portfolio.get("max_drawdown")),
        "max_excess_drawdown": _num(portfolio.get("max_excess_drawdown")),
        "max_rolling_underperformance": _num(portfolio.get("max_rolling_underperformance")),
        "mean_turnover": _num(portfolio.get("mean_turnover")),
    }


def summarize_error_for_leaderboard(
    spec: FeatureSelectionCandidateSpec,
    exc: BaseException,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "candidate_id": spec.candidate_id,
        "status": "error",
        "phase": spec.phase,
        "rationale": spec.rationale,
        "output_dir": "",
        "include_scopes": list(spec.include_scopes),
        "feature_count": int(len(spec.feature_allowlist)),
        "feature_warning_count": 0,
        "gate_passed": False,
        "failed_checks": ["runner_error"],
        "rank_ic_mean": 0.0,
        "rank_ic_p_value": 1.0,
        "top_bucket_spread_mean": 0.0,
        "positive_ratio": 0.0,
        "cv": 0.0,
        "stability_score": 0.0,
        "bootstrap_ci_low": 0.0,
        "bootstrap_ci_high": 0.0,
        "bootstrap_ci_crosses_zero": True,
        "sample_decay_late_minus_early": 0.0,
        "flag_late_decay": False,
        "flag_ic_decay": False,
        "flag_spread_decay": False,
        "flag_val_dominant": False,
        "net_mean_return": 0.0,
        "max_drawdown": 0.0,
        "max_excess_drawdown": 0.0,
        "max_rolling_underperformance": 0.0,
        "mean_turnover": 0.0,
        "elapsed_seconds": float(elapsed_seconds),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(limit=12),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AX1 feature-selection experiments")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--profile", default="personal_etf_core")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing baseline/leaderboard under output-root/run-tag and skip completed candidates.",
    )
    parser.add_argument(
        "--stop-after-first-pass",
        action="store_true",
        help="Stop once a candidate passes the promotion gate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile_path = resolve_profile_path(args.profile)
    profile_config = load_profile(profile_path)
    from skyeye.data import DataFacade

    facade = DataFacade()
    raw_df = AX1TrainingDataBuilder(data_facade=facade).build(
        AX1TrainingDataRequest(
            profile_config=profile_config,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    )
    root_dir = Path(args.output_root) / str(args.run_tag)
    root_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = _load_leaderboard(root_dir) if args.resume else []
    baseline_result = _load_baseline_result(root_dir) if args.resume else None
    if baseline_result is None:
        baseline_started = time.time()
        print("[ax1-feature-select] start baseline", flush=True)
        baseline_result = run_experiment(
            profile_path=profile_path,
            output_dir=root_dir / "baseline",
            raw_df=raw_df,
            raw_csv=None,
            experiment_name=f"{args.run_tag}_baseline",
            data_provider=facade,
        )
        baseline_elapsed = round(time.time() - baseline_started, 3)
        print(f"[ax1-feature-select] done baseline seconds={baseline_elapsed}", flush=True)
        leaderboard = [
            row for row in leaderboard
            if str(row.get("candidate_id", "")) != "baseline"
        ]
        leaderboard.insert(
            0,
            {
                **summarize_experiment_for_leaderboard(
                    FeatureSelectionCandidateSpec(
                        candidate_id="baseline",
                        rationale="unmodified profile",
                        phase="baseline",
                    ),
                    baseline_result,
                ),
                "elapsed_seconds": baseline_elapsed,
            },
        )
        _write_leaderboard(root_dir, leaderboard)
    else:
        print("[ax1-feature-select] resume baseline", flush=True)
        if "baseline" not in _completed_candidate_ids(leaderboard):
            leaderboard.insert(
                0,
                {
                    **summarize_experiment_for_leaderboard(
                        FeatureSelectionCandidateSpec(
                            candidate_id="baseline",
                            rationale="unmodified profile",
                            phase="baseline",
                        ),
                        baseline_result,
                    ),
                    "elapsed_seconds": 0.0,
                },
            )
            _write_leaderboard(root_dir, leaderboard)

    phase0_audit = _load_phase0_audit(root_dir) if args.resume else None
    if phase0_audit is None:
        phase0_audit = build_phase0_feature_audit(baseline_result.get("training_summary") or {})
        _write_json(root_dir / "phase0_audit.json", phase0_audit)

    specs = _load_candidate_specs(root_dir) if args.resume else []
    if not specs:
        specs = build_feature_selection_candidate_specs(
            baseline_features=baseline_result.get("training_summary", {}).get("feature_columns", []),
            phase0_audit=phase0_audit,
            max_candidates=args.max_candidates,
        )
        _write_json(root_dir / "candidate_specs.json", [spec.to_dict() for spec in specs])
    elif args.max_candidates is not None:
        specs = _limit_specs(specs, args.max_candidates)

    remaining_specs = _remaining_specs(specs, leaderboard) if args.resume else specs
    for spec in remaining_specs:
        started = time.time()
        try:
            print(f"[ax1-feature-select] start {spec.candidate_id} {spec.to_dict()}", flush=True)
            result = run_experiment(
                profile_path=profile_path,
                output_dir=root_dir / spec.candidate_id,
                raw_df=raw_df,
                raw_csv=None,
                experiment_name=f"{args.run_tag}_{spec.candidate_id}",
                config_override=spec.to_config_override(),
                data_provider=facade,
            )
            row = summarize_experiment_for_leaderboard(spec, result)
            row["elapsed_seconds"] = round(time.time() - started, 3)
            leaderboard.append(row)
            print(f"[ax1-feature-select] done {spec.candidate_id} gate={row['gate_passed']}", flush=True)
            _write_leaderboard(root_dir, leaderboard)
            if args.stop_after_first_pass and row["gate_passed"]:
                break
        except Exception as exc:  # pragma: no cover - exercised by live CLI failures
            row = summarize_error_for_leaderboard(spec, exc, elapsed_seconds=time.time() - started)
            leaderboard.append(row)
            print(f"[ax1-feature-select] error {spec.candidate_id} {type(exc).__name__}: {exc}", flush=True)
            _write_leaderboard(root_dir, leaderboard)

    _write_leaderboard(root_dir, leaderboard)
    print(str(root_dir))
    return 0


def _features_with_reason(phase0_features: dict[str, dict[str, Any]], reason: str) -> set[str]:
    return {
        feature
        for feature, payload in phase0_features.items()
        if reason in {str(item) for item in payload.get("reasons", []) or []}
    }


def _build_family_ablation_specs(ordered_features: tuple[str, ...]) -> list[FeatureSelectionCandidateSpec]:
    family_map: dict[str, list[str]] = {}
    for feature in ordered_features:
        family_map.setdefault(_feature_family(feature), []).append(feature)
    specs: list[FeatureSelectionCandidateSpec] = []
    for family, family_features in sorted(family_map.items()):
        if family == "other" or not family_features:
            continue
        family_set = set(family_features)
        drop_allowlist = tuple(feature for feature in ordered_features if feature not in family_set)
        if drop_allowlist:
            specs.append(
                FeatureSelectionCandidateSpec(
                    candidate_id=f"drop_family_{family}",
                    feature_allowlist=drop_allowlist,
                    rationale=f"drop {family} feature family",
                    phase="family_ablation",
                )
            )
        if len(family_features) >= 2:
            specs.append(
                FeatureSelectionCandidateSpec(
                    candidate_id=f"only_family_{family}",
                    feature_allowlist=tuple(family_features),
                    rationale=f"use only {family} feature family",
                    phase="family_ablation",
                )
            )
    return specs


def _build_top_stat_specs(
    ordered_features: tuple[str, ...],
    phase0_features: dict[str, dict[str, Any]],
) -> list[FeatureSelectionCandidateSpec]:
    ranked = _rank_phase0_features(ordered_features, phase0_features)
    specs: list[FeatureSelectionCandidateSpec] = []
    for size in (5, 10, 15):
        if len(ranked) < size:
            continue
        allowlist = tuple(feature for feature, _score in ranked[:size])
        specs.append(
            FeatureSelectionCandidateSpec(
                candidate_id=f"top_stat_{size}",
                feature_allowlist=allowlist,
                rationale=f"top {size} Phase 0 features by IC significance and fold stability",
                phase="stat_ranked_beam",
            )
        )
    return specs


def _rank_phase0_features(
    ordered_features: tuple[str, ...],
    phase0_features: dict[str, dict[str, Any]],
) -> list[tuple[str, float]]:
    order = {feature: idx for idx, feature in enumerate(ordered_features)}
    scored: list[tuple[str, float]] = []
    for feature in ordered_features:
        payload = phase0_features.get(feature) or {}
        if str(payload.get("decision", "")) == "hard_exclude":
            continue
        rank_ic = _num(payload.get("rank_ic_mean"))
        spread = _num(payload.get("top_bucket_spread_mean"))
        positive_ratio = _num(payload.get("positive_fold_ratio"))
        p_value = _num(payload.get("ic_p_value"), default=1.0)
        fold_count = _num(payload.get("fold_count"))
        significance = max(0.0, 1.0 - min(max(p_value, 0.0), 1.0))
        fold_support = min(fold_count / 6.0, 1.0)
        score = rank_ic + (0.10 * positive_ratio) + (0.05 * significance) + (0.02 * fold_support) + _signed_unit(spread) * 0.05
        scored.append((feature, float(score)))
    return sorted(scored, key=lambda item: (-item[1], order[item[0]]))


def _feature_family(feature: str) -> str:
    name = str(feature)
    if "interaction" in name:
        return "regime_interaction"
    if "regime" in name:
        return "regime"
    if "style" in name:
        return "style"
    if "volume_price_flow" in name:
        return "volume_flow"
    if "momentum" in name or "mom_" in name or "excess_mom" in name:
        return "momentum"
    if any(token in name for token in ("volatility", "risk", "liquidity", "dollar_volume", "cost", "amihud", "turnover")):
        return "risk_liquidity"
    if "fundamental" in name or name in {"feature_roe_ttm", "feature_net_profit_growth"}:
        return "fundamental"
    if "flow" in name or "northbound" in name or "margin" in name:
        return "flow"
    if "macro" in name or "yield" in name or "bond" in name:
        return "macro"
    if any(token in name for token in ("rsi", "macd", "boll", "atr", "ma_", "ema_", "kdj")):
        return "technical"
    return "other"


def _signed_unit(value: float) -> float:
    if value > 0:
        return min(abs(value) / 0.01, 1.0)
    if value < 0:
        return -min(abs(value) / 0.01, 1.0)
    return 0.0


def _dedupe_specs(specs: list[FeatureSelectionCandidateSpec]) -> list[FeatureSelectionCandidateSpec]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[FeatureSelectionCandidateSpec] = []
    for spec in specs:
        key = (spec.include_scopes, spec.feature_allowlist, spec.feature_blocklist)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


def _limit_specs(
    specs: list[FeatureSelectionCandidateSpec],
    max_candidates: int | None,
) -> list[FeatureSelectionCandidateSpec]:
    if max_candidates is None:
        return specs
    return specs[: max(0, int(max_candidates))]


def _remaining_specs(
    specs: list[FeatureSelectionCandidateSpec],
    leaderboard: list[dict[str, Any]],
) -> list[FeatureSelectionCandidateSpec]:
    completed = _completed_candidate_ids(leaderboard)
    return [spec for spec in specs if spec.candidate_id not in completed]


def _completed_candidate_ids(leaderboard: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("candidate_id", ""))
        for row in leaderboard
        if row.get("candidate_id") is not None
    }


def _load_leaderboard(root_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(root_dir / "leaderboard.json", default=[])
    if not isinstance(payload, list):
        return []
    return [dict(row or {}) for row in payload]


def _load_baseline_result(root_dir: Path) -> dict[str, Any] | None:
    payload = _read_json(root_dir / "baseline" / "experiment.json", default=None)
    return dict(payload) if isinstance(payload, dict) else None


def _load_phase0_audit(root_dir: Path) -> dict[str, Any] | None:
    payload = _read_json(root_dir / "phase0_audit.json", default=None)
    return dict(payload) if isinstance(payload, dict) else None


def _load_candidate_specs(root_dir: Path) -> list[FeatureSelectionCandidateSpec]:
    payload = _read_json(root_dir / "candidate_specs.json", default=[])
    if not isinstance(payload, list):
        return []
    return [FeatureSelectionCandidateSpec.from_dict(dict(item or {})) for item in payload]


def _write_leaderboard(root_dir: Path, leaderboard: list[dict[str, Any]]) -> None:
    _write_json(root_dir / "leaderboard.json", leaderboard)
    if not leaderboard:
        return
    fieldnames = list(leaderboard[0].keys())
    with (root_dir / "leaderboard.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(leaderboard)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _num(value: Any, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result or result in (float("inf"), float("-inf")):
        return default
    return result


if __name__ == "__main__":
    raise SystemExit(main())
