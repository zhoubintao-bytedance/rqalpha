# -*- coding: utf-8 -*-
"""
TX1 Robustness & Stability Module

Provides cross-fold stability scoring, overfit detection, and
regime-conditioned evaluation. Reuses stability scoring methodology
from skyeye/evaluation/rolling_score/engine.py with adaptations
for walk-forward fold analysis and experiment-vs-experiment comparison.

Core functions:
  1. compute_stability_score - CV + worst-fold + consecutive-bad-folds
  2. detect_overfit_flags    - validation-vs-test decay detection
  3. compute_regime_scores   - per-regime metric breakdown
  4. compare_experiments     - baseline vs candidate comparison
  5. render_comparison_report - merge-decision friendly text report
"""

from pathlib import Path

import numpy as np


def compute_stability_score(fold_metrics_list, metric_key="rank_ic_mean"):
    """Compute cross-fold stability score for a given metric.

    Three-dimensional score (mirrors rolling_score/engine.py stability):
      - CV (coefficient of variation): 50% weight
      - Worst fold value: 30% weight
      - Max consecutive below-median folds: 20% weight

    Args:
        fold_metrics_list: List of dicts, each containing prediction/portfolio metrics
        metric_key: Which metric to evaluate stability on

    Returns:
        Dict with stability_score (0-100), cv, worst_value, max_consecutive_low,
        and per-component scores
    """
    if not fold_metrics_list:
        return _empty_stability()

    values = _extract_metric_values(fold_metrics_list, metric_key)
    if len(values) == 0:
        return _empty_stability()

    arr = np.array(values, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=0))
    worst_val = float(np.min(arr))
    median_val = float(np.median(arr))

    cv = std_val / abs(mean_val) if abs(mean_val) > 1e-12 else float("inf")
    cv_score = _score_cv(cv)

    worst_score = _score_worst(worst_val, median_val)

    consecutive_low = _max_consecutive_below(arr, median_val)
    consec_score = _score_consecutive(consecutive_low, len(arr))

    stability = 0.5 * cv_score + 0.3 * worst_score + 0.2 * consec_score

    return {
        "stability_score": float(np.clip(stability, 0, 100)),
        "cv": cv,
        "worst_value": worst_val,
        "max_consecutive_low": consecutive_low,
        "cv_score": cv_score,
        "worst_score": worst_score,
        "consecutive_score": consec_score,
        "n_folds": len(arr),
        "mean": mean_val,
        "std": std_val,
        "metric_key": metric_key,
    }


def detect_overfit_flags(fold_results):
    """Detect potential overfitting by comparing validation vs test metrics.

    Flags:
      - val_test_ic_decay: Mean Rank IC drops significantly from val to test
      - val_test_spread_decay: Top bucket spread decays val -> test
      - val_dominant: Validation metrics consistently better than test

    Args:
        fold_results: List of fold result dicts with validation_metrics and prediction_metrics

    Returns:
        Dict with boolean flags and numeric decay ratios
    """
    if not fold_results:
        return _empty_overfit()

    ic_decays = []
    spread_decays = []
    val_wins = 0

    for fold in fold_results:
        val_m = fold.get("validation_metrics", {})
        test_m = fold.get("prediction_metrics", {})
        if not val_m or not test_m:
            continue

        val_ic = val_m.get("rank_ic_mean", 0.0)
        test_ic = test_m.get("rank_ic_mean", 0.0)
        ic_decays.append(val_ic - test_ic)

        val_spread = val_m.get("top_bucket_spread_mean", 0.0)
        test_spread = test_m.get("top_bucket_spread_mean", 0.0)
        spread_decays.append(val_spread - test_spread)

        val_score = val_ic + val_spread
        test_score = test_ic + test_spread
        if val_score > test_score:
            val_wins += 1

    n = len(ic_decays)
    if n == 0:
        return _empty_overfit()

    mean_ic_decay = float(np.mean(ic_decays))
    mean_spread_decay = float(np.mean(spread_decays))
    val_dominant_ratio = val_wins / n

    return {
        "val_test_ic_decay": mean_ic_decay,
        "val_test_spread_decay": mean_spread_decay,
        "val_dominant_ratio": val_dominant_ratio,
        "flag_ic_decay": mean_ic_decay > 0.02,
        "flag_spread_decay": mean_spread_decay > 0.005,
        "flag_val_dominant": val_dominant_ratio > 0.8,
        "n_folds_compared": n,
    }


def compute_regime_scores(fold_results, metric_key="rank_ic_mean"):
    """Score performance across different market regimes within each fold.

    Regimes are determined by portfolio return sign:
      - up: portfolio_return > 0 periods
      - down: portfolio_return <= 0 periods

    This is a simplified regime analysis; full regime detection
    (bull/bear/sideways) requires market-level data not available
    in the research proxy output.

    Args:
        fold_results: List of fold result dicts with portfolio_returns_df
        metric_key: Metric to report per-regime

    Returns:
        Dict with per-regime statistics
    """
    up_returns = []
    down_returns = []

    for fold in fold_results:
        port_df = fold.get("portfolio_returns_df")
        if port_df is None or len(port_df) == 0:
            continue
        returns = port_df["portfolio_return"].values
        up_mask = returns > 0
        down_mask = ~up_mask
        if up_mask.any():
            up_returns.extend(returns[up_mask].tolist())
        if down_mask.any():
            down_returns.extend(returns[down_mask].tolist())

    up_arr = np.array(up_returns) if up_returns else np.array([])
    down_arr = np.array(down_returns) if down_returns else np.array([])

    # Also compute fold-level metric consistency
    fold_values = _extract_metric_values(fold_results, metric_key)
    positive_folds = sum(1 for v in fold_values if v > 0)
    n_folds = len(fold_values)

    return {
        "up_regime": {
            "n_periods": len(up_arr),
            "mean_return": float(np.mean(up_arr)) if len(up_arr) > 0 else 0.0,
            "total_return": float(np.sum(up_arr)) if len(up_arr) > 0 else 0.0,
        },
        "down_regime": {
            "n_periods": len(down_arr),
            "mean_return": float(np.mean(down_arr)) if len(down_arr) > 0 else 0.0,
            "total_return": float(np.sum(down_arr)) if len(down_arr) > 0 else 0.0,
        },
        "metric_consistency": {
            "metric_key": metric_key,
            "positive_folds": positive_folds,
            "total_folds": n_folds,
            "positive_ratio": positive_folds / n_folds if n_folds > 0 else 0.0,
        },
    }


def load_experiment_reference(
    ref,
    artifacts_root=None,
    default_strategy_id="tx1.rolling_score",
):
    """Resolve an experiment reference into a loaded TX1 experiment.

    Supported forms:
      - experiment directory path
      - experiment.json path
      - full artifact reference: ``tx1.rolling_score@baseline_lgbm``
      - bare artifact line id: ``baseline_lgbm``

    Returns:
        Dict with loaded experiment payload plus resolution metadata.
    """
    from skyeye.products.tx1.artifacts import ArtifactLine, parse_artifact_line, resolve_artifact
    from skyeye.products.tx1.persistence import load_experiment

    ref_text = str(ref).strip()
    if not ref_text:
        raise ValueError("experiment reference must not be empty")

    root = Path(artifacts_root) if artifacts_root is not None else _default_artifacts_root()
    candidates = [Path(ref_text)]
    if not Path(ref_text).is_absolute():
        candidates.append(root / ref_text)

    for candidate in candidates:
        if candidate.is_file():
            experiment_dir = candidate.parent
            experiment = load_experiment(str(experiment_dir))
            return {
                "reference": ref_text,
                "label": _experiment_label_from_path(experiment_dir),
                "source_type": "experiment_json",
                "path": str(experiment_dir.resolve()),
                "artifact_line_id": "",
                "strategy_id": "",
                "experiment": experiment,
            }
        if candidate.is_dir() and (candidate / "experiment.json").is_file():
            experiment = load_experiment(str(candidate))
            return {
                "reference": ref_text,
                "label": _experiment_label_from_path(candidate),
                "source_type": "experiment_dir",
                "path": str(candidate.resolve()),
                "artifact_line_id": "",
                "strategy_id": "",
                "experiment": experiment,
            }

    artifact_line = None
    if "@" in ref_text:
        artifact_line = parse_artifact_line(ref_text)
    else:
        artifact_dir = root / "tx1_{}".format(ref_text)
        if artifact_dir.is_dir():
            artifact_line = ArtifactLine(
                strategy_id=default_strategy_id,
                artifact_line_id=ref_text,
            )

    if artifact_line is None:
        raise FileNotFoundError(
            "could not resolve TX1 experiment reference {!r}".format(ref_text)
        )

    resolved = resolve_artifact(artifact_line, root)
    experiment = load_experiment(str(resolved.artifact_root))
    return {
        "reference": ref_text,
        "label": resolved.artifact_line_id,
        "source_type": "artifact_line",
        "path": str(resolved.artifact_root.resolve()),
        "artifact_line_id": resolved.artifact_line_id,
        "strategy_id": resolved.strategy_id,
        "experiment": experiment,
    }


def summarize_experiment(experiment_result, label=None, metric_key="rank_ic_mean"):
    """Build a comparison-friendly experiment summary."""
    fold_results = experiment_result.get("fold_results", [])
    aggregate = experiment_result.get("aggregate_metrics", {}) or {}
    if fold_results:
        prediction = _aggregate_metric_group(fold_results, "prediction_metrics")
        portfolio = _aggregate_metric_group(fold_results, "portfolio_metrics")
        stability = compute_stability_score(fold_results, metric_key=metric_key)
        overfit = detect_overfit_flags(fold_results)
        regime = compute_regime_scores(fold_results, metric_key=metric_key)
        net_return_stability = compute_stability_score(fold_results, metric_key="net_mean_return")
        gross_return_stability = compute_stability_score(fold_results, metric_key="mean_return")
    else:
        prediction = aggregate.get("prediction") or {}
        portfolio = aggregate.get("portfolio") or {}
        robustness = aggregate.get("robustness") or {}
        stability = robustness.get("stability") or _empty_stability()
        overfit = robustness.get("overfit_flags") or _empty_overfit()
        regime = robustness.get("regime_scores") or compute_regime_scores([], metric_key=metric_key)
        net_return_stability = _empty_stability()
        gross_return_stability = _empty_stability()
    folds = [_build_fold_snapshot(fold, metric_key) for fold in fold_results]

    return {
        "label": label or experiment_result.get("experiment_name") or "experiment",
        "experiment_name": experiment_result.get("experiment_name"),
        "model_kind": experiment_result.get("model_kind", ""),
        "output_dir": experiment_result.get("output_dir"),
        "config": experiment_result.get("config", {}),
        "metric_key": metric_key,
        "prediction": prediction,
        "portfolio": portfolio,
        "robustness": {
            "stability": stability,
            "overfit_flags": overfit,
            "regime_scores": regime,
            "net_return_stability": net_return_stability,
            "gross_return_stability": gross_return_stability,
        },
        "fold_distribution": _compute_fold_distribution(folds, metric_key),
        "folds": folds,
    }


def compare_experiment_pair(
    baseline_summary,
    candidate_summary,
    metric_key="rank_ic_mean",
    return_metric="net_mean_return",
):
    """Compare one candidate experiment against a baseline summary."""
    fold_rows = _align_fold_rows(baseline_summary.get("folds", []), candidate_summary.get("folds", []))
    fold_level = _build_fold_level_comparison(fold_rows, return_metric=return_metric)
    regime_level = _build_regime_level_comparison(baseline_summary, candidate_summary)
    stability_level = _build_stability_level_comparison(baseline_summary, candidate_summary)
    cost_level = _build_cost_level_comparison(baseline_summary, candidate_summary)
    overfit_level = _build_overfit_level_comparison(baseline_summary, candidate_summary)
    decision = _build_merge_decision(
        fold_level=fold_level,
        stability_level=stability_level,
        cost_level=cost_level,
        overfit_level=overfit_level,
    )

    return {
        "metric_key": metric_key,
        "return_metric": return_metric,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "fold_level": fold_level,
        "regime_level": regime_level,
        "stability_level": stability_level,
        "cost_level": cost_level,
        "overfit_level": overfit_level,
        "flags": decision["flags"],
        "decision": decision,
    }


def compare_experiments(
    experiment_refs,
    baseline_ref=None,
    artifacts_root=None,
    metric_key="rank_ic_mean",
    return_metric="net_mean_return",
):
    """Compare multiple TX1 experiments against a baseline.

    If ``baseline_ref`` is omitted:
      - with 1 candidate ref, baseline defaults to ``baseline_lgbm``
      - with >=2 refs, the first ref is treated as baseline
    """
    refs = list(experiment_refs or [])
    if not refs and baseline_ref is None:
        raise ValueError("at least one experiment reference is required")

    if baseline_ref is None:
        if len(refs) == 1:
            baseline_ref = "baseline_lgbm"
        else:
            baseline_ref = refs[0]
            refs = refs[1:]

    if not refs:
        raise ValueError("at least one candidate experiment is required")

    baseline_loaded = load_experiment_reference(baseline_ref, artifacts_root=artifacts_root)
    baseline_summary = summarize_experiment(
        baseline_loaded["experiment"],
        label=baseline_loaded["label"],
        metric_key=metric_key,
    )
    baseline_summary["reference"] = baseline_loaded["reference"]
    baseline_summary["source_type"] = baseline_loaded["source_type"]
    baseline_summary["path"] = baseline_loaded["path"]
    baseline_summary["artifact_line_id"] = baseline_loaded.get("artifact_line_id", "")
    baseline_summary["strategy_id"] = baseline_loaded.get("strategy_id", "")

    comparisons = []
    experiment_summaries = [baseline_summary]
    for ref in refs:
        loaded = load_experiment_reference(ref, artifacts_root=artifacts_root)
        summary = summarize_experiment(
            loaded["experiment"],
            label=loaded["label"],
            metric_key=metric_key,
        )
        summary["reference"] = loaded["reference"]
        summary["source_type"] = loaded["source_type"]
        summary["path"] = loaded["path"]
        summary["artifact_line_id"] = loaded.get("artifact_line_id", "")
        summary["strategy_id"] = loaded.get("strategy_id", "")
        experiment_summaries.append(summary)
        comparisons.append(
            compare_experiment_pair(
                baseline_summary,
                summary,
                metric_key=metric_key,
                return_metric=return_metric,
            )
        )

    return {
        "metric_key": metric_key,
        "return_metric": return_metric,
        "baseline": baseline_summary,
        "experiments": experiment_summaries,
        "comparisons": comparisons,
    }


def render_comparison_report(comparison_result):
    """Render a human-readable multi-experiment comparison report."""
    baseline = comparison_result["baseline"]
    metric_key = comparison_result["metric_key"]
    return_metric = comparison_result["return_metric"]

    lines = [
        "TX1 Experiment Comparison Report",
        "=" * 80,
        "Baseline: {label}  model={model}  ref={ref}".format(
            label=baseline["label"],
            model=baseline.get("model_kind") or "?",
            ref=baseline.get("reference") or baseline.get("path") or "",
        ),
        "Metric key: {}  Return metric: {}".format(metric_key, return_metric),
        "",
        "Overview",
        "-" * 80,
    ]
    lines.extend(_render_overview_table(comparison_result["experiments"]))

    for item in comparison_result["comparisons"]:
        candidate = item["candidate"]
        fold_level = item["fold_level"]
        regime_level = item["regime_level"]
        stability_level = item["stability_level"]
        cost_level = item["cost_level"]
        overfit_level = item["overfit_level"]
        decision = item["decision"]

        lines.extend(
            [
                "",
                "Candidate: {label}  model={model}  ref={ref}".format(
                    label=candidate["label"],
                    model=candidate.get("model_kind") or "?",
                    ref=candidate.get("reference") or candidate.get("path") or "",
                ),
                "Decision: {status}  |  {reason}".format(
                    status=decision["status"],
                    reason="; ".join(decision["reasons"]) or "beats baseline without new warnings",
                ),
                "",
                "Fold-level",
                "  aligned_folds: {n}".format(n=fold_level["aligned_folds"]),
                "  {metric}: delta_mean={delta}  wins={wins}/{n}".format(
                    metric=metric_key,
                    delta=_fmt_signed(fold_level["metric_delta_mean"], digits=6),
                    wins=fold_level["metric_win_folds"],
                    n=max(fold_level["aligned_folds"], 1),
                ),
                "  {metric}: delta_mean={delta}  wins={wins}/{n}".format(
                    metric=return_metric,
                    delta=_fmt_signed(fold_level["return_delta_mean"], digits=6),
                    wins=fold_level["return_win_folds"],
                    n=max(fold_level["aligned_folds"], 1),
                ),
                "  收益来自少数窗口: {flag}  |  positive_uplift_folds={count}/{n}  top2_share={share}".format(
                    flag="YES" if fold_level["few_window_flag"] else "NO",
                    count=fold_level["positive_uplift_folds"],
                    n=max(fold_level["aligned_folds"], 1),
                    share=_fmt_percent(fold_level["top_2_uplift_share"]),
                ),
            ]
        )
        if fold_level["best_windows"]:
            best = ", ".join(
                "{}({})".format(row["window"], _fmt_signed(row["delta_return_metric"], digits=6))
                for row in fold_level["best_windows"]
            )
            lines.append("  best_windows: {}".format(best))
        if fold_level["worst_windows"]:
            worst = ", ".join(
                "{}({})".format(row["window"], _fmt_signed(row["delta_return_metric"], digits=6))
                for row in fold_level["worst_windows"]
            )
            lines.append("  worst_windows: {}".format(worst))

        lines.extend(
            [
                "",
                "Regime-level",
                "  up_regime mean_return delta: {}".format(
                    _fmt_signed(regime_level["up_mean_return_delta"], digits=6)
                ),
                "  down_regime mean_return delta: {}".format(
                    _fmt_signed(regime_level["down_mean_return_delta"], digits=6)
                ),
                "  positive_fold_ratio delta: {}".format(
                    _fmt_signed(regime_level["positive_ratio_delta"], digits=3)
                ),
                "",
                "Stability-level",
                "  {metric} stability: {base} -> {cand} ({delta})".format(
                    metric=metric_key,
                    base=_fmt_number(stability_level["baseline_metric_stability"], digits=1),
                    cand=_fmt_number(stability_level["candidate_metric_stability"], digits=1),
                    delta=_fmt_signed(stability_level["metric_stability_delta"], digits=1),
                ),
                "  net_mean_return stability: {base} -> {cand} ({delta})".format(
                    base=_fmt_number(stability_level["baseline_net_stability"], digits=1),
                    cand=_fmt_number(stability_level["candidate_net_stability"], digits=1),
                    delta=_fmt_signed(stability_level["net_stability_delta"], digits=1),
                ),
                "  stability regression: {}".format(
                    "YES" if stability_level["flag_regression"] else "NO"
                ),
                "",
                "Cost-level",
                "  gross_mean_return delta: {}".format(
                    _fmt_signed(cost_level["delta_gross_mean_return"], digits=6)
                ),
                "  net_mean_return delta: {}".format(
                    _fmt_signed(cost_level["delta_net_mean_return"], digits=6)
                ),
                "  成本后是否仍成立: {}".format(
                    "YES" if cost_level["survives_cost"] else "NO"
                ),
            ]
        )
        if cost_level["fades_after_cost"]:
            lines.append("  note: gross edge disappears after transaction costs")

        lines.extend(
            [
                "",
                "Overfit-level",
                "  new_overfit_flags: {}".format(
                    ", ".join(overfit_level["new_flags"]) if overfit_level["new_flags"] else "none"
                ),
                "  val_test_ic_decay delta: {}".format(
                    _fmt_signed(overfit_level["delta_ic_decay"], digits=4)
                ),
                "  过拟合提示: {}".format(
                    "YES" if overfit_level["warning"] else "NO"
                ),
            ]
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_metric_values(fold_results, metric_key):
    """Extract a metric from fold results, searching prediction_metrics then portfolio_metrics."""
    values = []
    for fold in fold_results:
        pred_m = fold.get("prediction_metrics", {})
        port_m = fold.get("portfolio_metrics", {})
        if metric_key in pred_m:
            values.append(float(pred_m[metric_key]))
        elif metric_key in port_m:
            values.append(float(port_m[metric_key]))
    return values


def _default_artifacts_root():
    return Path(__file__).resolve().parents[2] / "artifacts" / "experiments" / "tx1"


def _experiment_label_from_path(path):
    name = Path(path).name
    if name.startswith("tx1_"):
        return name[4:]
    return name


def _aggregate_metric_group(fold_results, group_key):
    if not fold_results:
        return {}
    keys = set()
    for fold in fold_results:
        keys.update((fold.get(group_key) or {}).keys())
    aggregated = {}
    for key in sorted(keys):
        values = []
        for fold in fold_results:
            group = fold.get(group_key) or {}
            if key in group:
                values.append(_coerce_float(group.get(key)))
        if values:
            aggregated[key] = float(np.mean(values))
    return aggregated


def _build_fold_snapshot(fold_result, metric_key):
    prediction = fold_result.get("prediction_metrics", {}) or {}
    validation = fold_result.get("validation_metrics", {}) or {}
    portfolio = fold_result.get("portfolio_metrics", {}) or {}
    date_range = fold_result.get("date_range", {}) or {}
    metric_value = _metric_from_groups(prediction, portfolio, metric_key)
    validation_metric = _metric_from_groups(validation, {}, metric_key)
    return {
        "fold_index": int(fold_result.get("fold_index", 0) or 0),
        "window": _window_label(date_range, fold_result.get("fold_index", 0)),
        "test_start": _normalize_date_text(date_range.get("test_start")),
        "test_end": _normalize_date_text(date_range.get("test_end")),
        "metric_value": metric_value,
        "rank_ic_mean": _coerce_float(prediction.get("rank_ic_mean")),
        "top_bucket_spread_mean": _coerce_float(prediction.get("top_bucket_spread_mean")),
        "validation_metric_value": validation_metric,
        "validation_rank_ic_mean": _coerce_float(validation.get("rank_ic_mean")),
        "mean_return": _coerce_float(portfolio.get("mean_return")),
        "net_mean_return": _coerce_float(
            portfolio.get("net_mean_return", portfolio.get("mean_return"))
        ),
        "max_drawdown": _coerce_float(portfolio.get("max_drawdown")),
        "mean_turnover": _coerce_float(portfolio.get("mean_turnover")),
        "cost_drag_annual": _coerce_float(portfolio.get("cost_drag_annual")),
        "breakeven_cost_bps": _coerce_float(portfolio.get("breakeven_cost_bps")),
    }


def _compute_fold_distribution(folds, metric_key):
    metric_values = [row["metric_value"] for row in folds]
    net_values = [row["net_mean_return"] for row in folds]
    return {
        "metric_key": metric_key,
        "n_folds": len(folds),
        "positive_metric_folds": sum(1 for value in metric_values if value > 0),
        "positive_metric_ratio": _positive_ratio(metric_values),
        "positive_net_folds": sum(1 for value in net_values if value > 0),
        "positive_net_ratio": _positive_ratio(net_values),
        "metric_median": _median(metric_values),
        "net_median": _median(net_values),
    }


def _align_fold_rows(baseline_folds, candidate_folds):
    def _build_map(rows, key_name):
        return {
            row.get(key_name): row
            for row in rows
            if row.get(key_name) not in (None, "")
        }

    baseline_start_map = _build_map(baseline_folds, "test_start")
    candidate_start_map = _build_map(candidate_folds, "test_start")
    baseline_index_map = _build_map(baseline_folds, "fold_index")
    candidate_index_map = _build_map(candidate_folds, "fold_index")

    start_keys = sorted(set(baseline_start_map) & set(candidate_start_map))
    index_keys = sorted(set(baseline_index_map) & set(candidate_index_map))

    if len(start_keys) >= len(index_keys) and start_keys:
        aligned_pairs = [(baseline_start_map[key], candidate_start_map[key]) for key in start_keys]
    else:
        aligned_pairs = [(baseline_index_map[key], candidate_index_map[key]) for key in index_keys]

    aligned_rows = []
    for baseline_row, candidate_row in aligned_pairs:
        aligned_rows.append(
            {
                "fold_index": baseline_row["fold_index"],
                "window": baseline_row["window"],
                "test_start": baseline_row["test_start"],
                "test_end": baseline_row["test_end"],
                "baseline_metric": baseline_row["metric_value"],
                "candidate_metric": candidate_row["metric_value"],
                "delta_metric": candidate_row["metric_value"] - baseline_row["metric_value"],
                "baseline_mean_return": baseline_row["mean_return"],
                "candidate_mean_return": candidate_row["mean_return"],
                "delta_mean_return": candidate_row["mean_return"] - baseline_row["mean_return"],
                "baseline_net_mean_return": baseline_row["net_mean_return"],
                "candidate_net_mean_return": candidate_row["net_mean_return"],
                "delta_net_mean_return": candidate_row["net_mean_return"] - baseline_row["net_mean_return"],
                "baseline_max_drawdown": baseline_row["max_drawdown"],
                "candidate_max_drawdown": candidate_row["max_drawdown"],
                "delta_max_drawdown": candidate_row["max_drawdown"] - baseline_row["max_drawdown"],
            }
        )
    return aligned_rows


def _build_fold_level_comparison(fold_rows, return_metric="net_mean_return"):
    metric_deltas = [row["delta_metric"] for row in fold_rows]
    return_deltas = [_return_delta_from_row(row, return_metric) for row in fold_rows]
    positive_uplifts = sorted(
        [delta for delta in return_deltas if delta > 0],
        reverse=True,
    )
    top_2_share = 0.0
    if positive_uplifts and sum(positive_uplifts) > 0:
        top_2_share = float(sum(positive_uplifts[:2]) / sum(positive_uplifts))
    positive_ratio = len(positive_uplifts) / len(fold_rows) if fold_rows else 0.0
    few_window_flag = bool(
        fold_rows
        and sum(return_deltas) > 0
        and (
            positive_ratio < 0.4
            or (len(positive_uplifts) <= 2 and len(fold_rows) >= 5)
            or top_2_share >= 0.65
        )
    )

    fold_rows = [
        {
            **row,
            "delta_return_metric": _return_delta_from_row(row, return_metric),
        }
        for row in fold_rows
    ]
    sorted_best = sorted(fold_rows, key=lambda row: row["delta_return_metric"], reverse=True)
    sorted_worst = sorted(fold_rows, key=lambda row: row["delta_return_metric"])

    return {
        "return_metric": return_metric,
        "aligned_folds": len(fold_rows),
        "metric_delta_mean": _mean(metric_deltas),
        "metric_delta_total": float(sum(metric_deltas)),
        "metric_win_folds": sum(1 for delta in metric_deltas if delta > 0),
        "metric_loss_folds": sum(1 for delta in metric_deltas if delta < 0),
        "return_delta_mean": _mean(return_deltas),
        "return_delta_total": float(sum(return_deltas)),
        "return_win_folds": sum(1 for delta in return_deltas if delta > 0),
        "return_loss_folds": sum(1 for delta in return_deltas if delta < 0),
        "positive_uplift_folds": len(positive_uplifts),
        "positive_uplift_ratio": positive_ratio,
        "top_2_uplift_share": top_2_share,
        "few_window_flag": few_window_flag,
        "best_windows": sorted_best[:3],
        "worst_windows": sorted_worst[:3],
        "fold_rows": fold_rows,
    }


def _build_regime_level_comparison(baseline_summary, candidate_summary):
    base = baseline_summary["robustness"]["regime_scores"]
    cand = candidate_summary["robustness"]["regime_scores"]
    base_up = base.get("up_regime", {})
    base_down = base.get("down_regime", {})
    cand_up = cand.get("up_regime", {})
    cand_down = cand.get("down_regime", {})
    base_consistency = base.get("metric_consistency", {})
    cand_consistency = cand.get("metric_consistency", {})

    return {
        "up_mean_return_delta": _coerce_float(cand_up.get("mean_return")) - _coerce_float(base_up.get("mean_return")),
        "down_mean_return_delta": _coerce_float(cand_down.get("mean_return")) - _coerce_float(base_down.get("mean_return")),
        "up_total_return_delta": _coerce_float(cand_up.get("total_return")) - _coerce_float(base_up.get("total_return")),
        "down_total_return_delta": _coerce_float(cand_down.get("total_return")) - _coerce_float(base_down.get("total_return")),
        "positive_ratio_delta": _coerce_float(cand_consistency.get("positive_ratio")) - _coerce_float(base_consistency.get("positive_ratio")),
        "baseline_positive_ratio": _coerce_float(base_consistency.get("positive_ratio")),
        "candidate_positive_ratio": _coerce_float(cand_consistency.get("positive_ratio")),
    }


def _build_stability_level_comparison(baseline_summary, candidate_summary):
    base_stability = baseline_summary["robustness"]["stability"]
    cand_stability = candidate_summary["robustness"]["stability"]
    base_net = baseline_summary["robustness"]["net_return_stability"]
    cand_net = candidate_summary["robustness"]["net_return_stability"]
    regime_cmp = _build_regime_level_comparison(baseline_summary, candidate_summary)

    metric_delta = _coerce_float(cand_stability.get("stability_score")) - _coerce_float(
        base_stability.get("stability_score")
    )
    net_delta = _coerce_float(cand_net.get("stability_score")) - _coerce_float(
        base_net.get("stability_score")
    )
    flag_regression = bool(
        metric_delta < -5.0
        or net_delta < -5.0
        or regime_cmp["positive_ratio_delta"] < -0.15
    )
    return {
        "baseline_metric_stability": _coerce_float(base_stability.get("stability_score")),
        "candidate_metric_stability": _coerce_float(cand_stability.get("stability_score")),
        "metric_stability_delta": metric_delta,
        "baseline_net_stability": _coerce_float(base_net.get("stability_score")),
        "candidate_net_stability": _coerce_float(cand_net.get("stability_score")),
        "net_stability_delta": net_delta,
        "baseline_cv": _coerce_float(base_stability.get("cv"), default=float("inf")),
        "candidate_cv": _coerce_float(cand_stability.get("cv"), default=float("inf")),
        "cv_delta": _coerce_float(cand_stability.get("cv"), default=float("inf")) - _coerce_float(
            base_stability.get("cv"), default=float("inf")
        ),
        "flag_regression": flag_regression,
    }


def _build_cost_level_comparison(baseline_summary, candidate_summary):
    base_port = baseline_summary.get("portfolio", {})
    cand_port = candidate_summary.get("portfolio", {})
    base_gross = _coerce_float(base_port.get("mean_return"))
    cand_gross = _coerce_float(cand_port.get("mean_return"))
    base_net = _coerce_float(base_port.get("net_mean_return", base_gross))
    cand_net = _coerce_float(cand_port.get("net_mean_return", cand_gross))
    return {
        "baseline_gross_mean_return": base_gross,
        "candidate_gross_mean_return": cand_gross,
        "delta_gross_mean_return": cand_gross - base_gross,
        "baseline_net_mean_return": base_net,
        "candidate_net_mean_return": cand_net,
        "delta_net_mean_return": cand_net - base_net,
        "survives_cost": cand_net > base_net,
        "fades_after_cost": cand_gross > base_gross and cand_net <= base_net,
        "candidate_profitable_after_cost": cand_net > 0,
    }


def _build_overfit_level_comparison(baseline_summary, candidate_summary):
    base = baseline_summary["robustness"]["overfit_flags"]
    cand = candidate_summary["robustness"]["overfit_flags"]
    flag_names = [
        "flag_ic_decay",
        "flag_spread_decay",
        "flag_val_dominant",
    ]
    new_flags = [name for name in flag_names if cand.get(name) and not base.get(name)]
    warning = bool(
        new_flags
        or _coerce_float(cand.get("val_test_ic_decay")) > _coerce_float(base.get("val_test_ic_decay")) + 0.01
        or _coerce_float(cand.get("val_test_spread_decay")) > _coerce_float(base.get("val_test_spread_decay")) + 0.003
        or _coerce_float(cand.get("val_dominant_ratio")) > _coerce_float(base.get("val_dominant_ratio")) + 0.15
    )
    return {
        "baseline": base,
        "candidate": cand,
        "new_flags": new_flags,
        "delta_ic_decay": _coerce_float(cand.get("val_test_ic_decay")) - _coerce_float(base.get("val_test_ic_decay")),
        "delta_spread_decay": _coerce_float(cand.get("val_test_spread_decay")) - _coerce_float(base.get("val_test_spread_decay")),
        "delta_val_dominant_ratio": _coerce_float(cand.get("val_dominant_ratio")) - _coerce_float(base.get("val_dominant_ratio")),
        "warning": warning,
    }


def _build_merge_decision(fold_level, stability_level, cost_level, overfit_level):
    flags = {
        "few_window_uplift": fold_level["few_window_flag"],
        "overfit_warning": overfit_level["warning"],
        "cost_not_hold": not cost_level["survives_cost"],
        "stability_regression": stability_level["flag_regression"],
    }
    reasons = []
    if flags["cost_not_hold"]:
        reasons.append("cost-adjusted return does not beat baseline")
    if flags["few_window_uplift"]:
        reasons.append("uplift is concentrated in a minority of windows")
    if flags["overfit_warning"]:
        reasons.append("overfit indicators worsened")
    if flags["stability_regression"]:
        reasons.append("stability regressed versus baseline")

    if flags["cost_not_hold"]:
        status = "REJECT"
    elif reasons:
        status = "HOLD"
    else:
        status = "MERGE"
    return {"status": status, "reasons": reasons, "flags": flags}


def _render_overview_table(experiments):
    rows = [("experiment", "model", "rank_ic", "gross", "net", "stab_ic", "stab_net", "pos_folds")]
    for item in experiments:
        rows.append(
            (
                item["label"],
                item.get("model_kind") or "?",
                _fmt_number(item.get("prediction", {}).get("rank_ic_mean"), digits=4),
                _fmt_number(item.get("portfolio", {}).get("mean_return"), digits=6),
                _fmt_number(item.get("portfolio", {}).get("net_mean_return"), digits=6),
                _fmt_number(item["robustness"]["stability"].get("stability_score"), digits=1),
                _fmt_number(item["robustness"]["net_return_stability"].get("stability_score"), digits=1),
                "{}/{}".format(
                    item["fold_distribution"]["positive_net_folds"],
                    item["fold_distribution"]["n_folds"],
                ),
            )
        )
    widths = [max(len(str(row[idx])) for row in rows) for idx in range(len(rows[0]))]
    lines = []
    for row_idx, row in enumerate(rows):
        rendered = "  ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row))
        lines.append(rendered)
        if row_idx == 0:
            lines.append("  ".join("-" * width for width in widths))
    return lines


def _metric_from_groups(primary_group, fallback_group, metric_key):
    if metric_key in primary_group:
        return _coerce_float(primary_group.get(metric_key))
    if metric_key in fallback_group:
        return _coerce_float(fallback_group.get(metric_key))
    return 0.0


def _window_label(date_range, fold_index):
    start = _normalize_date_text(date_range.get("test_start"))
    end = _normalize_date_text(date_range.get("test_end"))
    if start and end:
        return "{}..{}".format(start, end)
    if start:
        return start
    return "fold_{:03d}".format(int(fold_index or 0))


def _fold_alignment_key(row):
    key = row.get("test_start") or "fold_{:03d}".format(int(row.get("fold_index", 0)))
    return (int(row.get("fold_index", 0)), key)


def _return_delta_from_row(row, return_metric):
    if return_metric == "mean_return":
        return row["delta_mean_return"]
    return row["delta_net_mean_return"]


def _normalize_date_text(value):
    if value is None:
        return ""
    text = str(value)
    if " " in text:
        return text.split(" ", 1)[0]
    return text


def _coerce_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _positive_ratio(values):
    if not values:
        return 0.0
    return sum(1 for value in values if value > 0) / float(len(values))


def _mean(values):
    if not values:
        return 0.0
    return float(np.mean(np.array(values, dtype=float)))


def _median(values):
    if not values:
        return 0.0
    return float(np.median(np.array(values, dtype=float)))


def _fmt_number(value, digits=4):
    numeric = _coerce_float(value)
    if np.isinf(numeric):
        return "inf"
    return ("{0:." + str(digits) + "f}").format(numeric)


def _fmt_signed(value, digits=4):
    numeric = _coerce_float(value)
    if np.isinf(numeric):
        return "inf"
    return ("{0:+." + str(digits) + "f}").format(numeric)


def _fmt_percent(value):
    return "{:.1%}".format(_coerce_float(value))


def _score_cv(cv):
    """Score coefficient of variation (lower CV = more stable)."""
    if cv >= 0.5:
        return 0.0
    if cv <= 0.1:
        return 100.0
    return (0.5 - cv) / 0.4 * 100.0


def _score_worst(worst_val, median_val):
    """Score worst fold relative to median."""
    if median_val == 0:
        return 50.0
    ratio = worst_val / median_val if median_val != 0 else 0.0
    # ratio close to 1.0 means worst ≈ median → good
    # ratio <= 0 means worst is opposite sign → bad
    if ratio >= 0.8:
        return 100.0
    if ratio <= 0.0:
        return 0.0
    return ratio / 0.8 * 100.0


def _max_consecutive_below(arr, threshold):
    """Count max consecutive values below threshold."""
    max_run = 0
    current_run = 0
    for v in arr:
        if v < threshold:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _score_consecutive(max_consec, n_folds):
    """Score based on max consecutive below-median folds."""
    if n_folds <= 1:
        return 100.0
    ratio = max_consec / n_folds
    if ratio >= 0.5:
        return 0.0
    if ratio <= 0.1:
        return 100.0
    return (0.5 - ratio) / 0.4 * 100.0


def _empty_stability():
    return {
        "stability_score": 0.0,
        "cv": float("inf"),
        "worst_value": 0.0,
        "max_consecutive_low": 0,
        "cv_score": 0.0,
        "worst_score": 0.0,
        "consecutive_score": 0.0,
        "n_folds": 0,
        "mean": 0.0,
        "std": 0.0,
        "metric_key": "",
    }


def _empty_overfit():
    return {
        "val_test_ic_decay": 0.0,
        "val_test_spread_decay": 0.0,
        "val_dominant_ratio": 0.0,
        "flag_ic_decay": False,
        "flag_spread_decay": False,
        "flag_val_dominant": False,
        "n_folds_compared": 0,
    }
