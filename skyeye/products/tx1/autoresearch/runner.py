"""TX1 autoresearch 的实验执行包装层。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skyeye.products.tx1.persistence import load_experiment
from skyeye.products.tx1.robustness import summarize_experiment
from skyeye.products.tx1.run_baseline_experiment import _get_liquid_universe, build_raw_df
from skyeye.products.tx1.run_feature_experiment import run_feature_experiments


def build_run_root(runs_root: str | Path, run_tag: str) -> Path:
    """拼出本次 autoresearch run 的根目录。"""
    return Path(runs_root).resolve() / str(run_tag)


def build_experiment_root(run_root: str | Path, experiment_index: int) -> Path:
    """为单次实验分配独立产物目录，避免覆盖历史结果。"""
    return Path(run_root).resolve() / "experiments" / "exp_{:04d}".format(int(experiment_index))


def build_research_raw_df(
    *,
    universe_size: int = 300,
    start_date=None,
    end_date=None,
):
    """按研究侧旧口径构造 raw_df，供 autoresearch 复用。"""
    # universe 的截止日必须和 raw_df 一致，避免把更晚日期的信息带回研究样本。
    universe = _get_liquid_universe(universe_size, data_end=end_date)
    return build_raw_df(universe, start_date=start_date, end_date=end_date)


def summarize_metrics(summary: dict[str, Any] | None) -> dict[str, float]:
    """抽取结果记账最常用的几项核心指标。"""
    summary = dict(summary or {})
    prediction = dict(summary.get("prediction") or {})
    portfolio = dict(summary.get("portfolio") or {})
    robustness = dict(summary.get("robustness") or {})
    stability = dict(robustness.get("stability") or {})
    return {
        "rank_ic_mean": float(prediction.get("rank_ic_mean", 0.0)),
        "top_bucket_spread_mean": float(prediction.get("top_bucket_spread_mean", 0.0)),
        "net_mean_return": float(portfolio.get("net_mean_return", 0.0)),
        "max_drawdown": float(portfolio.get("max_drawdown", 0.0)),
        "stability_score": float(stability.get("stability_score", 0.0)),
    }


def load_summary_from_experiment_result(result: dict[str, Any]) -> dict[str, Any]:
    """把实验结果统一裁剪成 judge 关心的摘要结构。"""
    summary = summarize_experiment(result)
    summary["experiment_path"] = result.get("output_dir")
    return summary


def load_summary_from_experiment_path(experiment_path: str | Path) -> dict[str, Any]:
    """从实验目录加载结果，并抽取 judge 可消费的摘要。"""
    result = load_experiment(str(Path(experiment_path)))
    return load_summary_from_experiment_result(result)


def run_feature_trial(
    *,
    run_root: str | Path,
    experiment_index: int,
    raw_df,
    variant_name: str = "baseline_5f",
    model_kind: str = "lgbm",
    label_transform: str = "rank",
    horizon_days: int = 20,
    max_folds: int | None = None,
) -> dict[str, Any]:
    """执行一次基于 `run_feature_experiments(...)` 的研究试验并返回标准摘要。"""
    experiment_root = build_experiment_root(run_root, experiment_index=experiment_index)
    payload = run_feature_experiments(
        raw_df=raw_df,
        output_dir=str(experiment_root),
        variant_names=[variant_name],
        model_kind=model_kind,
        label_transform=label_transform,
        horizon_days=horizon_days,
        max_folds=max_folds,
    )
    variant = _extract_variant_payload(payload, variant_name=variant_name)
    result = {
        "experiment_name": variant_name,
        "model_kind": payload.get("model_kind", model_kind),
        "config": {
            "label_transform": payload.get("label_transform", label_transform),
            "horizon_days": int(horizon_days),
            "variant_name": variant_name,
        },
        "fold_results": list(variant.get("fold_results") or []),
        "output_dir": str(experiment_root),
    }
    summary = load_summary_from_experiment_result(result)
    summary["experiment_path"] = str(experiment_root)
    return summary


def run_baseline_trial(
    *,
    run_root: str | Path,
    raw_df,
    variant_name: str = "baseline_5f",
    model_kind: str = "lgbm",
    label_transform: str = "rank",
    horizon_days: int = 20,
    max_folds: int | None = None,
) -> dict[str, Any]:
    """执行 baseline 评估，统一 baseline 的实验索引约定。"""
    return run_feature_trial(
        run_root=run_root,
        experiment_index=0,
        raw_df=raw_df,
        variant_name=variant_name,
        model_kind=model_kind,
        label_transform=label_transform,
        horizon_days=horizon_days,
        max_folds=max_folds,
    )


def run_candidate_trial(
    *,
    run_root: str | Path,
    experiment_index: int,
    stage: str,
    raw_df,
    variant_name: str = "baseline_5f",
    model_kind: str = "lgbm",
    label_transform: str = "rank",
    horizon_days: int = 20,
    max_folds: int | None = None,
) -> dict[str, Any]:
    """执行候选评估，并把阶段标签显式落到摘要里。"""
    summary = run_feature_trial(
        run_root=run_root,
        experiment_index=experiment_index,
        raw_df=raw_df,
        variant_name=variant_name,
        model_kind=model_kind,
        label_transform=label_transform,
        horizon_days=horizon_days,
        max_folds=max_folds,
    )
    summary["stage"] = str(stage)
    return summary


def _extract_variant_payload(payload: dict[str, Any], *, variant_name: str) -> dict[str, Any]:
    """从 feature experiment payload 中取出指定 variant 的结果。"""
    for variant in payload.get("variants") or []:
        if variant.get("name") == variant_name:
            return dict(variant)
    raise KeyError("missing variant payload: {}".format(variant_name))
