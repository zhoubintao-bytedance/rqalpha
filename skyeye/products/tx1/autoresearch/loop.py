"""TX1 autoresearch 主循环入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skyeye.products.tx1.autoresearch.git_ops import (
    create_experiment_commit,
    ensure_read_only_paths_untouched,
    get_current_branch,
    get_current_commit,
    list_changed_paths,
    rollback_to_commit,
)
from skyeye.products.tx1.autoresearch.runner import (
    build_research_raw_df,
    build_run_root,
    run_feature_trial,
    summarize_metrics,
)
from skyeye.products.tx1.autoresearch.judge import judge_candidate
from skyeye.products.tx1.autoresearch.state import AutoresearchStateStore


READ_ONLY_ROOTS = [
    "skyeye/products/tx1/live_advisor/",
    "skyeye/products/tx1/strategies/rolling_score/runtime.py",
]


def run_loop(
    *,
    run_tag: str,
    runs_root: str | Path,
    workdir: str | Path | None = None,
    raw_df=None,
    build_raw_df_for_run: bool = False,
    evaluate_current: bool = False,
    universe_size: int = 300,
    start_date=None,
    end_date=None,
    variant_name: str = "baseline_5f",
    model_kind: str = "lgbm",
    label_transform: str = "rank",
    horizon_days: int = 20,
    max_experiments: int = 0,
    smoke_max_folds: int = 1,
    full_max_folds: int | None = None,
) -> dict[str, Any]:
    """初始化 run 状态，并返回最小可观测运行摘要。"""
    repo_root = Path(workdir).resolve() if workdir is not None else Path.cwd().resolve()
    run_root = build_run_root(runs_root, run_tag)
    store = AutoresearchStateStore(run_root)
    baseline_commit = get_current_commit(repo_root)
    branch_name = get_current_branch(repo_root)
    state = store.initialize(
        run_tag=run_tag,
        baseline_commit=baseline_commit,
        branch_name=branch_name,
        baseline_summary={},
    )
    if raw_df is None and (build_raw_df_for_run or evaluate_current):
        raw_df = build_research_raw_df(
            universe_size=universe_size,
            start_date=start_date,
            end_date=end_date,
        )

    baseline_summary = None
    if raw_df is not None:
        baseline_summary = run_feature_trial(
            run_root=run_root,
            experiment_index=0,
            raw_df=raw_df,
            variant_name=variant_name,
            model_kind=model_kind,
            label_transform=label_transform,
            horizon_days=horizon_days,
            max_folds=full_max_folds,
        )
        store.append_result(
            commit=baseline_commit,
            status="keep",
            metrics={
                "net_mean_return": baseline_summary.get("portfolio", {}).get("net_mean_return"),
                "max_drawdown": baseline_summary.get("portfolio", {}).get("max_drawdown"),
                "stability_score": baseline_summary.get("robustness", {}).get("stability", {}).get("stability_score"),
            },
            description="baseline",
            experiment_path=baseline_summary.get("experiment_path", ""),
        )
        state["baseline_summary"] = dict(baseline_summary)
        state["best_summary"] = dict(baseline_summary)
        store.save(state)

    candidate_result = None
    if evaluate_current:
        candidate_result = evaluate_current_candidate(
            workdir=repo_root,
            run_root=run_root,
            start_commit=baseline_commit,
            experiment_index=1,
            raw_df=raw_df,
            baseline_summary=baseline_summary,
            variant_name=variant_name,
            model_kind=model_kind,
            label_transform=label_transform,
            horizon_days=horizon_days,
            smoke_max_folds=smoke_max_folds,
            full_max_folds=full_max_folds,
        )
        _record_candidate_result(
            state_store=store,
            baseline_commit=baseline_commit,
            candidate_result=candidate_result,
        )

    return {
        "run_tag": state["run_tag"],
        "run_root": str(run_root),
        "workdir": str(repo_root),
        "state": store.load(),
        "state_store": store,
        "baseline_summary": baseline_summary,
        "candidate_result": candidate_result,
        "max_experiments": int(max_experiments),
        "smoke_max_folds": int(smoke_max_folds),
        "full_max_folds": None if full_max_folds is None else int(full_max_folds),
        "status": "initialized",
    }


def evaluate_current_candidate(
    *,
    workdir: str | Path,
    run_root: str | Path,
    start_commit: str,
    experiment_index: int,
    raw_df,
    baseline_summary: dict[str, Any] | None = None,
    variant_name: str = "baseline_5f",
    model_kind: str = "lgbm",
    label_transform: str = "rank",
    horizon_days: int = 20,
    smoke_max_folds: int = 1,
    full_max_folds: int | None = None,
) -> dict[str, Any]:
    """评估当前工作区里的源码改动，并给出 keep/discard/rollback 决策。"""
    repo_root = Path(workdir).resolve()
    changed_paths = list_changed_paths(repo_root)
    read_only_hits = ensure_read_only_paths_untouched(changed_paths, READ_ONLY_ROOTS)
    if read_only_hits:
        rollback_to_commit(repo_root, start_commit)
        return {
            "status": "invalid",
            "reason_code": "read_only_path_touched",
            "changed_paths": list(read_only_hits),
        }

    commit = create_experiment_commit(repo_root, "exp: autoresearch candidate {}".format(experiment_index))
    smoke_summary = run_feature_trial(
        run_root=run_root,
        experiment_index=experiment_index,
        raw_df=raw_df,
        variant_name=variant_name,
        model_kind=model_kind,
        label_transform=label_transform,
        horizon_days=horizon_days,
        max_folds=smoke_max_folds,
    )
    smoke_decision = judge_candidate(
        smoke_summary,
        baseline_summary=baseline_summary,
        stage="smoke",
    )
    if smoke_decision.get("status") != "keep":
        rollback_to_commit(repo_root, start_commit)
        return {
            "status": smoke_decision.get("status", "discard"),
            "reason_code": smoke_decision.get("reason_code"),
            "commit": commit,
            "smoke_summary": smoke_summary,
            "smoke_decision": smoke_decision,
        }

    full_summary = run_feature_trial(
        run_root=run_root,
        experiment_index=experiment_index,
        raw_df=raw_df,
        variant_name=variant_name,
        model_kind=model_kind,
        label_transform=label_transform,
        horizon_days=horizon_days,
        max_folds=full_max_folds,
    )
    full_decision = judge_candidate(
        full_summary,
        baseline_summary=baseline_summary,
        stage="full",
    )
    if full_decision.get("status") == "discard":
        rollback_to_commit(repo_root, start_commit)

    return {
        "status": full_decision.get("status", "discard"),
        "reason_code": full_decision.get("reason_code"),
        "commit": commit,
        "smoke_summary": smoke_summary,
        "smoke_decision": smoke_decision,
        "full_summary": full_summary,
        "full_decision": full_decision,
    }


def _record_candidate_result(
    *,
    state_store: AutoresearchStateStore,
    baseline_commit: str,
    candidate_result: dict[str, Any],
) -> dict[str, Any]:
    """把单轮候选结果同步写入 results.tsv 和 state.json。"""
    candidate_summary = _select_candidate_summary(candidate_result)
    commit = str(candidate_result.get("commit") or baseline_commit)
    status = str(candidate_result.get("status") or "discard")
    reason_code = str(candidate_result.get("reason_code") or status)

    state_store.append_result(
        commit=commit,
        status=status,
        metrics=summarize_metrics(candidate_summary),
        description=reason_code,
        experiment_path=candidate_summary.get("experiment_path", ""),
    )
    return state_store.update_after_decision(
        decision_status=status,
        commit=commit,
        candidate_summary=candidate_summary,
    )


def _select_candidate_summary(candidate_result: dict[str, Any] | None) -> dict[str, Any]:
    """优先选 full 结果，没有 full 时退回 smoke 结果，用于统一记账。"""
    payload = dict(candidate_result or {})
    if payload.get("full_summary"):
        return dict(payload["full_summary"])
    if payload.get("smoke_summary"):
        return dict(payload["smoke_summary"])
    return {}
