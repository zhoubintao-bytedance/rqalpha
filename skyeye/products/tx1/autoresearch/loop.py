"""TX1 autoresearch 主循环入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skyeye.products.tx1.autoresearch.git_ops import (
    assert_only_allowed_paths_changed,
    checkout_commit,
    collect_workspace_safety_checks,
    create_experiment_commit,
    ensure_read_only_paths_untouched,
    get_current_branch,
    get_current_commit,
    list_changed_paths,
    rollback_candidate_commit,
)
from skyeye.products.tx1.autoresearch.patch_source import detect_workspace_patch
from skyeye.products.tx1.autoresearch.runner import (
    build_research_raw_df,
    build_run_root,
    run_baseline_trial,
    run_candidate_trial,
    summarize_metrics,
)
from skyeye.products.tx1.autoresearch.judge import judge_candidate
from skyeye.products.tx1.autoresearch.state import AutoresearchStateStore


READ_ONLY_ROOTS = [
    "skyeye/products/tx1/live_advisor/",
    "skyeye/products/tx1/strategies/rolling_score/runtime.py",
]

DEFAULT_ALLOWED_WRITE_ROOTS = [
    "skyeye/products/tx1/autoresearch/",
    "skyeye/products/tx1/dataset_builder.py",
    "skyeye/products/tx1/label_builder.py",
    "skyeye/products/tx1/config.py",
    "skyeye/products/tx1/experiment_runner.py",
    "skyeye/products/tx1/main.py",
    "skyeye/products/tx1/preprocessor.py",
    "skyeye/products/tx1/baseline_models.py",
    "skyeye/products/tx1/run_feature_experiment.py",
    "skyeye/products/tx1/run_baseline_experiment.py",
    "skyeye/products/tx1/robustness.py",
    "skyeye/products/tx1/persistence.py",
    "skyeye/products/tx1/evaluator.py",
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
    allowed_write_roots: list[str] | None = None,
    read_only_roots: list[str] | None = None,
) -> dict[str, Any]:
    """初始化或恢复 run 状态，并在有候选 patch 时执行一轮评估。"""
    repo_root = Path(workdir).resolve() if workdir is not None else Path.cwd().resolve()
    run_root = build_run_root(runs_root, run_tag)
    store = AutoresearchStateStore(run_root)
    budget = {
        "max_experiments": int(max_experiments),
        "smoke_max_folds": int(smoke_max_folds),
        "full_max_folds": None if full_max_folds is None else int(full_max_folds),
    }
    raw_df_spec = {
        "universe_size": int(universe_size),
        "start_date": start_date,
        "end_date": end_date,
    }
    resolved_allowed_write_roots = list(allowed_write_roots or DEFAULT_ALLOWED_WRITE_ROOTS)
    resolved_read_only_roots = list(read_only_roots or READ_ONLY_ROOTS)
    workspace_checks = collect_workspace_safety_checks(repo_root)
    if workspace_checks.get("reason_code") is not None:
        if store.state_path.exists():
            state = store.load()
        else:
            state = store.initialize(
                run_tag=run_tag,
                baseline_commit="",
                branch_name="",
                baseline_summary={},
                budget=budget,
                raw_df_spec=raw_df_spec,
                allowed_write_roots=resolved_allowed_write_roots,
                read_only_roots=resolved_read_only_roots,
            )
        reason_code = str(workspace_checks["reason_code"])
        state = store.update_after_decision(
            experiment_index=int(state.get("next_experiment_index", 0)),
            decision_status="invalid",
            parent_commit=str(state.get("current_commit") or ""),
            commit=str(state.get("current_commit") or ""),
            candidate_summary=None,
            stage_reached="precheck",
            reason_code=reason_code,
            error_message=None,
        )
        return {
            "run_tag": run_tag,
            "run_root": str(run_root),
            "workdir": str(repo_root),
            "state": state,
            "state_store": store,
            "baseline_summary": None,
            "candidate_result": None,
            "max_experiments": int(max_experiments),
            "smoke_max_folds": int(smoke_max_folds),
            "full_max_folds": None if full_max_folds is None else int(full_max_folds),
            "status": "invalid",
            "reason_code": reason_code,
        }

    state = _bootstrap_or_resume_state(
        state_store=store,
        run_tag=run_tag,
        repo_root=repo_root,
        budget=budget,
        raw_df_spec=raw_df_spec,
        allowed_write_roots=resolved_allowed_write_roots,
        read_only_roots=resolved_read_only_roots,
    )
    baseline_commit = str(state.get("baseline_commit") or "")
    baseline_summary = dict(state.get("baseline_summary") or {})
    if not baseline_summary and build_raw_df_for_run:
        raw_df = _ensure_raw_df(
            current_raw_df=raw_df,
            build_raw_df_for_run=True,
            raw_df_spec=state.get("raw_df_spec") or raw_df_spec,
        )
    if not baseline_summary and raw_df is not None:
        baseline_summary = run_baseline_trial(
            run_root=run_root,
            raw_df=raw_df,
            variant_name=variant_name,
            model_kind=model_kind,
            label_transform=label_transform,
            horizon_days=horizon_days,
            max_folds=full_max_folds,
        )
        store.append_result(
            experiment_index=0,
            parent_commit="",
            commit=baseline_commit,
            status="keep",
            stage_reached="baseline",
            metrics={
                "net_mean_return": baseline_summary.get("portfolio", {}).get("net_mean_return"),
                "max_drawdown": baseline_summary.get("portfolio", {}).get("max_drawdown"),
                "stability_score": baseline_summary.get("robustness", {}).get("stability", {}).get("stability_score"),
            },
            reason_code="baseline",
            experiment_path=baseline_summary.get("experiment_path", ""),
        )
        state = store.load()
        state["baseline_summary"] = dict(baseline_summary)
        state["best_summary"] = dict(baseline_summary)
        state["next_experiment_index"] = max(int(state.get("next_experiment_index", 0)), 1)
        store.save(state)

    candidate_result = None
    candidate_enabled = bool(evaluate_current or int(max_experiments) > 0)
    if candidate_enabled:
        patch_candidate = detect_workspace_patch(repo_root)
        if patch_candidate is None:
            state = store.load()
            state["last_status"] = "waiting_for_patch"
            state["last_reason_code"] = "waiting_for_patch"
            state["last_error"] = None
            store.save(state)
            return {
                "run_tag": state["run_tag"],
                "run_root": str(run_root),
                "workdir": str(repo_root),
                "state": store.load(),
                "state_store": store,
                "baseline_summary": baseline_summary or None,
                "candidate_result": None,
                "max_experiments": int(max_experiments),
                "smoke_max_folds": int(smoke_max_folds),
                "full_max_folds": None if full_max_folds is None else int(full_max_folds),
                "status": "waiting_for_patch",
            }
        if raw_df is None:
            raw_df = _ensure_raw_df(
                current_raw_df=raw_df,
                build_raw_df_for_run=True,
                raw_df_spec=state.get("raw_df_spec") or raw_df_spec,
            )
        parent_commit = str(store.load().get("current_commit") or baseline_commit)
        experiment_index = max(int(store.load().get("next_experiment_index", 1)), 1)
        try:
            candidate_result = evaluate_current_candidate(
                workdir=repo_root,
                run_root=run_root,
                start_commit=parent_commit,
                experiment_index=experiment_index,
                raw_df=raw_df,
                baseline_summary=baseline_summary or None,
                best_summary=store.load().get("best_summary") or baseline_summary,
                variant_name=variant_name,
                model_kind=model_kind,
                label_transform=label_transform,
                horizon_days=horizon_days,
                smoke_max_folds=smoke_max_folds,
                full_max_folds=full_max_folds,
                allowed_write_roots=resolved_allowed_write_roots,
                read_only_roots=resolved_read_only_roots,
            )
        except Exception as exc:
            candidate_result = {
                "status": "crash",
                "reason_code": "candidate_crash",
                "commit": parent_commit,
                "error_message": str(exc),
                "smoke_summary": {},
                "stage_reached": "crash",
            }

        _record_candidate_result(
            state_store=store,
            baseline_commit=baseline_commit,
            parent_commit=parent_commit,
            experiment_index=experiment_index,
            candidate_result=candidate_result,
        )

    return {
        "run_tag": store.load()["run_tag"],
        "run_root": str(run_root),
        "workdir": str(repo_root),
        "state": store.load(),
        "state_store": store,
        "baseline_summary": baseline_summary or None,
        "candidate_result": candidate_result,
        "max_experiments": int(max_experiments),
        "smoke_max_folds": int(smoke_max_folds),
        "full_max_folds": None if full_max_folds is None else int(full_max_folds),
        "status": str((candidate_result or {}).get("status") or "initialized"),
    }


def evaluate_current_candidate(
    *,
    workdir: str | Path,
    run_root: str | Path,
    start_commit: str,
    experiment_index: int,
    raw_df,
    baseline_summary: dict[str, Any] | None = None,
    best_summary: dict[str, Any] | None = None,
    variant_name: str = "baseline_5f",
    model_kind: str = "lgbm",
    label_transform: str = "rank",
    horizon_days: int = 20,
    smoke_max_folds: int = 1,
    full_max_folds: int | None = None,
    allowed_write_roots: list[str] | None = None,
    read_only_roots: list[str] | None = None,
) -> dict[str, Any]:
    """评估当前工作区里的源码改动，并给出 keep/discard/rollback 决策。"""
    repo_root = Path(workdir).resolve()
    resolved_read_only_roots = list(read_only_roots or READ_ONLY_ROOTS)
    resolved_allowed_write_roots = list(allowed_write_roots or DEFAULT_ALLOWED_WRITE_ROOTS)
    changed_paths = list_changed_paths(repo_root, include_untracked=True)
    if not changed_paths:
        return {
            "status": "invalid",
            "reason_code": "missing_patch",
            "changed_paths": [],
            "stage_reached": "precheck",
        }
    read_only_hits = ensure_read_only_paths_untouched(changed_paths, resolved_read_only_roots)
    if read_only_hits:
        rollback_candidate_commit(repo_root, start_commit)
        return {
            "status": "invalid",
            "reason_code": "read_only_path_touched",
            "changed_paths": list(read_only_hits),
            "stage_reached": "precheck",
        }
    outside_allowed_hits = assert_only_allowed_paths_changed(
        changed_paths,
        resolved_allowed_write_roots,
        resolved_read_only_roots,
    )
    if outside_allowed_hits:
        rollback_candidate_commit(repo_root, start_commit)
        return {
            "status": "invalid",
            "reason_code": "path_outside_allowed_write_roots",
            "changed_paths": list(outside_allowed_hits),
            "stage_reached": "precheck",
        }

    commit = create_experiment_commit(
        repo_root,
        "exp: autoresearch candidate {}".format(experiment_index),
        allowed_paths=changed_paths,
    )
    smoke_summary = run_candidate_trial(
        run_root=run_root,
        experiment_index=experiment_index,
        stage="smoke",
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
        best_summary=best_summary,
        stage="smoke",
    )
    if smoke_decision.get("status") != "keep":
        rollback_candidate_commit(repo_root, start_commit)
        return {
            "status": smoke_decision.get("status", "discard"),
            "reason_code": smoke_decision.get("reason_code"),
            "commit": commit,
            "smoke_summary": smoke_summary,
            "smoke_decision": smoke_decision,
            "stage_reached": "smoke",
        }

    full_summary = run_candidate_trial(
        run_root=run_root,
        experiment_index=experiment_index,
        stage="full",
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
        best_summary=best_summary,
        stage="full",
    )
    if full_decision.get("status") == "discard":
        rollback_candidate_commit(repo_root, start_commit)

    return {
        "status": full_decision.get("status", "discard"),
        "reason_code": full_decision.get("reason_code"),
        "commit": commit,
        "smoke_summary": smoke_summary,
        "smoke_decision": smoke_decision,
        "full_summary": full_summary,
        "full_decision": full_decision,
        "stage_reached": "full",
    }


def _record_candidate_result(
    *,
    state_store: AutoresearchStateStore,
    baseline_commit: str,
    parent_commit: str,
    experiment_index: int,
    candidate_result: dict[str, Any],
) -> dict[str, Any]:
    """把单轮候选结果同步写入 results.tsv 和 state.json。"""
    candidate_summary = _select_candidate_summary(candidate_result)
    commit = str(candidate_result.get("commit") or baseline_commit)
    status = str(candidate_result.get("status") or "discard")
    reason_code = str(candidate_result.get("reason_code") or status)
    stage_reached = str(candidate_result.get("stage_reached") or _infer_stage(candidate_result))

    state_store.append_result(
        experiment_index=experiment_index,
        parent_commit=parent_commit,
        commit=commit,
        status=status,
        stage_reached=stage_reached,
        metrics=summarize_metrics(candidate_summary),
        reason_code=reason_code,
        experiment_path=candidate_summary.get("experiment_path", ""),
    )
    return state_store.update_after_decision(
        experiment_index=experiment_index,
        decision_status=status,
        parent_commit=parent_commit,
        commit=commit,
        candidate_summary=candidate_summary,
        stage_reached=stage_reached,
        reason_code=reason_code,
        error_message=candidate_result.get("error_message"),
    )


def _select_candidate_summary(candidate_result: dict[str, Any] | None) -> dict[str, Any]:
    """优先选 full 结果，没有 full 时退回 smoke 结果，用于统一记账。"""
    payload = dict(candidate_result or {})
    if payload.get("full_summary"):
        return dict(payload["full_summary"])
    if payload.get("smoke_summary"):
        return dict(payload["smoke_summary"])
    return {}


def _infer_stage(candidate_result: dict[str, Any] | None) -> str:
    """从候选结果中推断本轮实际走到的阶段。"""
    payload = dict(candidate_result or {})
    if payload.get("full_summary"):
        return "full"
    if payload.get("smoke_summary"):
        return "smoke"
    return "precheck"


def _bootstrap_or_resume_state(
    *,
    state_store: AutoresearchStateStore,
    run_tag: str,
    repo_root: Path,
    budget: dict[str, Any],
    raw_df_spec: dict[str, Any],
    allowed_write_roots: list[str],
    read_only_roots: list[str],
    allow_resume: bool = True,
) -> dict[str, Any]:
    """初始化新 run，或把现有 run 恢复到 current commit。"""
    if allow_resume and state_store.state_path.exists():
        state = state_store.load()
        current_commit = str(state.get("current_commit") or state.get("baseline_commit") or "")
        if current_commit:
            checkout_commit(repo_root, current_commit)
        state["budget"] = dict(state.get("budget") or budget)
        state["raw_df_spec"] = dict(state.get("raw_df_spec") or raw_df_spec)
        state["allowed_write_roots"] = list(state.get("allowed_write_roots") or allowed_write_roots)
        state["read_only_roots"] = list(state.get("read_only_roots") or read_only_roots)
        state_store.save(state)
        return state
    return state_store.initialize(
        run_tag=run_tag,
        baseline_commit=get_current_commit(repo_root),
        branch_name=get_current_branch(repo_root),
        baseline_summary={},
        budget=budget,
        raw_df_spec=raw_df_spec,
        allowed_write_roots=allowed_write_roots,
        read_only_roots=read_only_roots,
    )


def _ensure_raw_df(
    *,
    current_raw_df,
    build_raw_df_for_run: bool,
    raw_df_spec: dict[str, Any] | None,
):
    """按 run 固定的数据口径惰性构造 raw_df。"""
    if current_raw_df is not None:
        return current_raw_df
    if not build_raw_df_for_run:
        return None
    spec = dict(raw_df_spec or {})
    return build_research_raw_df(
        universe_size=int(spec.get("universe_size", 300)),
        start_date=spec.get("start_date"),
        end_date=spec.get("end_date"),
    )
