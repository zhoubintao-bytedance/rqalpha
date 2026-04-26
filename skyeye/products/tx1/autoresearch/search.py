"""TX1 autoresearch catalog 搜索编排。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skyeye.products.tx1.autoresearch.catalog import (
    DEFAULT_CATALOG_NAME,
    PORTFOLIO_COARSE_CATALOG_NAME,
    PORTFOLIO_REFINED_CATALOG_NAME,
    build_baseline_candidate,
    build_candidate_catalog,
    build_candidate_config,
)
from skyeye.products.tx1.autoresearch.judge import judge_candidate
from skyeye.products.tx1.autoresearch.portfolio_scorer import (
    batch_run_rolling_score,
    build_final_recommendation,
    build_final_recommendation_markdown,
    save_phase2_results,
)
from skyeye.products.tx1.autoresearch.runner import (
    build_research_raw_df,
    build_run_root,
    run_config_trial,
    summarize_metrics,
)


LEADERBOARD_HEADER = (
    "candidate_id\tstatus\tstage_reached\tnet_mean_return\tmax_drawdown\t"
    "stability_score\treason_code\texperiment_path\tdescription"
)


def run_catalog_search(
    *,
    run_tag: str,
    runs_root: str | Path,
    raw_df=None,
    universe_size: int = 300,
    start_date=None,
    end_date=None,
    catalog_name: str = DEFAULT_CATALOG_NAME,
    model_kind: str = "lgbm",
    label_transform: str = "rank",
    horizon_days: int = 20,
    max_experiments: int = 0,
    smoke_max_folds: int = 1,
    full_max_folds: int | None = None,
) -> dict[str, Any]:
    """运行一轮无需 git patch 的 catalog 搜索。"""
    run_root = build_run_root(runs_root, run_tag)
    run_root.mkdir(parents=True, exist_ok=True)

    if raw_df is None:
        raw_df = build_research_raw_df(
            universe_size=universe_size,
            start_date=start_date,
            end_date=end_date,
        )

    baseline_candidate = build_baseline_candidate()
    baseline_summary = run_config_trial(
        run_root=run_root,
        experiment_index=0,
        raw_df=raw_df,
        config=build_candidate_config(
            baseline_candidate,
            model_kind=model_kind,
            label_transform=label_transform,
            horizon_days=horizon_days,
        ),
        stage="baseline",
        max_folds=full_max_folds,
    )

    candidates = build_candidate_catalog(catalog_name)
    if int(max_experiments) > 0:
        candidates = candidates[: int(max_experiments)]

    leaderboard = []
    best_summary = dict(baseline_summary)
    champion_entry = None
    for experiment_index, candidate in enumerate(candidates, start=1):
        smoke_summary = run_config_trial(
            run_root=run_root,
            experiment_index=experiment_index,
            raw_df=raw_df,
            config=build_candidate_config(
                candidate,
                model_kind=model_kind,
                label_transform=label_transform,
                horizon_days=horizon_days,
            ),
            stage="smoke",
            max_folds=smoke_max_folds,
        )
        smoke_decision = judge_candidate(
            smoke_summary,
            baseline_summary=baseline_summary,
            best_summary=best_summary,
            stage="smoke",
        )
        if smoke_decision.get("status") != "keep":
            leaderboard.append(
                _build_leaderboard_entry(
                    candidate=candidate,
                    summary=smoke_summary,
                    decision=smoke_decision,
                    stage_reached="smoke",
                )
            )
            continue

        full_summary = run_config_trial(
            run_root=run_root,
            experiment_index=experiment_index,
            raw_df=raw_df,
            config=build_candidate_config(
                candidate,
                model_kind=model_kind,
                label_transform=label_transform,
                horizon_days=horizon_days,
            ),
            stage="full",
            max_folds=full_max_folds,
        )
        full_decision = judge_candidate(
            full_summary,
            baseline_summary=baseline_summary,
            best_summary=best_summary,
            stage="full",
        )
        entry = _build_leaderboard_entry(
            candidate=candidate,
            summary=full_summary,
            decision=full_decision,
            stage_reached="full",
        )
        leaderboard.append(entry)
        if full_decision.get("status") == "champion":
            best_summary = dict(full_summary)
            champion_entry = dict(entry)

    sorted_leaderboard = _sort_leaderboard(leaderboard)

    # 滚动打分器验证：对通过评估的候选进行最终排序
    rolling_recommendation = None
    rolling_results = None
    live_candidates = [e for e in sorted_leaderboard if e.get("status") not in ("crash", "invalid")]
    if len(live_candidates) >= 2:
        top_n = min(5, len(live_candidates))
        rolling_recommendation, rolling_results = _run_rolling_score_phase(
            run_root=run_root,
            leaderboard=sorted_leaderboard,
            top_n=top_n,
        )

        print("\n滚动打分器最终推荐:")
        print("  最佳候选: {}".format(rolling_recommendation.get("best_candidate_id")))
        print("  综合得分: {:.1f}".format(rolling_recommendation.get("best_composite_score") or 0))
        print("  稳定得分: {:.1f}".format(rolling_recommendation.get("best_stability_score") or 0))

    result = {
        "run_tag": str(run_tag),
        "run_root": str(run_root),
        "catalog_name": str(catalog_name),
        "baseline": {
            "candidate_id": baseline_candidate["id"],
            "summary": baseline_summary,
            "metrics": summarize_metrics(baseline_summary),
        },
        "leaderboard": sorted_leaderboard,
        "champion": champion_entry,
        "rolling_recommendation": rolling_recommendation,
        "rolling_results": rolling_results,
        "status": "completed",
    }
    _write_catalog_outputs(run_root=run_root, result=result)
    return result


def _build_leaderboard_entry(
    *,
    candidate: dict[str, Any],
    summary: dict[str, Any],
    decision: dict[str, Any],
    stage_reached: str,
) -> dict[str, Any]:
    """把候选摘要和 judge 结果整理成稳定的 leaderboard 行。"""
    return {
        "candidate_id": candidate["id"],
        "description": candidate["description"],
        "status": str(decision.get("status") or "discard"),
        "reason_code": str(decision.get("reason_code") or "unknown"),
        "stage_reached": str(stage_reached),
        "metrics": summarize_metrics(summary),
        "score_delta": dict(decision.get("score_delta") or {}),
        "best_score_delta": dict(decision.get("best_score_delta") or {}),
        "failed_guards": list(decision.get("failed_guards") or []),
        "experiment_path": str(summary.get("experiment_path") or ""),
    }


def _sort_leaderboard(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 champion/keep/discard 按收益和回撤优先级排序，便于晨间验收。"""
    status_rank = {
        "champion": 0,
        "keep": 1,
        "discard": 2,
        "crash": 3,
        "invalid": 4,
    }
    return sorted(
        entries,
        key=lambda item: (
            status_rank.get(item["status"], 99),
            -float(item["metrics"].get("net_mean_return", 0.0)),
            float(item["metrics"].get("max_drawdown", 1.0)),
            -float(item["metrics"].get("stability_score", 0.0)),
        ),
    )


def _write_catalog_outputs(run_root: str | Path, result: dict[str, Any]) -> None:
    """把 catalog 搜索结果同时写成 JSON 和可 grep 的 TSV。"""
    run_root = Path(run_root)
    json_path = run_root / "catalog_results.json"
    tsv_path = run_root / "catalog_leaderboard.tsv"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    rows = [LEADERBOARD_HEADER]
    for entry in result.get("leaderboard") or []:
        metrics = entry.get("metrics") or {}
        rows.append(
            "\t".join(
                [
                    str(entry["candidate_id"]),
                    str(entry["status"]),
                    str(entry["stage_reached"]),
                    "{:.6f}".format(float(metrics.get("net_mean_return", 0.0))),
                    "{:.6f}".format(float(metrics.get("max_drawdown", 0.0))),
                    "{:.6f}".format(float(metrics.get("stability_score", 0.0))),
                    str(entry["reason_code"]),
                    str(entry["experiment_path"]),
                    str(entry["description"]),
                ]
            )
        )
    tsv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


# ============================================================
# 三阶段Portfolio参数搜索
# ============================================================

def run_three_phase_portfolio_search(
    *,
    run_tag: str,
    runs_root: str | Path,
    raw_df=None,
    universe_size: int = 300,
    start_date=None,
    end_date=None,
    model_kind: str = "lgbm",
    label_transform: str = "rank",
    horizon_days: int = 20,
    top_n_for_refined: int = 2,
    top_n_for_final: int = 5,
    smoke_max_folds: int = 1,
    full_max_folds: int | None = None,
) -> dict[str, Any]:
    """三阶段Portfolio参数搜索：粗粒度探索 -> 中粒度筛选 -> 滚动打分验证。

    阶段0: 粗粒度探索 (~8分钟)
        - buy_top_k: [5, 15, 25]，步长10
        - hold_top_k: [15, 30, 45]，步长15
        - 约8个有效组合

    阶段1: 中粒度筛选 (~20分钟)
        - 基于阶段0 Top2候选的邻域
        - buy_top_k步长2，hold_top_k步长5
        - 约20个有效组合

    阶段2: 滚动打分验证 (~50分钟)
        - 对Top5候选执行13指标滚动评分
        - 生成最终推荐

    Args:
        run_tag: 本次运行的唯一标签
        runs_root: 运行目录根路径
        raw_df: 可选的预构建raw_df
        universe_size: liquid universe大小
        start_date: 样本起始日
        end_date: 样本截止日
        model_kind: 模型类型
        label_transform: 标签变换方式
        horizon_days: 标签horizon天数
        top_n_for_refined: 阶段0筛选后进入阶段1的候选数
        top_n_for_final: 阶段1筛选后进入阶段2的候选数
        smoke_max_folds: smoke阶段最大fold数
        full_max_folds: full阶段最大fold数

    Returns:
        三阶段搜索结果，包含各阶段输出和最终推荐
    """
    run_root = build_run_root(runs_root, run_tag)
    run_root.mkdir(parents=True, exist_ok=True)

    if raw_df is None:
        raw_df = build_research_raw_df(
            universe_size=universe_size,
            start_date=start_date,
            end_date=end_date,
        )

    # ========== 阶段0：粗粒度探索 ==========
    print("=" * 60)
    print("阶段0：粗粒度探索（约8分钟）")
    print("=" * 60)

    phase0_result = _run_single_phase_search(
        run_root=run_root,
        raw_df=raw_df,
        phase_name="phase0_coarse",
        catalog_name=PORTFOLIO_COARSE_CATALOG_NAME,
        model_kind=model_kind,
        label_transform=label_transform,
        horizon_days=horizon_days,
        smoke_max_folds=smoke_max_folds,
        full_max_folds=full_max_folds,
    )

    # 提取Top N候选用于阶段1
    top_candidates_for_refined = _extract_top_candidates(
        phase0_result.get("leaderboard", []),
        n=top_n_for_refined,
    )

    print("\n阶段0 Top {} 候选:".format(len(top_candidates_for_refined)))
    for i, c in enumerate(top_candidates_for_refined, 1):
        print("  {}. {} (net_return={:.4f})".format(
            i,
            c.get("candidate_id"),
            float(c.get("metrics", {}).get("net_mean_return", 0)),
        ))

    # ========== 阶段1：中粒度筛选 ==========
    print("\n" + "=" * 60)
    print("阶段1：中粒度筛选（约20分钟）")
    print("=" * 60)

    phase1_result = _run_single_phase_search(
        run_root=run_root,
        raw_df=raw_df,
        phase_name="phase1_refined",
        catalog_name=PORTFOLIO_REFINED_CATALOG_NAME,
        model_kind=model_kind,
        label_transform=label_transform,
        horizon_days=horizon_days,
        smoke_max_folds=smoke_max_folds,
        full_max_folds=full_max_folds,
        top_candidates=top_candidates_for_refined,
    )

    # 提取Top N候选用于阶段2
    top_candidates_for_final = _extract_top_candidates(
        phase1_result.get("leaderboard", []),
        n=top_n_for_final,
    )

    print("\n阶段1 Top {} 候选:".format(len(top_candidates_for_final)))
    for i, c in enumerate(top_candidates_for_final, 1):
        print("  {}. {} (net_return={:.4f})".format(
            i,
            c.get("candidate_id"),
            float(c.get("metrics", {}).get("net_mean_return", 0)),
        ))

    # ========== 阶段2：滚动打分验证 ==========
    final_recommendation, phase2_results = _run_rolling_score_phase(
        run_root=run_root,
        leaderboard=phase1_result.get("leaderboard", []),
        top_n=top_n_for_final,
    )

    # 生成Markdown报告
    md_report = build_final_recommendation_markdown(
        final_recommendation,
        phase1_summary=phase1_result,
    )
    md_path = run_root / "final_recommendation.md"
    md_path.write_text(md_report, encoding="utf-8")

    print("\n最终推荐:")
    print("  最佳候选: {}".format(final_recommendation.get("best_candidate_id")))
    print("  综合得分: {:.1f}".format(final_recommendation.get("best_composite_score") or 0))
    print("  稳定得分: {:.1f}".format(final_recommendation.get("best_stability_score") or 0))

    # ========== 汇总结果 ==========
    result = {
        "run_tag": str(run_tag),
        "run_root": str(run_root),
        "search_strategy": "coarse_to_fine_3phase",
        "phase0": {
            "name": "coarse",
            "catalog_name": PORTFOLIO_COARSE_CATALOG_NAME,
            "num_candidates": len(phase0_result.get("leaderboard", [])),
            "top_candidates": [
                {"candidate_id": c.get("candidate_id"), "metrics": c.get("metrics")}
                for c in top_candidates_for_refined
            ],
        },
        "phase1": {
            "name": "refined",
            "catalog_name": PORTFOLIO_REFINED_CATALOG_NAME,
            "num_candidates": len(phase1_result.get("leaderboard", [])),
            "top_candidates": [
                {"candidate_id": c.get("candidate_id"), "metrics": c.get("metrics")}
                for c in top_candidates_for_final
            ],
        },
        "phase2": {
            "name": "rolling_score",
            "output_path": str(run_root / "phase2_rolling_scores.json"),
            "results": phase2_results,
        },
        "final_recommendation": final_recommendation,
        "status": "completed",
    }

    # 保存完整结果
    full_result_path = run_root / "three_phase_search_results.json"
    full_result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return result


def _run_single_phase_search(
    *,
    run_root: Path,
    raw_df,
    phase_name: str,
    catalog_name: str,
    model_kind: str,
    label_transform: str,
    horizon_days: int,
    smoke_max_folds: int,
    full_max_folds: int | None,
    top_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """执行单阶段catalog搜索。"""
    phase_root = run_root / phase_name
    phase_root.mkdir(parents=True, exist_ok=True)

    baseline_candidate = build_baseline_candidate()
    baseline_summary = run_config_trial(
        run_root=phase_root,
        experiment_index=0,
        raw_df=raw_df,
        config=build_candidate_config(
            baseline_candidate,
            model_kind=model_kind,
            label_transform=label_transform,
            horizon_days=horizon_days,
        ),
        stage="baseline",
        max_folds=full_max_folds,
    )

    candidates = build_candidate_catalog(
        catalog_name,
        top_candidates=top_candidates,
    )

    leaderboard = []
    best_summary = dict(baseline_summary)

    for experiment_index, candidate in enumerate(candidates, start=1):
        print("  [{}/{}] 评估: {} ...".format(
            experiment_index, len(candidates), candidate["id"]
        ))

        smoke_summary = run_config_trial(
            run_root=phase_root,
            experiment_index=experiment_index,
            raw_df=raw_df,
            config=build_candidate_config(
                candidate,
                model_kind=model_kind,
                label_transform=label_transform,
                horizon_days=horizon_days,
            ),
            stage="smoke",
            max_folds=smoke_max_folds,
        )
        smoke_decision = judge_candidate(
            smoke_summary,
            baseline_summary=baseline_summary,
            best_summary=best_summary,
            stage="smoke",
        )
        if smoke_decision.get("status") != "keep":
            leaderboard.append(
                _build_leaderboard_entry(
                    candidate=candidate,
                    summary=smoke_summary,
                    decision=smoke_decision,
                    stage_reached="smoke",
                )
            )
            continue

        full_summary = run_config_trial(
            run_root=phase_root,
            experiment_index=experiment_index,
            raw_df=raw_df,
            config=build_candidate_config(
                candidate,
                model_kind=model_kind,
                label_transform=label_transform,
                horizon_days=horizon_days,
            ),
            stage="full",
            max_folds=full_max_folds,
        )
        full_decision = judge_candidate(
            full_summary,
            baseline_summary=baseline_summary,
            best_summary=best_summary,
            stage="full",
        )
        entry = _build_leaderboard_entry(
            candidate=candidate,
            summary=full_summary,
            decision=full_decision,
            stage_reached="full",
        )
        leaderboard.append(entry)
        if full_decision.get("status") == "champion":
            best_summary = dict(full_summary)

    result = {
        "phase_name": phase_name,
        "catalog_name": catalog_name,
        "baseline": {
            "candidate_id": baseline_candidate["id"],
            "metrics": summarize_metrics(baseline_summary),
        },
        "leaderboard": _sort_leaderboard(leaderboard),
    }

    # 保存阶段结果
    phase_result_path = phase_root / "{}_results.json".format(phase_name)
    phase_result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return result


def _extract_top_candidates(
    leaderboard: list[dict[str, Any]],
    n: int = 2,
) -> list[dict[str, Any]]:
    """从leaderboard提取Top N候选（排除crash/invalid状态）。"""
    valid = [
        entry for entry in leaderboard
        if entry.get("status") not in ("crash", "invalid")
    ]
    return valid[:n]


def _rebuild_candidates_for_phase2(
    top_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """从leaderboard条目重建完整候选定义（包含portfolio参数）。

    阶段2需要完整的候选定义来调用滚动打分器。
    """
    candidates = []
    for entry in top_candidates:
        candidate_id = entry.get("candidate_id", "")

        # 从candidate_id解析portfolio参数
        # 格式: refined_b{buy_k}_h{hold_k} 或 coarse_b{buy_k}_h{hold_k}
        import re
        match = re.search(r"b(\d+)_h(\d+)", candidate_id)
        if match:
            buy_k, hold_k = int(match.group(1)), int(match.group(2))
            candidates.append({
                "id": candidate_id,
                "portfolio": {
                    "buy_top_k": buy_k,
                    "hold_top_k": hold_k,
                    "rebalance_interval": 20,
                    "holding_bonus": 0.5,
                },
                "strategy_profile": "smooth",
            })

    return candidates


def _run_rolling_score_phase(
    run_root: Path,
    leaderboard: list[dict[str, Any]],
    top_n: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """滚动打分验证阶段：从 leaderboard 取 top N，重建候选，跑滚动打分器。

    供 ``run_catalog_search`` 和 ``run_three_phase_portfolio_search`` 共享使用。
    """
    top_candidates = _extract_top_candidates(leaderboard, n=top_n)
    candidates = _rebuild_candidates_for_phase2(top_candidates)

    print("\n" + "=" * 60)
    print("滚动打分器验证（约{}分钟 / 候选）".format(len(candidates) * 10))
    print("=" * 60)
    for i, c in enumerate(candidates, 1):
        print("  {}. {}".format(i, c.get("id")))

    phase2_results = batch_run_rolling_score(
        candidates=candidates,
        run_root=run_root,
    )

    phase2_output_path = run_root / "phase2_rolling_scores.json"
    save_phase2_results(phase2_results, phase2_output_path)

    final_recommendation = build_final_recommendation(
        phase2_results,
        top_n=top_n,
    )
    return final_recommendation, phase2_results
