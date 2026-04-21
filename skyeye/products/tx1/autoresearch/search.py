"""TX1 autoresearch catalog 搜索编排。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skyeye.products.tx1.autoresearch.catalog import (
    DEFAULT_CATALOG_NAME,
    build_baseline_candidate,
    build_candidate_catalog,
    build_candidate_config,
)
from skyeye.products.tx1.autoresearch.judge import judge_candidate
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

    result = {
        "run_tag": str(run_tag),
        "run_root": str(run_root),
        "catalog_name": str(catalog_name),
        "baseline": {
            "candidate_id": baseline_candidate["id"],
            "summary": baseline_summary,
            "metrics": summarize_metrics(baseline_summary),
        },
        "leaderboard": _sort_leaderboard(leaderboard),
        "champion": champion_entry,
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
