"""Portfolio候选滚动打分器封装。

提供对TX1 portfolio参数候选进行滚动打分器评估的能力。
用于三阶段搜索的第三阶段：对Top5候选进行精细化滚动打分验证。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skyeye.evaluation.rolling_score import run_rolling_backtests
from skyeye.evaluation.rolling_score import (
    compute_composite_score as compute_replay_composite_score,
)
from skyeye.evaluation.rolling_score import (
    compute_stability_score as compute_replay_stability_score,
)
from skyeye.evaluation.rolling_score import (
    detect_risk_alerts as detect_replay_risk_alerts,
)
from skyeye.evaluation.rolling_score import (
    project_to_quarters as project_replay_to_quarters,
)


ROLLING_SCORE_STRATEGY_FILE = (
    Path(__file__).resolve().parents[1]
    / "strategies"
    / "rolling_score"
    / "strategy.py"
)
TX1_DIAGNOSTIC_MOD_NAME = "tx1_diagnostics"
TX1_DIAGNOSTIC_MOD_LIB = "skyeye.products.tx1.autoresearch.rqalpha_mod_tx1_diagnostics"


def run_rolling_score_for_candidate(
    candidate: dict[str, Any],
    run_root: Path,
    selected_indices: list[int] | None = None,
    cash: float = 1_000_000,
) -> dict[str, Any]:
    """对单个portfolio候选执行滚动打分器评估。

    Args:
        candidate: 候选定义，需包含portfolio参数或artifact_line_id
        run_root: 运行根目录，用于存储输出
        selected_indices: 要执行的窗口编号列表，None表示全部
        cash: 初始资金

    Returns:
        滚动打分结果，包含composite_score、stability_score等
    """
    # 提取artifact_line_id
    artifact_line_id = _resolve_artifact_line_id(candidate)
    strategy_profile = str(candidate.get("strategy_profile") or "smooth")

    # 构建extra_config
    extra_config = {
        "strategy_profile": strategy_profile,
        "tx1_artifact_line": artifact_line_id,
    }
    if candidate.get("tx1_profile_overrides"):
        extra_config["tx1_profile_overrides"] = dict(
            candidate.get("tx1_profile_overrides") or {}
        )

    try:
        # 执行滚动回测
        raw_replay_output = run_rolling_backtests(
            str(ROLLING_SCORE_STRATEGY_FILE),
            cash=cash,
            selected_indices=selected_indices,
            extra_mods=[TX1_DIAGNOSTIC_MOD_NAME],
            mod_configs={
                TX1_DIAGNOSTIC_MOD_NAME: {"lib": TX1_DIAGNOSTIC_MOD_LIB},
            },
            extra_config=extra_config,
            return_details=True,
            include_trade_details=False,
        )

        # 解析结果
        if isinstance(raw_replay_output, dict) and "windows" in raw_replay_output:
            window_results = list(raw_replay_output.get("windows") or [])
            failed_windows = list(raw_replay_output.get("failed_windows") or [])
        else:
            window_results = list(raw_replay_output or [])
            failed_windows = []

        # 计算季度投影和综合得分
        quarterly_scores, quarterly_raw_indicators = project_replay_to_quarters(
            window_results
        )

        if quarterly_scores:
            composite_score, core_indicators = compute_replay_composite_score(
                quarterly_scores,
                quarterly_raw_indicators,
            )
        else:
            composite_score, core_indicators = 0.0, {
                "annualized_returns": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.0,
            }

        stability_score = compute_replay_stability_score(quarterly_scores)
        risk_alerts = detect_replay_risk_alerts(quarterly_scores)
        risk_codes = [str(item.get("code")) for item in risk_alerts]

        return {
            "candidate_id": candidate.get("id"),
            "artifact_line_id": artifact_line_id,
            "composite_score": float(composite_score),
            "stability_score": float(stability_score),
            "core_indicators": core_indicators,
            "risk_codes": risk_codes,
            "num_windows": len(window_results),
            "num_failed_windows": len(failed_windows),
            "status": "success",
        }

    except Exception as e:
        return {
            "candidate_id": candidate.get("id"),
            "composite_score": 0.0,
            "stability_score": 0.0,
            "error": str(e),
            "status": "failed",
        }


def batch_run_rolling_score(
    candidates: list[dict[str, Any]],
    run_root: Path,
    selected_indices: list[int] | None = None,
    cash: float = 1_000_000,
) -> list[dict[str, Any]]:
    """批量执行滚动打分器评估。

    Args:
        candidates: 候选列表
        run_root: 运行根目录
        selected_indices: 要执行的窗口编号列表
        cash: 初始资金

    Returns:
        每个候选的滚动打分结果列表
    """
    results = []
    for i, candidate in enumerate(candidates, 1):
        print(
            "[{}/{}] 正在评估: {} ...".format(
                i, len(candidates), candidate.get("id")
            )
        )
        result = run_rolling_score_for_candidate(
            candidate=candidate,
            run_root=run_root,
            selected_indices=selected_indices,
            cash=cash,
        )
        results.append(result)
    return results


def _resolve_artifact_line_id(candidate: dict[str, Any]) -> str:
    """从候选定义解析artifact_line_id。

    优先使用显式的artifact_line_id，否则从portfolio参数构造。
    """
    # 优先使用显式指定的artifact_line_id
    if candidate.get("artifact_line_id"):
        return str(candidate["artifact_line_id"])

    # 从portfolio参数构造artifact_line_id
    portfolio = candidate.get("portfolio", {})
    buy_k = portfolio.get("buy_top_k", 25)
    hold_k = portfolio.get("hold_top_k", 45)
    return "combo_b{}_h{}".format(buy_k, hold_k)


def build_final_recommendation(
    phase2_results: list[dict[str, Any]],
    top_n: int = 5,
) -> dict[str, Any]:
    """基于滚动打分结果生成最终推荐。

    Args:
        phase2_results: 滚动打分结果列表
        top_n: 返回的Top N数量

    Returns:
        最终推荐结果，包含best_candidate和ranked列表
    """
    # 过滤失败的结果并按composite_score排序
    valid_results = [
        r for r in phase2_results
        if r.get("status") == "success"
    ]

    sorted_results = sorted(
        valid_results,
        key=lambda x: float(x.get("composite_score", 0)),
        reverse=True,
    )

    top_results = sorted_results[:top_n]

    best = top_results[0] if top_results else None

    return {
        "best_candidate_id": best.get("candidate_id") if best else None,
        "best_composite_score": best.get("composite_score") if best else None,
        "best_stability_score": best.get("stability_score") if best else None,
        "best_core_indicators": best.get("core_indicators") if best else None,
        "ranked_candidates": [
            {
                "rank": i + 1,
                "candidate_id": r.get("candidate_id"),
                "composite_score": r.get("composite_score"),
                "stability_score": r.get("stability_score"),
                "risk_codes": r.get("risk_codes", []),
            }
            for i, r in enumerate(top_results)
        ],
        "total_evaluated": len(phase2_results),
        "total_valid": len(valid_results),
    }


def save_phase2_results(
    results: list[dict[str, Any]],
    output_path: Path,
) -> str:
    """保存阶段二滚动打分结果到文件。

    Args:
        results: 滚动打分结果列表
        output_path: 输出文件路径

    Returns:
        保存的文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "phase": "rolling_score_verification",
        "results": results,
    }

    output_path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return str(output_path)


def build_final_recommendation_markdown(
    recommendation: dict[str, Any],
    phase1_summary: dict[str, Any] | None = None,
) -> str:
    """生成人类可读的最终推荐Markdown报告。

    Args:
        recommendation: 最终推荐结果
        phase1_summary: 阶段一的摘要信息（可选）

    Returns:
        Markdown格式的报告文本
    """
    lines = [
        "# TX1 Portfolio参数搜索最终推荐",
        "",
        "## 推荐参数",
        "",
    ]

    best_id = recommendation.get("best_candidate_id")
    if best_id:
        # 从candidate_id解析参数
        # 格式: refined_b{buy_k}_h{hold_k} 或 coarse_b{buy_k}_h{hold_k}
        import re
        match = re.search(r"b(\d+)_h(\d+)", best_id)
        if match:
            buy_k, hold_k = int(match.group(1)), int(match.group(2))
            lines.extend([
                "| 参数 | 推荐值 |",
                "|------|--------|",
                "| buy_top_k | {} |".format(buy_k),
                "| hold_top_k | {} |".format(hold_k),
                "| rebalance_interval | 20 |",
                "| holding_bonus | 0.5 |",
                "",
            ])
        else:
            lines.append("**候选ID**: {}".format(best_id))
            lines.append("")

    lines.extend([
        "## 评分对比",
        "",
        "| 排名 | 候选 | 综合得分 | 稳定得分 | 风险标记 |",
        "|------|------|----------|----------|----------|",
    ])

    for r in recommendation.get("ranked_candidates", []):
        risk_str = ", ".join(r.get("risk_codes", [])) or "-"
        lines.append(
            "| {} | {} | {:.1f} | {:.1f} | {} |".format(
                r.get("rank", "-"),
                r.get("candidate_id", "-"),
                float(r.get("composite_score", 0)),
                float(r.get("stability_score", 0)),
                risk_str,
            )
        )

    lines.extend([
        "",
        "## 统计摘要",
        "",
        "- 评估候选数: {}".format(recommendation.get("total_evaluated", 0)),
        "- 有效候选数: {}".format(recommendation.get("total_valid", 0)),
        "",
    ])

    return "\n".join(lines)
