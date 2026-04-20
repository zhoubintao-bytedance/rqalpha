"""TX1 autoresearch 运行状态与结果记账。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


RESULTS_TSV_HEADER = (
    "commit\tstatus\tnet_mean_return\tmax_drawdown\tstability_score\tdescription\texperiment_path"
)


class AutoresearchStateStore(object):
    """管理 autoresearch run 目录内的 state 与结果摘要文件。"""

    def __init__(self, run_root: str | Path):
        # 统一把 run 根目录解析成 Path，后续所有状态文件都挂在这里。
        self.run_root = Path(run_root)
        self.state_path = self.run_root / "state.json"
        self.results_path = self.run_root / "results.tsv"

    def initialize(
        self,
        *,
        run_tag: str,
        baseline_commit: str,
        branch_name: str,
        baseline_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """初始化 run 根目录、state.json 与 results.tsv。"""
        self.run_root.mkdir(parents=True, exist_ok=True)
        state = {
            "run_tag": str(run_tag),
            "branch_name": str(branch_name),
            "baseline_commit": str(baseline_commit),
            "current_commit": str(baseline_commit),
            "best_commit": str(baseline_commit),
            "baseline_summary": dict(baseline_summary or {}),
            "best_summary": dict(baseline_summary or {}),
            "experiment_count": 0,
            "last_status": "initialized",
            "last_reason_code": None,
            "last_experiment_path": "",
            "last_error": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self.save(state)
        if not self.results_path.exists():
            self.results_path.write_text(RESULTS_TSV_HEADER + "\n", encoding="utf-8")
        return state

    def load(self) -> dict[str, Any]:
        """读取当前 run 的状态快照。"""
        with self.state_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, state: dict[str, Any]) -> None:
        """覆盖写入 state.json。"""
        working = dict(state)
        working["updated_at"] = datetime.now().isoformat()
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(working, handle, indent=2, ensure_ascii=False, default=str)

    def append_result(
        self,
        *,
        commit: str,
        status: str,
        metrics: dict[str, Any] | None,
        description: str,
        experiment_path: str | Path,
    ) -> None:
        """向 results.tsv 追加一行实验摘要，便于快速筛选。"""
        metrics = dict(metrics or {})
        row = [
            str(commit),
            str(status),
            self._format_metric(metrics.get("net_mean_return")),
            self._format_metric(metrics.get("max_drawdown")),
            self._format_metric(metrics.get("stability_score")),
            str(description),
            str(experiment_path),
        ]
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write("\t".join(row) + "\n")

    def update_after_decision(
        self,
        *,
        decision_status: str,
        commit: str,
        candidate_summary: dict[str, Any] | None,
        reason_code: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """根据 keep/discard/champion 等决策推进 state。"""
        state = self.load()
        state["experiment_count"] = int(state.get("experiment_count", 0)) + 1
        state["last_status"] = str(decision_status)
        state["last_reason_code"] = reason_code
        state["last_experiment_path"] = str((candidate_summary or {}).get("experiment_path", ""))
        state["last_error"] = error_message
        if decision_status in {"keep", "champion"}:
            # 被保留的候选可以推进当前工作基线，后续 patch 会在它的基础上继续演化。
            state["current_commit"] = str(commit)
        if decision_status == "champion":
            # champion 才代表“当前最优解”，因此只有它能刷新 best_* 字段。
            state["best_commit"] = str(commit)
            state["best_summary"] = dict(candidate_summary or {})
        self.save(state)
        return state

    @staticmethod
    def _format_metric(value: Any) -> str:
        """把数值指标规范成固定六位小数字符串。"""
        if value is None:
            return "0.000000"
        return "{:.6f}".format(float(value))
