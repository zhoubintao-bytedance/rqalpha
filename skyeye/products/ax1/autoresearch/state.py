"""Persistent state for AX1 autoresearch runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AX1AutoresearchStateStore:
    def __init__(self, run_root: str | Path):
        self.run_root = Path(run_root)
        self.state_path = self.run_root / "state.json"

    def initialize(
        self,
        *,
        run_tag: str,
        baseline_candidate_id: str,
        baseline_summary: dict[str, Any],
    ) -> dict[str, Any]:
        self.run_root.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": 1,
            "run_tag": str(run_tag),
            "baseline_candidate_id": str(baseline_candidate_id),
            "baseline_summary": dict(baseline_summary or {}),
            "champion_candidate_id": str(baseline_candidate_id),
            "leaderboard": [],
        }
        self.save(state)
        return state

    def load(self) -> dict[str, Any]:
        with self.state_path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def save(self, state: dict[str, Any]) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as file_obj:
            json.dump(state, file_obj, ensure_ascii=False, indent=2, sort_keys=True)

    def append_candidate(
        self,
        *,
        candidate_id: str,
        parent_candidate_id: str,
        status: str,
        reason_code: str,
        stage_reached: str,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self.load()
        entry = {
            "candidate_id": str(candidate_id),
            "parent_candidate_id": str(parent_candidate_id),
            "status": str(status),
            "reason_code": str(reason_code),
            "stage_reached": str(stage_reached),
            "metrics": dict(metrics or {}),
        }
        state.setdefault("leaderboard", []).append(entry)
        if status == "champion":
            state["champion_candidate_id"] = str(candidate_id)
        self.save(state)
        return state
