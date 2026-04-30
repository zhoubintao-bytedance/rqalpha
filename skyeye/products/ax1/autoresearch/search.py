"""Conflict-aware AX1 feature set search."""

from __future__ import annotations

from itertools import combinations
from typing import Any

from skyeye.products.ax1.autoresearch.feature_set import FeatureSetCandidate


def generate_feature_set_candidates(
    *,
    base_features: list[str] | tuple[str, ...],
    scorecards: dict[str, dict[str, Any]],
    conflicts: dict[str, Any] | None = None,
    data_audit_by_feature: dict[str, dict[str, Any]] | None = None,
    max_additions: int = 2,
    beam_width: int = 5,
    max_candidates: int | None = None,
    early_stop_score_floor: float | None = None,
    parent_candidate_id: str = "champion",
) -> list[FeatureSetCandidate]:
    base = tuple(str(feature) for feature in base_features)
    base_set = set(base)
    eligible = []
    for feature, scorecard in scorecards.items():
        feature_name = str(feature)
        if feature_name in base_set:
            continue
        audit = (data_audit_by_feature or {}).get(feature_name, {"passed": True})
        if not bool(audit.get("passed", False)):
            continue
        score = float(scorecard.get("final_factor_score", 0.0) or 0.0)
        if score <= 0.0:
            continue
        if early_stop_score_floor is not None and score < float(early_stop_score_floor):
            continue
        eligible.append((feature_name, score))
    eligible = sorted(eligible, key=lambda item: (-item[1], item[0]))

    conflict_sets = _conflict_sets(conflicts or {})
    candidates: list[FeatureSetCandidate] = []
    max_additions = max(1, int(max_additions))
    budget = int(beam_width)
    if max_candidates is not None:
        budget = min(budget, int(max_candidates))
    budget = max(0, budget)
    if budget == 0:
        return []
    for size in range(1, max_additions + 1):
        for combo in combinations([feature for feature, _ in eligible], size):
            if _violates_conflict(combo, conflict_sets):
                continue
            candidate_id = f"{'single_add' if size == 1 else 'combo_add'}_{len(candidates) + 1:03d}_{'_'.join(combo)}"
            candidates.append(
                FeatureSetCandidate(
                    candidate_id=candidate_id,
                    base_features=base,
                    added_features=tuple(combo),
                    removed_features=(),
                    rationale="selected by positive factor score and conflict guard",
                    parent_candidate_id=str(parent_candidate_id),
                    search_phase="single_add" if size == 1 else "combo_add",
                )
            )
            if len(candidates) >= budget:
                return candidates
    return candidates


def _conflict_sets(conflicts: dict[str, Any]) -> list[set[str]]:
    groups = []
    for group in conflicts.get("high_corr_groups", []) or []:
        features = {str(feature) for feature in group.get("features", [])}
        if len(features) > 1:
            groups.append(features)
    return groups


def _violates_conflict(features: tuple[str, ...], conflict_sets: list[set[str]]) -> bool:
    selected = set(features)
    return any(len(selected.intersection(group)) > 1 for group in conflict_sets)
