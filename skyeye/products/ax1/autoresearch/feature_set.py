"""Feature set candidate contracts for AX1 autoresearch."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSetCandidate:
    candidate_id: str
    base_features: tuple[str, ...]
    added_features: tuple[str, ...]
    removed_features: tuple[str, ...]
    rationale: str
    parent_candidate_id: str
    search_phase: str

    @property
    def features(self) -> tuple[str, ...]:
        removed = set(self.removed_features)
        return tuple([feature for feature in self.base_features if feature not in removed] + list(self.added_features))

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "base_features": list(self.base_features),
            "added_features": list(self.added_features),
            "removed_features": list(self.removed_features),
            "features": list(self.features),
            "rationale": self.rationale,
            "parent_candidate_id": self.parent_candidate_id,
            "search_phase": self.search_phase,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureSetCandidate":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            base_features=tuple(str(item) for item in payload.get("base_features", ())),
            added_features=tuple(str(item) for item in payload.get("added_features", ())),
            removed_features=tuple(str(item) for item in payload.get("removed_features", ())),
            rationale=str(payload.get("rationale", "")),
            parent_candidate_id=str(payload.get("parent_candidate_id", "")),
            search_phase=str(payload.get("search_phase", "")),
        )
