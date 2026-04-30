"""AX1 autoresearch runner utilities."""

from __future__ import annotations


def summarize_candidate_result(result: dict) -> dict:
    """Extract the stable fields used by AX1 candidate judging and state logs."""

    return {
        "metrics": dict(result.get("metrics") or {}),
        "data_audit": dict(result.get("data_audit") or {}),
        "feature_review_summary": dict(result.get("feature_review_summary") or {}),
    }
