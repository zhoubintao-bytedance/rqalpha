"""Confidence-aware adjustments for core-plus-tactical exposure."""

from __future__ import annotations


def confidence_adjusted_target(current_target, desired_target, confidence, core_floor):
    current = float(current_target)
    desired = float(desired_target)
    floor = float(core_floor)

    if desired <= current:
        return desired

    if confidence == "low":
        return min(desired, max(current, floor))

    return desired


def confidence_step_multiplier(confidence):
    if confidence == "low":
        return 0.5
    if confidence == "lowered":
        return 0.75
    return 1.0


def apply_cheap_trend_cap(
    desired_target,
    percentile_ref,
    trend_strength,
    cheap_entry_cap_percentile,
    cheap_entry_cap_trend_threshold,
    cheap_entry_cap_target,
):
    if (
        float(percentile_ref) <= float(cheap_entry_cap_percentile)
        and float(trend_strength) <= float(cheap_entry_cap_trend_threshold)
    ):
        return {
            "target": min(float(desired_target), float(cheap_entry_cap_target)),
            "mode": "cheap_trend_cap",
        }

    return {
        "target": float(desired_target),
        "mode": "none",
    }
