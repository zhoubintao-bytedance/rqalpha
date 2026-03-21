"""Confidence-aware exposure adjustments."""


def confidence_adjusted_target(current_target, desired_target, confidence):
    if confidence == "low" and desired_target > current_target:
        return current_target
    return desired_target


def confidence_step_multiplier(confidence):
    if confidence == "lowered":
        return 2.0 / 3.0
    return 1.0
