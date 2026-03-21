"""Target curve and rebalance-step helpers."""


def base_target_from_percentile(percentile):
    anchors = (
        (0.00, 0.90),
        (0.10, 0.90),
        (0.20, 0.80),
        (0.40, 0.70),
        (0.52, 0.55),
        (0.68, 0.55),
        (0.80, 0.40),
        (0.90, 0.30),
        (1.00, 0.25),
    )
    p = min(max(float(percentile), 0.0), 1.0)
    for (left_p, left_target), (right_p, right_target) in zip(anchors, anchors[1:]):
        if p <= right_p:
            if right_p == left_p:
                return right_target
            ratio = (p - left_p) / float(right_p - left_p)
            return left_target + ratio * (right_target - left_target)
    return anchors[-1][1]


def step_limit(percentile_ref, trend, current_target, desired_target, default_step_limit, initial_step_limit):
    if current_target < 0.01 and desired_target >= 0.40:
        return max(default_step_limit, min(initial_step_limit, desired_target))

    step = default_step_limit
    if percentile_ref <= 0.40:
        if trend > 0.03:
            return 0.18
        if trend < -0.03:
            return 0.08
    if percentile_ref >= 0.80:
        if trend > 0.03:
            return 0.18
        if trend < -0.03:
            return 0.08
    return step


def bounded_target(current_target, desired_target, max_step):
    delta = desired_target - current_target
    if delta > max_step:
        return current_target + max_step
    if delta < -max_step:
        return current_target - max_step
    return desired_target
