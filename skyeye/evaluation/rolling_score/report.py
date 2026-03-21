"""Shared report rendering helpers for rolling-score evaluation."""

import re
import unicodedata


def _strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def display_width(value):
    text = str(value)
    total = 0
    for ch in _strip_ansi(text):
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def render_box(lines, title=None):
    width = max(display_width(line) for line in lines)
    if title:
        width = max(width, display_width(title) + 4)
        pad_total = width - display_width(title) - 2
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        print("┌─" + "─" * pad_left + " " + title + " " + "─" * pad_right + "─┐")
    else:
        print("┌" + "─" * (width + 2) + "┐")
    for line in lines:
        pad = width - display_width(line)
        print("│ " + line + " " * pad + " │")
    print("└" + "─" * (width + 2) + "┘")


def build_strategy_card_lines(strategy_spec):
    if strategy_spec is None:
        return []
    raw = strategy_spec.raw
    lines = [
        "ID: {}".format(strategy_spec.strategy_id),
        "名称: {}".format(strategy_spec.name),
        "家族: {}".format(strategy_spec.family),
        "状态: {}".format(strategy_spec.status),
        "标的/基准: {} / {}".format(strategy_spec.instrument, strategy_spec.benchmark),
        "假设: {}".format(strategy_spec.hypothesis),
    ]
    if raw.get("positioning_rule"):
        lines.append("主仓位规则: {}".format(raw["positioning_rule"]))
    if raw.get("rebalance_frequency"):
        lines.append("调仓频率: {}".format(raw["rebalance_frequency"]))
    if raw.get("risk_controls"):
        lines.append("风控: {}".format("；".join(raw["risk_controls"])))
    if raw.get("expected_good_regimes"):
        lines.append("适用环境: {}".format("；".join(raw["expected_good_regimes"])))
    if raw.get("expected_bad_regimes"):
        lines.append("弱势环境: {}".format("；".join(raw["expected_bad_regimes"])))
    return lines
