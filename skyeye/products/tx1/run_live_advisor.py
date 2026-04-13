# -*- coding: utf-8 -*-
"""TX1 日频实盘辅助 CLI。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from functools import lru_cache

from skyeye.data import DataFacade
from skyeye.products.tx1.live_advisor.holdings_io import load_holdings_file
from skyeye.products.tx1.live_advisor.service import LiveAdvisorService

ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RESET = "\033[0m"
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def build_parser():
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Run TX1 live advisor for a given trade date")
    parser.add_argument("--package-id", required=True, help="Promoted TX1 live package id or package path")
    parser.add_argument("--trade-date", required=True, help="Trade date to score, format YYYY-MM-DD")
    parser.add_argument("--top-k", type=int, default=20, help="Number of recommendations to return")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    parser.add_argument("--include-dropped", action="store_true", help="Include dropped symbols in result output")
    parser.add_argument("--packages-root", help="Optional packages root override")
    parser.add_argument("--universe-size", type=int, default=300, help="Universe size when building live snapshot")
    parser.add_argument(
        "--universe-source",
        choices=["runtime_fast", "research"],
        default="runtime_fast",
        help="Universe resolver used by live runtime",
    )
    parser.add_argument("--universe-cache-root", help="Optional runtime universe cache root override")
    parser.add_argument("--market-cap-floor-quantile", type=float, help="Optional market-cap floor quantile")
    parser.add_argument("--market-cap-column", help="Optional market-cap column override")
    parser.add_argument("--holdings-file", help="Optional csv/json holdings file")
    parser.add_argument("--last-rebalance-date", help="Optional last rebalance date, format YYYY-MM-DD")
    return parser


def main(argv=None):
    """执行 live advisor，并把结果打印到 stdout。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    current_holdings = load_holdings_file(args.holdings_file) if args.holdings_file else None

    service = LiveAdvisorService(packages_root=args.packages_root)
    result = service.get_recommendations(
        args.package_id,
        trade_date=args.trade_date,
        top_k=args.top_k,
        include_dropped=args.include_dropped,
        universe_size=args.universe_size,
        universe_source=args.universe_source,
        universe_cache_root=args.universe_cache_root,
        market_cap_floor_quantile=args.market_cap_floor_quantile,
        market_cap_column=args.market_cap_column,
        current_holdings=current_holdings,
        last_rebalance_date=args.last_rebalance_date,
    )
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(render_table(result))
    return 0 if result.get("status") == "ok" else 2


def render_table(result: dict) -> str:
    """把 live advisor 结果渲染成可读表格。"""
    lines = [
        "package_id={}".format(result.get("package_id")),
        "gate_level={}".format(_format_gate_level(result.get("gate_level"))),
        "requested_trade_date={}".format(result.get("requested_trade_date")),
        "latest_available_trade_date={}".format(result.get("latest_available_trade_date")),
        "score_date={}".format(result.get("score_date")),
        "raw_data_end_date={}".format(result.get("raw_data_end_date")),
        "fit_end_date={}".format(result.get("fit_end_date")),
        "label_end_date={}".format(result.get("label_end_date")),
        "evidence_end_date={}".format(result.get("evidence_end_date")),
        "status={}".format(_format_status(result.get("status"))),
    ]
    warning_lines = _render_warning_lines(result.get("warnings", []))
    if warning_lines:
        lines = warning_lines + lines
    if result.get("status") != "ok":
        lines.append("reasons={}".format("; ".join(result.get("reasons", []))))
        return "\n".join(lines)

    recommendations = list(result.get("recommendations", []))
    order_book_ids = [str(item.get("order_book_id")) for item in recommendations]
    stock_name_map = _resolve_stock_name_map(order_book_ids)
    lines.extend(_build_result_interpretation(recommendations))
    lines.extend(_build_recent_canary_notes(recommendations))
    lines.extend(_build_portfolio_advice_notes(result.get("portfolio_advice", {})))
    lines.extend(_build_metric_notes())
    lines.append(_render_recommendation_table(recommendations, stock_name_map))
    return "\n".join(lines)


def _build_metric_notes() -> list[str]:
    """在表格前输出关键指标的中文说明，避免实盘阅读歧义。"""
    return [
        "指标说明:",
        "  分位: 当日横截面分位，100.0% 表示当日最高分。",
        "  胜率: 历史 OOS 同分位桶未来 20 日正收益占比。",
        "  均值收益: 历史 OOS 同分位桶未来 20 日平均收益。",
        "  中位收益: 历史 OOS 同分位桶未来 20 日收益中位数。",
        "  P25~P75: 历史 OOS 同分位桶未来 20 日收益四分位区间。",
        "  样本数: 历史 OOS 同分位桶样本数。",
    ]


def _build_recent_canary_notes(recommendations: list[dict]) -> list[str]:
    """补充近端 canary 证据，避免只看旧 OOS 统计。"""
    if not recommendations:
        return []
    first_item = recommendations[0]
    evidence = first_item.get("recent_canary_evidence") or {}
    window = evidence.get("window") or {}
    if not window:
        return []
    return [
        "近端Canary:",
        "  窗口 {} ~ {}，胜率 {}，均值收益 {}，中位收益 {}。".format(
            window.get("start_date", "-"),
            window.get("end_date", "-"),
            _format_percent(evidence.get("win_rate"), digits=1),
            _format_percent(evidence.get("mean_return"), digits=2, signed=True),
            _format_percent(evidence.get("median_return"), digits=2, signed=True),
        ),
    ]


def _build_portfolio_advice_notes(portfolio_advice: dict) -> list[str]:
    """输出组合建议摘要，直接回答是否调仓与主要动作。"""
    if not portfolio_advice:
        return []
    lines = [
        "组合建议:",
        "  建议等级: {}".format(portfolio_advice.get("advice_level", "ok")),
        "  需要调仓: {}".format("是" if portfolio_advice.get("rebalance_due") else "否"),
        "  预计换手: {}".format(_format_percent(portfolio_advice.get("estimated_turnover"), digits=1)),
    ]
    blockers = list(portfolio_advice.get("execution_blockers") or [])
    if blockers:
        lines.append("  执行阻塞:")
        for blocker in blockers:
            lines.append("    {}".format(blocker))
    preflight_checks = dict(portfolio_advice.get("preflight_checks") or {})
    if preflight_checks:
        lines.append("  执行前检查:")
        for check_name, check_payload in preflight_checks.items():
            status = "通过" if check_payload.get("passed") else "失败"
            lines.append("    {}: {}".format(check_name, status))
    actions = list(portfolio_advice.get("actions") or [])
    if not actions:
        return lines
    priority = {"add": 0, "reduce": 1, "exit": 2, "keep": 3}
    actions = sorted(
        actions,
        key=lambda item: (
            priority.get(item.get("action"), 99),
            -abs(float(item.get("delta_weight", 0.0))),
            str(item.get("order_book_id")),
        ),
    )
    lines.append("  调仓动作:")
    for action in actions[:8]:
        lines.append(
            "    {} {} -> {} (delta {})".format(
                action.get("action"),
                action.get("order_book_id"),
                _format_percent(action.get("target_weight"), digits=1),
                _format_percent(action.get("delta_weight"), digits=1, signed=True),
            )
        )
    return lines


def _render_warning_lines(warnings: list[dict]) -> list[str]:
    """把结构化 warning 渲染成表格顶部的醒目提示。"""
    lines = []
    for warning in warnings or []:
        message = str(warning.get("message", "")).strip()
        if not message:
            continue
        if warning.get("level") == "critical":
            lines.append(_colorize(message, ANSI_RED))
        elif warning.get("level") == "warning":
            lines.append(_colorize(message, ANSI_YELLOW))
        else:
            lines.append(message)
    return lines


def _resolve_stock_name_map(order_book_ids) -> dict[str, str]:
    """按股票代码解析显示名称，缺失时返回空映射。"""
    resolved = {}
    if not order_book_ids:
        return resolved
    all_names = _load_all_stock_names()
    for order_book_id in order_book_ids:
        order_book_id = str(order_book_id)
        if order_book_id in all_names:
            resolved[order_book_id] = all_names[order_book_id]
    return resolved


@lru_cache(maxsize=1)
def _load_all_stock_names() -> dict[str, str]:
    """加载全市场股票名映射，并在单进程内复用。"""
    facade = DataFacade()
    instruments = facade.all_instruments(type="CS")
    if instruments is None or len(instruments) == 0:
        return {}
    if "order_book_id" not in instruments.columns:
        return {}

    name_column = None
    for candidate in ("symbol", "display_name", "name"):
        if candidate in instruments.columns:
            name_column = candidate
            break
    if name_column is None:
        return {}

    name_frame = instruments.loc[:, ["order_book_id", name_column]].dropna()
    name_frame = name_frame.drop_duplicates(subset=["order_book_id"], keep="first")
    return {
        str(row["order_book_id"]): str(row[name_column])
        for _, row in name_frame.iterrows()
    }


def _build_result_interpretation(recommendations: list[dict]) -> list[str]:
    """在表格前直接解释胜率与收益分布，减少人工再解读。"""
    if not recommendations:
        return []

    first_item = recommendations[0]
    win_rate = float(first_item.get("win_rate", 0.0))
    mean_return = float(first_item.get("mean_return", 0.0))
    median_return = float(first_item.get("median_return", 0.0))
    lines = ["结果解读:"]

    if mean_return > 0 and median_return < 0:
        lines.append("  当前 top 桶更像右偏收益排序器，不是高胜率信号。")
        lines.append(
            "  胜率 {}，均值收益 {}，中位收益 {}。".format(
                _format_percent(win_rate, digits=1),
                _format_percent(mean_return, digits=2, signed=True),
                _format_percent(median_return, digits=2, signed=True),
            )
        )
        lines.append("  这通常意味着多数样本小亏小赚，少数大涨样本把均值拉正。")
        return lines

    lines.append(
        "  当前 top 桶历史统计: 胜率 {}，均值收益 {}，中位收益 {}。".format(
            _format_percent(win_rate, digits=1),
            _format_percent(mean_return, digits=2, signed=True),
            _format_percent(median_return, digits=2, signed=True),
        )
    )
    return lines


def _render_recommendation_table(recommendations: list[dict], stock_name_map: dict[str, str]) -> str:
    """使用显示宽度感知的终端表格，解决中文列错位问题。"""
    columns = [
        {"key": "order_book_id", "title": "代码", "align": "left"},
        {"key": "stock_name", "title": "股票名", "align": "left"},
        {"key": "rank", "title": "排名", "align": "right"},
        {"key": "percentile", "title": "分位", "align": "right"},
        {"key": "win_rate", "title": "胜率", "align": "right"},
        {"key": "mean_return", "title": "均值收益", "align": "right"},
        {"key": "median_return", "title": "中位收益", "align": "right"},
        {"key": "return_range", "title": "P25~P75", "align": "right"},
        {"key": "sample_count", "title": "样本数", "align": "right"},
    ]

    rows = []
    for item in recommendations:
        rows.append(
            {
                "order_book_id": str(item.get("order_book_id", "-")),
                "stock_name": stock_name_map.get(str(item.get("order_book_id")), "-"),
                "rank": str(int(item.get("rank", 0))),
                "percentile": _format_percent(item.get("percentile"), digits=1),
                "win_rate": _format_percent(item.get("win_rate"), digits=1),
                "mean_return": _format_percent(item.get("mean_return"), digits=2, signed=True),
                "median_return": _format_percent(item.get("median_return"), digits=2, signed=True),
                "return_range": _format_return_range(item.get("return_quantile_range", {})),
                "sample_count": _format_integer(item.get("sample_count")),
            }
        )

    widths = {}
    for column in columns:
        widths[column["key"]] = max(
            [_display_width(column["title"])] +
            [_display_width(row[column["key"]]) for row in rows]
        )

    lines = []
    header_cells = [
        _pad_text(column["title"], widths[column["key"]], align=column["align"])
        for column in columns
    ]
    lines.append("  ".join(header_cells))

    for row in rows:
        formatted_cells = []
        for column in columns:
            raw_text = row[column["key"]]
            padded_text = _pad_text(raw_text, widths[column["key"]], align=column["align"])
            formatted_cells.append(_style_cell(column["key"], raw_text, padded_text))
        lines.append("  ".join(formatted_cells))
    return "\n".join(lines)


def _format_return_range(return_quantile_range: dict) -> str:
    """把四分位区间格式化成紧凑百分比字符串。"""
    low = _format_percent(return_quantile_range.get("p25"), digits=2, signed=True)
    high = _format_percent(return_quantile_range.get("p75"), digits=2, signed=True)
    return "{}~{}".format(low, high)


def _format_percent(value, *, digits: int = 1, signed: bool = False) -> str:
    """统一百分比显示，避免长小数破坏可读性。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    format_spec = "{:+." + str(int(digits)) + "%}" if signed else "{:." + str(int(digits)) + "%}"
    return format_spec.format(number)


def _format_integer(value) -> str:
    """统一样本数显示，加入千分位分隔。"""
    try:
        return "{:,}".format(int(value))
    except (TypeError, ValueError):
        return "-"


def _format_status(status) -> str:
    """在支持颜色的终端里强调运行状态。"""
    text = str(status)
    if text == "ok":
        return _colorize(text, ANSI_GREEN)
    if text == "stopped":
        return _colorize(text, ANSI_YELLOW)
    return text


def _format_gate_level(gate_level) -> str:
    """在支持颜色的终端里强调 gate 等级。"""
    text = str(gate_level)
    if text == "default_live":
        return _colorize(text, ANSI_GREEN)
    if text == "canary_live":
        return _colorize(text, ANSI_YELLOW)
    return text


def _style_cell(column_key: str, raw_text: str, padded_text: str) -> str:
    """对关键收益列做颜色增强，同时保持列宽不乱。"""
    if column_key in {"mean_return", "median_return"}:
        return _colorize_signed_text(raw_text, padded_text)
    if column_key == "win_rate":
        try:
            ratio = float(str(raw_text).rstrip("%")) / 100.0
        except ValueError:
            return padded_text
        if ratio >= 0.5:
            return _colorize(padded_text, ANSI_GREEN)
        return _colorize(padded_text, ANSI_RED)
    return padded_text


def _colorize_signed_text(raw_text: str, padded_text: str) -> str:
    """按正负号给收益着色。"""
    if raw_text.startswith("+"):
        return _colorize(padded_text, ANSI_GREEN)
    if raw_text.startswith("-"):
        return _colorize(padded_text, ANSI_RED)
    return padded_text


def _colorize(text: str, color_code: str) -> str:
    """按终端能力决定是否输出 ANSI 颜色。"""
    if not _supports_color_output():
        return text
    return "{}{}{}".format(color_code, text, ANSI_RESET)


def _supports_color_output() -> bool:
    """只在交互终端里启用颜色，避免污染重定向输出。"""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("CLICOLOR_FORCE") == "1":
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _pad_text(text: str, width: int, *, align: str) -> str:
    """按终端显示宽度补齐字符串，兼容中文全角字符。"""
    text = str(text)
    display_width = _display_width(text)
    padding = max(int(width) - display_width, 0)
    if align == "right":
        return " " * padding + text
    return text + " " * padding


def _display_width(text: str) -> int:
    """计算终端显示宽度，忽略 ANSI 控制符。"""
    plain_text = ANSI_PATTERN.sub("", str(text))
    width = 0
    for char in plain_text:
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


if __name__ == "__main__":
    raise SystemExit(main())
