"""
策略滚动打分器 - 对输入的策略文件进行滚动窗口回测并综合评分。

用法:
    python -m skyeye.evaluation.rolling_score.cli strategy.py
    python -m skyeye.evaluation.rolling_score.cli strategy.py --cash 200000
    python -m skyeye.evaluation.rolling_score.cli strategy.py --log mid
    python -m skyeye.evaluation.rolling_score.cli --search "平安"
"""
import argparse
import datetime
import math
import os
import pickle
import re
import sys
import time
import warnings
import unicodedata
from collections import OrderedDict

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from rqalpha import run
from skyeye.evaluation.rolling_score.report import build_strategy_card_lines, render_box
from skyeye.products.registry import find_strategy_spec_by_file
from skyeye.data import DataFacade

# ============================================================
# 1. 常量与配置
# ============================================================

# 13个指标配置：多项式系数(c0~c4)、锚点范围、外推斜率、是否取反
# coeffs 顺序: [c0, c1, c2, c3, c4]，即 score = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4
INDICATORS = {
    # --- 收益类 ---
    "total_returns": {
        "coeffs": [0.0, 385.873016, 36.772487, -5579.365079, 16243.386243],
        "x_min": -0.15, "x_max": 0.3,
        "slope_left": 144.4444, "slope_right": 655.7937,
        "negate": False,
    },
    "annualized_returns": {
        "coeffs": [-17.142857, 306.190476, 1033.333333, -6476.190476, 9523.809524],
        "x_min": -0.1, "x_max": 0.4,
        "slope_left": 130.0, "slope_right": 462.381,
        "negate": False,
    },
    "excess_annual_returns": {
        "coeffs": [19.312452, 329.895595, -690.603514, 8041.762159, -13750.954927],
        "x_min": -0.1, "x_max": 0.2,
        "slope_left": 764.273, "slope_right": 578.6351,
        "negate": False,
    },
    # --- 风险类（negate=True，越低越好，输入时取反 x_input = -x_raw）---
    "max_drawdown": {
        "coeffs": [166.875, 1702.916667, 8391.666667, 22833.333333, 23333.333333],
        "x_min": -0.35, "x_max": -0.05,
        "slope_left": 218.3333, "slope_right": 1023.3333,
        "negate": True,
    },
    "max_drawdown_duration_days": {
        "coeffs": [549.2455242967089, 9.365380789997912, 0.06447013829687052,
                   0.00019543904518330954, 2.1194468125416533e-07],
        "x_min": -350, "x_max": -90,
        "slope_left": 0.25, "slope_right": 1.8919,
        "negate": True,
    },
    "excess_max_drawdown": {
        "coeffs": [229.705882, 6249.509804, 76062.091503, 419934.640523, 816993.464052],
        "x_min": -0.2, "x_max": -0.03,
        "slope_left": 382.3529, "slope_right": 2731.3725,
        "negate": True,
    },
    "tracking_error": {
        "coeffs": [160.0, 1423.333333, 4633.333333, 2666.666667, -13333.333333],
        "x_min": -0.3, "x_max": -0.05,
        "slope_left": 803.3333, "slope_right": 986.6667,
        "negate": True,
    },
    # --- 风险调整收益类 ---
    "sharpe": {
        "coeffs": [0.0, 70.434783, -80.942029, 55.652174, -10.144928],
        "x_min": -0.3, "x_max": 2.0,
        "slope_left": 135.1217, "slope_right": 89.8551,
        "negate": False,
    },
    "sortino": {
        "coeffs": [0.0, 73.145743, -77.676768, 37.822671, -5.451339],
        "x_min": -0.3, "x_max": 3.0,
        "slope_left": 130.5527, "slope_right": 39.5527,
        "negate": False,
    },
    "information_ratio": {
        "coeffs": [0.0, 76.260684, -59.850427, 59.496676, -15.906933],
        "x_min": -0.3, "x_max": 1.5,
        "slope_left": 129.953, "slope_right": 83.5684,
        "negate": False,
    },
    # --- 交易统计类 ---
    "win_rate": {
        "coeffs": [-1920.0, 13426.666667, -34366.666667, 37333.333333, -13333.333333],
        "x_min": 0.35, "x_max": 0.6,
        "slope_left": 803.3333, "slope_right": 986.6667,
        "negate": False,
    },
    "profit_loss_rate": {
        "coeffs": [-121.671068, 253.344465, -164.707265, 52.293645, -5.776856],
        "x_min": 0.5, "x_max": 3.0,
        "slope_left": 124.969, "slope_right": 53.1288,
        "negate": False,
    },
    # --- 月度指标 ---
    "monthly_excess_win_rate": {
        "coeffs": [-111.666667, 215.15873, 1013.492063, -2825.396825, 2222.222222],
        "x_min": 0.25, "x_max": 0.7,
        "slope_left": 331.0317, "slope_right": 529.6032,
        "negate": False,
    },
}

# 13个指标的最终权重（加总 = 100%）
WEIGHTS = {
    "total_returns": 0.12,
    "annualized_returns": 0.12,
    "excess_annual_returns": 0.06,
    "max_drawdown": 0.10,
    "max_drawdown_duration_days": 0.075,
    "excess_max_drawdown": 0.0375,
    "tracking_error": 0.0375,
    "sharpe": 0.10,
    "sortino": 0.10,
    "information_ratio": 0.05,
    "win_rate": 0.04,
    "profit_loss_rate": 0.06,
    "monthly_excess_win_rate": 0.10,
}

LAMBDA = 0.03           # 时间衰减参数
BENCHMARK = "000300.XSHG"
MIN_WINDOW_ACTIVE_DAYS = 60
MIN_WINDOW_ACTIVE_RATIO = 0.25
MAX_FIRST_ACTIVE_DELAY_DAYS = 60

# 核心展示指标（用于输出行"核心指标"）
CORE_INDICATORS = ["annualized_returns", "max_drawdown", "sharpe", "win_rate"]

# 评级体系（E/M+/M/M-/I），灵感来自字节跳动绩效评级
# 颜色: E=亮绿, M+=绿, M=黄, M-=红, I=亮红
def rating_color(score, use_color=True):
    """根据分数返回 ANSI 颜色码（不含标签文本）"""
    if not use_color:
        return ""
    if score >= 60:
        return "\033[92m"       # 亮绿 — 卓越
    elif score >= 30:
        return "\033[96m"       # 亮青 — 稳健
    elif score >= 0:
        return "\033[93m"       # 亮黄 — 警示
    elif score >= -20:
        return "\033[91m"       # 亮红 — 危险
    else:
        return "\033[41;97m"    # 红底白字 — 淘汰


def rating_label(score, use_color=True):
    """根据分数返回带颜色的评级标签"""
    if score >= 60:
        grade = "E"
    elif score >= 30:
        grade = "M+"
    elif score >= 0:
        grade = "M"
    elif score >= -20:
        grade = "M-"
    else:
        grade = "I"
    if use_color:
        color = rating_color(score, True)
        return f"{color}{grade}\033[0m"
    return grade


def indicator_color(indicator_name, raw_value, use_color=True):
    """将指标原始值通过评分函数映射到颜色码"""
    score = score_indicator(indicator_name, raw_value)
    return rating_color(score, use_color)


def rating_legend(use_color=True):
    """生成带颜色的评级图例"""
    RST = "\033[0m" if use_color else ""
    items = [
        ("\033[92m", "E",  ">=60 优秀"),
        ("\033[96m", "M+", ">=30 良好"),
        ("\033[93m", "M",  ">=0 一般"),
        ("\033[91m", "M-", ">=-20 较差"),
        ("\033[41;97m", "I",  "<-20 很差"),
    ]
    parts = []
    for color, grade, desc in items:
        if use_color:
            parts.append(f"{color}{grade}{RST}({desc})")
        else:
            parts.append(f"{grade}({desc})")
    return "  评级说明: " + "  ".join(parts)

# ============================================================
# 2. 评分函数
# ============================================================

def score_indicator(name, value):
    """对单个指标评分，返回浮点分数（不封顶不封底）"""
    cfg = INDICATORS[name]
    c0, c1, c2, c3, c4 = cfg["coeffs"]
    x_min, x_max = cfg["x_min"], cfg["x_max"]
    slope_left, slope_right = cfg["slope_left"], cfg["slope_right"]

    x = -value if cfg["negate"] else value

    if x < x_min:
        y_min = c0 + c1 * x_min + c2 * x_min**2 + c3 * x_min**3 + c4 * x_min**4
        return y_min + slope_left * (x - x_min)
    elif x > x_max:
        y_max = c0 + c1 * x_max + c2 * x_max**2 + c3 * x_max**3 + c4 * x_max**4
        return y_max + slope_right * (x - x_max)
    else:
        return c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4

# ============================================================
# 3. 窗口分数
# ============================================================

def score_window(summary):
    """计算单个窗口的加权综合分"""
    total = 0.0
    for name, weight in WEIGHTS.items():
        raw = summary.get(name)
        if pd.isna(raw):
            s = 0.0  # 无交易时无信号贡献，与 total_returns(0)=0、sharpe(0)=0 保持一致
        else:
            s = score_indicator(name, raw)
        total += s * weight
    return total

# ============================================================
# 4. 滚动窗口回测
# ============================================================

def generate_windows():
    """生成37个滚动窗口的起止日期"""
    base_start = datetime.date(2016, 2, 1)
    windows = []
    for i in range(37):
        start = base_start + relativedelta(months=3 * i)
        end = start + relativedelta(years=1) - relativedelta(days=1)
        windows.append((start, end))
    return windows


def extract_strategy_floor_date(strategy_file):
    with open(strategy_file, "r", encoding="utf-8") as f:
        source = f.read()
    match = re.search(
        r'^\s*(?:ROLLING_SCORE_START_DATE|STRATEGY_SCORER_START_DATE)\s*=\s*[\'"](\d{4}-\d{2}-\d{2})[\'"]\s*$',
        source,
        re.MULTILINE,
    )
    if not match:
        return None
    try:
        return datetime.datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_window_arg(window_str):
    """解析 --window 参数，返回窗口编号列表（1-based）

    支持格式:
        37       → [37]
        35-37    → [35, 36, 37]
        1,10,37  → [1, 10, 37]
    """
    nums = set()
    for part in window_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a.strip()), int(b.strip())
            nums.update(range(a, b + 1))
        else:
            nums.add(int(part))
    # 校验范围
    for n in nums:
        if n < 1 or n > 37:
            print(f"错误: 窗口编号 {n} 超出范围 (1-37)")
            sys.exit(1)
    return sorted(nums)


def parse_cli_scalar(value):
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in ("none", "null"):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"[+-]?\d+", stripped):
        return int(stripped)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", stripped):
        return float(stripped)
    return value


def parse_mod_config_args(items):
    mod_configs, _ = parse_runtime_config_args(items)
    return mod_configs


def parse_runtime_config_args(items):
    mod_configs = OrderedDict()
    extra_config = OrderedDict()
    for key, raw_value in items or []:
        if "." not in key:
            print("错误: --mod-config 键必须是 <mod>.<field>，例如 dividend_scorer.prior_blend")
            sys.exit(1)
        scope_name, field_name = key.split(".", 1)
        if not scope_name or not field_name:
            print("错误: --mod-config 键必须是 <mod>.<field>，例如 dividend_scorer.prior_blend")
            sys.exit(1)
        parsed_value = parse_cli_scalar(raw_value)
        if scope_name == "extra":
            extra_config[field_name] = parsed_value
            continue
        mod_configs.setdefault(scope_name, OrderedDict())
        mod_configs[scope_name][field_name] = parsed_value
    return mod_configs, extra_config


def analyze_window_sample(start, trades, portfolio):
    """识别样本不足窗口，避免把明显失真的年化风险指标当成正常结果解读。"""
    diagnostics = {
        "total_days": 0,
        "active_days": 0,
        "active_ratio": 0.0,
        "trade_count": 0,
        "sell_count": 0,
        "first_active_date": None,
        "first_active_delay_days": None,
        "warning_text": "",
        "sparse": False,
    }

    if trades is not None and not trades.empty:
        diagnostics["trade_count"] = int(len(trades))
        if "side" in trades.columns:
            diagnostics["sell_count"] = int((trades["side"] == "SELL").sum())

    if portfolio is None or portfolio.empty or "market_value" not in portfolio.columns:
        diagnostics["warning_text"] = "组合净值明细缺失"
        diagnostics["sparse"] = True
        return diagnostics

    diagnostics["total_days"] = int(len(portfolio))
    active_mask = portfolio["market_value"].fillna(0) > 0
    diagnostics["active_days"] = int(active_mask.sum())
    diagnostics["active_ratio"] = (
        diagnostics["active_days"] / diagnostics["total_days"] if diagnostics["total_days"] else 0.0
    )

    warnings_list = []
    if diagnostics["active_days"] == 0:
        warnings_list.append("窗口内无持仓日")
    else:
        first_active_ts = portfolio.index[active_mask][0]
        first_active_date = first_active_ts.date()
        diagnostics["first_active_date"] = first_active_date
        diagnostics["first_active_delay_days"] = (first_active_date - start).days

        if diagnostics["first_active_delay_days"] > MAX_FIRST_ACTIVE_DELAY_DAYS:
            warnings_list.append("首次建仓滞后 {} 天".format(diagnostics["first_active_delay_days"]))
        if diagnostics["active_days"] < MIN_WINDOW_ACTIVE_DAYS:
            warnings_list.append(
                "持仓仅 {}/{} 个交易日".format(diagnostics["active_days"], diagnostics["total_days"])
            )
        elif diagnostics["active_ratio"] < MIN_WINDOW_ACTIVE_RATIO:
            warnings_list.append("持仓覆盖率仅 {:.0%}".format(diagnostics["active_ratio"]))

    diagnostics["sparse"] = bool(warnings_list)
    diagnostics["warning_text"] = "；".join(warnings_list)
    return diagnostics


def run_rolling_backtests(
    strategy_file,
    cash,
    selected_indices=None,
    extra_mods=None,
    mod_configs=None,
    extra_config=None,
    benchmark_id=None,
    return_details=False,
    include_trade_details=True,
):
    """对策略文件执行滚动窗口回测，返回结果列表

    selected_indices: 要执行的窗口编号列表（1-based），None 表示全部37个
    extra_mods: 额外启用的 mod 名称列表
    mod_configs: 额外的 mod 配置，格式为 {mod_name: {key: value}}
    """
    with open(strategy_file) as f:
        source_code = f.read()

    if benchmark_id is None:
        strategy_spec = find_strategy_spec_by_file(strategy_file)
        benchmark_id = getattr(strategy_spec, "benchmark", None) or BENCHMARK

    windows = generate_windows()
    if selected_indices is not None:
        run_list = [(i, windows[i - 1]) for i in selected_indices]
    else:
        run_list = [(i + 1, w) for i, w in enumerate(windows)]

    total = len(run_list)
    results = []
    failed = 0
    failed_windows = []

    for seq, (idx, (start, end)) in enumerate(run_list, 1):
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        print(f"  窗口 {idx:2d} [{seq}/{total}]: {start_str} ~ {end_str} ... ", end="", flush=True)

        config = {
            "base": {
                "start_date": start_str,
                "end_date": end_str,
                "accounts": {"stock": cash},
                "frequency": "1d",
            },
            "extra": {"log_level": "error"},
            "mod": {
                "sys_analyser": {"benchmark": benchmark_id, "plot": False},
                "sys_progress": {"enabled": False},
            },
        }
        config["extra"].update(extra_config or {})
        enabled_mods = []
        for mod_name in extra_mods or []:
            if mod_name not in enabled_mods:
                enabled_mods.append(mod_name)
        for mod_name in (mod_configs or {}).keys():
            if mod_name not in enabled_mods:
                enabled_mods.append(mod_name)
        for mod_name in enabled_mods:
            config["mod"][mod_name] = {"enabled": True}
        for mod_name, mod_config in (mod_configs or {}).items():
            config["mod"].setdefault(mod_name, {"enabled": True})
            config["mod"][mod_name]["enabled"] = True
            config["mod"][mod_name].update(mod_config)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run(config, source_code=source_code)

            if result is None:
                print("失败(返回None)")
                failed += 1
                failed_windows.append({
                    "idx": idx,
                    "start": start_str,
                    "end": end_str,
                    "error": "result_none",
                })
                continue

            summary = result["sys_analyser"]["summary"]
            trades = result["sys_analyser"]["trades"]
            portfolio = result["sys_analyser"].get("portfolio")
            sample_diagnostics = analyze_window_sample(start, trades, portfolio)
            # 把 sys_analyser 之外的 mod 返回透传出来，便于上层做策略诊断。
            extra_results = {
                str(key): value
                for key, value in dict(result).items()
                if str(key) != "sys_analyser"
            }

            window_score = score_window(summary)
            use_color = sys.stdout.isatty()
            sc = rating_color(window_score, use_color)
            rst = "\033[0m" if use_color else ""
            message = f"得分 {sc}{window_score:.1f}{rst}"
            if sample_diagnostics["warning_text"]:
                message += "  [样本不足: {}]".format(sample_diagnostics["warning_text"])
            print(message)

            window_entry = {
                "idx": idx,
                "start": start,
                "end": end,
                "summary": summary,
                "score": window_score,
            }
            if include_trade_details:
                window_entry["trades"] = trades
                window_entry["sample_diagnostics"] = sample_diagnostics
                window_entry["mod_results"] = extra_results
            results.append(window_entry)
        except Exception as e:
            print(f"异常: {e}")
            failed += 1
            failed_windows.append({
                "idx": idx,
                "start": start_str,
                "end": end_str,
                "error": str(e),
            })

    if failed > 0:
        print(f"\n  警告: {failed} 个窗口回测失败")

    if return_details:
        return {
            "windows": results,
            "failed_windows": failed_windows,
            "total_windows": total,
            "successful_windows": len(results),
        }

    return results

# ============================================================
# 5. 交易日志
# ============================================================

def flatten_trades(trades_df):
    """将 trades_df 展开为逐笔交易流水，跟踪每只股票的持仓和累计盈亏"""
    if trades_df is None or trades_df.empty:
        return []

    records = []
    # 按股票分组，维护每只股票的持仓成本
    holdings = {}  # {order_book_id: {"quantity": int, "cost": float(总成本)}}
    total_realized_pnl = 0.0  # 组合级别累计盈亏（所有股票）

    trades_df = trades_df.sort_index()

    for _, row in trades_df.iterrows():
        trade_dt = row.name
        ob_id = row["order_book_id"]
        symbol = row["symbol"]
        side = row["side"]
        price = row["last_price"]
        qty = row["last_quantity"]
        transaction_cost = float(row.get("transaction_cost", 0.0) or 0.0)

        if ob_id not in holdings:
            holdings[ob_id] = {"quantity": 0, "cost": 0.0, "realized_pnl": 0.0}
        h = holdings[ob_id]

        if side == "BUY":
            # trade 记录来自 sys_analyser 时通常带有 transaction_cost（佣金+税费等）
            # 买入时将交易成本计入持仓成本，否则后续均价/已实现盈亏会偏高。
            h["cost"] += price * qty + transaction_cost
            h["quantity"] += qty
            records.append({
                "order_book_id": ob_id,
                "symbol": symbol,
                "side": "买入",
                "datetime": trade_dt,
                "price": price,
                "quantity": qty,
                "amount": price * qty,
                "pnl": None,
                "transaction_cost": transaction_cost,
                "holding_qty": h["quantity"],
                "holding_cost": h["cost"],
                "realized_pnl": h["realized_pnl"],
                "total_realized_pnl": total_realized_pnl,
            })
        elif side == "SELL":
            sell_qty = min(qty, h["quantity"])
            avg_cost = h["cost"] / h["quantity"] if h["quantity"] > 0 else 0
            # 卖出时从已实现盈亏中扣除卖出交易成本。
            # 若 sell_qty < qty（异常数据/超卖保护），按数量比例分摊本笔交易成本。
            sell_cost = transaction_cost
            if qty:
                sell_cost = transaction_cost * (sell_qty / qty)
            pnl = (price - avg_cost) * sell_qty - sell_cost
            h["realized_pnl"] += pnl
            total_realized_pnl += pnl
            h["cost"] -= avg_cost * sell_qty
            h["quantity"] -= sell_qty
            records.append({
                "order_book_id": ob_id,
                "symbol": symbol,
                "side": "卖出",
                "datetime": trade_dt,
                "price": price,
                "quantity": sell_qty,
                "amount": price * sell_qty,
                "pnl": pnl,
                "transaction_cost": sell_cost,
                "holding_qty": h["quantity"],
                "holding_cost": h["cost"],
                "realized_pnl": h["realized_pnl"],
                "total_realized_pnl": total_realized_pnl,
            })

    return records


def display_width(value):
    text = str(value)
    total = 0
    for ch in text:
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def pad_cell(value, width, align, colored_value=None):
    text = str(value)
    pad_len = width - display_width(text)
    out = text if colored_value is None else str(colored_value)
    if pad_len <= 0:
        return out
    if align == "left":
        return out + " " * pad_len
    return " " * pad_len + out


def build_trade_log(window_results, level, cash):
    """根据日志级别筛选窗口，逐笔输出交易流水"""
    if level == "low":
        selected = window_results[-1:]
    elif level == "mid":
        selected = window_results[-4:]
    else:
        selected = window_results

    all_records = []
    for w in selected:
        idx = w["idx"]
        records = flatten_trades(w["trades"])
        for r in records:
            r["window"] = f"#{idx}"
        all_records.extend(records)

    if not all_records:
        print("\n【交易日志】无交易记录")
        return

    # 汇总统计
    sells = [r for r in all_records if r["side"] == "卖出"]
    buys = [r for r in all_records if r["side"] == "买入"]
    n_sell = len(sells)
    n_win = sum(1 for r in sells if r["pnl"] is not None and r["pnl"] > 0)
    win_pct = n_win / n_sell * 100 if n_sell > 0 else 0
    unique_sell_dates = len({r["datetime"].date() for r in sells}) if sells else 0

    level_desc = {"low": "最近1个窗口", "mid": "最近4个窗口", "high": "全部窗口"}
    use_color_hdr = sys.stdout.isatty()
    if use_color_hdr and n_sell > 0:
        wp_color = "\033[32m" if win_pct >= 50 else "\033[31m"
        wp_rst = "\033[0m"
        wp_str = f"{wp_color}{win_pct:.1f}%{wp_rst}"
    else:
        wp_str = f"{win_pct:.1f}%"
    header = f"【交易日志】({level_desc[level]}, 买入{len(buys)}笔 卖出{n_sell}笔, 卖出胜率 {wp_str})"
    print(f"\n{header}")
    print("注: 卖出胜率只统计已平仓卖出，不包含仍持仓仓位的浮盈浮亏。")
    if unique_sell_dates and unique_sell_dates < n_sell:
        print(f"注: 卖出 {n_sell} 笔只对应 {unique_sell_dates} 个实际卖点日，滚动窗口重叠会重复统计同一天的卖出。")

    headers = [
        "#", "窗口", "方向", "证券代码", "股票名称",
        "日期", "价格", "数量", "金额",
        "盈亏", "持仓", "累计盈亏"
    ]

    # ANSI 颜色码
    GREEN = "\033[32m"
    RED = "\033[31m"
    RESET = "\033[0m"
    use_color = sys.stdout.isatty()

    def colorize(text, value):
        """根据数值正负着色"""
        if not use_color or value is None or value == 0:
            return text
        if value > 0:
            return f"{GREEN}{text}{RESET}"
        return f"{RED}{text}{RESET}"

    rows = []
    row_colors = []  # 每行的着色版本，用于输出
    def fmt_num(n):
        """数字格式化：大数用万，加千分位"""
        if abs(n) >= 10000:
            return f"{n/10000:,.1f}万"
        return f"{n:,.0f}"

    def fmt_pnl(n):
        """盈亏格式化：带正负号，大数用万"""
        if abs(n) >= 10000:
            return f"{n/10000:+,.1f}万"
        return f"{n:+,.0f}"

    for i, r in enumerate(all_records, 1):
        dt = r["datetime"].strftime("%Y-%m-%d") if hasattr(r["datetime"], "strftime") else str(r["datetime"])[:10]
        pnl_str = fmt_pnl(r['pnl']) if r["pnl"] is not None else ""
        cum_str = fmt_pnl(r['total_realized_pnl'])
        side_str = r["side"]
        plain_row = [
            str(i), r["window"], side_str, r["order_book_id"], r["symbol"],
            dt, f"{r['price']:.2f}", fmt_num(r["quantity"]), fmt_num(r["amount"]),
            pnl_str, fmt_num(r["holding_qty"]), cum_str
        ]
        colored_row = list(plain_row)
        # 方向列着色
        if use_color:
            colored_row[2] = f"{RED}{side_str}{RESET}" if side_str == "卖出" else f"{GREEN}{side_str}{RESET}"
        # 盈亏列着色
        colored_row[9] = colorize(pnl_str, r["pnl"])
        # 累计盈亏列着色
        colored_row[11] = colorize(cum_str, r["total_realized_pnl"])
        rows.append(plain_row)
        row_colors.append(colored_row)

    alignments = [
        "right", "left", "left", "left", "left",
        "right", "right", "right", "right",
        "right", "right", "right"
    ]

    columns = list(zip(headers, *rows)) if rows else [(h,) for h in headers]
    widths = [max(display_width(cell) for cell in col) for col in columns]

    # 边框字符
    SEP = "│"
    def make_line(left, mid, right, fill="─"):
        return left + mid.join(fill * (w + 2) for w in widths) + right

    top_line = make_line("┌", "┬", "┐")
    header_sep = make_line("├", "┼", "┤")
    bottom_line = make_line("└", "┴", "┘")

    def render_row(cells_list):
        parts = []
        for i, cell in enumerate(cells_list):
            padded = pad_cell(cell, widths[i], alignments[i])
            parts.append(f" {padded} ")
        return SEP + SEP.join(parts) + SEP

    print(top_line)
    print(render_row(headers))
    print(header_sep)
    for plain_row, colored_row in zip(rows, row_colors):
        parts = []
        for i in range(len(headers)):
            cv = colored_row[i] if colored_row[i] != plain_row[i] else None
            padded = pad_cell(plain_row[i], widths[i], alignments[i], colored_value=cv)
            parts.append(f" {padded} ")
        print(SEP + SEP.join(parts) + SEP)
    print(bottom_line)

    # 汇总: 按窗口分别输出（每个窗口是独立的100万回测）
    for w in selected:
        idx = w["idx"]
        w_records = [r for r in all_records if r["window"] == f"#{idx}"]
        if not w_records:
            continue

        summary = w["summary"]
        engine_total_returns = summary.get("total_returns", 0)
        total_asset = cash * (1 + engine_total_returns)
        total_pnl = total_asset - cash
        total_ret_pct = engine_total_returns * 100

        # 该窗口的已实现盈亏（最后一条记录的组合级别累计）
        realized_pnl = w_records[-1]["total_realized_pnl"]

        # 该窗口的期末持仓
        w_holdings = {}
        for r in w_records:
            ob_id = r["order_book_id"]
            w_holdings[ob_id] = {"qty": r["holding_qty"], "symbol": r["symbol"], "price": r["price"]}
        w_holding_stocks = {k: v for k, v in w_holdings.items() if v["qty"] > 0}

        # 持仓成本（按股票分别获取）
        w_holding_costs = {}
        for ob_id in w_holding_stocks:
            for r in reversed(w_records):
                if r["order_book_id"] == ob_id:
                    w_holding_costs[ob_id] = r.get("holding_cost", 0)
                    break

        # 窗口标题（多窗口时才显示）
        if len(selected) > 1:
            print(f"\n  --- 窗口 #{idx} ({w['start']} ~ {w['end']}) ---")
        else:
            print()

        # 总资产
        asset_text = f"{total_asset:,.0f}"
        total_pnl_text = f"{total_pnl:+,.0f}"
        total_pnl_colored = colorize(total_pnl_text, total_pnl)
        total_ret_text = f"{total_ret_pct:+.1f}%"
        total_ret_colored = colorize(total_ret_text, total_ret_pct)
        print(f"  总资产: {asset_text}  总盈亏: {total_pnl_colored}  总收益率: {total_ret_colored}")

        # 落袋/浮盈
        realized_text = f"{realized_pnl:+,.0f}"
        realized_colored = colorize(realized_text, realized_pnl)
        if w_holding_stocks:
            unrealized_pnl = total_pnl - realized_pnl
            unrealized_text = f"{unrealized_pnl:+,.0f}"
            unrealized_colored = colorize(unrealized_text, unrealized_pnl)
            unrealized_label = "浮盈" if unrealized_pnl >= 0 else "浮亏"
            realized_label = "落袋为安" if realized_pnl >= 0 else "落袋亏损"
            realized_label_colored = colorize(realized_label, realized_pnl)
            unrealized_label_colored = colorize(unrealized_label, unrealized_pnl)
            print(f"  {realized_label_colored}: {realized_colored}  {unrealized_label_colored}: {unrealized_colored}")
        else:
            realized_label = "落袋为安" if realized_pnl >= 0 else "落袋亏损"
            realized_label_colored = colorize(realized_label, realized_pnl)
            print(f"  {realized_label_colored}: {realized_colored}")

        # 期末持仓
        if w_holding_stocks:
            end_date = w["end"]
            parts = []
            for k, v in w_holding_stocks.items():
                qty = int(v['qty'])
                cost = w_holding_costs.get(k, 0)
                avg_price = cost / qty if qty > 0 else 0
                # 从日线数据获取期末收盘价计算市值（向前查10天，防止end_date是非交易日）
                bars = read_daily_bars(k, end_date - datetime.timedelta(days=10), end_date)
                if bars is not None and not bars.empty:
                    last_price = bars["close"].iloc[-1]
                else:
                    last_price = v["price"]  # 回退到最后成交价
                mv = qty * last_price
                parts.append(f"{v['symbol']}({k}) {qty}股 均价{avg_price:.2f} 成本{cost:,.0f} 市值{mv:,.0f}")
            print(f"  期末持仓: {', '.join(parts)}")
        else:
            print(f"  期末持仓: 无（已清仓）")

# ============================================================
# 6. 季度网格投影
# ============================================================

def get_covered_quarters(start_date, end_date):
    """返回窗口 [start_date, end_date] 覆盖的所有季度列表"""
    quarters = []
    q_start = (start_date.year, (start_date.month - 1) // 3 + 1)
    q_end = (end_date.year, (end_date.month - 1) // 3 + 1)
    year, q = q_start
    while (year, q) <= q_end:
        quarters.append((year, q))
        q += 1
        if q > 4:
            q = 1
            year += 1
    return quarters


def project_to_quarters(window_results):
    """将窗口分数和原始指标投影到季度网格"""
    quarter_scores = {}   # {(year,q): [scores]}
    quarter_raw = {}      # {(year,q): {indicator: [values]}}

    for w in window_results:
        quarters = get_covered_quarters(w["start"], w["end"])
        for qk in quarters:
            quarter_scores.setdefault(qk, []).append(w["score"])
            quarter_raw.setdefault(qk, {ind: [] for ind in CORE_INDICATORS})
            for ind in CORE_INDICATORS:
                val = w["summary"].get(ind)
                if not pd.isna(val):
                    quarter_raw[qk][ind].append(val)

    # 计算季度平均值
    quarterly_scores = OrderedDict()
    quarterly_raw_indicators = OrderedDict()
    for qk in sorted(quarter_scores.keys()):
        quarterly_scores[qk] = np.mean(quarter_scores[qk])
        raw_avg = {}
        for ind in CORE_INDICATORS:
            vals = quarter_raw[qk][ind]
            raw_avg[ind] = np.mean(vals) if vals else 0.0
        quarterly_raw_indicators[qk] = raw_avg

    return quarterly_scores, quarterly_raw_indicators

# ============================================================
# 7. 综合得分与核心指标
# ============================================================

def compute_composite_score(quarterly_scores, quarterly_raw_indicators):
    """时间衰减加权计算综合得分和核心指标均值"""
    sorted_quarters = sorted(quarterly_scores.keys(), reverse=True)  # 最近在前
    n = len(sorted_quarters)

    weights = [math.exp(-LAMBDA * i) for i in range(n)]
    total_weight = sum(weights)

    composite = sum(quarterly_scores[sorted_quarters[i]] * weights[i] for i in range(n)) / total_weight

    core_weighted = {}
    for ind in CORE_INDICATORS:
        core_weighted[ind] = sum(
            quarterly_raw_indicators[sorted_quarters[i]][ind] * weights[i]
            for i in range(n)
        ) / total_weight

    return composite, core_weighted

# ============================================================
# 8. 稳定性评分
# ============================================================

def compute_stability_score(quarterly_scores):
    """计算稳定性评分（0~100）"""
    scores = list(quarterly_scores.values())
    if not scores:
        return 0.0

    mean_s = np.mean(scores)
    std_s = np.std(scores, ddof=0)

    # 维度1: CV（变异系数），权重 50%
    if abs(mean_s) < 1e-6:
        cv = float("inf")
    else:
        cv = std_s / abs(mean_s)

    if cv >= 0.5:
        cv_score = 0
    elif cv >= 0.25:
        cv_score = (0.5 - cv) / (0.5 - 0.25) * 50
    elif cv >= 0.1:
        cv_score = 50 + (0.25 - cv) / (0.25 - 0.1) * 50
    else:
        cv_score = 100

    # 维度2: 最差季度得分，权重 30%
    worst = min(scores)
    if worst <= -10:
        worst_score = 0
    elif worst <= 25:
        worst_score = (worst - (-10)) / (25 - (-10)) * 50
    elif worst <= 50:
        worst_score = 50 + (worst - 25) / (50 - 25) * 50
    else:
        worst_score = 100

    # 维度3: 最长连续低分（<30分）季度数，权重 20%
    low_threshold = 30
    max_consec = 0
    cur_consec = 0
    for s in scores:
        if s < low_threshold:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    if max_consec >= 5:
        consec_score = 0
    elif max_consec >= 2:
        consec_score = (5 - max_consec) / (5 - 2) * 50
    else:
        consec_score = 50 + (2 - max_consec) / (2 - 0) * 50

    return 0.5 * cv_score + 0.3 * worst_score + 0.2 * consec_score

# ============================================================
# 9. 市场环境适应性
# ============================================================

def read_daily_bars(order_book_id, start_date, end_date):
    df = DataFacade().get_daily_bars(order_book_id, start_date, end_date, fields=["close"], adjust_type="none")
    if df is None or df.empty:
        return None
    return _normalize_single_instrument_bars(df, order_book_id)


def plot_trades(window_results):
    """为每个窗口绘制价格走势 + 买卖标注图，保存为 PNG"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    for w in window_results:
        records = flatten_trades(w["trades"])
        if not records:
            continue

        # 获取该窗口交易的所有股票
        ob_ids = list(dict.fromkeys(r["order_book_id"] for r in records))

        for ob_id in ob_ids:
            # 读取日线数据
            df = read_daily_bars(ob_id, w["start"], w["end"])
            if df is None or df.empty:
                print(f"  警告: 无法读取 {ob_id} 的日线数据，跳过绘图")
                continue

            # 筛选该股票的交易记录
            stock_records = [r for r in records if r["order_book_id"] == ob_id]
            buys = [r for r in stock_records if r["side"] == "买入"]
            sells = [r for r in stock_records if r["side"] == "卖出"]
            symbol = stock_records[0]["symbol"]

            # 绘图
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(df.index, df["close"], color="#555555", linewidth=1, label="收盘价")

            # 买入标注
            if buys:
                buy_dates = [r["datetime"] for r in buys]
                buy_prices = [r["price"] for r in buys]
                ax.scatter(buy_dates, buy_prices, marker="^", color="#2ca02c",
                           s=120, zorder=5, label=f"买入({len(buys)}笔)")
                for r in buys:
                    ax.annotate(f"{r['price']:.2f}", (r["datetime"], r["price"]),
                                textcoords="offset points", xytext=(0, 10), ha="center",
                                fontsize=8, color="#2ca02c")

            # 卖出标注
            if sells:
                sell_dates = [r["datetime"] for r in sells]
                sell_prices = [r["price"] for r in sells]
                ax.scatter(sell_dates, sell_prices, marker="v", color="#d62728",
                           s=120, zorder=5, label=f"卖出({len(sells)}笔)")
                for r in sells:
                    ax.annotate(f"{r['price']:.2f}", (r["datetime"], r["price"]),
                                textcoords="offset points", xytext=(0, -14), ha="center",
                                fontsize=8, color="#d62728")

            # 格式化
            idx = w["idx"]
            ax.set_title(f"窗口 #{idx}  {symbol}({ob_id})  得分 {w['score']:.1f}", fontsize=14)
            ax.set_ylabel("价格")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            fig.autofmt_xdate(rotation=30)
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = f"trade_w{idx}_{ob_id.replace('.', '_')}.png"
            fig.savefig(filename, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  图表已保存: {os.path.basename(filename)}")


def _normalize_single_instrument_bars(df, order_book_id=None):
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        if order_book_id is not None and "order_book_id" in df.index.names:
            try:
                df = df.xs(order_book_id, level="order_book_id")
            except KeyError:
                pass
        if isinstance(df.index, pd.MultiIndex):
            if "date" in df.index.names:
                levels_to_drop = [level for level in df.index.names if level != "date"]
                if levels_to_drop:
                    df = df.reset_index(level=levels_to_drop, drop=True)
            else:
                df = df.reset_index(level=list(range(df.index.nlevels - 1)), drop=True)
    if "order_book_id" in df.columns:
        df = df.drop(columns=["order_book_id"])
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.name = "date"
    return df


def get_benchmark_quarterly_returns():
    # Keep the benchmark slice inside pandas' timestamp bounds. The scoring
    # windows are historical, so using today's date as the upper bound is
    # sufficient and avoids out-of-bounds datetime parsing.
    benchmark_end = datetime.date.today().strftime("%Y-%m-%d")
    df = DataFacade().get_daily_bars(
        "000300.XSHG",
        "2000-01-01",
        benchmark_end,
        fields=["close"],
        adjust_type="none",
    )
    if df is None or df.empty:
        return {}
    df = _normalize_single_instrument_bars(df, "000300.XSHG")
    quarterly_close = df["close"].resample("QE").last()
    quarterly_returns = quarterly_close.pct_change().dropna()
    result = {}
    for date, ret in quarterly_returns.items():
        result[(date.year, (date.month - 1) // 3 + 1)] = ret
    return result


def compute_market_env_scores(quarterly_scores, benchmark_quarterly_returns):
    """按市场环境分组计算平均分"""
    env_scores = {"牛市": [], "震荡": [], "熊市": []}

    for qk, score in quarterly_scores.items():
        ret = benchmark_quarterly_returns.get(qk)
        if ret is None:
            continue
        if ret > 0.08:
            env_scores["牛市"].append(score)
        elif ret < -0.08:
            env_scores["熊市"].append(score)
        else:
            env_scores["震荡"].append(score)

    result = {}
    for env, scores in env_scores.items():
        result[env] = np.mean(scores) if scores else "N/A"
    return result


def summarize_window_results(window_results, benchmark_quarterly_returns=None):
    quarterly_scores, quarterly_raw_indicators = project_to_quarters(window_results)
    composite, core_indicators = compute_composite_score(quarterly_scores, quarterly_raw_indicators)
    stability = compute_stability_score(quarterly_scores)
    benchmark_returns = benchmark_quarterly_returns
    if benchmark_returns is None:
        benchmark_returns = get_benchmark_quarterly_returns()
    market_env = compute_market_env_scores(quarterly_scores, benchmark_returns)
    risk_alerts = detect_risk_alerts(quarterly_scores)
    risk_flags = [format_risk_alert(alert) for alert in risk_alerts]
    return {
        "quarterly_scores": quarterly_scores,
        "quarterly_raw_indicators": quarterly_raw_indicators,
        "composite": composite,
        "core_indicators": core_indicators,
        "stability": stability,
        "market_env": market_env,
        "risk_alerts": risk_alerts,
        "risk_flags": risk_flags,
        # Backward compatibility for older callers.
        "overfit_flags": risk_flags,
    }


def detect_risk_alerts(quarterly_scores):
    """检测季度分布中的稳定性与阶段性风险。

    这些信号是启发式风险提示，不等同于严格的“过拟合”判定。
    """
    scores = list(quarterly_scores.values())
    if len(scores) < 4:
        return []

    alerts = []

    # 1. 近期衰退风险
    mid = len(scores) // 2
    first_half = float(np.mean(scores[:mid]))
    second_half = float(np.mean(scores[mid:]))
    decay = first_half - second_half
    if decay >= 15:
        alerts.append({
            "code": "performance_decay",
            "title": "近期衰退风险",
            "severity": "high" if decay >= 25 else "medium",
            "detail": "前半段均分 {:.0f} → 后半段 {:.0f}，近期表现明显转弱".format(
                first_half, second_half
            ),
        })

    # 2. 收益集中风险
    best = float(max(scores))
    median = float(np.median(scores))
    gap = best - median
    top_n = min(max(3, len(scores) // 5), len(scores))
    top_mean = float(np.mean(sorted(scores, reverse=True)[:top_n]))
    top_gap = top_mean - median
    if gap >= 45 or (gap >= 30 and top_gap >= 20):
        alerts.append({
            "code": "return_concentration",
            "title": "收益集中风险",
            "severity": "high" if gap >= 60 or top_gap >= 35 else "medium",
            "detail": (
                "最佳季度 {:.0f}，中位数 {:.0f}，头部{}季度均分 {:.0f}，高分主要集中在少数时段"
            ).format(best, median, top_n, top_mean),
        })

    # 3. 波动过大风险
    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores, ddof=0))
    if abs(mean_s) < 1e-6:
        cv = float("inf")
    else:
        cv = std_s / abs(mean_s)
    if cv >= 0.55:
        alerts.append({
            "code": "score_volatility",
            "title": "波动过大风险",
            "severity": "high" if cv >= 0.8 else "medium",
            "detail": "季度得分 CV {:.2f}，波动幅度显著高于均值".format(cv),
        })

    # 4. 连续低迷风险
    low_threshold = 30
    max_consec = 0
    cur_consec = 0
    for score in scores:
        if score < low_threshold:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0
    if max_consec >= 3:
        alerts.append({
            "code": "prolonged_slump",
            "title": "连续低迷风险",
            "severity": "high" if max_consec >= 5 else "medium",
            "detail": "存在连续 {} 个季度得分低于 {} 分，弱势阶段偏长".format(
                max_consec, low_threshold
            ),
        })

    # 5. 尾部脆弱风险
    worst = float(min(scores))
    if worst <= 10:
        alerts.append({
            "code": "weak_tail",
            "title": "尾部脆弱风险",
            "severity": "high" if worst <= 0 else "medium",
            "detail": "最差季度得分 {:.0f}，尾部表现偏弱".format(worst),
        })

    priority = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }
    alerts.sort(key=lambda item: (priority.get(item["severity"], 99), item["title"]))
    return alerts


def format_risk_alert(alert):
    return "{}：{}".format(alert["title"], alert["detail"])


def detect_overfit_flags(quarterly_scores):
    """Backward-compatible alias for older callers."""
    return [format_risk_alert(alert) for alert in detect_risk_alerts(quarterly_scores)]


# ============================================================
# 10. 股票搜索
# ============================================================

SEARCH_TYPES = {"CS", "ETF", "LOF", "INDX"}
TYPE_LABELS = {"CS": "股票", "ETF": "ETF", "LOF": "LOF", "INDX": "指数"}
EXCHANGE_LABELS = {"XSHE": "深交所", "XSHG": "上交所"}

def search_instruments(pattern):
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        print(f"错误: 无效的正则表达式 '{pattern}': {e}")
        sys.exit(1)
    facade = DataFacade()
    frames = []
    for t in ["CS", "ETF", "LOF", "INDX"]:
        df = facade.all_instruments(type=t)
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        print(f'搜索: "{pattern}" — 未找到匹配项')
        return
    ins_df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    ins_df = ins_df[ins_df.get("status") == "Active"]
    matches = []
    for _, ins in ins_df.iterrows():
        ob_id = ins.get("order_book_id", "")
        symbol = ins.get("symbol", "")
        if pd.isna(ob_id):
            ob_id = ""
        if pd.isna(symbol):
            symbol = ""
        if regex.search(str(ob_id)) or regex.search(str(symbol)):
            matches.append(ins.to_dict())

    matches.sort(key=lambda x: x.get("order_book_id", ""))

    if not matches:
        print(f'搜索: "{pattern}" — 未找到匹配项')
        return

    print(f'搜索: "{pattern}" (共 {len(matches)} 条结果)\n')

    headers = ["代码", "名称", "类型", "交易所", "行业", "上市日期"]
    rows = []
    for ins in matches:
        rows.append([
            ins.get("order_book_id", ""),
            ins.get("symbol", ""),
            TYPE_LABELS.get(ins.get("type", ""), ins.get("type", "")),
            EXCHANGE_LABELS.get(ins.get("exchange", ""), ins.get("exchange", "")),
            ins.get("industry_name", "") or "",
            ins.get("listed_date", "") or "",
        ])

    columns = list(zip(headers, *rows))
    widths = [max(display_width(cell) for cell in col) for col in columns]
    alignments = ["left"] * len(headers)

    # 表头
    header_parts = []
    for i, h in enumerate(headers):
        header_parts.append(pad_cell(h, widths[i], alignments[i]))
    print("  " + "  ".join(header_parts))
    print("  " + "  ".join("─" * w for w in widths))

    # 数据行
    for row in rows:
        parts = []
        for i, cell in enumerate(row):
            parts.append(pad_cell(cell, widths[i], alignments[i]))
        print("  " + "  ".join(parts))


# ============================================================
# 11. 主函数
# ============================================================

HELP_TEXT = """\
策略打分器 - 对策略文件进行10年滚动窗口回测并综合评分

用法:
    python -m skyeye.evaluation.rolling_score.cli <策略文件> [选项]

参数:
    strategy_file       策略文件路径（.py），需包含 init/handle_bar 等标准接口

选项:
    --cash N            初始资金，默认 1000000（100万）
    --window, -w SPEC   指定回测窗口编号（1-37），默认全部
                          37       只跑第37个窗口
                          35-37    跑第35到37窗口
                          1,10,37  跑指定的几个窗口
                        窗口不足5个时跳过综合评分，直接输出单窗口得分+交易日志
    --log LEVEL         交易日志详细程度，可选 none/low/mid/high，默认 low
                          none = 不输出交易日志
                          low  = 最近 1 个窗口
                          mid  = 最近 4 个窗口（约1年）
                          high = 全部 37 个窗口
    --plot              绘制价格走势图，在收盘价曲线上标注买卖点（保存为 PNG 文件）
    --mod MOD [MOD ...]
                        启用额外的 mod，例如 --mod dividend_scorer
    --mod-config, -mc KEY VALUE
                        设置 mod 配置，可重复传入，例如
                          -mc dividend_scorer.prior_blend 0.7
                          -mc dividend_scorer.dynamic_diagnostic false
    --search, -s PATTERN  搜索股票代码或名称（正则匹配，不跑回测）
                          匹配 order_book_id 或 symbol，仅显示 Active 的股票/ETF/指数
    --help              显示此帮助信息
    --eg                显示完整使用示例

评分机制:
    1. 滚动窗口: 最近10年数据，1年窗口，3个月步长，共37个窗口
    2. 13项指标: 收益(3) + 风险(4) + 风险调整收益(3) + 交易统计(2) + 月度(1)
    3. 多项式评分: 每个指标通过4次多项式映射到分数（-30~100分锚点范围）
    4. 季度投影: 37个窗口分数投影到40个季度，消除重叠窗口的重复计分
    5. 时间衰减: 近期季度权重更高（λ=0.03，近2年贡献约50%权重）

输出:
    【策略综合得分】 加权总分（可为负分）
    【策略稳定得分】 0~100分，反映策略泛化性（CV + 最差季度 + 连续低分）
    【核心指标】     年化收益率 | 最大回撤 | 夏普率 | 日胜率
    【市场环境】     牛市/震荡/熊市 三种环境下的平均得分
    【交易日志】     逐笔买卖交易流水
"""

EXAMPLE_TEXT = """\
=== 策略打分器使用示例 ===

1. 基本用法 - 对策略文件一键打分:

    python -m skyeye.evaluation.rolling_score.cli my_strategy.py

2. 指定初始资金（默认100万）:

    python -m skyeye.evaluation.rolling_score.cli my_strategy.py --cash 200000

3. 只跑指定窗口（快速调试）:

    python -m skyeye.evaluation.rolling_score.cli my_strategy.py -w 37
    python -m skyeye.evaluation.rolling_score.cli my_strategy.py -w 35-37
    python -m skyeye.evaluation.rolling_score.cli my_strategy.py -w 1,10,20,37

4. 查看更多交易日志:

    python -m skyeye.evaluation.rolling_score.cli my_strategy.py --log mid
    python -m skyeye.evaluation.rolling_score.cli my_strategy.py --log high

5. 绘制价格走势 + 买卖点图表:

    python -m skyeye.evaluation.rolling_score.cli my_strategy.py -w 37 --plot
    python -m skyeye.evaluation.rolling_score.cli my_strategy.py -w 35-37 --plot

6. 启用额外 mod 并传递配置:

    python -m skyeye.evaluation.rolling_score.cli my_strategy.py --mod dividend_scorer
    python -m skyeye.evaluation.rolling_score.cli my_strategy.py --mod dividend_scorer -mc dividend_scorer.prior_blend 0.7

7. 用自带的示例策略测试:

    python -m skyeye.evaluation.rolling_score.cli rqalpha/examples/demo_strategy.py
    python -m skyeye.evaluation.rolling_score.cli rqalpha/examples/demo_strategy.py --cash 500000 --log mid

8. 策略文件格式要求:

    策略文件需包含 rqalpha 标准接口函数，最小示例：

    ```python
    from rqalpha.apis import *

    def init(context):
        context.stock = "000001.XSHE"

    def handle_bar(context, bar_dict):
        prices = history_bars(context.stock, 20, '1d', 'close')
        if prices is None or len(prices) < 20:
            return
        ma_short = prices[-5:].mean()
        ma_long = prices.mean()
        if ma_short > ma_long:
            order_target_percent(context.stock, 0.95)
        elif ma_short < ma_long:
            order_target_percent(context.stock, 0)
    ```

9. 输出示例:

    ==================================================
    【策略综合得分】 32.5 分 | 收益 + 风险 + 指标加权总分
    【策略稳定得分】  45 分 | 策略综合得分相同时，反映策略的泛化性
    【核心指标】年化 12.3% | 回撤 15.2% | 夏普 0.85 | 胜率 52.1%
    【市场环境】牛市 55.2 | 震荡 28.3 | 熊市 -5.1
    ==================================================

    【交易日志】(最近1个窗口, 买入4笔 卖出4笔, 卖出胜率 75.0%)
     #   窗口  方向  证券代码      股票名称  日期        价格    数量   金额      盈亏       持仓  累计盈亏
     1   #37   买入  000001.XSHE  平安银行  2025-02-10  11.43  1000   11430               1000  +0.00
     2   #37   卖出  000001.XSHE  平安银行  2025-03-03  11.51   500    5755    +40.00       500  +40.00
     3   #37   卖出  000001.XSHE  平安银行  2025-03-10  11.60   500    5800    +85.00         0  +125.00

10. 搜索股票代码:

    python -m skyeye.evaluation.rolling_score.cli --search "平安"
    python -m skyeye.evaluation.rolling_score.cli -s "000001"
    python -m skyeye.evaluation.rolling_score.cli -s "510300"
    python -m skyeye.evaluation.rolling_score.cli -s "^600[0-9]{3}"
    python -m skyeye.evaluation.rolling_score.cli -s "银行"

11. 评分参考:

    E  (>=60分)  优秀策略，各维度表现均衡
    M+ (>=30分)  良好策略，有一定优势
    M  (>=0分)   一般策略，需要优化
    M- (>=-20分) 较差策略，存在明显缺陷
    I  (<-20分)  淘汰策略，需要重新设计
"""


def print_logo():
    """打印 Sky eye logo，使用蓝色到青色的渐变"""
    logo_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "assets", "logo.txt")
    )
    if not os.path.isfile(logo_path):
        return
    with open(logo_path, "r") as f:
        lines = f.readlines()
    # 蓝→青渐变色 (RGB)，灵感来自字节跳动 Sky eye 配色
    colors = [
        (17, 95, 179),   # #115fb3
        (27, 113, 179),  # #1b71b3
        (37, 130, 178),  # #2582b2
        (47, 148, 178),  # #2f94b2
        (56, 166, 178),  # #38a6b2
        (66, 183, 177),  # #42b7b1
        (76, 201, 177),  # #4cc9b1
    ]
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            print()
            continue
        n = len(line)
        colored = []
        for i, ch in enumerate(line):
            if ch == ' ':
                colored.append(ch)
            else:
                t = i / max(n - 1, 1)
                idx = t * (len(colors) - 1)
                ci = min(int(idx), len(colors) - 2)
                frac = idx - ci
                r = int(colors[ci][0] + (colors[ci + 1][0] - colors[ci][0]) * frac)
                g = int(colors[ci][1] + (colors[ci + 1][1] - colors[ci][1]) * frac)
                b = int(colors[ci][2] + (colors[ci + 1][2] - colors[ci][2]) * frac)
                colored.append(f"\033[38;2;{r};{g};{b}m{ch}\033[0m")
        print("".join(colored))
    print()


def main():
    print_logo()
    parser = argparse.ArgumentParser(description="策略打分器", add_help=False)
    parser.add_argument("strategy_file", nargs="?", default=None, help="策略文件路径")
    parser.add_argument("--cash", type=int, default=1000000, help="初始资金（默认100万）")
    parser.add_argument("--window", "-w", type=str, default=None,
                        help="指定窗口编号，如 37、35-37、1,10,37（默认全部）")
    parser.add_argument("--log", choices=["none", "low", "mid", "high"], default="low",
                        help="交易日志详细程度（默认 low，none=不输出）")
    parser.add_argument("--help", "-h", action="store_true", help="显示帮助信息")
    parser.add_argument("--eg", action="store_true", help="显示完整使用示例")
    parser.add_argument("--plot", action="store_true", help="绘制价格走势+买卖点图表")
    parser.add_argument("--mod", type=str, nargs="*", default=None,
                        help="启用额外的 mod，如 --mod dividend_scorer")
    parser.add_argument(
        "--mod-config",
        "-mc",
        metavar=("KEY", "VALUE"),
        nargs=2,
        action="append",
        default=None,
        help="设置 mod 配置，如 -mc dividend_scorer.prior_blend 0.7",
    )
    parser.add_argument("--search", "-s", type=str, default=None,
                        help="搜索股票代码或名称（正则匹配）")
    args = parser.parse_args()

    if args.eg:
        print(EXAMPLE_TEXT)
        sys.exit(0)

    if args.search:
        search_instruments(args.search)
        sys.exit(0)

    if args.help or args.strategy_file is None:
        print(HELP_TEXT)
        sys.exit(0)

    if not os.path.isfile(args.strategy_file):
        print(f"错误: 策略文件不存在: {args.strategy_file}")
        sys.exit(1)

    selected_indices = None
    if args.window:
        selected_indices = parse_window_arg(args.window)
    else:
        strategy_floor_date = extract_strategy_floor_date(args.strategy_file)
        if strategy_floor_date is not None:
            selected_indices = [
                i + 1 for i, (start, _) in enumerate(generate_windows()) if start >= strategy_floor_date
            ]

    mod_configs, extra_config = parse_runtime_config_args(args.mod_config)

    all_windows = generate_windows()
    if selected_indices:
        starts = [all_windows[i - 1][0] for i in selected_indices]
        ends = [all_windows[i - 1][1] for i in selected_indices]
        start_dt = min(starts).strftime("%Y-%m")
        end_dt = max(ends).strftime("%Y-%m")
        window_desc = f"{start_dt} ~ {end_dt}"
    else:
        window_desc = f"{all_windows[0][0].strftime('%Y-%m')} ~ {all_windows[-1][1].strftime('%Y-%m')}"

    strategy_spec = find_strategy_spec_by_file(args.strategy_file)
    benchmark_id = getattr(strategy_spec, "benchmark", None) or BENCHMARK

    print(f"策略文件: {os.path.basename(args.strategy_file)}")
    print(f"初始资金: {args.cash:,}")
    print(f"基准指数: {benchmark_id}")
    print(f"回测时间段: {window_desc}")
    print()
    strategy_card_lines = build_strategy_card_lines(strategy_spec)
    if strategy_card_lines:
        render_box(strategy_card_lines, title="策略卡片")
        print()

    # 滚动回测
    print("=" * 50)
    print("开始滑动窗口回测")
    print("=" * 50)
    t_start = time.time()
    window_results = run_rolling_backtests(
        args.strategy_file,
        args.cash,
        selected_indices,
        extra_mods=args.mod,
        mod_configs=mod_configs,
        extra_config=extra_config,
        benchmark_id=benchmark_id,
    )

    if not window_results:
        print("\n错误: 没有成功的回测窗口")
        sys.exit(1)

    total_expected = len(selected_indices) if selected_indices else 37
    print(f"\n成功完成 {len(window_results)}/{total_expected} 个窗口回测")
    sparse_windows = [w for w in window_results if w.get("sample_diagnostics", {}).get("sparse")]
    if sparse_windows:
        sparse_desc = ", ".join(
            "#{}({})".format(w["idx"], w["sample_diagnostics"]["warning_text"]) for w in sparse_windows
        )
        print("样本不足窗口: {}".format(sparse_desc))

    # 窗口数少于5个时，跳过综合评分，只输出单窗口得分和交易日志
    if len(window_results) < 5:
        use_color = sys.stdout.isatty()
        RST = "\033[0m" if use_color else ""

        print()
        print("=" * 50)
        for w in window_results:
            sc = rating_color(w['score'], use_color)
            rl = rating_label(w['score'], use_color)
            print(f"窗口 {w['start']} ~ {w['end']}  得分: {sc}{w['score']:.1f}{RST} [{rl}]")
            if w.get("sample_diagnostics", {}).get("warning_text"):
                print("  样本不足: {}".format(w["sample_diagnostics"]["warning_text"]))
            s = w["summary"]
            ann_ret = s.get("annualized_returns", 0) * 100
            max_dd = abs(s.get("max_drawdown", 0)) * 100
            sharpe_val = s.get("sharpe", 0)
            win_rate_val = s.get("win_rate", 0) * 100
            ac = indicator_color("annualized_returns", ann_ret / 100, use_color)
            dc = indicator_color("max_drawdown", max_dd / 100, use_color)
            shc = indicator_color("sharpe", sharpe_val, use_color)
            wc = indicator_color("win_rate", win_rate_val / 100, use_color)
            print(f"  年化 {ac}{ann_ret:.1f}%{RST} | 回撤 {dc}{max_dd:.1f}%{RST} | 夏普 {shc}{sharpe_val:.2f}{RST} | 胜率 {wc}{win_rate_val:.1f}%{RST}")
        print()
        print(rating_legend(use_color))
        print("=" * 50)
        # 少量窗口时，交易日志默认显示全部（除非 --log none）
        if args.log != "none":
            build_trade_log(window_results, "high", args.cash)
        if args.plot:
            plot_trades(window_results)
        elapsed = time.time() - t_start
        print(f"\n回测总耗时: {elapsed:.1f} 秒")
        return

    # 季度网格投影
    summary = summarize_window_results(window_results)
    quarterly_scores = summary["quarterly_scores"]
    composite = summary["composite"]
    core_indicators = summary["core_indicators"]
    stability = summary["stability"]
    market_env = summary["market_env"]

    # 输出报告
    use_color = sys.stdout.isatty()
    RST = "\033[0m" if use_color else ""

    print()
    ann_ret = core_indicators["annualized_returns"] * 100
    max_dd = abs(core_indicators["max_drawdown"]) * 100
    sharpe_val = core_indicators["sharpe"]
    win_rate_val = core_indicators["win_rate"] * 100

    comp_c = rating_color(composite, use_color)
    comp_rl = rating_label(composite, use_color)
    stab_c = rating_color(stability, use_color)
    stab_rl = rating_label(stability, use_color)
    summary_lines = []
    summary_lines.append(f"【策略综合得分】 {comp_c}{composite:.1f}{RST} 分 [{comp_rl}]")
    summary_lines.append(f"【策略稳定得分】  {stab_c}{stability:.0f}{RST} 分 [{stab_rl}]")
    ac = indicator_color("annualized_returns", ann_ret / 100, use_color)
    dc = indicator_color("max_drawdown", max_dd / 100, use_color)
    shc = indicator_color("sharpe", sharpe_val, use_color)
    wc = indicator_color("win_rate", win_rate_val / 100, use_color)
    summary_lines.append(f"【核心指标】年化 {ac}{ann_ret:.1f}%{RST} | 回撤 {dc}{max_dd:.1f}%{RST} | 夏普 {shc}{sharpe_val:.2f}{RST} | 胜率 {wc}{win_rate_val:.1f}%{RST}")
    risk_flags = summary.get("risk_flags") or summary.get("overfit_flags") or []
    if risk_flags:
        warn_color = "\033[41;97m" if use_color else ""
        summary_lines.append(f"{warn_color}提示: ⚑风险提示{RST}")
        for f in risk_flags:
            summary_lines.append(f"  - {f}")

    env_parts = []
    for env_name in ["牛市", "震荡", "熊市"]:
        v = market_env[env_name]
        if isinstance(v, (int, float)):
            rl = rating_label(v, use_color)
            vc = rating_color(v, use_color)
            env_parts.append(f"{env_name} {vc}{v:.1f}{RST}[{rl}]")
        else:
            env_parts.append(f"{env_name} {v}")
    summary_lines.append(f"【市场环境】{' | '.join(env_parts)}")
    summary_lines.append("")
    summary_lines.append(rating_legend(use_color))

    render_box(summary_lines, title="回测结论")

    elapsed = time.time() - t_start
    print(f"\n回测总耗时: {elapsed:.1f} 秒")

    # 交易日志
    if args.log != "none":
        build_trade_log(window_results, args.log, args.cash)

    # 图表
    if args.plot:
        plot_trades(window_results)


if __name__ == "__main__":
    main()
