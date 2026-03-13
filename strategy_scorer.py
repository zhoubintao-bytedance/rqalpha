"""
策略打分器 - 对输入的策略文件进行滚动窗口回测并综合评分

用法:
    python strategy_scorer.py strategy.py
    python strategy_scorer.py strategy.py --cash 200000
    python strategy_scorer.py strategy.py --log mid
"""
import argparse
import datetime
import math
import os
import sys
import warnings
import unicodedata
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from rqalpha import run

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
    return "  评级: " + "  ".join(parts)

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
            if name == "profit_loss_rate":
                s = 100.0
            else:
                s = 0.0
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


def run_rolling_backtests(strategy_file, cash, selected_indices=None):
    """对策略文件执行滚动窗口回测，返回结果列表

    selected_indices: 要执行的窗口编号列表（1-based），None 表示全部37个
    """
    with open(strategy_file) as f:
        source_code = f.read()

    windows = generate_windows()
    if selected_indices is not None:
        run_list = [(i, windows[i - 1]) for i in selected_indices]
    else:
        run_list = [(i + 1, w) for i, w in enumerate(windows)]

    total = len(run_list)
    results = []
    failed = 0

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
                "sys_analyser": {"benchmark": BENCHMARK, "plot": False},
                "sys_progress": {"enabled": False},
            },
        }

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run(config, source_code=source_code)

            if result is None:
                print("失败(返回None)")
                failed += 1
                continue

            summary = result["sys_analyser"]["summary"]
            trades = result["sys_analyser"]["trades"]
            window_score = score_window(summary)
            use_color = sys.stdout.isatty()
            sc = rating_color(window_score, use_color)
            rst = "\033[0m" if use_color else ""
            print(f"得分 {sc}{window_score:.1f}{rst}")

            results.append({
                "idx": idx,
                "start": start,
                "end": end,
                "summary": summary,
                "score": window_score,
                "trades": trades,
            })
        except Exception as e:
            print(f"异常: {e}")
            failed += 1

    if failed > 0:
        print(f"\n  警告: {failed} 个窗口回测失败")

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

    for _, row in trades_df.iterrows():
        trade_dt = row.name
        ob_id = row["order_book_id"]
        symbol = row["symbol"]
        side = row["side"]
        price = row["last_price"]
        qty = row["last_quantity"]

        if ob_id not in holdings:
            holdings[ob_id] = {"quantity": 0, "cost": 0.0, "realized_pnl": 0.0}
        h = holdings[ob_id]

        if side == "BUY":
            h["cost"] += price * qty
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
                "holding_qty": h["quantity"],
                "holding_cost": h["cost"],
                "realized_pnl": h["realized_pnl"],
                "total_realized_pnl": total_realized_pnl,
            })
        elif side == "SELL":
            avg_cost = h["cost"] / h["quantity"] if h["quantity"] > 0 else 0
            pnl = (price - avg_cost) * qty
            h["realized_pnl"] += pnl
            total_realized_pnl += pnl
            h["cost"] -= avg_cost * qty
            h["quantity"] -= qty
            records.append({
                "order_book_id": ob_id,
                "symbol": symbol,
                "side": "卖出",
                "datetime": trade_dt,
                "price": price,
                "quantity": qty,
                "amount": price * qty,
                "pnl": pnl,
                "holding_qty": h["quantity"],
                "holding_cost": h["cost"],
                "realized_pnl": h["realized_pnl"],
                "total_realized_pnl": total_realized_pnl,
            })

    return records


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
        cum_str = fmt_pnl(r['realized_pnl'])
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
        colored_row[11] = colorize(cum_str, r["realized_pnl"])
        rows.append(plain_row)
        row_colors.append(colored_row)

    def display_width(value):
        text = str(value)
        total = 0
        for ch in text:
            total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        return total

    def pad_cell(value, width, align):
        text = str(value)
        pad_len = width - display_width(text)
        if pad_len <= 0:
            return text
        if align == "left":
            return text + " " * pad_len
        return " " * pad_len + text

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
            padded = pad_cell(plain_row[i], widths[i], alignments[i])
            if colored_row[i] != plain_row[i]:
                padded = padded.replace(plain_row[i], colored_row[i], 1)
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

        # 该窗口的已实现盈亏（最后一条记录的累计）
        realized_pnl = w_records[-1]["realized_pnl"]

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
            # 市值 = 总资产 - 剩余现金，剩余现金 = 初始资金 + 已实现盈亏 - 持仓成本
            total_holding_cost = sum(w_holding_costs.get(k, 0) for k in w_holding_stocks)
            holding_market_value = total_asset - (cash + realized_pnl - total_holding_cost)
            parts = []
            for k, v in w_holding_stocks.items():
                qty = int(v['qty'])
                cost = w_holding_costs.get(k, 0)
                avg_price = cost / qty if qty > 0 else 0
                # 按持仓比例分配市值
                share = cost / total_holding_cost if total_holding_cost > 0 else 1
                mv = holding_market_value * share
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
    """从 bundle 读取指定股票的日线数据"""
    bundle_path = os.path.expanduser("~/.rqalpha/bundle")
    for filename in ["stocks.h5", "indexes.h5", "funds.h5"]:
        filepath = os.path.join(bundle_path, filename)
        if not os.path.exists(filepath):
            continue
        with h5py.File(filepath, "r") as h5:
            if order_book_id in h5:
                bars = h5[order_book_id][:]
                df = pd.DataFrame(bars)
                df["date"] = pd.to_datetime(df["datetime"].astype(str).str[:8], format="%Y%m%d")
                df = df.set_index("date").sort_index()
                return df.loc[str(start_date):str(end_date)]
    return None


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


def get_benchmark_quarterly_returns():
    """从 bundle 读取沪深300季度涨跌幅"""
    bundle_path = os.path.expanduser("~/.rqalpha/bundle")
    with h5py.File(os.path.join(bundle_path, "indexes.h5"), "r") as h5:
        bars = h5["000300.XSHG"][:]

    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["datetime"].astype(str), format="%Y%m%d%H%M%S")
    df = df.set_index("date").sort_index()

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

# ============================================================
# 10. 主函数
# ============================================================

HELP_TEXT = """\
策略打分器 - 对策略文件进行10年滚动窗口回测并综合评分

用法:
    python strategy_scorer.py <策略文件> [选项]

参数:
    strategy_file       策略文件路径（.py），需包含 init/handle_bar 等标准接口

选项:
    --cash N            初始资金，默认 1000000（100万）
    --window, -w SPEC   指定回测窗口编号（1-37），默认全部
                          37       只跑第37个窗口
                          35-37    跑第35到37窗口
                          1,10,37  跑指定的几个窗口
                        窗口不足5个时跳过综合评分，直接输出单窗口得分+交易日志
    --log LEVEL         交易日志详细程度，可选 low/mid/high，默认 low
                          low  = 最近 1 个窗口
                          mid  = 最近 4 个窗口（约1年）
                          high = 全部 37 个窗口
    --plot              绘制价格走势图，在收盘价曲线上标注买卖点（弹窗显示）
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

    python strategy_scorer.py my_strategy.py

2. 指定初始资金（默认100万）:

    python strategy_scorer.py my_strategy.py --cash 200000

3. 只跑指定窗口（快速调试）:

    python strategy_scorer.py my_strategy.py -w 37          # 只跑最新窗口
    python strategy_scorer.py my_strategy.py -w 35-37       # 跑最近3个窗口
    python strategy_scorer.py my_strategy.py -w 1,10,20,37  # 跑指定窗口

4. 查看更多交易日志:

    python strategy_scorer.py my_strategy.py --log mid    # 最近4个窗口
    python strategy_scorer.py my_strategy.py --log high   # 全部37个窗口

5. 绘制价格走势 + 买卖点图表:

    python strategy_scorer.py my_strategy.py -w 37 --plot        # 单窗口图表
    python strategy_scorer.py my_strategy.py -w 35-37 --plot     # 多窗口各一张图

6. 用自带的示例策略测试:

    python strategy_scorer.py rqalpha/examples/demo_strategy.py
    python strategy_scorer.py rqalpha/examples/demo_strategy.py --cash 500000 --log mid

7. 策略文件格式要求:

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

8. 输出示例:

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

9. 评分参考:

    E  (>=60分)  优秀策略，各维度表现均衡
    M+ (>=30分)  良好策略，有一定优势
    M  (>=0分)   一般策略，需要优化
    M- (>=-20分) 较差策略，存在明显缺陷
    I  (<-20分)  淘汰策略，需要重新设计
"""


def main():
    parser = argparse.ArgumentParser(description="策略打分器", add_help=False)
    parser.add_argument("strategy_file", nargs="?", default=None, help="策略文件路径")
    parser.add_argument("--cash", type=int, default=1000000, help="初始资金（默认100万）")
    parser.add_argument("--window", "-w", type=str, default=None,
                        help="指定窗口编号，如 37、35-37、1,10,37（默认全部）")
    parser.add_argument("--log", choices=["low", "mid", "high"], default="low",
                        help="交易日志详细程度（默认 low）")
    parser.add_argument("--help", "-h", action="store_true", help="显示帮助信息")
    parser.add_argument("--eg", action="store_true", help="显示完整使用示例")
    parser.add_argument("--plot", action="store_true", help="绘制价格走势+买卖点图表")
    args = parser.parse_args()

    if args.eg:
        print(EXAMPLE_TEXT)
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

    all_windows = generate_windows()
    if selected_indices:
        starts = [all_windows[i - 1][0] for i in selected_indices]
        ends = [all_windows[i - 1][1] for i in selected_indices]
        start_dt = min(starts).strftime("%Y-%m")
        end_dt = max(ends).strftime("%Y-%m")
        window_desc = f"{start_dt} ~ {end_dt}"
    else:
        window_desc = f"{all_windows[0][0].strftime('%Y-%m')} ~ {all_windows[-1][1].strftime('%Y-%m')}"

    print(f"策略文件: {os.path.basename(args.strategy_file)}")
    print(f"初始资金: {args.cash:,}")
    print(f"基准指数: {BENCHMARK}")
    print(f"回测时间段: {window_desc}")
    print()

    # 滚动回测
    print("=" * 50)
    print("开始滑动窗口回测")
    print("=" * 50)
    window_results = run_rolling_backtests(args.strategy_file, args.cash, selected_indices)

    if not window_results:
        print("\n错误: 没有成功的回测窗口")
        sys.exit(1)

    total_expected = len(selected_indices) if selected_indices else 37
    print(f"\n成功完成 {len(window_results)}/{total_expected} 个窗口回测")

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
            s = w["summary"]
            ann_ret = s.get("annualized_returns", 0) * 100
            max_dd = s.get("max_drawdown", 0) * 100
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
        # 少量窗口时，交易日志默认显示全部
        build_trade_log(window_results, "high", args.cash)
        if args.plot:
            plot_trades(window_results)
        return

    # 季度网格投影
    quarterly_scores, quarterly_raw_indicators = project_to_quarters(window_results)

    # 综合得分 + 核心指标
    composite, core_indicators = compute_composite_score(quarterly_scores, quarterly_raw_indicators)

    # 稳定性评分
    stability = compute_stability_score(quarterly_scores)

    # 市场环境
    benchmark_qr = get_benchmark_quarterly_returns()
    market_env = compute_market_env_scores(quarterly_scores, benchmark_qr)

    # 输出报告
    use_color = sys.stdout.isatty()
    RST = "\033[0m" if use_color else ""

    print()
    print("=" * 50)
    ann_ret = core_indicators["annualized_returns"] * 100
    max_dd = core_indicators["max_drawdown"] * 100
    sharpe_val = core_indicators["sharpe"]
    win_rate_val = core_indicators["win_rate"] * 100

    comp_c = rating_color(composite, use_color)
    comp_rl = rating_label(composite, use_color)
    stab_c = rating_color(stability, use_color)
    stab_rl = rating_label(stability, use_color)
    print(f"【策略综合得分】 {comp_c}{composite:.1f}{RST} 分 [{comp_rl}]")
    print(f"【策略稳定得分】  {stab_c}{stability:.0f}{RST} 分 [{stab_rl}]")
    ac = indicator_color("annualized_returns", ann_ret / 100, use_color)
    dc = indicator_color("max_drawdown", max_dd / 100, use_color)
    shc = indicator_color("sharpe", sharpe_val, use_color)
    wc = indicator_color("win_rate", win_rate_val / 100, use_color)
    print(f"【核心指标】年化 {ac}{ann_ret:.1f}%{RST} | 回撤 {dc}{max_dd:.1f}%{RST} | 夏普 {shc}{sharpe_val:.2f}{RST} | 胜率 {wc}{win_rate_val:.1f}%{RST}")

    env_parts = []
    for env_name in ["牛市", "震荡", "熊市"]:
        v = market_env[env_name]
        if isinstance(v, (int, float)):
            rl = rating_label(v, use_color)
            vc = rating_color(v, use_color)
            env_parts.append(f"{env_name} {vc}{v:.1f}{RST}[{rl}]")
        else:
            env_parts.append(f"{env_name} {v}")
    print(f"【市场环境】{' | '.join(env_parts)}")
    print()
    print(rating_legend(use_color))
    print("=" * 50)

    # 交易日志
    build_trade_log(window_results, args.log, args.cash)

    # 图表
    if args.plot:
        plot_trades(window_results)


if __name__ == "__main__":
    main()
