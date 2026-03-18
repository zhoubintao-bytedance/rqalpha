"""
红利低波打分器 vs 512890 走势对比示例

用途:
1. 复用 RQAlpha 内置 `plot(...)` 画图能力
2. 把 512890 的真实收盘价归一化到 [0, 10]
3. 与红利低波打分器输出的 `total_score` 放到同一坐标轴上对比

运行命令见 docs/dividend_scorer_usage.md
"""

import datetime

import numpy as np

from rqalpha.apis import *
from rqalpha.environment import Environment
from rqalpha.utils.datetime_func import convert_date_to_int


SCORE_MIN = 0.0
SCORE_MAX = 10.0


def init(context):
    context.etf = "512890.XSHG"
    update_universe(context.etf)
    _prepare_price_range(context)


def _prepare_price_range(context):
    env = Environment.get_instance()
    start_date = env.config.base.start_date
    end_date = env.config.base.end_date
    end_dt = datetime.datetime.combine(end_date, datetime.time.min)
    context.price_anchor_dt = end_dt
    bars = env.data_proxy.history_bars(
        context.etf,
        None,
        "1d",
        ["datetime", "close"],
        end_dt,
        skip_suspended=False,
        adjust_type="pre",
        adjust_orig=end_dt,
    )
    if bars is None or len(bars) == 0:
        raise RuntimeError("missing ETF bars for {}".format(context.etf))

    start_int = np.uint64(convert_date_to_int(start_date))
    end_int = np.uint64(convert_date_to_int(end_date))
    bars = bars[(bars["datetime"] >= start_int) & (bars["datetime"] <= end_int)]
    if len(bars) == 0:
        raise RuntimeError(
            "missing ETF bars for {} between {} and {}".format(context.etf, start_date, end_date)
        )

    prices = bars["close"].astype(float)
    context.price_min = float(np.nanmin(prices))
    context.price_max = float(np.nanmax(prices))
    if np.isclose(context.price_min, context.price_max):
        context.price_max = context.price_min + 1.0

    logger.info(
        "normalize {} close into [{:.1f}, {:.1f}] with window {} -> {}, min={:.4f}, max={:.4f}".format(
            context.etf,
            SCORE_MIN,
            SCORE_MAX,
            start_date,
            end_date,
            context.price_min,
            context.price_max,
        )
    )


def _normalize_price(context, close):
    scale = (close - context.price_min) / (context.price_max - context.price_min)
    normalized = SCORE_MIN + scale * (SCORE_MAX - SCORE_MIN)
    return float(np.clip(normalized, SCORE_MIN, SCORE_MAX))


def handle_bar(context, bar_dict):
    env = Environment.get_instance()
    close_series = env.data_proxy.history_bars(
        context.etf,
        1,
        "1d",
        "close",
        env.calendar_dt,
        skip_suspended=False,
        adjust_type="pre",
        adjust_orig=context.price_anchor_dt,
    )
    if close_series is None or len(close_series) == 0:
        return

    close = float(close_series[-1])
    if np.isnan(close):
        return

    plot("512890_close_norm", _normalize_price(context, close))

    score = get_dividend_score()
    score_value = float("nan")
    if score and not score.get("error") and score.get("total_score") is not None:
        score_value = float(score["total_score"])
    plot("dividend_score", score_value)


def before_trading(context):
    pass


def after_trading(context):
    pass
