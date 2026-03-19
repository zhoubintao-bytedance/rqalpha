"""
红利低波打分器示例策略

运行前请先准备:
1. RQAlpha bundle（用于回测）
2. 红利低波打分器 SQLite 缓存（用于估值打分）

示例命令见 docs/dividend_scorer_usage.md
"""
from rqalpha.apis import *


def init(context):
    context.etf = "512890.XSHG"
    context.max_target_percent = 0.95
    context.buy_percentile = 0.20
    context.sell_percentile = 0.80
    context.last_score_date = None
    context.signal_target_percent = 0.0
    update_universe(context.etf)


def before_trading(context):
    pass


def handle_bar(context, bar_dict):
    score = get_dividend_score()
    if not score or score.get("error"):
        return

    score_date = score.get("date")
    if score_date == context.last_score_date:
        return
    context.last_score_date = score_date

    total_score = score.get("total_score")
    score_percentile = score.get("score_percentile")
    if total_score is None or score_percentile is None:
        return

    confidence = score.get("confidence")
    confidence_multiplier = {
        "normal": 1.0,
        "lowered": 0.5,
        "low": 0.0,
    }.get(confidence, 0.0)

    logger.info(
        "dividend score signal_date={} trade_date={} score={:.2f} percentile={:.1%} confidence={} method={}".format(
            score_date,
            score.get("trade_date"),
            total_score,
            score_percentile,
            confidence,
            score.get("model_meta", {}).get("method"),
        )
    )

    if score_percentile <= context.buy_percentile:
        context.signal_target_percent = context.max_target_percent
        logger.info(
            "buy regime triggered: percentile {:.1%} <= {:.1%}".format(
                score_percentile, context.buy_percentile
            )
        )
    elif score_percentile >= context.sell_percentile:
        context.signal_target_percent = 0.0
        logger.info(
            "sell regime triggered: percentile {:.1%} >= {:.1%}".format(
                score_percentile, context.sell_percentile
            )
        )

    desired_target_percent = context.signal_target_percent * confidence_multiplier
    position = get_position(context.etf)
    current_market_value = position.market_value if position is not None else 0.0
    portfolio_value = context.portfolio.total_value
    current_target_percent = current_market_value / portfolio_value if portfolio_value > 0 else 0.0

    if abs(desired_target_percent - current_target_percent) >= 0.05:
        logger.info(
            "rebalance target: signal_target={:.0%} confidence_target={:.0%} current={:.0%}".format(
                context.signal_target_percent,
                desired_target_percent,
                current_target_percent,
            )
        )
        order_target_percent(context.etf, desired_target_percent)


def after_trading(context):
    pass
