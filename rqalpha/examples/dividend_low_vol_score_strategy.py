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
    context.target_percent = 0.95
    context.buy_threshold = 3.5
    context.sell_threshold = 6.5
    context.last_score_date = None
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
    if total_score is None:
        return

    position = get_position(context.etf)
    has_position = position.quantity > 0

    logger.info(
        "dividend score date={} score={:.2f} confidence={} method={}".format(
            score_date,
            total_score,
            score.get("confidence"),
            score.get("model_meta", {}).get("method"),
        )
    )

    if total_score < context.buy_threshold and not has_position:
        logger.info("buy signal triggered: {:.2f} < {:.2f}".format(total_score, context.buy_threshold))
        order_target_percent(context.etf, context.target_percent)
    elif total_score > context.sell_threshold and has_position:
        logger.info("sell signal triggered: {:.2f} > {:.2f}".format(total_score, context.sell_threshold))
        order_target_percent(context.etf, 0)


def after_trading(context):
    pass
