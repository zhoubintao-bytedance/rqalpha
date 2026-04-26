"""TX1 autoresearch 候选 catalog 定义。"""

from __future__ import annotations

from copy import deepcopy

from skyeye.products.tx1.evaluator import (
    BASELINE_5F_COLUMNS,
    LIQUIDITY_FEATURE_COLUMNS,
    TREND_FEATURE_COLUMNS,
    collect_feature_columns,
)


DEFAULT_CATALOG_NAME = "risk_reward_v1"
PORTFOLIO_COARSE_CATALOG_NAME = "portfolio_coarse_v1"
PORTFOLIO_REFINED_CATALOG_NAME = "portfolio_refined_v1"


def _dedupe_features(features):
    """按声明顺序去重，保证候选特征集稳定可复现。"""
    ordered = []
    for feature in features:
        if feature not in ordered:
            ordered.append(feature)
    return ordered


def _build_candidate(
    candidate_id,
    *,
    description,
    features,
    portfolio=None,
    preprocessing=None,
    multi_output=None,
):
    """构造单个候选定义，避免 catalog 内部字段散落。"""
    return {
        "id": str(candidate_id),
        "description": str(description),
        "features": _dedupe_features(list(features)),
        "portfolio": dict(portfolio or {}),
        "preprocessing": deepcopy(preprocessing) if preprocessing else None,
        "multi_output": deepcopy(multi_output) if multi_output else None,
    }


def build_portfolio_coarse_catalog():
    """阶段0：粗粒度探索，快速锁定参数区间。

    buy_top_k: [5, 15, 25]，步长10
    hold_top_k: [15, 30, 45]，步长15
    有效组合：约8个
    预计耗时：~8分钟
    """
    baseline_features = list(BASELINE_5F_COLUMNS)
    candidates = []

    buy_range = [5, 15, 25]
    hold_range = [15, 30, 45]

    for buy_k in buy_range:
        for hold_k in hold_range:
            if hold_k < buy_k:
                continue
            candidates.append(_build_candidate(
                "coarse_b{}_h{}".format(buy_k, hold_k),
                description="粗粒度探索: buy_top_k={}, hold_top_k={}".format(buy_k, hold_k),
                features=baseline_features,
                portfolio={
                    "buy_top_k": buy_k,
                    "hold_top_k": hold_k,
                    "rebalance_interval": 20,
                    "holding_bonus": 0.5,
                },
            ))
    return candidates


def build_portfolio_refined_catalog(top_candidates):
    """阶段1：基于阶段0结果生成中粒度候选集。

    从Top2候选提取邻域范围，步长更细：
    - buy_top_k 步长2
    - hold_top_k 步长5

    Args:
        top_candidates: 阶段0的Top候选列表，至少包含portfolio参数

    Returns:
        中粒度候选列表，约20个组合
    """
    baseline_features = list(BASELINE_5F_COLUMNS)

    if not top_candidates:
        # 如果没有传入候选，使用默认范围
        buy_values = [12, 15, 18, 22, 25, 28]
        hold_values = [25, 30, 35, 40, 45]
    else:
        # 从Top候选提取邻域范围
        buy_values = set()
        hold_values = set()

        for c in top_candidates:
            params = c.get("portfolio", {})
            buy_k = params.get("buy_top_k", 25)
            hold_k = params.get("hold_top_k", 45)

            # 扩展邻域：buy_top_k ±6，步长2
            for delta in range(-6, 7, 2):
                buy_values.add(buy_k + delta)
            # 扩展邻域：hold_top_k ±10，步长5
            for delta in range(-10, 11, 5):
                hold_values.add(hold_k + delta)

        # 过滤有效范围
        buy_values = sorted([v for v in buy_values if 5 <= v <= 30])
        hold_values = sorted([v for v in hold_values if 10 <= v <= 50])

    candidates = []
    for buy_k in buy_values:
        for hold_k in hold_values:
            if hold_k < buy_k:
                continue
            candidates.append(_build_candidate(
                "refined_b{}_h{}".format(buy_k, hold_k),
                description="中粒度筛选: buy_top_k={}, hold_top_k={}".format(buy_k, hold_k),
                features=baseline_features,
                portfolio={
                    "buy_top_k": buy_k,
                    "hold_top_k": hold_k,
                    "rebalance_interval": 20,
                    "holding_bonus": 0.5,
                },
            ))
    return candidates


def build_candidate_catalog(catalog_name=DEFAULT_CATALOG_NAME, top_candidates=None):
    """返回本轮 autoresearch 要遍历的候选集合。

    Args:
        catalog_name: catalog名称，支持以下值：
            - "risk_reward_v1": 默认风险收益搜索catalog
            - "portfolio_coarse_v1": 粗粒度portfolio参数探索
            - "portfolio_refined_v1": 中粒度portfolio参数筛选
        top_candidates: 仅用于portfolio_refined_v1，传入阶段0的Top候选
    """
    name = str(catalog_name or DEFAULT_CATALOG_NAME)

    # 粗粒度portfolio参数探索
    if name == PORTFOLIO_COARSE_CATALOG_NAME:
        return build_portfolio_coarse_catalog()

    # 中粒度portfolio参数筛选
    if name == PORTFOLIO_REFINED_CATALOG_NAME:
        return build_portfolio_refined_catalog(top_candidates)

    # 默认catalog
    if name != DEFAULT_CATALOG_NAME:
        raise ValueError("unknown TX1 autoresearch catalog: {}".format(name))

    baseline_features = list(BASELINE_5F_COLUMNS)
    liquidity_plus_features = _dedupe_features(
        baseline_features + list(LIQUIDITY_FEATURE_COLUMNS)
    )
    trend_liquidity_features = _dedupe_features(
        collect_feature_columns("baseline_5f", "trend", "liquidity")
    )
    light_guard = {
        "enabled": True,
        "volatility": {"enabled": True, "transform": "rank"},
        "max_drawdown": {"enabled": True, "transform": "rank"},
        "prediction": {
            "combine_auxiliary": True,
            "volatility_weight": 0.05,
            "max_drawdown_weight": 0.05,
        },
        "reliability_score": {"enabled": True},
    }

    return [
        _build_candidate(
            "baseline_5f_preproc",
            description="给默认 5 因子加截面预处理，优先试图提升稳定性而不额外扩特征。",
            features=baseline_features,
            preprocessing={
                "enabled": True,
                "neutralize": True,
                "winsorize_scale": 5.0,
                "standardize": True,
            },
        ),
        _build_candidate(
            "baseline_5f_slow",
            description="保持默认信号，只把组合层改成更慢的 30 日调仓和更高持仓惯性，降低换手与回撤。",
            features=baseline_features,
            portfolio={
                "rebalance_interval": 30,
                "holding_bonus": 0.8,
            },
        ),
        _build_candidate(
            "baseline_5f_wide_hold",
            description="保持默认信号，适度放宽持有缓冲区，尝试减少抖动卖出。",
            features=baseline_features,
            portfolio={
                "buy_top_k": 25,
                "hold_top_k": 50,
                "holding_bonus": 0.8,
            },
        ),
        _build_candidate(
            "liquidity_plus",
            description="复用历史最接近 baseline 的流动性增强方向，争取在不牺牲太多收益的前提下降回撤。",
            features=liquidity_plus_features,
        ),
        _build_candidate(
            "liquidity_plus_slow",
            description="流动性增强叠加慢调仓，优先探索更稳的收益兑现路径。",
            features=liquidity_plus_features,
            portfolio={
                "rebalance_interval": 30,
                "holding_bonus": 0.8,
            },
        ),
        _build_candidate(
            "trend_liquidity_plus",
            description="把趋势结构和流动性约束一起引入，尝试减少高位追涨后的回撤暴露。",
            features=trend_liquidity_features,
        ),
        _build_candidate(
            "trend_liquidity_plus_slow",
            description="趋势+流动性增强再叠加更慢的组合层，优先压低回撤与换手。",
            features=trend_liquidity_features,
            portfolio={
                "rebalance_interval": 30,
                "holding_bonus": 0.8,
            },
        ),
        _build_candidate(
            "baseline_5f_light_guard",
            description="只给默认 5 因子加极轻量的波动率/回撤辅助头，避免重演历史保护版过度惩罚收益的问题。",
            features=baseline_features,
            multi_output=light_guard,
        ),
        _build_candidate(
            "liquidity_plus_light_guard",
            description="流动性增强加极轻量风险辅助头，探索更平滑的风险收益前沿。",
            features=liquidity_plus_features,
            multi_output=light_guard,
        ),
    ]


def build_baseline_candidate():
    """返回 catalog 搜索的固定 baseline 定义。"""
    return _build_candidate(
        "baseline_5f_default",
        description="TX1 当前默认 5 因子 + Top25/Top45 执行口径。",
        features=list(BASELINE_5F_COLUMNS),
    )


def build_candidate_config(
    candidate,
    *,
    model_kind="lgbm",
    label_transform="rank",
    horizon_days=20,
):
    """把候选定义展开成 `main(...)` 可直接消费的配置。"""
    candidate = dict(candidate)
    candidate_model = deepcopy(candidate.get("model") or {})
    resolved_model = {
        "kind": str(candidate_model.get("kind") or model_kind),
    }
    if candidate_model.get("params") is not None:
        # 候选级模型参数优先于全局默认，便于 autoresearch 显式搜索正则强度。
        resolved_model["params"] = deepcopy(candidate_model.get("params") or {})
    config = {
        "experiment_name": "tx1_autoresearch_{}".format(candidate["id"]),
        "model": resolved_model,
        "features": list(candidate["features"]),
        "labels": {
            "transform": label_transform,
        },
        "portfolio": dict(candidate.get("portfolio") or {}),
    }
    if candidate.get("preprocessing") is not None:
        config["preprocessing"] = dict(candidate["preprocessing"])
    if candidate.get("multi_output") is not None:
        config["multi_output"] = deepcopy(candidate["multi_output"])
    if int(horizon_days) != 20:
        # 研究默认冻结 horizon=20；只有显式搜索其它 horizon 时才解锁。
        config["labels"]["horizon"] = int(horizon_days)
        config["labels"]["allow_horizon_override"] = True
    return config
