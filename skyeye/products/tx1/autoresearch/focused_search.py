"""TX1 面向长预算夜跑的 two-phase focused autoresearch。"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skyeye.evaluation.rolling_score import (
    compute_composite_score as compute_replay_composite_score,
)
from skyeye.evaluation.rolling_score import (
    compute_stability_score as compute_replay_stability_score,
)
from skyeye.evaluation.rolling_score import (
    detect_risk_alerts as detect_replay_risk_alerts,
)
from skyeye.evaluation.rolling_score import (
    project_to_quarters as project_replay_to_quarters,
)
from skyeye.evaluation.rolling_score import run_rolling_backtests
from skyeye.products.tx1.autoresearch.catalog import build_candidate_config
from skyeye.products.tx1.autoresearch.judge import judge_candidate
from skyeye.products.tx1.autoresearch.runner import (
    build_research_raw_df,
    build_run_root,
    build_stage_experiment_root,
    summarize_metrics,
)
from skyeye.products.tx1.baseline_models import create_model, supports_validation
from skyeye.products.tx1.config import normalize_config
from skyeye.products.tx1.cost_layer import CostConfig
from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.evaluator import (
    BASELINE_5F_COLUMNS,
    LIQUIDITY_FEATURE_COLUMNS,
    build_portfolio_returns,
    evaluate_portfolios,
    evaluate_predictions,
)
from skyeye.products.tx1.label_builder import LabelBuilder
from skyeye.products.tx1.persistence import save_experiment
from skyeye.products.tx1.portfolio_proxy import PortfolioProxy
from skyeye.products.tx1.preprocessor import FeaturePreprocessor
from skyeye.products.tx1.robustness import (
    compute_regime_scores,
    compute_stability_score as compute_proxy_stability_score,
    detect_overfit_flags,
    summarize_experiment,
)
from skyeye.products.tx1.splitter import WalkForwardSplitter
from skyeye.products.tx1.strategies.rolling_score.params import load_profile


LEADERBOARD_HEADER = (
    "candidate_id\tphase\tevaluation_mode\tfamily\tstatus\tstage_reached\tsearch_round\t"
    "net_mean_return\tmax_drawdown\tstability_score\tcomposite_score\tnum_windows\treplay_health_code\treason_code\t"
    "artifact_line_id\tstrategy_profile\tprofile_overrides\texperiment_path\tdescription"
)
TARGET_PROXY_FLAGS = (
    "flag_ic_decay",
    "flag_spread_decay",
    "flag_val_dominant",
)
DEFAULT_PREPROCESSING = {
    "enabled": True,
    "neutralize": True,
    "winsorize_scale": 5.0,
    "standardize": True,
}
ROLLING_SCORE_STRATEGY_FILE = (
    Path(__file__).resolve().parents[1]
    / "strategies"
    / "rolling_score"
    / "strategy.py"
)
TX1_DIAGNOSTIC_MOD_NAME = "tx1_diagnostics"
TX1_DIAGNOSTIC_MOD_LIB = (
    "skyeye.products.tx1.autoresearch.rqalpha_mod_tx1_diagnostics"
)
LGBM_REGULARIZATION_PROFILES = {
    "default": {},
    "heavy_reg": {
        "num_leaves": 16,
        "max_depth": 4,
        "learning_rate": 0.025,
        "n_estimators": 240,
        "subsample": 0.70,
        "subsample_freq": 1,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.8,
        "reg_lambda": 4.5,
        "min_child_samples": 120,
        "early_stopping_rounds": 30,
    },
    "ultra_reg": {
        "num_leaves": 12,
        "max_depth": 3,
        "learning_rate": 0.02,
        "n_estimators": 280,
        "subsample": 0.65,
        "subsample_freq": 1,
        "colsample_bytree": 0.65,
        "reg_alpha": 1.2,
        "reg_lambda": 6.0,
        "min_child_samples": 180,
        "early_stopping_rounds": 35,
    },
    "leaf_guard": {
        "num_leaves": 10,
        "max_depth": 3,
        "learning_rate": 0.03,
        "n_estimators": 220,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.60,
        "reg_alpha": 0.6,
        "reg_lambda": 5.5,
        "min_child_samples": 220,
        "early_stopping_rounds": 30,
    },
    "slow_lr": {
        "num_leaves": 18,
        "max_depth": 4,
        "learning_rate": 0.015,
        "n_estimators": 320,
        "subsample": 0.70,
        "subsample_freq": 1,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.9,
        "reg_lambda": 4.0,
        "min_child_samples": 140,
        "early_stopping_rounds": 35,
    },
    "tiny_leaf": {
        "num_leaves": 8,
        "max_depth": 3,
        "learning_rate": 0.02,
        "n_estimators": 300,
        "subsample": 0.60,
        "subsample_freq": 1,
        "colsample_bytree": 0.60,
        "reg_alpha": 1.0,
        "reg_lambda": 6.5,
        "min_child_samples": 240,
        "early_stopping_rounds": 40,
    },
    "subsample_guard": {
        "num_leaves": 14,
        "max_depth": 4,
        "learning_rate": 0.018,
        "n_estimators": 340,
        "subsample": 0.55,
        "subsample_freq": 1,
        "colsample_bytree": 0.55,
        "reg_alpha": 0.9,
        "reg_lambda": 5.0,
        "min_child_samples": 180,
        "early_stopping_rounds": 40,
    },
}


def _dedupe_features(features: list[str]) -> list[str]:
    """按声明顺序去重，保证候选特征集稳定可复现。"""
    ordered: list[str] = []
    for feature in features:
        if feature not in ordered:
            ordered.append(feature)
    return ordered


def _liquidity_plus_features() -> list[str]:
    """返回 liquidity_plus 主线使用的固定特征集合。"""
    return _dedupe_features(
        list(BASELINE_5F_COLUMNS) + list(LIQUIDITY_FEATURE_COLUMNS)
    )


def _candidate(
    candidate_id: str,
    *,
    phase: str,
    evaluation_mode: str,
    family: str,
    description: str,
    search_round: int = 0,
    features: list[str] | None = None,
    portfolio: dict[str, Any] | None = None,
    universe_filter: dict[str, str] | None = None,
    model: dict[str, Any] | None = None,
    preprocessing: dict[str, Any] | None = None,
    artifact_line_id: str | None = None,
    strategy_profile: str | None = None,
    tx1_profile_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """统一构造 focused search 候选定义，避免字段散落。"""
    return {
        "id": str(candidate_id),
        "phase": str(phase),
        "evaluation_mode": str(evaluation_mode),
        "family": str(family),
        "description": str(description),
        "search_round": int(search_round),
        "features": _dedupe_features(list(features or [])),
        "portfolio": dict(portfolio or {}),
        "universe_filter": dict(universe_filter or {}),
        "model": deepcopy(model) if model else None,
        "preprocessing": deepcopy(preprocessing) if preprocessing else None,
        "artifact_line_id": artifact_line_id,
        "strategy_profile": strategy_profile,
        "tx1_profile_overrides": dict(tx1_profile_overrides or {}),
    }


def _proxy_candidate_id(
    round_index: int,
    model_kind: str,
    reg_profile: str | None,
    use_preprocessing: bool,
) -> str:
    """把 phase-1 配置编码成稳定可读的候选 ID。"""
    reg_token = reg_profile or "default"
    preproc_token = "preproc" if use_preprocessing else "raw"
    return "liq_stab_r{}_{}_{}_{}".format(
        int(round_index),
        str(model_kind),
        str(reg_token),
        preproc_token,
    )


def _build_proxy_candidate(
    *,
    round_index: int,
    model_kind: str,
    reg_profile: str | None = None,
    use_preprocessing: bool = False,
) -> dict[str, Any]:
    """构造 phase-1 稳化搜索候选。"""
    model_payload = {"kind": str(model_kind)}
    if reg_profile and reg_profile != "default":
        model_payload["params"] = deepcopy(
            LGBM_REGULARIZATION_PROFILES.get(reg_profile, {})
        )
    description = (
        "liquidity_plus 稳化搜索：model={} reg={} preprocessing={}，优先改善 stability/cv/positive_ratio。"
    ).format(
        model_kind,
        reg_profile or "default",
        "on" if use_preprocessing else "off",
    )
    return _candidate(
        _proxy_candidate_id(
            round_index=round_index,
            model_kind=model_kind,
            reg_profile=reg_profile,
            use_preprocessing=use_preprocessing,
        ),
        phase="phase1",
        evaluation_mode="proxy",
        family="stabilization",
        description=description,
        search_round=round_index,
        features=_liquidity_plus_features(),
        model=model_payload,
        preprocessing=DEFAULT_PREPROCESSING if use_preprocessing else None,
    )


def build_liquidity_focus_candidates(
    *,
    round_index: int = 0,
    frontier_entries: list[dict[str, Any]] | None = None,
    tried_signatures: set[str] | None = None,
) -> list[dict[str, Any]]:
    """构造 phase-1 稳化搜索候选，只保留 raw lgbm 正则邻域。"""
    del frontier_entries, tried_signatures
    phase_schedule = {
        0: ["default", "heavy_reg", "ultra_reg"],
        1: ["leaf_guard", "slow_lr"],
        2: ["tiny_leaf", "subsample_guard"],
    }
    profiles = list(phase_schedule.get(int(round_index), []))
    return [
        _build_proxy_candidate(
            round_index=round_index,
            model_kind="lgbm",
            reg_profile=profile_name,
            use_preprocessing=False,
        )
        for profile_name in profiles
    ]


def _format_override_value(value: Any) -> str:
    """把 override 值编码成可读的 token。"""
    if isinstance(value, float):
        text = "{:.3f}".format(value).rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace(".", "p").replace("-", "m")


def _build_replay_candidate_id(
    round_index: int,
    artifact_line_id: str,
    strategy_profile: str,
    tx1_profile_overrides: dict[str, Any] | None,
) -> str:
    """把 phase-2 replay 配置编码成稳定可读的候选 ID。"""
    override_payload = dict(tx1_profile_overrides or {})
    if not override_payload:
        return "replay_r{}_{}_{}".format(
            int(round_index),
            str(artifact_line_id),
            str(strategy_profile),
        )
    override_tokens = [
        "{}{}".format(str(key), _format_override_value(value))
        for key, value in sorted(override_payload.items())
    ]
    return "replay_r{}_{}_{}_{}".format(
        int(round_index),
        str(artifact_line_id),
        str(strategy_profile),
        "_".join(override_tokens),
    )


def _build_replay_candidate(
    *,
    round_index: int,
    artifact_line_id: str,
    description: str,
    strategy_profile: str = "smooth",
    tx1_profile_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 phase-2 rolling_score replay 搜索候选。"""
    normalized_overrides = _normalize_replay_profile_overrides(
        strategy_profile=strategy_profile,
        tx1_profile_overrides=tx1_profile_overrides,
    )
    return _candidate(
        _build_replay_candidate_id(
            round_index=round_index,
            artifact_line_id=artifact_line_id,
            strategy_profile=strategy_profile,
            tx1_profile_overrides=normalized_overrides,
        ),
        phase="phase2",
        evaluation_mode="replay",
        family="rolling_score",
        description=description,
        search_round=round_index,
        artifact_line_id=artifact_line_id,
        strategy_profile=strategy_profile,
        tx1_profile_overrides=normalized_overrides,
    )


def _normalize_replay_profile_overrides(
    *,
    strategy_profile: str,
    tx1_profile_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """剔除与 profile 默认值等价的 no-op override，避免重复 replay。"""
    override_payload = dict(tx1_profile_overrides or {})
    if not override_payload:
        return {}
    profile_payload = load_profile(strategy_profile)
    normalized: dict[str, Any] = {}
    for key, value in override_payload.items():
        base_value = profile_payload.get(key)
        if _profile_values_equal(base_value, value):
            continue
        normalized[str(key)] = value
    return normalized


def _profile_values_equal(base_value: Any, candidate_value: Any) -> bool:
    """比较 profile 默认值和 override 是否等价，兼容数值类型。"""
    numeric_types = (int, float, np.integer, np.floating)
    if isinstance(base_value, numeric_types) and isinstance(candidate_value, numeric_types):
        return abs(float(base_value) - float(candidate_value)) < 1e-12
    return base_value == candidate_value


def _merged_profile_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    """解析 replay 候选的最终执行 profile，便于做邻域扩表。"""
    profile_name = str(candidate.get("strategy_profile") or "smooth")
    payload = load_profile(profile_name)
    payload.update(dict(candidate.get("tx1_profile_overrides") or {}))
    return payload


def _replay_record_is_expandable(record: dict[str, Any]) -> bool:
    """只允许围绕真实跑出窗口的 replay 候选继续扩表。"""
    entry = dict(record.get("entry") or {})
    if str(entry.get("status") or "") in {"crash", "invalid"}:
        return False
    return int(_metric(record.get("summary") or {}, "replay", "num_windows")) > 0


def _build_replay_probe_neighbors(
    *,
    round_index: int,
    source_candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    """围绕单个健康 replay 候选生成少量可解释执行 probe。"""
    artifact_line_id = str(source_candidate.get("artifact_line_id") or "combo_b25_h45")
    profile_name = str(source_candidate.get("strategy_profile") or "smooth")
    merged_profile = _merged_profile_payload(source_candidate)
    existing_overrides = dict(source_candidate.get("tx1_profile_overrides") or {})

    base_cap = round(float(merged_profile.get("single_stock_cap", 0.10) or 0.10), 3)
    base_turnover = round(
        float(merged_profile.get("turnover_threshold", 0.30) or 0.30), 3
    )
    base_halflife = int(float(merged_profile.get("ema_halflife", 5) or 5))
    base_min_weight = round(
        float(merged_profile.get("ema_min_weight", 0.005) or 0.005), 3
    )

    neighbor_overrides: list[dict[str, Any]] = []
    if not existing_overrides:
        neighbor_overrides = [
            {"single_stock_cap": max(0.06, round(base_cap - 0.01, 3))},
            {"turnover_threshold": max(0.15, round(base_turnover - 0.10, 3))},
            {
                "ema_halflife": min(12, base_halflife + 3),
                "ema_min_weight": max(0.002, round(base_min_weight - 0.001, 3)),
            },
        ]
    elif "single_stock_cap" in existing_overrides:
        neighbor_overrides = [
            {"single_stock_cap": max(0.06, round(base_cap - 0.01, 3))},
            {"single_stock_cap": min(0.12, round(base_cap + 0.01, 3))},
            {
                "single_stock_cap": round(base_cap, 3),
                "turnover_threshold": max(0.15, round(base_turnover - 0.05, 3)),
            },
        ]
    elif "turnover_threshold" in existing_overrides:
        neighbor_overrides = [
            {"turnover_threshold": max(0.10, round(base_turnover - 0.05, 3))},
            {"turnover_threshold": min(0.40, round(base_turnover + 0.05, 3))},
            {
                "turnover_threshold": round(base_turnover, 3),
                "single_stock_cap": max(0.07, round(base_cap - 0.01, 3)),
            },
        ]
    elif "ema_halflife" in existing_overrides or "ema_min_weight" in existing_overrides:
        neighbor_overrides = [
            {
                "ema_halflife": max(3, base_halflife - 2),
                "ema_min_weight": min(0.008, round(base_min_weight + 0.001, 3)),
            },
            {
                "ema_halflife": min(14, base_halflife + 2),
                "ema_min_weight": max(0.002, round(base_min_weight - 0.001, 3)),
            },
        ]

    return [
        candidate
        for overrides in neighbor_overrides
        for candidate in [
            _build_replay_candidate(
                round_index=round_index,
                artifact_line_id=artifact_line_id,
                strategy_profile=profile_name,
                tx1_profile_overrides=overrides,
                description=(
                    "围绕 {}@{} 的真实 replay probe 扩表，只继续验证可执行执行层邻域。"
                ).format(artifact_line_id, profile_name),
            )
        ]
        if candidate.get("tx1_profile_overrides")
    ]


def build_replay_search_candidates(
    *,
    round_index: int = 0,
    replay_entries: list[dict[str, Any]] | None = None,
    tried_signatures: set[str] | None = None,
) -> list[dict[str, Any]]:
    """围绕 `combo_b25_h45` 邻域构造真实 replay 搜索候选。"""
    del tried_signatures
    if round_index <= 0:
        return [
            _build_replay_candidate(
                round_index=0,
                artifact_line_id="combo_h45_bonus1",
                strategy_profile="smooth",
                description="PLAYBOOK 证明过的 Top20/Top45 + bonus1 邻域，对照默认执行线。",
            ),
            _build_replay_candidate(
                round_index=0,
                artifact_line_id="combo_h40_bonus1",
                strategy_profile="smooth",
                description="PLAYBOOK 证明过的 Top20/Top40 邻域基线，对照 `combo_b25_h45` 的真实 replay。",
            ),
            _build_replay_candidate(
                round_index=0,
                artifact_line_id="combo_b25_h45",
                strategy_profile="smooth",
                tx1_profile_overrides={"single_stock_cap": 0.08},
                description="默认 executable line 降单票上限，先验证仓位分散 override 是否可执行。",
            ),
            _build_replay_candidate(
                round_index=0,
                artifact_line_id="combo_b25_h45",
                strategy_profile="smooth",
                tx1_profile_overrides={"turnover_threshold": 0.25},
                description="围绕新默认 executable line 略收紧换手门槛，先验证收益与稳定性的再平衡。",
            ),
            _build_replay_candidate(
                round_index=0,
                artifact_line_id="combo_b25_h45",
                strategy_profile="smooth",
                tx1_profile_overrides={
                    "ema_halflife": 8,
                    "ema_min_weight": 0.004,
                },
                description="默认 executable line 调整 EMA 粘性，先验证平滑 override 是否可执行。",
            ),
        ]

    ranked_entries = list(replay_entries or [])
    healthy_records = [
        record for record in ranked_entries if _replay_record_is_expandable(record)
    ]
    if not healthy_records:
        return []

    candidates: list[dict[str, Any]] = []
    for record in healthy_records[:2]:
        candidates.extend(
            _build_replay_probe_neighbors(
                round_index=round_index,
                source_candidate=dict(record.get("candidate") or {}),
            )
        )
    return candidates


def apply_cross_section_filter(
    frame: pd.DataFrame,
    universe_filter: dict[str, str] | None,
) -> pd.DataFrame:
    """按日期做截面过滤，但保留每只股票完整历史的特征计算结果。"""
    if not universe_filter:
        return frame.copy()
    filtered_parts = []
    for _, day_df in frame.groupby("date", sort=True):
        mask = pd.Series(True, index=day_df.index)
        for column_name, condition in (universe_filter or {}).items():
            if column_name not in day_df.columns or day_df[column_name].isna().all():
                continue
            if condition == "above_median":
                median_value = day_df[column_name].median()
                mask = mask & (day_df[column_name] >= median_value)
        filtered_parts.append(day_df.loc[mask])
    if not filtered_parts:
        return frame.iloc[:0].copy()
    return pd.concat(filtered_parts, ignore_index=True)


def build_labeled_panel(
    *,
    raw_df: pd.DataFrame,
    label_transform: str = "rank",
    horizon_days: int = 20,
) -> pd.DataFrame:
    """构建一次可复用的 labeled panel，避免每个候选都重复算特征。"""
    dataset = DatasetBuilder(input_window=60).build(raw_df)
    return LabelBuilder(horizon=horizon_days, transform=label_transform).build(dataset)


def _build_cost_config(config: dict[str, Any]) -> CostConfig | None:
    """按实验配置解析成本层参数。"""
    costs_cfg = dict(config.get("costs") or {})
    if not costs_cfg.get("enabled", False):
        return None
    return CostConfig(
        commission_rate=costs_cfg.get("commission_rate", 0.0008),
        stamp_tax_rate=costs_cfg.get("stamp_tax_rate", 0.0005),
        slippage_bps=costs_cfg.get("slippage_bps", 5.0),
    )


def _build_preprocessor(config: dict[str, Any]) -> FeaturePreprocessor | None:
    """按实验配置构建可选的截面预处理器。"""
    preprocessing_cfg = dict(config.get("preprocessing") or {})
    if not preprocessing_cfg.get("enabled", False):
        return None
    return FeaturePreprocessor(
        neutralize=preprocessing_cfg.get("neutralize", True),
        winsorize_scale=preprocessing_cfg.get("winsorize_scale", 5.0),
        standardize=preprocessing_cfg.get("standardize", True),
    )


def _aggregate_metric_group(
    fold_results: list[dict[str, Any]],
    metric_key: str,
) -> dict[str, float]:
    """聚合 fold 级指标，产出 experiment.json 可读的均值摘要。"""
    if not fold_results:
        return {}
    metric_names = sorted((fold_results[0].get(metric_key) or {}).keys())
    return {
        metric_name: float(
            np.mean(
                [
                    float((fold_result.get(metric_key) or {}).get(metric_name, 0.0))
                    for fold_result in fold_results
                ]
            )
        )
        for metric_name in metric_names
    }


def _build_prediction_frame(test_df: pd.DataFrame) -> pd.DataFrame:
    """裁出最关键的预测列，避免实验产物过大。"""
    keep_columns = [
        column_name
        for column_name in [
            "date",
            "order_book_id",
            "prediction",
            "label_return_raw",
            "target_label",
        ]
        if column_name in test_df.columns
    ]
    return test_df.loc[:, keep_columns].copy()


def run_focused_candidate_trial(
    *,
    run_root: str | Path,
    experiment_index: int,
    labeled_df: pd.DataFrame,
    config: dict[str, Any],
    stage: str | None = None,
    max_folds: int | None = None,
) -> dict[str, Any]:
    """在预先构建好的 labeled panel 上执行单个 proxy 候选。"""
    cfg = normalize_config(deepcopy(config))
    trial_df = labeled_df.copy()
    feature_cols = [
        column_name
        for column_name in (cfg.get("features") or [])
        if column_name in trial_df.columns
    ]
    if not feature_cols:
        raise ValueError("focused candidate has no available feature columns")

    trial_df = (
        trial_df.dropna(subset=feature_cols)
        .sort_values(["date", "order_book_id"])
        .reset_index(drop=True)
    )
    splitter = WalkForwardSplitter(
        train_years=cfg["splitter"]["train_years"],
        val_months=cfg["splitter"]["val_months"],
        test_months=cfg["splitter"]["test_months"],
        embargo_days=cfg["splitter"]["embargo_days"],
    )
    folds = splitter.split(trial_df)
    if max_folds is not None:
        folds = folds[: int(max_folds)]

    portfolio_builder = PortfolioProxy(
        buy_top_k=cfg["portfolio"]["buy_top_k"],
        hold_top_k=cfg["portfolio"]["hold_top_k"],
        rebalance_interval=cfg["portfolio"]["rebalance_interval"],
        holding_bonus=cfg["portfolio"]["holding_bonus"],
    )
    preprocessor = _build_preprocessor(cfg)
    cost_config = _build_cost_config(cfg)

    fold_results = []
    for fold_index, fold in enumerate(folds, start=1):
        train_df = fold["train_df"].copy()
        val_df = fold["val_df"].copy()
        test_df = fold["test_df"].copy()

        # 预处理必须严格限定在 fold 内，避免把未来信息泄漏回训练集。
        if preprocessor is not None:
            train_df = preprocessor.transform(train_df, feature_cols)
            val_df = preprocessor.transform(val_df, feature_cols)
            test_df = preprocessor.transform(test_df, feature_cols)

        model = create_model(cfg["model"]["kind"], params=cfg["model"].get("params"))
        fit_kwargs = {}
        if supports_validation(model):
            fit_kwargs["val_X"] = val_df[feature_cols]
            fit_kwargs["val_y"] = val_df["target_label"]
        model.fit(train_df[feature_cols], train_df["target_label"], **fit_kwargs)

        val_df["prediction"] = model.predict(val_df[feature_cols])
        test_df["prediction"] = model.predict(test_df[feature_cols])

        prediction_metrics = evaluate_predictions(
            test_df, top_k=cfg["evaluation"]["top_k"]
        )
        validation_metrics = evaluate_predictions(
            val_df, top_k=cfg["evaluation"]["top_k"]
        )
        weights_df = portfolio_builder.build(
            test_df[["date", "order_book_id", "prediction"]]
        )
        portfolio_returns = build_portfolio_returns(
            test_df,
            weights_df,
            horizon_days=cfg["labels"]["horizon"],
        )
        portfolio_metrics = evaluate_portfolios(
            portfolio_returns, cost_config=cost_config
        )

        fold_results.append(
            {
                "fold_index": fold_index,
                "prediction_metrics": prediction_metrics,
                "validation_metrics": validation_metrics,
                "portfolio_metrics": portfolio_metrics,
                "row_counts": {
                    "train": int(len(train_df)),
                    "val": int(len(val_df)),
                    "test": int(len(test_df)),
                },
                "date_range": {
                    "train_end": fold.get("train_end"),
                    "val_start": fold.get("val_start"),
                    "val_end": fold.get("val_end"),
                    "test_start": fold.get("test_start"),
                    "test_end": fold.get("test_end"),
                },
                "predictions_df": _build_prediction_frame(test_df),
                "weights_df": weights_df,
                "portfolio_returns_df": portfolio_returns,
                "model_heads": ["return"],
            }
        )

    experiment_result = {
        "experiment_name": cfg.get("experiment_name"),
        "model_kind": cfg["model"]["kind"],
        "model_heads": ["return"],
        "prediction_columns": ["prediction"],
        "feature_columns": list(feature_cols),
        "fold_results": fold_results,
        "aggregate_metrics": {
            "prediction": _aggregate_metric_group(fold_results, "prediction_metrics"),
            "portfolio": _aggregate_metric_group(fold_results, "portfolio_metrics"),
            "robustness": {
                "stability": compute_proxy_stability_score(
                    fold_results, metric_key="rank_ic_mean"
                ),
                "overfit_flags": detect_overfit_flags(fold_results),
                "regime_scores": compute_regime_scores(
                    fold_results, metric_key="rank_ic_mean"
                ),
            },
        },
    }

    experiment_root = build_stage_experiment_root(
        run_root,
        experiment_index=experiment_index,
        stage=stage,
    )
    saved_path = save_experiment(
        experiment_result,
        str(experiment_root),
        config=cfg,
        experiment_name=cfg.get("experiment_name"),
    )
    experiment_result["output_dir"] = saved_path
    summary = summarize_experiment(experiment_result)
    summary["experiment_path"] = saved_path
    if stage is not None:
        summary["stage"] = str(stage)
    return summary


def _quarter_key(year_quarter: tuple[int, int]) -> str:
    """把季度 tuple 编码成稳定字符串，便于 JSON 落盘。"""
    return "{}Q{}".format(int(year_quarter[0]), int(year_quarter[1]))


def _replay_cv(quarterly_scores: dict[Any, float]) -> float:
    """按 rolling_score 季度分布计算 CV。"""
    values = [float(value) for value in quarterly_scores.values()]
    if not values:
        return 0.0
    mean_value = float(np.mean(values))
    if abs(mean_value) < 1e-9:
        return 0.0
    return float(np.std(values, ddof=0) / abs(mean_value))


def _replay_positive_ratio(quarterly_scores: dict[Any, float]) -> float:
    """用季度得分中高于 30 分的比例近似 replay 侧的正向覆盖率。"""
    values = [float(value) for value in quarterly_scores.values()]
    if not values:
        return 0.0
    return float(sum(value >= 30.0 for value in values) / len(values))


def _extract_tx1_replay_mod_payload(window_result: dict[str, Any]) -> dict[str, Any]:
    """从窗口结果里提取 TX1 诊断 mod 返回。"""
    mod_results = dict(window_result.get("mod_results") or {})
    return dict(mod_results.get(TX1_DIAGNOSTIC_MOD_NAME) or {})


def _summarize_failed_replay_windows(
    failed_windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """裁剪失败窗口信息，便于在结果文件里直接定位异常。"""
    return [
        {
            "idx": int(item.get("idx", 0) or 0),
            "start": str(item.get("start") or ""),
            "end": str(item.get("end") or ""),
            "error": str(item.get("error") or ""),
        }
        for item in list(failed_windows or [])
    ]


def _summarize_replay_run_diagnostics(
    *,
    window_results: list[dict[str, Any]],
    failed_windows: list[dict[str, Any]],
    total_requested_windows: int,
) -> dict[str, Any]:
    """汇总 replay 运行健康度，避免把空跑误判成策略失败。"""
    diagnostics = {
        "health_code": "ok",
        "total_requested_windows": int(total_requested_windows),
        "successful_windows": int(len(window_results)),
        "failed_window_count": int(len(failed_windows)),
        "active_window_count": 0,
        "sparse_window_count": 0,
        "no_position_window_count": 0,
        "rebalance_checks_total": 0,
        "executed_rebalances_total": 0,
        "missing_signal_days_total": 0,
        "turnover_skips_total": 0,
        "failure_examples": _summarize_failed_replay_windows(failed_windows)[:5],
    }

    for item in window_results:
        sample_diagnostics = dict(item.get("sample_diagnostics") or {})
        active_days = int(sample_diagnostics.get("active_days", 0) or 0)
        if bool(sample_diagnostics.get("sparse")):
            diagnostics["sparse_window_count"] += 1
        if active_days > 0:
            diagnostics["active_window_count"] += 1
        else:
            diagnostics["no_position_window_count"] += 1

        tx1_payload = _extract_tx1_replay_mod_payload(item)
        strategy_diagnostics = dict(tx1_payload.get("strategy_diagnostics") or {})
        diagnostics["rebalance_checks_total"] += int(
            strategy_diagnostics.get("rebalance_checks", 0) or 0
        )
        diagnostics["executed_rebalances_total"] += int(
            strategy_diagnostics.get("executed_rebalances", 0) or 0
        )
        diagnostics["missing_signal_days_total"] += int(
            strategy_diagnostics.get("missing_signal_days", 0) or 0
        )
        diagnostics["turnover_skips_total"] += int(
            strategy_diagnostics.get("turnover_skips", 0) or 0
        )

    # 优先把“完全没跑起来”的情况标成基础设施/执行链路异常。
    if not window_results:
        diagnostics["health_code"] = (
            "window_exception" if failed_windows else "no_successful_window"
        )
    elif diagnostics["active_window_count"] <= 0:
        diagnostics["health_code"] = "no_active_positions"
    elif diagnostics["executed_rebalances_total"] <= 0:
        if diagnostics["missing_signal_days_total"] > 0:
            diagnostics["health_code"] = "missing_signal_dominant"
        elif diagnostics["turnover_skips_total"] > 0:
            diagnostics["health_code"] = "turnover_blocked"
        else:
            diagnostics["health_code"] = "no_rebalance_executed"
    elif diagnostics["failed_window_count"] > 0:
        diagnostics["health_code"] = "partial_window_failures"

    return diagnostics


def _summarize_replay_windows(window_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 rolling_score 窗口结果裁剪成可读且可落盘的摘要。"""
    summarized = []
    for item in window_results:
        raw_summary = {
            str(key): float(value)
            for key, value in dict(item.get("summary") or {}).items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
        tx1_payload = _extract_tx1_replay_mod_payload(item)
        summarized.append(
            {
                "idx": int(item.get("idx", 0)),
                "start": str(item.get("start")),
                "end": str(item.get("end")),
                "score": float(item.get("score", 0.0)),
                "summary": raw_summary,
                "sample_diagnostics": dict(item.get("sample_diagnostics") or {}),
                "strategy_diagnostics": dict(
                    tx1_payload.get("strategy_diagnostics") or {}
                ),
                "runtime_diagnostics": dict(tx1_payload.get("runtime") or {}),
                "last_signal_date": tx1_payload.get("last_signal_date"),
                "pending_turnover": float(
                    tx1_payload.get("pending_turnover", 0.0) or 0.0
                ),
            }
        )
    return summarized


def _save_replay_candidate_result(
    *,
    run_root: str | Path,
    experiment_index: int,
    candidate: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    """把 replay 候选的窗口摘要落到独立目录，便于后续复盘。"""
    experiment_root = build_stage_experiment_root(
        run_root,
        experiment_index=experiment_index,
        stage="replay",
    )
    experiment_root.mkdir(parents=True, exist_ok=True)
    summary_path = experiment_root / "replay_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "candidate": deepcopy(candidate),
                "summary": deepcopy(payload),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    return str(experiment_root)


def run_replay_candidate_trial(
    *,
    run_root: str | Path,
    experiment_index: int,
    candidate: dict[str, Any],
    cash: float = 1_000_000,
    selected_indices: list[int] | None = None,
) -> dict[str, Any]:
    """执行单个真实 rolling_score replay 候选，并返回统一摘要结构。"""
    artifact_line_id = str(candidate.get("artifact_line_id") or "combo_b25_h45")
    strategy_profile = str(candidate.get("strategy_profile") or "smooth")
    extra_config = {
        "strategy_profile": strategy_profile,
        "tx1_artifact_line": artifact_line_id,
    }
    if candidate.get("tx1_profile_overrides"):
        extra_config["tx1_profile_overrides"] = dict(
            candidate.get("tx1_profile_overrides") or {}
        )

    raw_replay_output = run_rolling_backtests(
        str(ROLLING_SCORE_STRATEGY_FILE),
        cash=float(cash),
        selected_indices=selected_indices,
        extra_mods=[TX1_DIAGNOSTIC_MOD_NAME],
        mod_configs={
            TX1_DIAGNOSTIC_MOD_NAME: {"lib": TX1_DIAGNOSTIC_MOD_LIB},
        },
        extra_config=extra_config,
        return_details=True,
    )
    if isinstance(raw_replay_output, dict) and "windows" in raw_replay_output:
        window_results = list(raw_replay_output.get("windows") or [])
        failed_windows = _summarize_failed_replay_windows(
            list(raw_replay_output.get("failed_windows") or [])
        )
        total_requested_windows = int(
            raw_replay_output.get("total_windows") or len(window_results)
        )
    else:
        # 兼容旧测试桩或旧调用约定，未提供详情时退化成“仅有成功窗口”。
        window_results = list(raw_replay_output or [])
        failed_windows = []
        total_requested_windows = int(
            len(selected_indices) if selected_indices is not None else len(window_results)
        )

    quarterly_scores, quarterly_raw_indicators = project_replay_to_quarters(
        window_results
    )
    if quarterly_scores:
        composite_score, core_indicators = compute_replay_composite_score(
            quarterly_scores,
            quarterly_raw_indicators,
        )
    else:
        composite_score, core_indicators = 0.0, {
            "annualized_returns": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
        }
    stability_score = compute_replay_stability_score(quarterly_scores)
    risk_alerts = detect_replay_risk_alerts(quarterly_scores)
    risk_codes = [str(item.get("code")) for item in risk_alerts]
    replay_diagnostics = _summarize_replay_run_diagnostics(
        window_results=window_results,
        failed_windows=failed_windows,
        total_requested_windows=total_requested_windows,
    )
    summary = {
        "prediction": {
            "rank_ic_mean": float(composite_score),
            "top_bucket_spread_mean": float(composite_score),
        },
        "portfolio": {
            "net_mean_return": float(core_indicators.get("annualized_returns", 0.0)),
            "max_drawdown": abs(float(core_indicators.get("max_drawdown", 0.0))),
            "sharpe": float(core_indicators.get("sharpe", 0.0)),
            "win_rate": float(core_indicators.get("win_rate", 0.0)),
            "mean_turnover": 0.0,
        },
        "robustness": {
            "stability": {
                "stability_score": float(stability_score),
                "cv": _replay_cv(quarterly_scores),
            },
            "regime_scores": {
                "metric_consistency": {
                    "positive_ratio": _replay_positive_ratio(quarterly_scores),
                }
            },
            "risk_alerts": deepcopy(risk_alerts),
            "risk_codes": list(risk_codes),
        },
        "replay": {
            "artifact_line_id": artifact_line_id,
            "strategy_profile": strategy_profile,
            "profile_overrides": dict(candidate.get("tx1_profile_overrides") or {}),
            "num_windows": int(len(window_results)),
            "total_requested_windows": int(total_requested_windows),
            "failed_window_count": int(len(failed_windows)),
            "composite_score": float(composite_score),
            "stability_score": float(stability_score),
            "quarterly_scores": {
                _quarter_key(key): float(value)
                for key, value in quarterly_scores.items()
            },
            "quarterly_raw_indicators": {
                _quarter_key(key): {
                    str(indicator): float(indicator_value)
                    for indicator, indicator_value in dict(values).items()
                }
                for key, values in quarterly_raw_indicators.items()
            },
            "core_indicators": {
                str(key): float(value)
                for key, value in dict(core_indicators).items()
            },
            "risk_codes": list(risk_codes),
            "risk_alerts": deepcopy(risk_alerts),
            "diagnostics": replay_diagnostics,
            "windows": _summarize_replay_windows(window_results),
            "failed_windows": failed_windows,
        },
    }
    experiment_path = _save_replay_candidate_result(
        run_root=run_root,
        experiment_index=experiment_index,
        candidate=candidate,
        payload=summary,
    )
    summary["experiment_path"] = experiment_path
    summary["stage"] = "replay"
    return summary


def _proxy_target_flags(summary: dict[str, Any]) -> list[str]:
    """提取 phase-1 关心的三个 overfit flags。"""
    failed = []
    for flag_name in TARGET_PROXY_FLAGS:
        if _metric(summary, "robustness", "overfit_flags", flag_name, default=False):
            failed.append(flag_name)
    return failed


def judge_phase1_candidate(
    candidate_summary: dict[str, Any] | None,
    *,
    anchor_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """phase-1 只做“稳化 pruning”，不直接产出 champion。"""
    if not candidate_summary:
        return {
            "status": "crash",
            "reason_code": "missing_summary",
            "failed_guards": ["summary_missing"],
            "score_delta": {},
            "best_score_delta": {},
        }

    anchor_summary = dict(anchor_summary or {})
    base_decision = judge_candidate(
        candidate_summary,
        baseline_summary=anchor_summary,
        best_summary=anchor_summary,
        stage="full",
    )
    candidate_flags = _proxy_target_flags(candidate_summary)
    anchor_flags = _proxy_target_flags(anchor_summary)
    stability_delta = float(
        _metric(
            candidate_summary,
            "robustness",
            "stability",
            "stability_score",
        )
    ) - float(
        _metric(
            anchor_summary,
            "robustness",
            "stability",
            "stability_score",
        )
    )
    positive_ratio_delta = float(
        _metric(
            candidate_summary,
            "robustness",
            "regime_scores",
            "metric_consistency",
            "positive_ratio",
        )
    ) - float(
        _metric(
            anchor_summary,
            "robustness",
            "regime_scores",
            "metric_consistency",
            "positive_ratio",
        )
    )
    net_return_delta = float(
        _metric(candidate_summary, "portfolio", "net_mean_return")
    ) - float(_metric(anchor_summary, "portfolio", "net_mean_return"))
    non_target_failed = [
        item
        for item in list(base_decision.get("failed_guards") or [])
        if item not in TARGET_PROXY_FLAGS
    ]

    if not candidate_flags and not non_target_failed:
        return {
            "status": "frontier_seed",
            "reason_code": "flags_cleared",
            "failed_guards": [],
            "score_delta": dict(base_decision.get("score_delta") or {}),
            "best_score_delta": dict(base_decision.get("best_score_delta") or {}),
        }

    if (
        len(candidate_flags) < len(anchor_flags)
        and stability_delta >= 0.0
        and positive_ratio_delta >= -0.02
    ):
        return {
            "status": "frontier_seed",
            "reason_code": "flags_reduced",
            "failed_guards": list(non_target_failed) + list(candidate_flags),
            "score_delta": dict(base_decision.get("score_delta") or {}),
            "best_score_delta": dict(base_decision.get("best_score_delta") or {}),
        }

    if (
        not candidate_flags
        and stability_delta >= 0.0
        and net_return_delta >= -0.0002
    ):
        return {
            "status": "frontier_seed",
            "reason_code": "stable_borderline",
            "failed_guards": list(non_target_failed),
            "score_delta": dict(base_decision.get("score_delta") or {}),
            "best_score_delta": dict(base_decision.get("best_score_delta") or {}),
        }

    return {
        "status": "discard",
        "reason_code": str(base_decision.get("reason_code") or "proxy_rejected"),
        "failed_guards": list(base_decision.get("failed_guards") or []),
        "score_delta": dict(base_decision.get("score_delta") or {}),
        "best_score_delta": dict(base_decision.get("best_score_delta") or {}),
    }


def _build_replay_score_delta(
    candidate_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> dict[str, float]:
    """构造 replay 侧相对 baseline 的关键差值。"""
    return {
        "composite_score": float(
            _metric(candidate_summary, "replay", "composite_score")
        ) - float(_metric(baseline_summary, "replay", "composite_score")),
        "annualized_returns": float(
            _metric(candidate_summary, "portfolio", "net_mean_return")
        ) - float(_metric(baseline_summary, "portfolio", "net_mean_return")),
        "max_drawdown": float(
            _metric(candidate_summary, "portfolio", "max_drawdown")
        ) - float(_metric(baseline_summary, "portfolio", "max_drawdown")),
        "stability_score": float(
            _metric(candidate_summary, "replay", "stability_score")
        ) - float(_metric(baseline_summary, "replay", "stability_score")),
        "win_rate": float(
            _metric(candidate_summary, "portfolio", "win_rate")
        ) - float(_metric(baseline_summary, "portfolio", "win_rate")),
    }


def _replay_has_material_improvement(score_delta: dict[str, float]) -> bool:
    """判断 replay 候选是否形成材料性改进。"""
    return (
        float(score_delta.get("composite_score", 0.0)) > 0.5
        and float(score_delta.get("annualized_returns", 0.0)) >= -0.002
        and float(score_delta.get("max_drawdown", 0.0)) <= 0.01
        and float(score_delta.get("stability_score", 0.0)) >= -2.0
    )


def judge_replay_candidate(
    candidate_summary: dict[str, Any] | None,
    *,
    baseline_summary: dict[str, Any] | None,
    best_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """phase-2 以 rolling_score replay 为准，允许产出 keep / champion。"""
    if not candidate_summary:
        return {
            "status": "crash",
            "reason_code": "missing_summary",
            "failed_guards": ["summary_missing"],
            "score_delta": {},
            "best_score_delta": {},
        }

    baseline_summary = dict(baseline_summary or {})
    best_summary = dict(best_summary or baseline_summary)
    num_windows = int(_metric(candidate_summary, "replay", "num_windows"))
    replay_health_code = str(
        _metric(candidate_summary, "replay", "diagnostics", "health_code", default="")
    )
    if num_windows <= 0:
        return {
            "status": "crash",
            "reason_code": "replay_infra_failure",
            "failed_guards": ["num_windows", replay_health_code or "empty_replay"],
            "score_delta": _build_replay_score_delta(
                candidate_summary, baseline_summary
            ),
            "best_score_delta": _build_replay_score_delta(
                candidate_summary, best_summary
            ),
        }

    if replay_health_code in {
        "no_active_positions",
        "missing_signal_dominant",
        "turnover_blocked",
        "no_rebalance_executed",
    }:
        return {
            "status": "crash",
            "reason_code": "replay_inactive",
            "failed_guards": [replay_health_code],
            "score_delta": _build_replay_score_delta(
                candidate_summary, baseline_summary
            ),
            "best_score_delta": _build_replay_score_delta(
                candidate_summary, best_summary
            ),
        }

    risk_codes = set(_metric(candidate_summary, "replay", "risk_codes", default=[]))
    score_delta = _build_replay_score_delta(candidate_summary, baseline_summary)
    best_score_delta = _build_replay_score_delta(candidate_summary, best_summary)

    if (
        "performance_decay" in risk_codes
        and float(score_delta.get("composite_score", 0.0)) < 1.0
    ):
        return {
            "status": "discard",
            "reason_code": "replay_risk_alert",
            "failed_guards": ["performance_decay"],
            "score_delta": score_delta,
            "best_score_delta": best_score_delta,
        }

    if _replay_has_material_improvement(score_delta):
        if _replay_has_material_improvement(best_score_delta):
            return {
                "status": "champion",
                "reason_code": "replay_improved",
                "failed_guards": [],
                "score_delta": score_delta,
                "best_score_delta": best_score_delta,
            }
        return {
            "status": "keep",
            "reason_code": "replay_pass_not_best",
            "failed_guards": [],
            "score_delta": score_delta,
            "best_score_delta": best_score_delta,
        }

    return {
        "status": "discard",
        "reason_code": "replay_no_material_improvement",
        "failed_guards": list(risk_codes),
        "score_delta": score_delta,
        "best_score_delta": best_score_delta,
    }


def _candidate_signature(candidate: dict[str, Any]) -> str:
    """对候选关键字段做稳定序列化，用于扩表去重。"""
    payload = {
        "phase": candidate.get("phase"),
        "evaluation_mode": candidate.get("evaluation_mode"),
        "features": list(candidate.get("features") or []),
        "portfolio": dict(candidate.get("portfolio") or {}),
        "universe_filter": dict(candidate.get("universe_filter") or {}),
        "model": deepcopy(candidate.get("model") or {}),
        "preprocessing": deepcopy(candidate.get("preprocessing") or {}),
        "artifact_line_id": candidate.get("artifact_line_id"),
        "strategy_profile": candidate.get("strategy_profile"),
        "tx1_profile_overrides": dict(candidate.get("tx1_profile_overrides") or {}),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def _build_entry_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """统一抽取 proxy / replay 两侧 leaderboard 会用到的指标。"""
    metrics = summarize_metrics(summary)
    replay_payload = dict(summary.get("replay") or {})
    portfolio_payload = dict(summary.get("portfolio") or {})
    metrics["composite_score"] = float(replay_payload.get("composite_score", 0.0))
    metrics["annualized_returns"] = float(
        portfolio_payload.get("net_mean_return", 0.0)
    )
    metrics["win_rate"] = float(portfolio_payload.get("win_rate", 0.0))
    metrics["num_windows"] = float(replay_payload.get("num_windows", 0.0))
    return metrics


def _build_leaderboard_entry(
    *,
    candidate: dict[str, Any],
    summary: dict[str, Any],
    decision: dict[str, Any],
    stage_reached: str,
) -> dict[str, Any]:
    """把候选摘要整理成统一 leaderboard 行。"""
    return {
        "candidate_id": candidate["id"],
        "candidate_signature": _candidate_signature(candidate),
        "phase": str(candidate.get("phase") or ""),
        "evaluation_mode": str(candidate.get("evaluation_mode") or ""),
        "family": candidate.get("family", ""),
        "description": candidate["description"],
        "status": str(decision.get("status") or "discard"),
        "reason_code": str(decision.get("reason_code") or "unknown"),
        "stage_reached": str(stage_reached),
        "search_round": int(candidate.get("search_round") or 0),
        "metrics": _build_entry_metrics(summary),
        "score_delta": dict(decision.get("score_delta") or {}),
        "best_score_delta": dict(decision.get("best_score_delta") or {}),
        "failed_guards": list(decision.get("failed_guards") or []),
        "artifact_line_id": candidate.get("artifact_line_id"),
        "strategy_profile": candidate.get("strategy_profile"),
        "profile_overrides": dict(candidate.get("tx1_profile_overrides") or {}),
        "risk_codes": list(_metric(summary, "replay", "risk_codes", default=[])),
        "num_windows": int(_metric(summary, "replay", "num_windows", default=0)),
        "replay_health_code": str(
            _metric(summary, "replay", "diagnostics", "health_code", default="")
        ),
        "experiment_path": str(summary.get("experiment_path") or ""),
    }


def _sort_leaderboard(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按候选命运和 replay/proxy 价值排序，方便晨间快速验收。"""
    status_rank = {
        "champion": 0,
        "keep": 1,
        "frontier_seed": 2,
        "crash": 3,
        "discard": 4,
        "invalid": 5,
    }
    phase_rank = {
        "phase2": 0,
        "phase1": 1,
    }
    return sorted(
        entries,
        key=lambda item: (
            status_rank.get(item["status"], 99),
            phase_rank.get(item.get("phase"), 99),
            -float((item.get("metrics") or {}).get("composite_score", 0.0)),
            -float((item.get("metrics") or {}).get("net_mean_return", 0.0)),
            float((item.get("metrics") or {}).get("max_drawdown", 1.0)),
            -float((item.get("metrics") or {}).get("stability_score", 0.0)),
        ),
    )


def _write_outputs(run_root: str | Path, result: dict[str, Any]) -> None:
    """持续把 focused search 的当前状态写成 JSON 和 TSV。"""
    run_root = Path(run_root)
    json_path = run_root / "focused_results.json"
    tsv_path = run_root / "focused_leaderboard.tsv"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    rows = [LEADERBOARD_HEADER]
    for entry in result.get("leaderboard") or []:
        metrics = entry.get("metrics") or {}
        rows.append(
            "\t".join(
                [
                    str(entry["candidate_id"]),
                    str(entry.get("phase", "")),
                    str(entry.get("evaluation_mode", "")),
                    str(entry.get("family", "")),
                    str(entry["status"]),
                    str(entry["stage_reached"]),
                    str(entry.get("search_round", 0)),
                    "{:.6f}".format(float(metrics.get("net_mean_return", 0.0))),
                    "{:.6f}".format(float(metrics.get("max_drawdown", 0.0))),
                    "{:.6f}".format(float(metrics.get("stability_score", 0.0))),
                    "{:.6f}".format(float(metrics.get("composite_score", 0.0))),
                    str(entry.get("num_windows", 0)),
                    str(entry.get("replay_health_code", "")),
                    str(entry["reason_code"]),
                    str(entry.get("artifact_line_id") or ""),
                    str(entry.get("strategy_profile") or ""),
                    json.dumps(
                        entry.get("profile_overrides") or {},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    str(entry["experiment_path"]),
                    str(entry["description"]),
                ]
            )
        )
    tsv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _count_phase_entries(
    entries: list[dict[str, Any]],
    *,
    phase: str,
    status: str | None = None,
) -> int:
    """统计某个 phase 下的候选数量。"""
    return int(
        sum(
            1
            for entry in entries
            if entry.get("phase") == phase
            and (status is None or entry.get("status") == status)
        )
    )


def _build_partial_result(
    *,
    run_tag: str,
    run_root: Path,
    baselines: dict[str, Any],
    leaderboard: list[dict[str, Any]],
    champion_entry: dict[str, Any] | None,
    candidates_total: int,
    candidates_evaluated: int,
    started_at: datetime,
    deadline_at: datetime,
    status: str,
    stabilization_rounds_started: int,
    replay_rounds_started: int,
) -> dict[str, Any]:
    """构造可持续落盘的 focused search 状态快照。"""
    now = datetime.now()
    sorted_leaderboard = _sort_leaderboard(list(leaderboard))
    frontier_entries = [
        entry for entry in sorted_leaderboard if entry.get("status") == "frontier_seed"
    ]
    return {
        "run_tag": str(run_tag),
        "run_root": str(run_root),
        "search_name": "liquidity_focus_2phase_budget",
        "status": str(status),
        "started_at": started_at.isoformat(),
        "deadline_at": deadline_at.isoformat(),
        "updated_at": now.isoformat(),
        "elapsed_minutes": round((now - started_at).total_seconds() / 60.0, 2),
        "candidates_total": int(candidates_total),
        "candidates_evaluated": int(candidates_evaluated),
        "baseline": deepcopy(baselines.get("phase1_proxy")),
        "baselines": deepcopy(baselines),
        "leaderboard": sorted_leaderboard,
        "frontier": frontier_entries[:10],
        "champion": deepcopy(champion_entry),
        "phase_progress": {
            "stabilization_rounds_started": int(stabilization_rounds_started),
            "replay_rounds_started": int(replay_rounds_started),
            "phase1_frontier_count": _count_phase_entries(
                sorted_leaderboard, phase="phase1", status="frontier_seed"
            ),
            "phase2_keep_count": _count_phase_entries(
                sorted_leaderboard, phase="phase2", status="keep"
            )
            + _count_phase_entries(
                sorted_leaderboard, phase="phase2", status="champion"
            ),
            "phase2_crash_count": _count_phase_entries(
                sorted_leaderboard, phase="phase2", status="crash"
            ),
            "phase1_evaluated": _count_phase_entries(
                sorted_leaderboard, phase="phase1"
            ),
            "phase2_evaluated": _count_phase_entries(
                sorted_leaderboard, phase="phase2"
            ),
        },
    }


def _build_candidate_labeled_df(
    labeled_df: pd.DataFrame,
    candidate: dict[str, Any],
) -> pd.DataFrame:
    """为单个 proxy 候选准备截面过滤后的 labeled panel。"""
    filtered = apply_cross_section_filter(
        labeled_df, candidate.get("universe_filter")
    )
    return filtered.sort_values(["date", "order_book_id"]).reset_index(drop=True)


def _register_candidates(
    candidates: list[dict[str, Any]],
    discovered_signatures: set[str],
) -> list[dict[str, Any]]:
    """把新一轮候选做去重注册，避免 budget loop 来回踩同一配置。"""
    fresh_candidates: list[dict[str, Any]] = []
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        signature = _candidate_signature(candidate)
        if signature in discovered_signatures:
            continue
        discovered_signatures.add(signature)
        fresh_candidates.append(candidate)
    return fresh_candidates


def build_default_run_tag(now: datetime | None = None) -> str:
    """生成 focused runner 默认使用的 run_tag。"""
    current = now or datetime.now()
    return current.strftime("tx1_autoresearch_4h_%Y%m%d_%H%M%S")


def _metric(payload: dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    """安全读取嵌套指标，缺失时返回默认值。"""
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def run_liquidity_focus_search(
    *,
    run_tag: str,
    runs_root: str | Path,
    raw_df: pd.DataFrame | None = None,
    universe_size: int = 300,
    start_date=None,
    end_date=None,
    model_kind: str = "lgbm",
    label_transform: str = "rank",
    horizon_days: int = 20,
    max_runtime_hours: float = 4.0,
    smoke_max_folds: int = 1,
    full_max_folds: int | None = 5,
    max_stabilization_rounds: int = 3,
    max_replay_rounds: int = 4,
    replay_cash: float = 1_000_000,
    replay_selected_indices: list[int] | None = None,
) -> dict[str, Any]:
    """运行以 liquidity_plus 稳化 + rolling_score replay 为主线的预算驱动搜索。"""
    run_root = build_run_root(runs_root, run_tag)
    run_root.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now()
    safe_hours = max(float(max_runtime_hours), 0.0)
    deadline_at = started_at + timedelta(hours=safe_hours)

    if raw_df is None:
        raw_df = build_research_raw_df(
            universe_size=universe_size,
            start_date=start_date,
            end_date=end_date,
        )
    labeled_df = build_labeled_panel(
        raw_df=raw_df,
        label_transform=label_transform,
        horizon_days=horizon_days,
    )

    baselines: dict[str, Any] = {}
    leaderboard: list[dict[str, Any]] = []
    champion_entry: dict[str, Any] | None = None
    frontier_records: list[dict[str, Any]] = []
    replay_records: list[dict[str, Any]] = []
    discovered_signatures: set[str] = set()
    next_experiment_index = 0
    evaluated = 0
    status = "running"

    proxy_anchor_candidate = _candidate(
        "liquidity_plus_anchor",
        phase="phase1",
        evaluation_mode="proxy",
        family="stabilization_anchor",
        description="phase-1 稳化搜索锚点：liquidity_plus 默认 lgbm + 默认预处理关闭。",
        search_round=0,
        features=_liquidity_plus_features(),
        model={"kind": str(model_kind)},
    )
    proxy_anchor_summary = run_focused_candidate_trial(
        run_root=run_root,
        experiment_index=next_experiment_index,
        labeled_df=labeled_df,
        config=build_candidate_config(
            proxy_anchor_candidate,
            model_kind=model_kind,
            label_transform=label_transform,
            horizon_days=horizon_days,
        ),
        stage="proxy_anchor",
        max_folds=full_max_folds,
    )
    baselines["phase1_proxy"] = {
        "candidate_id": proxy_anchor_candidate["id"],
        "summary": proxy_anchor_summary,
        "metrics": _build_entry_metrics(proxy_anchor_summary),
    }
    next_experiment_index += 1

    stabilization_rounds_started = 1
    replay_rounds_started = 0
    phase1_round_index = 0
    phase2_round_index = 0
    phase1_queue = _register_candidates(
        build_liquidity_focus_candidates(round_index=0),
        discovered_signatures,
    )
    phase2_queue: list[dict[str, Any]] = []
    replay_anchor_ready = False
    best_replay_summary: dict[str, Any] | None = None

    partial = _build_partial_result(
        run_tag=run_tag,
        run_root=run_root,
        baselines=baselines,
        leaderboard=leaderboard,
        champion_entry=champion_entry,
        candidates_total=len(discovered_signatures),
        candidates_evaluated=evaluated,
        started_at=started_at,
        deadline_at=deadline_at,
        status="running",
        stabilization_rounds_started=stabilization_rounds_started,
        replay_rounds_started=replay_rounds_started,
    )
    _write_outputs(run_root, partial)

    while True:
        did_work = False

        if phase1_queue:
            candidate = phase1_queue.pop(0)
            candidate_labeled_df = _build_candidate_labeled_df(labeled_df, candidate)
            smoke_summary = run_focused_candidate_trial(
                run_root=run_root,
                experiment_index=next_experiment_index,
                labeled_df=candidate_labeled_df,
                config=build_candidate_config(
                    candidate,
                    model_kind=model_kind,
                    label_transform=label_transform,
                    horizon_days=horizon_days,
                ),
                stage="smoke",
                max_folds=smoke_max_folds,
            )
            smoke_decision = judge_candidate(
                smoke_summary,
                baseline_summary=proxy_anchor_summary,
                best_summary=proxy_anchor_summary,
                stage="smoke",
            )
            if smoke_decision.get("status") != "keep":
                leaderboard.append(
                    _build_leaderboard_entry(
                        candidate=candidate,
                        summary=smoke_summary,
                        decision=smoke_decision,
                        stage_reached="smoke",
                    )
                )
            else:
                full_summary = run_focused_candidate_trial(
                    run_root=run_root,
                    experiment_index=next_experiment_index,
                    labeled_df=candidate_labeled_df,
                    config=build_candidate_config(
                        candidate,
                        model_kind=model_kind,
                        label_transform=label_transform,
                        horizon_days=horizon_days,
                    ),
                    stage="full",
                    max_folds=full_max_folds,
                )
                full_decision = judge_phase1_candidate(
                    full_summary,
                    anchor_summary=proxy_anchor_summary,
                )
                entry = _build_leaderboard_entry(
                    candidate=candidate,
                    summary=full_summary,
                    decision=full_decision,
                    stage_reached="full",
                )
                leaderboard.append(entry)
                if full_decision.get("status") == "frontier_seed":
                    frontier_records.append(
                        {
                            "candidate": dict(candidate),
                            "summary": dict(full_summary),
                            "entry": dict(entry),
                        }
                    )
            next_experiment_index += 1
            evaluated += 1
            did_work = True

        elif not replay_anchor_ready:
            replay_anchor_candidate = _build_replay_candidate(
                round_index=0,
                artifact_line_id="combo_b25_h45",
                strategy_profile="smooth",
                description="phase-2 replay 锚点：当前默认 TX1 executable line。",
            )
            replay_anchor_summary = run_replay_candidate_trial(
                run_root=run_root,
                experiment_index=next_experiment_index,
                candidate=replay_anchor_candidate,
                cash=replay_cash,
                selected_indices=replay_selected_indices,
            )
            baselines["phase2_replay"] = {
                "candidate_id": replay_anchor_candidate["id"],
                "summary": replay_anchor_summary,
                "metrics": _build_entry_metrics(replay_anchor_summary),
            }
            best_replay_summary = dict(replay_anchor_summary)
            next_experiment_index += 1
            replay_anchor_ready = True
            phase2_queue = _register_candidates(
                build_replay_search_candidates(round_index=0),
                discovered_signatures,
            )
            replay_rounds_started = 1 if phase2_queue else 0
            did_work = True

        elif phase2_queue:
            candidate = phase2_queue.pop(0)
            replay_summary = run_replay_candidate_trial(
                run_root=run_root,
                experiment_index=next_experiment_index,
                candidate=candidate,
                cash=replay_cash,
                selected_indices=replay_selected_indices,
            )
            replay_decision = judge_replay_candidate(
                replay_summary,
                baseline_summary=_metric(
                    baselines,
                    "phase2_replay",
                    "summary",
                    default={},
                ),
                best_summary=best_replay_summary or _metric(
                    baselines,
                    "phase2_replay",
                    "summary",
                    default={},
                ),
            )
            entry = _build_leaderboard_entry(
                candidate=candidate,
                summary=replay_summary,
                decision=replay_decision,
                stage_reached="replay",
            )
            leaderboard.append(entry)
            replay_records.append(
                {
                    "candidate": dict(candidate),
                    "summary": dict(replay_summary),
                    "entry": dict(entry),
                }
            )
            if replay_decision.get("status") == "champion":
                best_replay_summary = dict(replay_summary)
                champion_entry = dict(entry)
            next_experiment_index += 1
            evaluated += 1
            did_work = True

        else:
            expanded = False

            if replay_anchor_ready and phase2_round_index + 1 < max_replay_rounds:
                next_round = phase2_round_index + 1
                ranked_replay_records = sorted(
                    replay_records,
                    key=lambda item: (
                        -float(
                            _metric(
                                item.get("summary") or {},
                                "replay",
                                "composite_score",
                            )
                        ),
                        float(
                            _metric(
                                item.get("summary") or {},
                                "portfolio",
                                "max_drawdown",
                            )
                        ),
                        -float(
                            _metric(
                                item.get("summary") or {},
                                "replay",
                                "stability_score",
                            )
                        ),
                    ),
                )
                next_candidates = _register_candidates(
                    build_replay_search_candidates(
                        round_index=next_round,
                        replay_entries=ranked_replay_records,
                        tried_signatures=discovered_signatures,
                    ),
                    discovered_signatures,
                )
                if next_candidates:
                    phase2_round_index = next_round
                    replay_rounds_started += 1
                    phase2_queue = next_candidates
                    expanded = True

            if (not expanded) and phase1_round_index + 1 < max_stabilization_rounds:
                next_round = phase1_round_index + 1
                next_candidates = _register_candidates(
                    build_liquidity_focus_candidates(
                        round_index=next_round,
                        frontier_entries=frontier_records,
                        tried_signatures=discovered_signatures,
                    ),
                    discovered_signatures,
                )
                if next_candidates:
                    phase1_round_index = next_round
                    stabilization_rounds_started += 1
                    phase1_queue = next_candidates
                    expanded = True

            if not expanded:
                replay_crash_count = int(
                    sum(
                        1
                        for record in replay_records
                        if str((record.get("entry") or {}).get("status") or "")
                        == "crash"
                    )
                )
                if replay_crash_count > 0:
                    status = "infra_stalled"
                else:
                    status = "completed"
                break
            did_work = True

        partial = _build_partial_result(
            run_tag=run_tag,
            run_root=run_root,
            baselines=baselines,
            leaderboard=leaderboard,
            champion_entry=champion_entry,
            candidates_total=len(discovered_signatures),
            candidates_evaluated=evaluated,
            started_at=started_at,
            deadline_at=deadline_at,
            status="running",
            stabilization_rounds_started=stabilization_rounds_started,
            replay_rounds_started=replay_rounds_started,
        )
        _write_outputs(run_root, partial)

        # 时间预算是上限而不是硬停点，因此只在当前单元工作完成后检查。
        if did_work and datetime.now() >= deadline_at:
            status = "time_budget_reached"
            break

    result = _build_partial_result(
        run_tag=run_tag,
        run_root=run_root,
        baselines=baselines,
        leaderboard=leaderboard,
        champion_entry=champion_entry,
        candidates_total=len(discovered_signatures),
        candidates_evaluated=evaluated,
        started_at=started_at,
        deadline_at=deadline_at,
        status=status,
        stabilization_rounds_started=stabilization_rounds_started,
        replay_rounds_started=replay_rounds_started,
    )
    _write_outputs(run_root, result)
    return result
