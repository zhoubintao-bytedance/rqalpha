# -*- coding: utf-8 -*-
"""TX1 live advisor 服务编排。"""

from __future__ import annotations

import pandas as pd

from skyeye.data import DataFacade
from skyeye.products.tx1.baseline_models import load_model_bundle
from skyeye.products.tx1.live_advisor.calibration import lookup_calibration_bucket
from skyeye.products.tx1.live_advisor.package_io import load_live_package
from skyeye.products.tx1.live_advisor.runtime_gates import (
    evaluate_score_runtime_gates,
    evaluate_snapshot_runtime_gates,
)
from skyeye.products.tx1.live_advisor.snapshot import build_live_snapshot
from skyeye.products.tx1.portfolio_proxy import PortfolioProxy
from skyeye.products.tx1.preprocessor import FeaturePreprocessor
from skyeye.products.tx1.strategies.rolling_score.replay import compute_turnover_ratio


DATA_FACADE = DataFacade()


class LiveAdvisorService(object):
    def __init__(self, packages_root=None):
        # packages_root 允许测试与正式环境使用不同的产包根目录。
        self.packages_root = packages_root

    def get_recommendations(
        self,
        package_ref,
        *,
        trade_date,
        top_k=20,
        raw_df=None,
        universe=None,
        include_dropped=False,
        universe_size=300,
        market_cap_floor_quantile=None,
        market_cap_column=None,
        universe_source="runtime_fast",
        universe_cache_root=None,
        current_holdings=None,
        last_rebalance_date=None,
    ) -> dict:
        """基于 promoted package 生成指定交易日的推荐卡片。"""
        package_payload = load_live_package(package_ref, packages_root=self.packages_root)
        manifest = package_payload["manifest"]
        feature_schema = package_payload["feature_schema"]
        feature_columns = list(feature_schema.get("feature_columns") or manifest.get("required_features") or [])

        snapshot = build_live_snapshot(
            trade_date=trade_date,
            raw_df=raw_df,
            universe=universe,
            required_features=feature_columns,
            universe_size=universe_size,
            market_cap_floor_quantile=market_cap_floor_quantile,
            market_cap_column=market_cap_column,
            universe_source=universe_source,
            universe_cache_root=universe_cache_root,
        )
        snapshot_gate = evaluate_snapshot_runtime_gates(
            snapshot,
            required_features=feature_columns,
        )
        if not snapshot_gate["passed"]:
            return self._build_stop_result(
                manifest,
                snapshot,
                reasons=snapshot_gate["reasons"],
                warnings=snapshot_gate.get("warnings"),
                include_dropped=include_dropped,
            )

        preprocessor = FeaturePreprocessor.from_bundle(package_payload["preprocessor_bundle"])
        model = load_model_bundle(package_payload["model_bundle"])
        scored_frame = snapshot["snapshot_features"].copy()
        if preprocessor is not None:
            scored_frame = preprocessor.transform(scored_frame, feature_columns)

        predictions = model.predict(scored_frame[feature_columns])
        scored_frame = self._finalize_predictions(
            scored_frame,
            predictions,
            package_payload["model_bundle"],
        )
        score_gate = evaluate_score_runtime_gates(scored_frame, package_payload["calibration_bundle"])
        if not score_gate["passed"]:
            return self._build_stop_result(
                manifest,
                snapshot,
                reasons=score_gate["reasons"],
                warnings=None,
                include_dropped=include_dropped,
            )

        scored_frame = scored_frame.sort_values("prediction", ascending=False).reset_index(drop=True)
        scored_frame["rank"] = range(1, len(scored_frame) + 1)
        scored_frame["percentile"] = scored_frame["rank"].map(
            lambda rank_value: 1.0 - float(rank_value - 1) / float(len(scored_frame))
        )

        recommendations = []
        recent_canary_bundle = package_payload.get("recent_canary_bundle")
        for row in scored_frame.head(int(top_k)).itertuples(index=False):
            calibration = lookup_calibration_bucket(
                package_payload["calibration_bundle"],
                getattr(row, "percentile"),
            )
            recent_canary_evidence = None
            if isinstance(recent_canary_bundle, dict):
                try:
                    recent_canary_stats = lookup_calibration_bucket(
                        recent_canary_bundle,
                        getattr(row, "percentile"),
                    )
                    recent_canary_evidence = {
                        "window": dict(recent_canary_bundle.get("window", {})),
                        "sample_count": int(recent_canary_stats.get("sample_count", 0)),
                        "win_rate": float(recent_canary_stats.get("win_rate", 0.0)),
                        "mean_return": float(recent_canary_stats.get("mean_return", 0.0)),
                        "median_return": float(recent_canary_stats.get("median_return", 0.0)),
                        "return_quantile_range": {
                            "p25": float(recent_canary_stats.get("return_quantiles", {}).get("p25", 0.0)),
                            "p75": float(recent_canary_stats.get("return_quantiles", {}).get("p75", 0.0)),
                        },
                    }
                except Exception:
                    recent_canary_evidence = None
            recommendations.append(
                {
                    "order_book_id": str(getattr(row, "order_book_id")),
                    "rank": int(getattr(row, "rank")),
                    "percentile": float(getattr(row, "percentile")),
                    "win_rate": float(calibration["win_rate"]),
                    "mean_return": float(calibration["mean_return"]),
                    "median_return": float(calibration["median_return"]),
                    "return_quantile_range": {
                        "p25": float(calibration["return_quantiles"]["p25"]),
                        "p75": float(calibration["return_quantiles"]["p75"]),
                    },
                    "max_drawdown_quantile_range": {
                        "p25": float(calibration["max_drawdown_quantiles"]["p25"]),
                        "p75": float(calibration["max_drawdown_quantiles"]["p75"]),
                    },
                    "sample_count": int(calibration["sample_count"]),
                    "historical_evidence": {
                        "sample_count": int(calibration["sample_count"]),
                        "win_rate": float(calibration["win_rate"]),
                        "mean_return": float(calibration["mean_return"]),
                        "median_return": float(calibration["median_return"]),
                    },
                    "recent_canary_evidence": recent_canary_evidence,
                    "warnings": [],
                }
            )

        warnings = []
        if not isinstance(recent_canary_bundle, dict):
            warnings.append(
                {
                    "level": "warning",
                    "code": "recent_canary_missing",
                    "message": "WARNING: promoted package does not contain recent canary evidence; only historical OOS calibration is available.",
                }
            )
        portfolio_advice = self._build_portfolio_advice(
            scored_frame,
            package_payload["portfolio_policy"],
            current_holdings=current_holdings,
            last_rebalance_date=last_rebalance_date,
            trade_date=snapshot["trade_date"],
        )
        result = {
            "status": "ok",
            "package_id": manifest["package_id"],
            "gate_level": manifest.get("gate_summary", {}).get("gate_level"),
            "requested_trade_date": snapshot.get("requested_trade_date"),
            "latest_available_trade_date": snapshot.get("latest_available_trade_date"),
            "score_date": snapshot["trade_date"],
            "raw_data_end_date": snapshot.get("raw_data_end_date"),
            "fit_end_date": manifest.get("fit_end_date"),
            "recommendations": recommendations,
            "warnings": warnings,
            "portfolio_advice": portfolio_advice,
        }
        if include_dropped:
            result["dropped"] = snapshot["dropped_reasons"]
        return result

    @staticmethod
    def _finalize_predictions(frame, predictions, model_bundle):
        """兼容单头与多头模型输出，并统一生成 active ranking score。"""
        result = frame.copy()
        prediction_config = model_bundle.get("prediction_config", {})
        if isinstance(predictions, dict):
            if "return" in predictions:
                result["prediction_ret"] = predictions["return"]
            if "volatility" in predictions:
                result["prediction_vol"] = predictions["volatility"]
            if "max_drawdown" in predictions:
                result["prediction_mdd"] = predictions["max_drawdown"]
            result["prediction"] = result["prediction_ret"]
            if prediction_config.get("combine_auxiliary", False):
                combined = _cross_section_rank(result, "prediction_ret")
                if (
                    prediction_config.get("volatility_weight", 0.0) > 0
                    and "prediction_vol" in result.columns
                ):
                    combined = combined - prediction_config["volatility_weight"] * _cross_section_rank(result, "prediction_vol")
                if (
                    prediction_config.get("max_drawdown_weight", 0.0) > 0
                    and "prediction_mdd" in result.columns
                ):
                    combined = combined - prediction_config["max_drawdown_weight"] * _cross_section_rank(result, "prediction_mdd")
                result["prediction"] = combined
        else:
            result["prediction"] = predictions
            result["prediction_ret"] = predictions
        return result

    @staticmethod
    def _build_stop_result(manifest, snapshot, *, reasons, warnings=None, include_dropped):
        """统一返回 stop-serve 结果，避免静默降级。"""
        result = {
            "status": "stopped",
            "package_id": manifest["package_id"],
            "gate_level": manifest.get("gate_summary", {}).get("gate_level"),
            "requested_trade_date": snapshot.get("requested_trade_date"),
            "latest_available_trade_date": snapshot.get("latest_available_trade_date"),
            "score_date": snapshot["trade_date"],
            "raw_data_end_date": snapshot.get("raw_data_end_date"),
            "fit_end_date": manifest.get("fit_end_date"),
            "reasons": list(reasons),
            "warnings": list(warnings or []),
            "recommendations": [],
        }
        if include_dropped:
            result["dropped"] = snapshot.get("dropped_reasons", {})
        return result

    @staticmethod
    def _build_portfolio_advice(
        scored_frame,
        portfolio_policy,
        *,
        current_holdings=None,
        last_rebalance_date=None,
        trade_date=None,
    ) -> dict:
        """基于组合规则和当前持仓，生成调仓建议与目标权重。"""
        current_weights = {
            str(order_book_id): float(weight)
            for order_book_id, weight in (current_holdings or {}).items()
            if float(weight) > 0
        }
        total_current = sum(current_weights.values())
        if total_current > 0:
            current_weights = {
                order_book_id: weight / total_current
                for order_book_id, weight in current_weights.items()
            }

        proxy = PortfolioProxy(
            buy_top_k=portfolio_policy.get("buy_top_k", 25),
            hold_top_k=portfolio_policy.get("hold_top_k", 45),
            rebalance_interval=portfolio_policy.get("rebalance_interval", 20),
            holding_bonus=portfolio_policy.get("holding_bonus", 0.5),
        )
        rebalance_due = _is_rebalance_due(
            trade_date=trade_date,
            last_rebalance_date=last_rebalance_date,
            rebalance_interval=int(portfolio_policy.get("rebalance_interval", 20)),
            has_current_holdings=bool(current_weights),
        )
        target_weights = proxy.build_target_weights(
            scored_frame,
            current_holdings=current_weights,
            should_rebalance=rebalance_due,
        )
        estimated_turnover = compute_turnover_ratio(current_weights, target_weights)
        action_rows = []
        for order_book_id in sorted(set(current_weights) | set(target_weights)):
            current_weight = float(current_weights.get(order_book_id, 0.0))
            target_weight = float(target_weights.get(order_book_id, 0.0))
            delta_weight = target_weight - current_weight
            if current_weight <= 0 and target_weight > 0:
                action = "add"
            elif current_weight > 0 and target_weight <= 0:
                action = "exit"
            elif delta_weight > 1e-9:
                action = "add"
            elif delta_weight < -1e-9:
                action = "reduce"
            else:
                action = "keep"
            action_rows.append(
                {
                    "order_book_id": order_book_id,
                    "action": action,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "delta_weight": delta_weight,
                }
            )
        return {
            "rebalance_due": bool(rebalance_due),
            "last_rebalance_date": last_rebalance_date,
            "current_weights": current_weights,
            "target_weights": target_weights,
            "weight_deltas": {
                row["order_book_id"]: row["delta_weight"]
                for row in action_rows
            },
            "estimated_turnover": float(estimated_turnover),
            "actions": action_rows,
        }


def _cross_section_rank(frame, column):
    """复用研究侧的横截面分位定义，保持 live bucket 映射一致。"""
    return frame.groupby("date")[column].rank(method="average", pct=True)


def _is_rebalance_due(
    *,
    trade_date,
    last_rebalance_date,
    rebalance_interval: int,
    has_current_holdings: bool,
) -> bool:
    """根据上次调仓日和组合规则判断今天是否应调仓。"""
    if not has_current_holdings:
        return True
    if not last_rebalance_date:
        return True
    trade_ts = pd.Timestamp(trade_date).normalize()
    last_rebalance_ts = pd.Timestamp(last_rebalance_date).normalize()
    if trade_ts <= last_rebalance_ts:
        return False
    trading_dates = []
    try:
        trading_dates = list(DATA_FACADE.get_trading_dates(last_rebalance_ts, trade_ts) or [])
    except Exception:
        trading_dates = []
    if trading_dates:
        gap = max(len(trading_dates) - 1, 0)
    else:
        gap = max(len(pd.bdate_range(last_rebalance_ts, trade_ts)) - 1, 0)
    return gap >= int(rebalance_interval)
