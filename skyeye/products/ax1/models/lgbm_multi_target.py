"""AX1 LGBM cost-aware multi-target predictor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from skyeye.products.ax1.models.registry import Predictor
from skyeye.products.tx1.baseline_models import create_multi_head_model


@dataclass
class LGBMMultiTargetPredictor(Predictor):
    """LGBM 独立多头 predictor。"""

    horizons: Sequence[int] = (5, 10, 20)
    training_horizons: Sequence[int] = (5, 10, 20)
    risk_horizon: int = 10
    stability_horizon: int = 20
    feature_columns: Sequence[str] | None = None
    liquidity_column: str = "feature_dollar_volume"
    params: Mapping[str, Any] | None = None
    confidence_method: str = "sign_consistency"

    _multi_head: Any = field(default=None, init=False, repr=False)
    _trained_feature_columns: tuple[str, ...] = field(default=(), init=False, repr=False)
    _head_configs: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame | None = None,
        *,
        val_features: pd.DataFrame | None = None,
        val_labels: pd.DataFrame | None = None,
    ) -> "LGBMMultiTargetPredictor":
        if features is None or len(features) == 0:
            raise ValueError("LGBMMultiTargetPredictor.fit requires non-empty features")
        if labels is None or len(labels) == 0:
            raise ValueError("LGBMMultiTargetPredictor.fit requires labels")

        feat_cols = self._resolve_feature_columns(features)
        self._trained_feature_columns = feat_cols

        head_configs = self._build_head_configs()
        self._head_configs = head_configs
        self._validate_has_columns(
            labels,
            [config["target_column"] for config in head_configs.values()],
            frame_name="labels",
        )

        train_X = features.loc[:, list(feat_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        train_targets = labels.loc[:, [config["target_column"] for config in head_configs.values()]]

        if val_features is not None and val_labels is not None and len(val_features) > 0 and len(val_labels) > 0:
            val_X = val_features.loc[:, list(feat_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            val_targets = val_labels.loc[:, [config["target_column"] for config in head_configs.values()]]
        else:
            val_X = None
            val_targets = None

        self._multi_head = create_multi_head_model(
            kind="lgbm",
            head_configs=head_configs,
            params=dict(self.params or {}),
        )
        self._multi_head.fit(train_X, train_targets, val_X=val_X, val_targets=val_targets)
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._multi_head is None:
            raise RuntimeError("LGBMMultiTargetPredictor must be fit before predict")
        if features is None or len(features) == 0:
            return pd.DataFrame()

        feat_cols = list(self._trained_feature_columns)
        self._validate_has_columns(features, feat_cols, frame_name="features")
        X = features.loc[:, feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        raw = self._multi_head.predict(X)  # dict: head_name -> ndarray

        id_columns = [column for column in ["date", "order_book_id"] if column in features.columns]
        out = features[id_columns].copy() if id_columns else pd.DataFrame(index=features.index)

        for horizon in self.training_horizons:
            head_name = f"relative_net_return_{int(horizon)}d"
            out[f"expected_relative_net_return_{int(horizon)}d"] = pd.Series(raw[head_name], index=features.index, dtype=float)

        risk_raw = pd.Series(raw["risk"], index=features.index, dtype=float)
        out["risk_forecast"] = risk_raw.clip(lower=0.0).fillna(0.0)

        out["liquidity_score"] = self._derive_liquidity(features)
        out["cost_forecast"] = self._derive_cost_forecast(out["liquidity_score"])
        out["confidence_raw"] = self._derive_confidence(out)
        out["confidence"] = out["confidence_raw"]

        for horizon in self.horizons:
            column = f"expected_relative_net_return_{int(horizon)}d"
            if column in out.columns:
                continue
            out[column] = self._fallback_net_return(out, int(horizon))

        return out

    def feature_importance(self) -> dict[str, Any]:
        """Return JSON-friendly per-head and aggregate gain/split importance."""
        if self._multi_head is None:
            raise RuntimeError("LGBMMultiTargetPredictor must be fit before feature_importance")
        feature_columns = list(self._trained_feature_columns)
        if not feature_columns:
            raise RuntimeError("LGBMMultiTargetPredictor has no trained feature columns")

        heads: dict[str, Any] = {}
        gain_totals = np.zeros(len(feature_columns), dtype=float)
        split_totals = np.zeros(len(feature_columns), dtype=float)
        head_count = 0
        for head_name in sorted(self._multi_head.models_):
            head_state = self._multi_head.models_[head_name]
            model = head_state.get("model")
            booster = getattr(model, "_model", None)
            gain_values = _extract_lgbm_importance(booster, feature_columns, importance_type="gain")
            split_values = _extract_lgbm_importance(booster, feature_columns, importance_type="split")
            gain_totals += np.asarray(gain_values, dtype=float)
            split_totals += np.asarray(split_values, dtype=float)
            head_count += 1
            heads[str(head_name)] = {
                "target_column": str(head_state.get("target_column", "")),
                "gain": _format_importance(feature_columns, gain_values),
                "split": _format_importance(feature_columns, split_values),
            }

        if head_count > 0:
            gain_totals = gain_totals / float(head_count)
            split_totals = split_totals / float(head_count)
        return _json_ready_importance(
            {
                "schema_version": 1,
                "feature_columns": feature_columns,
                "heads": heads,
                "aggregate": {
                    "gain": _format_importance(feature_columns, gain_totals.tolist()),
                    "split": _format_importance(feature_columns, split_totals.tolist()),
                },
            }
        )

    # ---- internal helpers ----

    def _resolve_feature_columns(self, features: pd.DataFrame) -> tuple[str, ...]:
        if self.feature_columns:
            columns = tuple(self.feature_columns)
            missing = [column for column in columns if column not in features.columns]
            if missing:
                raise ValueError(f"features missing configured feature columns: {missing}")
            return columns
        numeric = features.select_dtypes(include="number")
        reserved_prefixes = ("expected_", "label_")
        reserved_exact = {"risk_forecast", "confidence", "liquidity_score", "cost_forecast"}
        columns = tuple(
            column
            for column in numeric.columns
            if column not in reserved_exact and not any(column.startswith(prefix) for prefix in reserved_prefixes)
        )
        if not columns:
            raise ValueError("LGBMMultiTargetPredictor could not autodetect any feature column")
        return columns

    def _build_head_configs(self) -> dict[str, dict[str, Any]]:
        head_configs: dict[str, dict[str, Any]] = {}
        for horizon in self.training_horizons:
            horizon = int(horizon)
            head_configs[f"relative_net_return_{horizon}d"] = {"target_column": f"label_relative_net_return_{horizon}d"}
        head_configs["risk"] = {"target_column": f"label_volatility_{int(self.risk_horizon)}d"}
        return head_configs

    @staticmethod
    def _validate_has_columns(frame: pd.DataFrame, required: Sequence[str], *, frame_name: str) -> None:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{frame_name} missing columns: {missing}")

    def _fallback_net_return(self, out: pd.DataFrame, horizon: int) -> pd.Series:
        trained = sorted(int(h) for h in self.training_horizons)
        if not trained:
            return pd.Series(0.0, index=out.index)
        anchor = 10 if 10 in trained else trained[-1]
        anchor_col = f"expected_relative_net_return_{anchor}d"
        if anchor_col not in out.columns:
            return pd.Series(0.0, index=out.index)
        return out[anchor_col] * (float(horizon) / float(anchor))

    def _derive_liquidity(self, features: pd.DataFrame) -> pd.Series:
        if self.liquidity_column not in features.columns:
            return pd.Series(1.0, index=features.index)
        liquidity = pd.to_numeric(features[self.liquidity_column], errors="coerce").fillna(0.0)
        if "date" in features.columns:
            ranks = liquidity.groupby(pd.to_datetime(features["date"]), sort=False).rank(method="average", pct=True)
        else:
            ranks = liquidity.rank(method="average", pct=True)
        return ranks.fillna(1.0).clip(0.0, 1.0)

    @staticmethod
    def _derive_cost_forecast(liquidity_score: pd.Series) -> pd.Series:
        return (1.0 - liquidity_score).clip(0.0, 1.0) * 0.005

    def _derive_confidence(self, out: pd.DataFrame) -> pd.Series:
        if self.confidence_method != "sign_consistency":
            raise ValueError(f"unsupported confidence_method: {self.confidence_method}")
        if "expected_relative_net_return_5d" not in out.columns or "expected_relative_net_return_10d" not in out.columns:
            return pd.Series(0.0, index=out.index)
        r5 = out["expected_relative_net_return_5d"].astype(float)
        r10 = out["expected_relative_net_return_10d"].astype(float)
        risk = out["risk_forecast"].astype(float).clip(lower=1e-6)
        sign_agree = (np.sign(r5) == np.sign(r10)).astype(float)
        abs_r5 = r5.abs()
        abs_r10 = r10.abs()
        max_abs = np.maximum(abs_r5, abs_r10) + 1e-9
        min_abs = np.minimum(abs_r5, abs_r10)
        mag_ratio = min_abs / max_abs
        confidence = sign_agree * mag_ratio / (1.0 + risk)
        return confidence.clip(0.0, 1.0).fillna(0.0)


def _extract_lgbm_importance(booster: Any, feature_columns: Sequence[str], *, importance_type: str) -> list[float]:
    if booster is None or not hasattr(booster, "feature_importance"):
        return [0.0 for _ in feature_columns]
    raw = booster.feature_importance(importance_type=importance_type)
    values = [float(value) for value in np.asarray(raw, dtype=float).tolist()]
    if len(values) < len(feature_columns):
        values.extend([0.0] * (len(feature_columns) - len(values)))
    return values[: len(feature_columns)]


def _format_importance(feature_columns: Sequence[str], values: Sequence[float]) -> list[dict[str, Any]]:
    pairs = [(str(feature), _finite_float(value)) for feature, value in zip(feature_columns, values)]
    total = sum(max(value, 0.0) for _, value in pairs)
    sorted_pairs = sorted(pairs, key=lambda item: (-item[1], item[0]))
    return [
        {
            "feature": feature,
            "importance": value,
            "normalized_importance": float(max(value, 0.0) / total) if total > 0 else 0.0,
            "rank": rank,
        }
        for rank, (feature, value) in enumerate(sorted_pairs, start=1)
    ]


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def _json_ready_importance(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready_importance(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready_importance(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready_importance(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _finite_float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else 0.0
    return value
