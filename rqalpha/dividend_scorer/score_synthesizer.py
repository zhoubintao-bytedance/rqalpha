# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from rqalpha.dividend_scorer.config import (
    CONFIDENCE_EXTREME_HIGH,
    CONFIDENCE_EXTREME_LOW,
    CONFIDENCE_FEATURES,
    FEATURES,
    LABEL_WINDOW_N,
    MIN_DIMENSIONS,
    PARAMS_VERSION,
    SUBSAMPLE_INTERVAL_S,
    VALUATION_FEATURES,
)


class ScoreUnavailableError(RuntimeError):
    pass


class ScoreSynthesizer(object):
    def sub_score(self, percentile_value):
        return 10.0 * float(percentile_value)

    def synthesize(self, feature_snapshot, weight_result, freshness=None):
        weight_map = OrderedDict(weight_result.get("weights", {}))
        effective_features = OrderedDict()
        effective_weights = OrderedDict()

        for feature_name in VALUATION_FEATURES:
            feature_info = dict(feature_snapshot.get(feature_name, {}))
            normalized = feature_info.get("normalized")
            feature_weight = float(weight_map.get(feature_name, 0.0) or 0.0)
            if normalized is None or feature_weight <= 0:
                continue
            effective_features[feature_name] = feature_info
            effective_weights[feature_name] = feature_weight

        contributing_dimensions = set(
            FEATURES[feature_name]["dimension"]
            for feature_name in effective_features.keys()
        )
        if "dividend" not in contributing_dimensions:
            raise ScoreUnavailableError("dividend dimension is unavailable")
        if len(contributing_dimensions) < MIN_DIMENSIONS:
            raise ScoreUnavailableError("at least {} dimensions are required".format(MIN_DIMENSIONS))

        weight_sum = float(sum(effective_weights.values()))
        if weight_sum <= 0:
            raise ScoreUnavailableError("no effective feature weights remain")

        features = OrderedDict()
        total_score = 0.0
        for feature_name in VALUATION_FEATURES:
            feature_info = dict(feature_snapshot.get(feature_name, {}))
            normalized = feature_info.get("normalized")
            percentile = feature_info.get("percentile")
            sub_score = self.sub_score(normalized) if normalized is not None else None
            effective_weight = effective_weights.get(feature_name, 0.0) / weight_sum if feature_name in effective_weights else 0.0
            if sub_score is not None:
                total_score += effective_weight * sub_score
            feature_info.update({
                "percentile": percentile,
                "normalized": normalized,
                "sub_score": sub_score,
                "weight": effective_weight,
            })
            features[feature_name] = feature_info

        confidence_modifiers = OrderedDict()
        extreme_count = 0
        for feature_name in CONFIDENCE_FEATURES:
            feature_info = dict(feature_snapshot.get(feature_name, {}))
            percentile = feature_info.get("percentile")
            status = "normal"
            if percentile is not None and (
                percentile <= CONFIDENCE_EXTREME_LOW or percentile >= CONFIDENCE_EXTREME_HIGH
            ):
                status = "extreme"
                extreme_count += 1
            confidence_modifiers[feature_name] = {
                "raw": feature_info.get("raw"),
                "percentile": percentile,
                "status": status,
            }

        warnings = []
        if freshness:
            stale_sources = [name for name, meta in freshness.items() if meta.get("status") in ("stale", "expired")]
            if stale_sources:
                warnings.append("stale_sources: {}".format(", ".join(sorted(stale_sources))))

        if any(features[feature_name].get("under_sampled") for feature_name in features):
            warnings.append("under_sampled_percentile_window")

        confidence = "normal"
        if extreme_count == 1 or warnings:
            confidence = "lowered"
        if extreme_count >= 2:
            confidence = "low"

        return {
            "total_score": round(total_score, 4),
            "confidence": confidence,
            "features": features,
            "confidence_modifiers": confidence_modifiers,
            "warnings": warnings,
            "model_meta": {
                "method": weight_result.get("method"),
                "fallback_reason": weight_result.get("fallback_reason"),
                "test_ic_avg": weight_result.get("test_ic_avg"),
                "test_ic_ir_avg": weight_result.get("test_ic_ir_avg"),
                "label_window_n": LABEL_WINDOW_N,
                "subsample_interval": SUBSAMPLE_INTERVAL_S,
                "params_version": PARAMS_VERSION,
            },
        }
