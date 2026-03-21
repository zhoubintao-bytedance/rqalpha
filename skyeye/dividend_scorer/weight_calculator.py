# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

from skyeye.dividend_scorer.config import (
    COSINE_SIM_MIN,
    DIMENSION_FEATURES,
    DIVIDEND_WEIGHT_MIN,
    DOMAIN_WEIGHTS,
    FEATURES,
    IC_IR_THRESHOLD,
    IC_MIN_POINTS,
    IC_MIN_SURVIVING,
    IC_PVALUE_THRESHOLD,
    IC_ROLLING_K,
    LABEL_WINDOW_N,
    PERCENTILE_WINDOW,
    SAMPLE_OUT_YEARS,
    SUBSAMPLE_INTERVAL_S,
    TRADING_DAYS_A_YEAR,
    VALUATION_FEATURES,
    WEIGHT_PRIOR_BLEND,
    WEIGHT_CONCENTRATION_MAX,
)


class WeightCalculator(object):
    def __init__(
        self,
        label_window_n=LABEL_WINDOW_N,
        subsample_interval=SUBSAMPLE_INTERVAL_S,
        ic_rolling_k=IC_ROLLING_K,
        sample_out_years=SAMPLE_OUT_YEARS,
    ):
        self.label_window_n = label_window_n
        self.subsample_interval = subsample_interval
        self.ic_rolling_k = ic_rolling_k
        self.sample_out_years = sample_out_years

    def calculate_ic_ir_weights(self, feature_matrix, price_series, enforce_structure=True):
        feature_matrix = feature_matrix.reindex(columns=VALUATION_FEATURES)
        joined = feature_matrix.join(price_series.rename("price"), how="inner").sort_index()
        if joined.empty:
            return self._fallback_result("missing_training_data", feature_matrix)

        training = self._select_training_sample(joined)
        if len(training) < max(PERCENTILE_WINDOW, self.label_window_n + self.subsample_interval):
            return self._fallback_result("training_history_too_short", feature_matrix)

        feature_training = training.loc[:, list(VALUATION_FEATURES)]
        future_returns = training["price"].shift(-self.label_window_n) / training["price"] - 1.0
        sampled = feature_training.iloc[::self.subsample_interval].copy()
        sampled["future_return"] = future_returns.iloc[::self.subsample_interval]
        sampled = sampled.dropna(subset=["future_return"], how="any")
        if len(sampled) < IC_MIN_POINTS:
            return self._fallback_result("independent_samples_too_few", feature_matrix)

        ic_stats = OrderedDict()
        raw_weights = OrderedDict()
        surviving_features = []

        for feature_name in VALUATION_FEATURES:
            sample_frame = sampled[[feature_name, "future_return"]].dropna()
            ic_series = self._compute_ic_series(sample_frame, feature_name)
            ic_mean = float(ic_series.mean()) if len(ic_series) else np.nan
            ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else np.nan
            if np.isnan(ic_std) or ic_std == 0:
                ic_ir = 0.0
            else:
                ic_ir = abs(ic_mean) / ic_std
            if len(ic_series) > 1:
                t_test = ttest_1samp(ic_series, 0.0, nan_policy="omit")
                pvalue = float(t_test.pvalue) if not np.isnan(t_test.pvalue) else 1.0
            else:
                pvalue = 1.0
            survived = bool(
                len(sample_frame) >= IC_MIN_POINTS and
                len(ic_series) > 0 and
                ic_mean < 0 and
                pvalue <= IC_PVALUE_THRESHOLD and
                ic_ir >= IC_IR_THRESHOLD
            )
            ic_stats[feature_name] = {
                "sample_count": int(len(sample_frame)),
                "ic_observation_count": int(len(ic_series)),
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "pvalue": pvalue,
                "survived": survived,
            }
            raw_weights[feature_name] = ic_ir if survived else 0.0
            if survived:
                surviving_features.append(feature_name)

        if len(surviving_features) < IC_MIN_SURVIVING:
            return self._fallback_result("surviving_features_too_few", feature_matrix, ic_stats=ic_stats)

        normalized_weights = self._normalize_weights(raw_weights)
        fallback_reason = self._should_fallback(normalized_weights)
        if fallback_reason is not None and enforce_structure:
            return self._fallback_result(fallback_reason, feature_matrix, ic_stats=ic_stats)

        return {
            "weights": normalized_weights,
            "method": "ic_ir",
            "fallback_reason": fallback_reason,
            "ic_stats": ic_stats,
            "training_start": training.index[0].strftime("%Y-%m-%d"),
            "training_end": training.index[-1].strftime("%Y-%m-%d"),
            "sample_count": int(len(sampled)),
            "test_ic_avg": float(np.nanmean([stat["ic_mean"] for stat in ic_stats.values()])),
            "test_ic_ir_avg": float(np.nanmean([stat["ic_ir"] for stat in ic_stats.values()])),
        }

    def calculate_prior_weights(self, feature_matrix):
        available_features = [
            feature_name
            for feature_name in VALUATION_FEATURES
            if feature_name in feature_matrix.columns and feature_matrix[feature_name].notna().any()
        ]
        return {
            "weights": self._domain_weights(available_features),
            "method": "fixed_domain_prior",
            "fallback_reason": None,
            "ic_stats": {},
            "training_start": None,
            "training_end": None,
            "sample_count": 0,
            "test_ic_avg": np.nan,
            "test_ic_ir_avg": np.nan,
        }

    def calculate_shrunk_weights(
        self,
        feature_matrix,
        price_series,
        prior_blend=WEIGHT_PRIOR_BLEND,
        compute_diagnostics=False,
    ):
        domain_result = self.calculate_prior_weights(feature_matrix)
        diagnostic_result = None

        if prior_blend >= 1.0:
            result = dict(domain_result)
            result["prior_blend"] = float(prior_blend)
            if compute_diagnostics:
                diagnostic_result = self.calculate_ic_ir_weights(feature_matrix, price_series)
                result["diagnostic_weight_result"] = diagnostic_result
            return result

        dynamic_result = self.calculate_ic_ir_weights(
            feature_matrix,
            price_series,
            enforce_structure=False,
        )
        if dynamic_result.get("method") != "ic_ir":
            result = dict(domain_result)
            result["method"] = "domain_knowledge_fallback"
            result["prior_blend"] = float(prior_blend)
            result["fallback_reason"] = dynamic_result.get("fallback_reason")
            if compute_diagnostics:
                result["diagnostic_weight_result"] = dynamic_result
            return result

        blended_weights = OrderedDict()
        for feature_name in VALUATION_FEATURES:
            domain_weight = float(domain_result["weights"].get(feature_name, 0.0) or 0.0)
            dynamic_weight = float(dynamic_result["weights"].get(feature_name, 0.0) or 0.0)
            blended_weights[feature_name] = prior_blend * domain_weight + (1.0 - prior_blend) * dynamic_weight
        blended_weights = self._normalize_weights(blended_weights)
        fallback_reason = self._should_fallback(blended_weights)
        if fallback_reason is not None:
            result = dict(domain_result)
            result["method"] = "domain_knowledge_fallback"
            result["prior_blend"] = float(prior_blend)
            result["fallback_reason"] = fallback_reason
            if compute_diagnostics:
                result["diagnostic_weight_result"] = dynamic_result
            return result

        result = {
            "weights": blended_weights,
            "method": "shrunk_ic_ir",
            "fallback_reason": None,
            "ic_stats": dynamic_result.get("ic_stats", {}),
            "training_start": dynamic_result.get("training_start"),
            "training_end": dynamic_result.get("training_end"),
            "sample_count": dynamic_result.get("sample_count", 0),
            "test_ic_avg": dynamic_result.get("test_ic_avg"),
            "test_ic_ir_avg": dynamic_result.get("test_ic_ir_avg"),
            "prior_blend": float(prior_blend),
        }
        if compute_diagnostics:
            result["diagnostic_weight_result"] = dynamic_result
        return result

    def _select_training_sample(self, joined):
        if len(joined) > self.sample_out_years * TRADING_DAYS_A_YEAR:
            return joined.iloc[:-self.sample_out_years * TRADING_DAYS_A_YEAR].copy()
        return joined.copy()

    def _compute_ic_series(self, sample_frame, feature_name):
        if len(sample_frame) < IC_MIN_POINTS:
            return pd.Series(dtype="float64")
        window_size = min(self.ic_rolling_k, len(sample_frame))
        min_points = min(window_size, IC_MIN_POINTS)
        values = []
        index = []
        for end in range(min_points, len(sample_frame) + 1):
            window = sample_frame.iloc[max(0, end - window_size):end]
            if window[feature_name].nunique(dropna=True) <= 1 or window["future_return"].nunique(dropna=True) <= 1:
                continue
            ic_value = spearmanr(window[feature_name], window["future_return"], nan_policy="omit").correlation
            if np.isnan(ic_value):
                continue
            values.append(float(ic_value))
            index.append(window.index[-1])
        return pd.Series(values, index=index)

    def _should_fallback(self, normalized_weights):
        if not normalized_weights:
            return "empty_ic_weights"
        dimension_weights = OrderedDict()
        for dimension, features in DIMENSION_FEATURES.items():
            dimension_weights[dimension] = float(sum(normalized_weights.get(feature, 0.0) for feature in features))
        if any(weight > WEIGHT_CONCENTRATION_MAX for weight in dimension_weights.values()):
            return "single_dimension_overweight"
        if dimension_weights.get("dividend", 0.0) < DIVIDEND_WEIGHT_MIN:
            return "dividend_dimension_underweight"
        cosine_similarity = self._cosine_similarity(
            [normalized_weights.get(feature, 0.0) for feature in VALUATION_FEATURES],
            [DOMAIN_WEIGHTS.get(feature, 0.0) for feature in VALUATION_FEATURES],
        )
        if cosine_similarity < COSINE_SIM_MIN:
            return "weight_structure_deviates_from_domain_knowledge"
        return None

    def _fallback_result(self, fallback_reason, feature_matrix, ic_stats=None):
        available_features = [
            feature_name
            for feature_name in VALUATION_FEATURES
            if feature_name in feature_matrix.columns and feature_matrix[feature_name].notna().any()
        ]
        weights = self._domain_weights(available_features)
        return {
            "weights": weights,
            "method": "domain_knowledge_fallback",
            "fallback_reason": fallback_reason,
            "ic_stats": ic_stats or {},
            "training_start": None,
            "training_end": None,
            "sample_count": 0,
            "test_ic_avg": np.nan,
            "test_ic_ir_avg": np.nan,
        }

    @staticmethod
    def _domain_weights(available_features):
        weights = OrderedDict((feature_name, 0.0) for feature_name in VALUATION_FEATURES)
        for feature_name in available_features:
            weights[feature_name] = DOMAIN_WEIGHTS.get(feature_name, 0.0)
        return WeightCalculator._normalize_weights(weights)

    @staticmethod
    def _normalize_weights(weights):
        total = float(sum(weight for weight in weights.values() if weight is not None and weight > 0))
        normalized = OrderedDict()
        for feature_name in VALUATION_FEATURES:
            weight = float(weights.get(feature_name, 0.0) or 0.0)
            normalized[feature_name] = weight / total if total > 0 and weight > 0 else 0.0
        return normalized

    @staticmethod
    def _cosine_similarity(left, right):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))
