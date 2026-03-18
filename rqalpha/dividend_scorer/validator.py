# -*- coding: utf-8 -*-

import importlib.util
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rqalpha.dividend_scorer.config import LABEL_WINDOW_N, SUBSAMPLE_INTERVAL_S, VALUATION_FEATURES


class Validator(object):
    def validate_ic_stability(self, feature_matrix, price_series):
        feature_matrix = feature_matrix.reindex(columns=VALUATION_FEATURES)
        future_returns = price_series.shift(-LABEL_WINDOW_N) / price_series - 1.0
        joined = feature_matrix.join(future_returns.rename("future_return"), how="inner").dropna(how="all")

        result = OrderedDict()
        for feature_name in VALUATION_FEATURES:
            yearly_ics = OrderedDict()
            feature_frame = joined[[feature_name, "future_return"]].dropna()
            if feature_frame.empty:
                result[feature_name] = {
                    "yearly_ics": yearly_ics,
                    "negative_year_ratio": 0.0,
                }
                continue
            for year, frame in feature_frame.groupby(feature_frame.index.year):
                sample = frame.iloc[::SUBSAMPLE_INTERVAL_S]
                if len(sample) < 3:
                    continue
                if sample[feature_name].nunique(dropna=True) <= 1 or sample["future_return"].nunique(dropna=True) <= 1:
                    continue
                ic_value = spearmanr(sample[feature_name], sample["future_return"], nan_policy="omit").correlation
                if np.isnan(ic_value):
                    continue
                yearly_ics[int(year)] = float(ic_value)
            negative_years = [value for value in yearly_ics.values() if value < 0]
            negative_year_ratio = float(len(negative_years)) / float(len(yearly_ics)) if yearly_ics else 0.0
            result[feature_name] = {
                "yearly_ics": yearly_ics,
                "negative_year_ratio": negative_year_ratio,
            }
        return result

    def validate_strategy_closure(self, strategy_file, cash=1000000):
        strategy_scorer = self._load_strategy_scorer()
        windows = strategy_scorer.run_rolling_backtests(strategy_file, cash)
        quarterly_scores, quarterly_raw = strategy_scorer.project_to_quarters(windows)
        composite, core_indicators = strategy_scorer.compute_composite_score(quarterly_scores, quarterly_raw)
        stability = strategy_scorer.compute_stability_score(quarterly_scores)
        market_env = strategy_scorer.compute_market_env_scores(
            quarterly_scores,
            strategy_scorer.get_benchmark_quarterly_returns(),
        )
        return {
            "window_count": len(windows),
            "strategy_score": composite,
            "stability_score": stability,
            "market_env_score": market_env,
            "core_indicators": core_indicators,
        }

    @staticmethod
    def _load_strategy_scorer():
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        module_path = os.path.join(root_dir, "strategy_scorer.py")
        if not os.path.exists(module_path):
            raise RuntimeError("strategy_scorer.py is not available at {}".format(module_path))
        spec = importlib.util.spec_from_file_location("strategy_scorer", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
