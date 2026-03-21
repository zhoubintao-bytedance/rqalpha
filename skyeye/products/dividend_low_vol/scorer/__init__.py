# -*- coding: utf-8 -*-

from skyeye.products.dividend_low_vol.scorer.config import (
    CONFIDENCE_FEATURES,
    ETF_CODE,
    FEATURES,
    INDEX_CODE,
    VALUATION_FEATURES,
)
from skyeye.products.dividend_low_vol.scorer.data_fetcher import DataFetcher, DataGapError
from skyeye.products.dividend_low_vol.scorer.feature_engine import FeatureEngine
from skyeye.products.dividend_low_vol.scorer.score_synthesizer import ScoreSynthesizer, ScoreUnavailableError
from skyeye.products.dividend_low_vol.scorer.validator import Validator
from skyeye.products.dividend_low_vol.scorer.weight_calculator import WeightCalculator

__all__ = [
    "CONFIDENCE_FEATURES",
    "DataFetcher",
    "DataGapError",
    "DividendScorer",
    "ETF_CODE",
    "FEATURES",
    "FeatureEngine",
    "INDEX_CODE",
    "ScoreSynthesizer",
    "ScoreUnavailableError",
    "Validator",
    "VALUATION_FEATURES",
    "WeightCalculator",
]


def __getattr__(name):
    if name == "DividendScorer":
        from skyeye.products.dividend_low_vol.scorer.main import DividendScorer
        return DividendScorer
    raise AttributeError(name)
