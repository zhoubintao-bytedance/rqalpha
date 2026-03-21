# -*- coding: utf-8 -*-

from skyeye.dividend_scorer.config import (
    CONFIDENCE_FEATURES,
    ETF_CODE,
    FEATURES,
    INDEX_CODE,
    VALUATION_FEATURES,
)
from skyeye.dividend_scorer.data_fetcher import DataFetcher, DataGapError
from skyeye.dividend_scorer.feature_engine import FeatureEngine
from skyeye.dividend_scorer.score_synthesizer import ScoreSynthesizer, ScoreUnavailableError
from skyeye.dividend_scorer.validator import Validator
from skyeye.dividend_scorer.weight_calculator import WeightCalculator

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
        from skyeye.dividend_scorer.main import DividendScorer
        return DividendScorer
    raise AttributeError(name)
