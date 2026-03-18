# -*- coding: utf-8 -*-

from rqalpha.dividend_scorer.config import (
    CONFIDENCE_FEATURES,
    ETF_CODE,
    FEATURES,
    INDEX_CODE,
    VALUATION_FEATURES,
)
from rqalpha.dividend_scorer.data_fetcher import DataFetcher, DataGapError
from rqalpha.dividend_scorer.feature_engine import FeatureEngine
from rqalpha.dividend_scorer.score_synthesizer import ScoreSynthesizer, ScoreUnavailableError
from rqalpha.dividend_scorer.validator import Validator
from rqalpha.dividend_scorer.weight_calculator import WeightCalculator

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
        from rqalpha.dividend_scorer.main import DividendScorer
        return DividendScorer
    raise AttributeError(name)
