"""Predictor interface tests for AX1 LGBM-only model contract."""

from __future__ import annotations

import pandas as pd
import pytest

from skyeye.products.ax1.models.registry import ModelRegistry, Predictor


class _DummyPredictor(Predictor):
    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame | None = None,
        *,
        val_features: pd.DataFrame | None = None,
        val_labels: pd.DataFrame | None = None,
    ) -> "_DummyPredictor":
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": features["date"],
                "order_book_id": features["order_book_id"],
                "expected_relative_net_return_10d": 0.0,
            }
        )


def _make_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "order_book_id": ["A", "B"],
            "feature_z_excess_mom_20d": [1.0, -1.0],
        }
    )


def test_predictor_base_methods_are_explicitly_not_implemented():
    predictor = Predictor()

    with pytest.raises(NotImplementedError, match="Predictor.fit"):
        predictor.fit(_make_features())
    with pytest.raises(NotImplementedError, match="Predictor.predict"):
        predictor.predict(_make_features())


def test_model_registry_registers_explicit_predictor_instances():
    registry = ModelRegistry()
    predictor = _DummyPredictor()

    returned = registry.register("lgbm_test_double", predictor)
    predictions = registry.get("lgbm_test_double").fit(_make_features()).predict(_make_features())

    assert returned is predictor
    assert registry.list_names() == ["lgbm_test_double"]
    assert list(predictions["order_book_id"]) == ["A", "B"]
    assert "expected_relative_net_return_10d" in predictions.columns


def test_model_registry_rejects_duplicate_or_unknown_names():
    registry = ModelRegistry()
    registry.register("lgbm_test_double", _DummyPredictor())

    with pytest.raises(ValueError, match="model already registered"):
        registry.register("lgbm_test_double", _DummyPredictor())
    with pytest.raises(KeyError, match="unknown model"):
        registry.get("missing")
