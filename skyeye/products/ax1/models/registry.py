"""AX1 predictor registry."""

from __future__ import annotations

from typing import Any

import pandas as pd


class Predictor:
    """模型接口：真实模型必须显式实现 fit/predict。"""

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame | None = None,
        *,
        val_features: pd.DataFrame | None = None,
        val_labels: pd.DataFrame | None = None,
    ) -> "Predictor":
        raise NotImplementedError("Predictor.fit is not implemented")

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Predictor.predict is not implemented")


class ModelRegistry:
    """按名字管理 predictor 实例。"""

    def __init__(self) -> None:
        self._predictors: dict[str, Predictor] = {}

    def register(self, name: str, predictor: Predictor) -> Predictor:
        if not name:
            raise ValueError("model name must be non-empty")
        if name in self._predictors:
            raise ValueError(f"model already registered: {name}")
        self._predictors[name] = predictor
        return predictor

    def get(self, name: str) -> Predictor:
        try:
            return self._predictors[name]
        except KeyError as exc:
            raise KeyError(f"unknown model: {name}") from exc

    def list_names(self) -> list[str]:
        return sorted(self._predictors)
