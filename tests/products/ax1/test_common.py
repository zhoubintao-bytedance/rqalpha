"""Tests for skyeye.products.ax1._common utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skyeye.products.ax1._common import (
    allocate_with_cap,
    apply_group_cap,
    coerce_cost_config,
    normalize_asset_type,
    require_columns,
)


class TestRequireColumns:
    def test_raises_on_none_frame(self):
        with pytest.raises(ValueError, match="must not be None"):
            require_columns(None, ["a"], entity="df")

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="missing required columns.*b"):
            require_columns(df, ["a", "b"], entity="df")

    def test_passes_when_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        require_columns(df, ["a", "b"], entity="df")


class TestNormalizeAssetType:
    @pytest.mark.parametrize("value", [None, np.nan, pd.NA, "", "nan", "none", "null", "<na>"])
    def test_fallback_to_stock(self, value):
        assert normalize_asset_type(value) == "stock"

    @pytest.mark.parametrize("value", ["ETF", "etf", "fund", "index_fund"])
    def test_etf_variants(self, value):
        assert normalize_asset_type(value) == "etf"

    @pytest.mark.parametrize("value", ["stock", "Stock", "CS", "equity", "common_stock"])
    def test_stock_variants(self, value):
        assert normalize_asset_type(value) == "stock"

    def test_unknown_passthrough(self):
        assert normalize_asset_type("bond") == "bond"


class TestAllocateWithCap:
    def test_empty_scores(self):
        assert allocate_with_cap(pd.Series(dtype=float), budget=1.0, max_weight=0.1) == {}

    def test_uniform_allocation(self):
        scores = pd.Series([1.0, 1.0], index=["A", "B"])
        result = allocate_with_cap(scores, budget=1.0, max_weight=0.6)
        assert pytest.approx(result["A"], abs=1e-9) == 0.5
        assert pytest.approx(result["B"], abs=1e-9) == 0.5

    def test_capping(self):
        scores = pd.Series([10.0, 1.0, 1.0], index=["A", "B", "C"])
        result = allocate_with_cap(scores, budget=1.0, max_weight=0.4)
        assert result["A"] == pytest.approx(0.4, abs=1e-9)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_zero_budget(self):
        scores = pd.Series([1.0], index=["A"])
        assert allocate_with_cap(scores, budget=0, max_weight=0.5) == {}


class TestApplyGroupCap:
    def test_empty_weights(self):
        assert apply_group_cap({}, {}, budget=1.0, max_group_weight=0.5) == {}

    def test_group_capping(self):
        weights = {"A": 0.4, "B": 0.3, "C": 0.3}
        group_map = {"A": "tech", "B": "tech", "C": "fin"}
        result = apply_group_cap(weights, group_map, budget=1.0, max_group_weight=0.5)
        tech_weight = result.get("A", 0) + result.get("B", 0)
        assert tech_weight <= 0.5 + 1e-9


class TestCoerceCostConfig:
    def test_none_returns_none(self):
        assert coerce_cost_config(None) is None

    def test_dict_disabled(self):
        assert coerce_cost_config({"enabled": False}) is None

    def test_dict_creates_config(self):
        result = coerce_cost_config({"commission_rate": 0.001})
        assert result is not None
        assert result.commission_rate == 0.001

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            coerce_cost_config(42)
