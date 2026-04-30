import pandas as pd
import pytest

from skyeye.products.ax1.optimizer.allocation import OpportunityPoolOptimizer


def _views(**overrides) -> pd.DataFrame:
    n = len(next(iter(overrides.values()))) if overrides else 6
    data = {
        "date": pd.to_datetime(["2024-01-02"] * 6),
        "order_book_id": ["CORE_LOW", "CORE_OK", "IND_HIGH", "IND_MID", "STYLE_POS", "STYLE_NEG"],
        "universe_layer": ["core", "core", "industry", "industry", "style", "style"],
        "exposure_group": ["broad_beta", "broad_beta", "sector", "sector", "style_factor", "style_factor"],
        "asset_type": ["etf"] * 6,
        "industry": ["broad", "broad", "tech", "bank", "growth", "value"],
        "expected_relative_net_return_10d": [0.01, 0.03, 0.20, 0.06, 0.04, -0.02],
        "confidence": [1.0, 0.9, 1.0, 0.8, 0.7, 1.0],
        "view_score_10d": [0.01, 0.03, 0.20, 0.06, 0.04, -0.02],
    }
    data = {key: value[:n] for key, value in data.items()}
    data.update(overrides)
    return pd.DataFrame(data)


def _constraints(**overrides) -> dict:
    result = {
        "target_gross_exposure": 1.0,
        "cash_buffer": 0.03,
        "max_single_weight": 0.50,
        "max_industry_weight": 1.0,
        "max_position_count": 6,
        "min_position_count": 1,
        "max_portfolio_volatility": None,
    }
    result.update(overrides)
    return result


def _allocation_config(**overrides) -> dict:
    result = {
        "kind": "opportunity_pool_optimizer",
        "score_column": "expected_relative_net_return_10d",
        "allow_gross_underfill": True,
        "min_allocatable_score": 0.0,
        "cash_fallback": {"enabled": True},
        "exposure_groups": {
            "broad_beta": {"max_weight": 0.60},
            "sector": {"max_weight": 0.70},
            "style_factor": {"max_weight": 0.35},
        },
    }
    result.update(overrides)
    return result


def test_opportunity_pool_allocates_by_global_score_not_layer_budget():
    targets = OpportunityPoolOptimizer().optimize(
        _views(),
        constraints=_constraints(max_single_weight=0.60),
        allocation_config=_allocation_config(
            exposure_groups={
                "broad_beta": {"max_weight": 1.0},
                "sector": {"max_weight": 1.0},
                "style_factor": {"max_weight": 1.0},
            }
        ),
    )
    weights = targets.set_index("order_book_id")["target_weight"]

    assert targets["target_weight"].sum() == pytest.approx(0.97)
    assert weights["IND_HIGH"] > weights["CORE_LOW"]
    assert weights["IND_HIGH"] > weights["CORE_OK"]
    assert "STYLE_NEG" not in set(targets["order_book_id"])
    assert set(targets["component"]) == {"opportunity_pool"}


def test_opportunity_pool_ignores_max_industry_weight_when_all_unknown():
    """When all ETFs have industry='Unknown', max_industry_weight should be ignored.

    Rationale: ETF industry classification is ambiguous. When data source lacks industry info,
    all ETFs default to 'Unknown', making max_industry_weight a global cap instead of sector risk control.
    The exposure_group constraint already provides proper sector-level risk management.
    """
    views = _views(
        # All ETFs have industry='Unknown' (simulating missing industry data)
        industry=["Unknown"] * 6,
        expected_relative_net_return_10d=[0.01, 0.03, 0.20, 0.06, 0.04, -0.02],
        confidence=[1.0, 0.9, 1.0, 0.8, 0.7, 1.0],
    )

    # With max_industry_weight=0.20, if applied to 'Unknown' group,
    # it would cap total portfolio at 20% (incorrect behavior)
    targets = OpportunityPoolOptimizer().optimize(
        views,
        constraints=_constraints(
            max_single_weight=0.40,
            max_industry_weight=0.20,  # Should be ignored when all are 'Unknown'
        ),
        allocation_config=_allocation_config(
            exposure_groups={
                "broad_beta": {"max_weight": 0.40},
                "sector": {"max_weight": 0.55},  # This should control sector exposure
                "style_factor": {"max_weight": 0.35},
            }
        ),
    )

    weights = targets.set_index("order_book_id")["target_weight"]

    # Portfolio should allocate normally (not capped at 20%)
    assert targets["target_weight"].sum() > 0.60, "Portfolio should not be capped by max_industry_weight when all industry='Unknown'"

    # IND_HIGH (highest alpha) should get meaningful allocation
    assert weights["IND_HIGH"] > 0.15, "Highest alpha ETF should get significant weight"

    # Sector group (IND_HIGH + IND_MID) should respect exposure_group cap of 0.55
    sector_weight = weights.get("IND_HIGH", 0) + weights.get("IND_MID", 0)
    assert sector_weight <= 0.55 + 1e-6, "Sector exposure should respect exposure_group cap"


def test_opportunity_pool_keeps_cash_when_all_alpha_is_not_positive():
    views = _views(
        order_book_id=["CORE_A", "CORE_B", "IND_A"],
        universe_layer=["core", "core", "industry"],
        exposure_group=["broad_beta", "broad_beta", "sector"],
        industry=["broad", "broad", "tech"],
        expected_relative_net_return_10d=[-0.01, -0.03, -0.20],
        confidence=[1.0, 1.0, 1.0],
        view_score_10d=[-0.01, -0.03, -0.20],
    )

    targets = OpportunityPoolOptimizer().optimize(
        views,
        constraints=_constraints(max_position_count=3, min_position_count=3),
        allocation_config=_allocation_config(min_allocatable_score=0.0),
    )

    assert targets.empty


def test_opportunity_pool_respects_exposure_group_caps_after_global_ranking():
    targets = OpportunityPoolOptimizer().optimize(
        _views(),
        constraints=_constraints(max_single_weight=0.80),
        allocation_config=_allocation_config(
            exposure_groups={
                "broad_beta": {"max_weight": 0.20},
                "sector": {"max_weight": 0.80},
                "style_factor": {"max_weight": 0.80},
            }
        ),
    )

    exposure = targets.groupby("exposure_group")["target_weight"].sum()

    assert exposure["broad_beta"] <= 0.20 + 1e-12
    assert exposure["sector"] > exposure["broad_beta"]


class _CorrelatedRiskModel:
    def get_covariance_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                [0.04, 0.039, 0.000],
                [0.039, 0.04, 0.000],
                [0.000, 0.000, 0.04],
            ],
            index=["CORR_A", "CORR_B", "LOW_CORR"],
            columns=["CORR_A", "CORR_B", "LOW_CORR"],
        )


def test_opportunity_pool_penalizes_correlated_cluster_from_full_covariance():
    views = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "order_book_id": ["CORR_A", "CORR_B", "LOW_CORR"],
            "universe_layer": ["core", "industry", "style"],
            "exposure_group": ["broad_beta", "sector", "style_factor"],
            "asset_type": ["etf"] * 3,
            "industry": ["broad", "tech", "value"],
            "expected_relative_net_return_10d": [0.05, 0.05, 0.05],
            "confidence": [1.0, 1.0, 1.0],
            "view_score_10d": [0.05, 0.05, 0.05],
        }
    )

    targets = OpportunityPoolOptimizer().optimize(
        views,
        constraints=_constraints(max_single_weight=1.0, max_position_count=3),
        allocation_config=_allocation_config(
            exposure_groups={
                "broad_beta": {"max_weight": 1.0},
                "sector": {"max_weight": 1.0},
                "style_factor": {"max_weight": 1.0},
            }
        ),
        risk_model=_CorrelatedRiskModel(),
    )
    weights = targets.set_index("order_book_id")["target_weight"]

    assert weights["LOW_CORR"] > weights["CORR_A"]
    assert weights["LOW_CORR"] > weights["CORR_B"]


class _NegativeCorrRiskModel:
    """One asset is negatively correlated (diversifier), others are positively correlated."""

    def get_covariance_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                [0.04, 0.039, -0.038],
                [0.039, 0.04, -0.037],
                [-0.038, -0.037, 0.04],
            ],
            index=["POS_A", "POS_B", "NEG_DIV"],
            columns=["POS_A", "POS_B", "NEG_DIV"],
        )


def test_opportunity_pool_rewards_negative_correlation_diversification():
    """Assets with negative average correlation (diversifiers) should get a risk
    discount and therefore higher weight than positively-correlated peers with
    the same alpha score."""
    views = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "order_book_id": ["POS_A", "POS_B", "NEG_DIV"],
            "universe_layer": ["core", "industry", "style"],
            "exposure_group": ["broad_beta", "sector", "style_factor"],
            "asset_type": ["etf"] * 3,
            "industry": ["broad", "tech", "value"],
            "expected_relative_net_return_10d": [0.05, 0.05, 0.05],
            "confidence": [1.0, 1.0, 1.0],
            "view_score_10d": [0.05, 0.05, 0.05],
        }
    )

    targets = OpportunityPoolOptimizer().optimize(
        views,
        constraints=_constraints(max_single_weight=1.0, max_position_count=3),
        allocation_config=_allocation_config(
            exposure_groups={
                "broad_beta": {"max_weight": 1.0},
                "sector": {"max_weight": 1.0},
                "style_factor": {"max_weight": 1.0},
            }
        ),
        risk_model=_NegativeCorrRiskModel(),
    )
    weights = targets.set_index("order_book_id")["target_weight"]

    # NEG_DIV has negative average off-diagonal correlation → diversification
    # discount → lower risk penalty → higher weight than POS_A/POS_B.
    assert weights["NEG_DIV"] > weights["POS_A"]
    assert weights["NEG_DIV"] > weights["POS_B"]


class _HighVolRiskModel:
    def get_covariance_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[0.60, 0.0], [0.0, 0.60]],
            index=["HIGH_A", "HIGH_B"],
            columns=["HIGH_A", "HIGH_B"],
        )


def test_opportunity_pool_respects_portfolio_volatility_cap():
    views = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 2),
            "order_book_id": ["HIGH_A", "HIGH_B"],
            "universe_layer": ["industry", "industry"],
            "exposure_group": ["sector", "sector"],
            "asset_type": ["etf", "etf"],
            "industry": ["tech", "bank"],
            "expected_relative_net_return_10d": [0.10, 0.10],
            "confidence": [1.0, 1.0],
            "view_score_10d": [0.10, 0.10],
        }
    )

    targets = OpportunityPoolOptimizer().optimize(
        views,
        constraints=_constraints(
            max_single_weight=1.0,
            max_position_count=2,
            max_portfolio_volatility=0.20,
        ),
        allocation_config=_allocation_config(exposure_groups={"sector": {"max_weight": 1.0}}),
        risk_model=_HighVolRiskModel(),
    )

    assert 0.0 < targets["target_weight"].sum() < 0.97
