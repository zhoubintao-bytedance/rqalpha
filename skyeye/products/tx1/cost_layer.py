# -*- coding: utf-8 -*-
"""
TX1 Transaction Cost Layer

Models transaction costs for research portfolio evaluation.
Reuses cost logic from rqalpha_mod_sys_transaction_cost/deciders.py
with simplified vectorized interface suitable for research backtesting.

Cost components:
  - Commission: both sides, default 0.08% (0.0008)
  - Stamp tax: sell side only, default 0.05% (0.0005) post 2023-08-28
  - Slippage: both sides, default 5 bps (0.0005)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    """Transaction cost parameters.

    All rates are expressed as decimals (e.g. 0.0008 = 0.08%).
    """

    commission_rate: float = 0.0008
    stamp_tax_rate: float = 0.0005
    slippage_bps: float = 5.0
    min_commission: float = 0.0

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10000.0

    @property
    def one_way_cost(self) -> float:
        """Buy-side cost rate (commission + slippage)."""
        return self.commission_rate + self.slippage_rate

    @property
    def round_trip_cost(self) -> float:
        """Full round-trip cost rate (buy + sell including tax)."""
        buy = self.commission_rate + self.slippage_rate
        sell = self.commission_rate + self.slippage_rate + self.stamp_tax_rate
        return buy + sell


DEFAULT_COST_CONFIG = CostConfig()

CONSERVATIVE_COST_CONFIG = CostConfig(
    commission_rate=0.001,
    stamp_tax_rate=0.0005,
    slippage_bps=10.0,
)

STRESS_COST_CONFIG = CostConfig(
    commission_rate=0.0015,
    stamp_tax_rate=0.001,
    slippage_bps=20.0,
)


def apply_transaction_costs(portfolio_returns_df, cost_config=None):
    """Deduct transaction costs from portfolio returns using turnover.

    For each rebalancing day, cost = turnover * round_trip_cost / 2.
    Turnover is already defined as 0.5 * sum(|w_new - w_old|) in evaluator,
    so the full round-trip cost on the traded portion is turnover * round_trip_cost.

    Args:
        portfolio_returns_df: DataFrame with columns [date, portfolio_return, turnover, overlap]
        cost_config: CostConfig instance, defaults to DEFAULT_COST_CONFIG

    Returns:
        New DataFrame with added columns: cost, net_return
    """
    if portfolio_returns_df is None or len(portfolio_returns_df) == 0:
        return pd.DataFrame(columns=["date", "portfolio_return", "turnover", "overlap", "cost", "net_return"])

    cfg = cost_config or DEFAULT_COST_CONFIG
    result = portfolio_returns_df.copy()
    result["cost"] = result["turnover"] * cfg.round_trip_cost
    result["net_return"] = result["portfolio_return"] - result["cost"]
    return result


def estimate_annual_cost_drag(mean_turnover, cost_config=None, rebalance_freq=252):
    """Estimate annualized cost drag from average single-period turnover.

    Args:
        mean_turnover: Average per-period turnover ratio
        cost_config: CostConfig instance
        rebalance_freq: Number of rebalancing periods per year

    Returns:
        Annualized cost drag as a float
    """
    cfg = cost_config or DEFAULT_COST_CONFIG
    return mean_turnover * cfg.round_trip_cost * rebalance_freq


def compute_cost_metrics(portfolio_returns_df, cost_config=None):
    """Compute cost-layer evaluation metrics.

    Args:
        portfolio_returns_df: DataFrame with columns [date, portfolio_return, turnover]
        cost_config: CostConfig instance

    Returns:
        Dict with cost-layer metrics
    """
    if portfolio_returns_df is None or len(portfolio_returns_df) == 0:
        return {
            "annual_turnover": 0.0,
            "mean_turnover_per_period": 0.0,
            "cost_drag_annual": 0.0,
            "cost_erosion_ratio": 0.0,
            "net_mean_return": 0.0,
            "breakeven_cost_bps": 0.0,
        }

    cfg = cost_config or DEFAULT_COST_CONFIG
    costed = apply_transaction_costs(portfolio_returns_df, cfg)

    mean_turnover = float(costed["turnover"].mean())
    annual_turnover = mean_turnover * 252.0
    cost_drag = estimate_annual_cost_drag(mean_turnover, cfg)

    gross_mean = float(costed["portfolio_return"].mean())
    net_mean = float(costed["net_return"].mean())

    if abs(gross_mean) > 1e-12:
        cost_erosion = 1.0 - (net_mean / gross_mean) if gross_mean > 0 else float("inf")
    else:
        cost_erosion = 0.0

    # Breakeven: how high could round_trip_cost be before net_mean <= 0
    mean_cost_per_period = float(costed["cost"].mean())
    if mean_turnover > 1e-12 and gross_mean > 0:
        breakeven_rate = gross_mean / mean_turnover
        breakeven_bps = breakeven_rate * 10000.0
    else:
        breakeven_bps = 0.0

    return {
        "annual_turnover": float(np.clip(annual_turnover, 0, None)),
        "mean_turnover_per_period": mean_turnover,
        "cost_drag_annual": cost_drag,
        "cost_erosion_ratio": float(np.clip(cost_erosion, 0, None)),
        "net_mean_return": net_mean,
        "breakeven_cost_bps": breakeven_bps,
    }
