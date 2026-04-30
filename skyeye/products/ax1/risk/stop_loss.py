# -*- coding: utf-8 -*-
"""Portfolio-level drawdown stop-loss for AX1.

Three-level mechanism:
  Level 1 (Yellow):  DD >= 10%  → exposure cap 70%
  Level 2 (Orange):  DD >= 15%  → exposure cap 40%
  Level 3 (Red):     DD >= 20%  → exposure cap  0%  + cooldown

Recovery from Red requires cooldown expiry AND (regime not risk_off or gating disabled).
After recovery, the effective level is re-evaluated from current drawdown (gradual recovery).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Level enum
# ---------------------------------------------------------------------------

class StopLossLevel(Enum):
    """Stop-loss severity level."""
    NORMAL = "normal"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StopLossLevelConfig:
    """Configuration for a single stop-loss level."""
    name: str
    drawdown_threshold: float  # e.g. 0.10 for 10%
    target_exposure: float     # e.g. 0.70 for 70%


@dataclass
class StopLossConfig:
    """Full stop-loss configuration."""
    enabled: bool = False
    levels: list[StopLossLevelConfig] = field(default_factory=lambda: [
        StopLossLevelConfig(name="yellow", drawdown_threshold=0.10, target_exposure=0.70),
        StopLossLevelConfig(name="orange", drawdown_threshold=0.15, target_exposure=0.40),
        StopLossLevelConfig(name="red", drawdown_threshold=0.20, target_exposure=0.00),
    ])
    cooldown_trading_days: int = 10
    recovery_requires_regime_not_risk_off: bool = True


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class StopLossState:
    """Mutable state tracked across trading days.

    Designed for serialization so it can persist across live-trading sessions.
    """
    peak_equity: float = 1.0
    current_equity: float = 1.0
    current_drawdown: float = 0.0
    active_level: StopLossLevel = StopLossLevel.NORMAL
    cooldown_remaining: int = 0
    red_trigger_date: pd.Timestamp | None = None
    last_update_date: pd.Timestamp | None = None
    exposure_cap: float = 1.0
    bankrupt: bool = False
    trigger_log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "current_drawdown": self.current_drawdown,
            "active_level": self.active_level.value,
            "cooldown_remaining": self.cooldown_remaining,
            "red_trigger_date": str(self.red_trigger_date) if self.red_trigger_date is not None else None,
            "last_update_date": str(self.last_update_date) if self.last_update_date is not None else None,
            "exposure_cap": self.exposure_cap,
            "bankrupt": self.bankrupt,
            "trigger_log": deepcopy(self.trigger_log),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StopLossState:
        """Deserialize state from persistence."""
        level_str = data.get("active_level", "normal")
        try:
            active_level = StopLossLevel(level_str)
        except ValueError:
            active_level = StopLossLevel.NORMAL

        red_date = data.get("red_trigger_date")
        last_date = data.get("last_update_date")

        return cls(
            peak_equity=float(data.get("peak_equity", 1.0)),
            current_equity=float(data.get("current_equity", 1.0)),
            current_drawdown=float(data.get("current_drawdown", 0.0)),
            active_level=active_level,
            cooldown_remaining=int(data.get("cooldown_remaining", 0)),
            red_trigger_date=pd.Timestamp(red_date) if red_date else None,
            last_update_date=pd.Timestamp(last_date) if last_date else None,
            exposure_cap=float(data.get("exposure_cap", 1.0)),
            bankrupt=bool(data.get("bankrupt", False)),
            trigger_log=list(data.get("trigger_log", [])),
        )


# ---------------------------------------------------------------------------
# Config coercion helper
# ---------------------------------------------------------------------------

def _coerce_stop_loss_config(config: StopLossConfig | dict[str, Any] | None) -> StopLossConfig:
    """Convert various config formats into StopLossConfig."""
    if config is None:
        return StopLossConfig()
    if isinstance(config, StopLossConfig):
        return config
    if isinstance(config, dict):
        default_cfg = StopLossConfig()
        levels_raw = config.get("levels", None)
        if levels_raw is None:
            levels = default_cfg.levels
        else:
            levels = []
            for lv in levels_raw:
                if isinstance(lv, StopLossLevelConfig):
                    levels.append(lv)
                elif isinstance(lv, dict):
                    levels.append(StopLossLevelConfig(
                    name=str(lv.get("name", "")),
                    drawdown_threshold=float(lv.get("drawdown_threshold", 0.0)),
                    target_exposure=float(lv.get("target_exposure", 0.0)),
                ))
        return StopLossConfig(
            enabled=bool(config.get("enabled", False)),
            levels=levels,
            cooldown_trading_days=int(config.get("cooldown_trading_days", 10)),
            recovery_requires_regime_not_risk_off=bool(
                config.get("recovery_requires_regime_not_risk_off", True)
            ),
        )
    raise TypeError(f"Unsupported stop-loss config type: {type(config)}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PortfolioStopLoss:
    """Three-level portfolio drawdown stop-loss.

    Usage in backtest::

        stop_loss = PortfolioStopLoss(config)
        for date in sorted_dates:
            target_weights_date = ...
            stop_loss.update_equity(date, portfolio_return)
            scaled_weights = stop_loss.apply(target_weights_date, date, regime_state)

    Usage in live trading::

        stop_loss = PortfolioStopLoss(config, state=loaded_state)
        stop_loss.update_equity(today, portfolio_return)
        scaled = stop_loss.apply(target_weights, today, regime_state)
        save_state(stop_loss.state)
    """

    def __init__(
        self,
        config: StopLossConfig | dict[str, Any] | None = None,
        state: StopLossState | None = None,
    ) -> None:
        self._config = _coerce_stop_loss_config(config)
        self._state = state or StopLossState()
        # Sort levels by threshold descending for evaluation order
        self._sorted_levels = sorted(
            self._config.levels,
            key=lambda lv: lv.drawdown_threshold,
            reverse=True,
        )

    @property
    def config(self) -> StopLossConfig:
        return self._config

    @property
    def state(self) -> StopLossState:
        return self._state

    # ---- Equity tracking ----

    def update_equity(self, date: pd.Timestamp, portfolio_return: float) -> None:
        """Update the equity curve and drawdown from a single-day return.

        Must be called BEFORE ``apply()`` for each trading day.
        """
        self._state.current_equity *= (1.0 + portfolio_return)

        # Detect bankruptcy: negative equity means the portfolio is underwater
        if self._state.current_equity <= 0.0:
            if not self._state.bankrupt:
                self._state.bankrupt = True
                logger.critical(
                    "BANKRUPTCY detected: equity=%.6f (return=%.4f%%, date=%s). "
                    "Forcing RED level with target_exposure=0.",
                    self._state.current_equity, portfolio_return * 100, date,
                )

        self._state.peak_equity = max(self._state.peak_equity, self._state.current_equity)

        if self._state.peak_equity > 0:
            self._state.current_drawdown = 1.0 - self._state.current_equity / self._state.peak_equity
        else:
            self._state.current_drawdown = 1.0

        # Clamp: negative equity yields DD > 1.0, but 1.0 (total loss) is the
        # meaningful ceiling for level determination.  The bankrupt flag
        # distinguishes total-loss from underwater.
        self._state.current_drawdown = max(0.0, min(self._state.current_drawdown, 1.0))
        self._state.last_update_date = pd.Timestamp(date) if date is not None else None

    # ---- Main entry point ----

    def apply(
        self,
        target_weights: pd.DataFrame,
        date: pd.Timestamp,
        regime_state: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Apply stop-loss exposure scaling to target weights.

        Returns a DataFrame with the same schema but with ``target_weight``
        scaled by the stop-loss exposure cap.
        """
        if target_weights is None or target_weights.empty:
            return target_weights

        if not self._config.enabled:
            return target_weights

        # Decrement cooldown each day
        if self._state.cooldown_remaining > 0:
            self._state.cooldown_remaining -= 1

        # Determine effective level
        new_level = self._determine_level(regime_state)
        prev_level = self._state.active_level

        # Compute exposure cap from the effective level
        exposure_cap = self._exposure_cap_for_level(new_level)

        # Update state
        self._state.active_level = new_level
        self._state.exposure_cap = exposure_cap

        # Log level transitions
        if new_level != prev_level:
            self._log_trigger(date, new_level, prev_level, regime_state, exposure_cap)

        # Scale weights
        return self._scale_weights(target_weights, exposure_cap)

    # ---- Internal helpers ----

    def _determine_level(self, regime_state: dict[str, Any] | None) -> StopLossLevel:
        """Determine the effective stop-loss level.

        If currently in RED cooldown, stay RED until cooldown expires and
        recovery conditions are met.

        If bankrupt, always force RED with no recovery possible.
        """
        # Bankrupt portfolio: permanently locked at RED
        if self._state.bankrupt:
            return StopLossLevel.RED

        dd = self._state.current_drawdown

        # If currently in RED (or cooldown), check recovery first
        if self._state.active_level == StopLossLevel.RED:
            if not self._can_recover_from_red(regime_state):
                return StopLossLevel.RED
            # Recovery allowed: fall through to re-evaluate based on drawdown

        # Evaluate from highest severity to lowest
        for lv in self._sorted_levels:
            if dd >= lv.drawdown_threshold:
                # Map level name to enum
                level = self._level_enum_from_name(lv.name)
                # If transitioning to RED, start cooldown
                if level == StopLossLevel.RED and self._state.active_level != StopLossLevel.RED:
                    self._state.cooldown_remaining = self._config.cooldown_trading_days
                    self._state.red_trigger_date = self._state.last_update_date
                return level

        return StopLossLevel.NORMAL

    def _can_recover_from_red(self, regime_state: dict[str, Any] | None) -> bool:
        """Check if recovery from Level 3 (Red) is allowed.

        Requires:
        1. Cooldown period has elapsed
        2. Market regime is NOT risk_off (if gating is enabled)
        """
        if self._state.cooldown_remaining > 0:
            return False

        if self._config.recovery_requires_regime_not_risk_off:
            risk_state = (regime_state or {}).get("risk_state", "unknown")
            if risk_state == "risk_off":
                return False

        return True

    def _exposure_cap_for_level(self, level: StopLossLevel) -> float:
        """Get the exposure cap for a given level."""
        for lv in self._config.levels:
            if self._level_enum_from_name(lv.name) == level:
                return lv.target_exposure
        # NORMAL → full exposure
        if level == StopLossLevel.NORMAL:
            return 1.0
        # Fallback (should not happen)
        return 1.0

    def _scale_weights(self, target_weights: pd.DataFrame, exposure_cap: float) -> pd.DataFrame:
        """Scale all target_weight values by exposure cap.

        Preserves relative weights and all other columns.
        """
        if exposure_cap >= 1.0:
            return target_weights

        result = target_weights.copy()
        if "target_weight" in result.columns:
            result["target_weight"] = pd.to_numeric(
                result["target_weight"], errors="coerce"
            ).fillna(0.0) * exposure_cap
        return result

    def _level_enum_from_name(self, name: str) -> StopLossLevel:
        """Map a level config name to StopLossLevel enum."""
        mapping = {
            "yellow": StopLossLevel.YELLOW,
            "orange": StopLossLevel.ORANGE,
            "red": StopLossLevel.RED,
        }
        return mapping.get(name.lower(), StopLossLevel.NORMAL)

    def _log_trigger(
        self,
        date: pd.Timestamp,
        new_level: StopLossLevel,
        prev_level: StopLossLevel,
        regime_state: dict[str, Any] | None,
        exposure_cap: float,
    ) -> None:
        """Record a trigger event for diagnostics."""
        entry = {
            "date": str(date),
            "from_level": prev_level.value,
            "to_level": new_level.value,
            "drawdown": round(self._state.current_drawdown, 6),
            "exposure_cap": exposure_cap,
            "cooldown_remaining": self._state.cooldown_remaining,
            "regime_risk_state": (regime_state or {}).get("risk_state", "unknown"),
        }
        self._state.trigger_log.append(entry)
        logger.info(
            "Stop-loss level change: %s → %s (DD=%.2f%%, cap=%.0f%%, date=%s)",
            prev_level.value, new_level.value,
            self._state.current_drawdown * 100, exposure_cap * 100,
            date,
        )
