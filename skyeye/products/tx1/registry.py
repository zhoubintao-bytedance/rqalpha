# -*- coding: utf-8 -*-
"""Registry for TX1 strategies.

Mirrors the pattern from skyeye/products/dividend_low_vol/registry.py
but scoped to the TX1 product line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


PRODUCT_ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = PRODUCT_ROOT / "strategies"


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    name: str
    family: str
    benchmark: str
    instrument: str
    status: str
    hypothesis: str
    entrypoint: str
    rolling_score_start_date: str | None
    raw: dict

    @property
    def entrypoint_path(self) -> Path:
        return PRODUCT_ROOT / self.entrypoint


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def list_strategy_specs() -> list[StrategySpec]:
    specs = []
    for spec_path in sorted(STRATEGIES_ROOT.glob("*/spec.yaml")):
        raw = _load_yaml(spec_path)
        specs.append(
            StrategySpec(
                strategy_id=raw["strategy_id"],
                name=raw["name"],
                family=raw["family"],
                benchmark=raw["benchmark"],
                instrument=raw["instrument"],
                status=raw["status"],
                hypothesis=raw["hypothesis"],
                entrypoint=raw["entrypoint"],
                rolling_score_start_date=raw.get("rolling_score_start_date"),
                raw=raw,
            )
        )
    return specs


def get_strategy_spec(strategy_id: str) -> StrategySpec:
    for spec in list_strategy_specs():
        if spec.strategy_id == strategy_id:
            return spec
    raise KeyError("unknown strategy_id: {}".format(strategy_id))


def find_strategy_spec_by_file(strategy_file: str | Path) -> StrategySpec | None:
    target = Path(strategy_file).resolve()
    for spec in list_strategy_specs():
        if spec.entrypoint_path.resolve() == target:
            return spec
    return None
