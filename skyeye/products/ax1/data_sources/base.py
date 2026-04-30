"""AX1 data source contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DataSourceCapability:
    name: str
    source_family: str
    asset_type: str
    point_in_time: bool
    observable_lag_days: int | None
    requires_as_of_date: bool
    status: str
    reason_code: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source_family": self.source_family,
            "asset_type": self.asset_type,
            "point_in_time": bool(self.point_in_time),
            "observable_lag_days": self.observable_lag_days,
            "requires_as_of_date": bool(self.requires_as_of_date),
            "status": self.status,
            "reason_code": self.reason_code,
            "description": self.description,
        }


class AX1DataSource:
    source_family = "unknown"

    def capabilities(self) -> list[DataSourceCapability]:
        raise NotImplementedError

    def load_panel(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError


class DataSourceRegistry:
    def __init__(self, capabilities: Iterable[DataSourceCapability]):
        self._capabilities = list(capabilities)
        self._by_name = {capability.name: capability for capability in self._capabilities}
        if len(self._by_name) != len(self._capabilities):
            raise ValueError("AX1 data source registry contains duplicate capability names")

    def capabilities(self) -> list[DataSourceCapability]:
        return list(self._capabilities)

    def get(self, name: str) -> DataSourceCapability:
        try:
            return self._by_name[str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown AX1 data source capability: {name}") from exc

    def to_dict(self) -> dict:
        return {"capabilities": [capability.to_dict() for capability in self._capabilities]}
