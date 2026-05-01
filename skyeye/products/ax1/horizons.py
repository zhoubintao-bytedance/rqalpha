"""Helpers for AX1 score/label horizon contracts."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


_HORIZON_RE = re.compile(r"_(\d+)d$")

_LABEL_KINDS = (
    ("label_relative_net_return_", "relative_net_return"),
    ("label_net_return_", "net_return"),
    ("label_return_", "return"),
    ("label_volatility_", "volatility"),
)


def horizon_from_column(column: str | None) -> int | None:
    if not column:
        return None
    match = _HORIZON_RE.search(str(column))
    return int(match.group(1)) if match else None


def label_kind_from_column(column: str | None) -> str | None:
    if not column:
        return None
    value = str(column)
    for prefix, kind in _LABEL_KINDS:
        if value.startswith(prefix):
            return kind
    return None


def select_label_column_for_horizon(
    frame: pd.DataFrame,
    *,
    horizon: int | None,
    prefixes: Iterable[str],
) -> str | None:
    columns = set(frame.columns)
    if horizon is not None:
        for prefix in prefixes:
            candidate = f"{prefix}_{int(horizon)}d"
            if candidate in columns:
                return candidate
    for prefix in prefixes:
        matching = sorted(column for column in columns if column.startswith(f"{prefix}_"))
        if matching:
            return matching[0]
    return None
