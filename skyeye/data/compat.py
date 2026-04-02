# -*- coding: utf-8 -*-
"""
AKShare fallback for data not available in rqdatac.

Currently only northbound aggregate net flow needs AKShare.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

DateLike = Union[str, int, "pd.Timestamp"]


def get_northbound_flow_akshare(
    start_date: DateLike, end_date: DateLike
) -> Optional[pd.DataFrame]:
    """Fetch northbound aggregate net flow via AKShare.

    Returns DataFrame with columns: date, north_net_flow.
    """
    try:
        import akshare as ak
    except ImportError:
        logger.warning("akshare not installed, northbound flow unavailable")
        return None

    frames = []
    for label in ("沪股通", "深股通"):
        try:
            raw = ak.stock_hsgt_hist_em(symbol=label)
            if raw is not None and not raw.empty:
                frames.append(raw)
        except Exception as exc:
            logger.warning("AKShare northbound fetch failed for %s: %s", label, exc)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)

    date_col = "日期" if "日期" in combined.columns else combined.columns[0]
    flow_col = None
    for candidate in ("当日成交净买额", "当日净流入"):
        if candidate in combined.columns:
            flow_col = candidate
            break
    if flow_col is None:
        flow_col = combined.columns[1]

    combined["date"] = pd.to_datetime(combined[date_col])
    combined["north_net_flow"] = pd.to_numeric(combined[flow_col], errors="coerce")

    result = (
        combined.groupby("date")["north_net_flow"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    mask = (result["date"] >= pd.Timestamp(str(start_date))) & (
        result["date"] <= pd.Timestamp(str(end_date))
    )
    return result.loc[mask].reset_index(drop=True)
