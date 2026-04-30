# -*- coding: utf-8 -*-
"""
AKShare fallback for data not available in rqdatac.

Currently northbound aggregate net flow and PMI rely on AKShare fallback.
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


def get_macro_pmi_akshare(
    start_date: DateLike, end_date: DateLike
) -> Optional[pd.DataFrame]:
    """Fetch official manufacturing PMI via AKShare.

    Returns DataFrame with columns: ``date``, ``pmi``.
    """
    try:
        import akshare as ak
    except ImportError:
        logger.warning("akshare not installed, macro PMI unavailable")
        return None

    candidate_functions = [
        "macro_china_pmi_yearly",
        "macro_china_pmi",
    ]

    for function_name in candidate_functions:
        fetcher = getattr(ak, function_name, None)
        if not callable(fetcher):
            continue
        try:
            raw = fetcher()
        except TypeError:
            continue
        except Exception as exc:
            logger.warning("AKShare PMI fetch failed via %s: %s", function_name, exc)
            continue

        normalized = _normalize_macro_pmi_frame(raw)
        if normalized is None or normalized.empty:
            continue

        mask = (normalized["date"] >= pd.Timestamp(str(start_date))) & (
            normalized["date"] <= pd.Timestamp(str(end_date))
        )
        result = normalized.loc[mask].reset_index(drop=True)
        if not result.empty:
            return result

    logger.warning("macro PMI unavailable from AKShare")
    return None


def _normalize_macro_pmi_frame(raw: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if raw is None or raw.empty:
        return None

    frame = raw.copy()
    date_col = None
    for candidate in ("date", "Date", "日期", "月份", "month"):
        if candidate in frame.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = frame.columns[0]

    pmi_col = None
    for candidate in (
        "pmi",
        "PMI",
        "制造业PMI",
        "制造业采购经理指数(%)",
        "官方制造业PMI",
        "value",
        "今值",
    ):
        if candidate in frame.columns:
            pmi_col = candidate
            break

    if pmi_col is None:
        numeric_candidates = [
            column
            for column in frame.columns
            if column != date_col and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not numeric_candidates:
            return None
        pmi_col = numeric_candidates[0]

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_col], errors="coerce"),
            "pmi": pd.to_numeric(frame[pmi_col], errors="coerce"),
        }
    )
    result = result.dropna(subset=["date"])
    if result.empty:
        return None
    result = result.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return result.reset_index(drop=True)
