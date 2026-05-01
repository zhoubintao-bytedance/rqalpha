"""AX1-specific ETF metadata fallback helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd


AX1_ETF_INDUSTRY_FALLBACK: dict[str, str] = {
    "510050.XSHG": "sse50",
    "510300.XSHG": "broad",
    "510500.XSHG": "broad",
    "512100.XSHG": "csi1000",
    "588000.XSHG": "star50",
    "512800.XSHG": "bank",
    "512880.XSHG": "brokerage",
    "512000.XSHG": "brokerage",
    "512070.XSHG": "non_bank",
    "515000.XSHG": "technology",
    "515260.XSHG": "electronics",
    "512480.XSHG": "semiconductor",
    "159995.XSHE": "semiconductor",
    "512760.XSHG": "semiconductor",
    "512720.XSHG": "computing",
    "515050.XSHG": "telecom_5g",
    "515880.XSHG": "telecom",
    "515790.XSHG": "pv",
    "515030.XSHG": "new_energy_vehicle",
    "516160.XSHG": "new_energy",
    "512170.XSHG": "healthcare",
    "512010.XSHG": "healthcare",
    "159929.XSHE": "healthcare",
    "159928.XSHE": "consumer",
    "512690.XSHG": "liquor",
    "512400.XSHG": "metals",
    "515220.XSHG": "coal",
    "515210.XSHG": "steel",
    "512980.XSHG": "media",
    "512660.XSHG": "defense",
    "516970.XSHG": "infrastructure",
    "159865.XSHE": "agriculture",
    "159611.XSHE": "utilities",
    "516110.XSHG": "auto",
    "159819.XSHE": "ai",
    "562500.XSHG": "robotics",
    "159639.XSHE": "carbon_neutral",
    "159870.XSHE": "chemical",
    "512200.XSHG": "real_estate",
    "510880.XSHG": "dividend",
    "515180.XSHG": "dividend",
    "512890.XSHG": "low_vol",
    "159901.XSHE": "shenzhen100",
    "159902.XSHE": "midcap100",
    "159949.XSHE": "growth",
    "159915.XSHE": "growth",
    "513050.XSHG": "internet",
    "513130.XSHG": "overseas_tech",
    "513100.XSHG": "overseas_broad",
    "513500.XSHG": "overseas_broad",
    "518880.XSHG": "commodity_gold",
    "159985.XSHE": "commodity_agriculture",
    "511010.XSHG": "bond_government_5y",
    "511260.XSHG": "bond_government_10y",
    "511220.XSHG": "bond_credit",
}

_MISSING_INDUSTRY_VALUES = {"", "unknown", "none", "nan", "nat", "null"}


def normalize_industry_value(value: Any) -> str | None:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in _MISSING_INDUSTRY_VALUES:
        return None
    return text


def resolve_ax1_etf_industry_map(
    order_book_ids: Sequence[str],
    *sources: Mapping[str, Any] | None,
    default: str = "Unknown",
) -> dict[str, str]:
    normalized_sources: list[dict[str, str]] = []
    for source in sources:
        if not source:
            continue
        normalized_sources.append(
            {
                str(order_book_id): normalized
                for order_book_id, value in source.items()
                if (normalized := normalize_industry_value(value)) is not None
            }
        )
    fallback = {
        str(order_book_id): normalized
        for order_book_id, value in AX1_ETF_INDUSTRY_FALLBACK.items()
        if (normalized := normalize_industry_value(value)) is not None
    }
    resolved: dict[str, str] = {}
    for order_book_id in order_book_ids:
        key = str(order_book_id)
        value = None
        for source in normalized_sources:
            value = source.get(key)
            if value is not None:
                break
        if value is None:
            value = fallback.get(key)
        resolved[key] = value if value is not None else default
    return resolved
