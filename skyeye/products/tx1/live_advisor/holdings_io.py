# -*- coding: utf-8 -*-
"""TX1 live advisor 的持仓输入解析。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_holdings_file(path: str | Path) -> dict[str, float]:
    """加载 CSV/JSON 持仓文件，并归一化成权重字典。"""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(file_path)
        if "order_book_id" not in frame.columns or "weight" not in frame.columns:
            raise ValueError("holdings csv must contain order_book_id and weight columns")
        rows = frame.to_dict("records")
        raw_holdings = {
            str(row.get("order_book_id", "")): row.get("weight")
            for row in rows
        }
        return normalize_holdings(raw_holdings)

    if suffix == ".json":
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("holdings json must be an object mapping order_book_id to weight")
        return normalize_holdings(payload)

    raise ValueError("unsupported holdings file type: {}".format(file_path.suffix))


def normalize_holdings(raw_holdings: dict) -> dict[str, float]:
    """过滤非法权重，并把剩余持仓归一化到 1.0。"""
    normalized = {}
    for order_book_id, raw_weight in (raw_holdings or {}).items():
        if not order_book_id:
            continue
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        normalized[str(order_book_id)] = weight

    total_weight = sum(normalized.values())
    if total_weight <= 0:
        raise ValueError("holdings file contains no valid positive weights")
    return {
        order_book_id: float(weight) / float(total_weight)
        for order_book_id, weight in normalized.items()
    }
