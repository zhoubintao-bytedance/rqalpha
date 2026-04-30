"""AX1 time-series splitter。

提供两种 splitter：
  - ``SingleSplitSplitter``: MVP 单次 train/val/test 三段切分。
  - ``WalkForwardSplitter``: 滚动多 fold，fold schema 与 SingleSplit 对齐。

接口（``split(labeled_df) -> list[dict]``）统一，使得 runner 按 ``splitter.kind``
切换 splitter 时无需修改下游模型/评估/persistence。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from dateutil.relativedelta import relativedelta


class SingleSplitSplitter:
    """MVP 单次 train/val/test 三段切分器。

    Fold schema 与 TX1 ``WalkForwardSplitter`` 完全一致，方便下游接口兼容：
        {
            "fold_id": 0,
            "train_df", "val_df", "test_df",
            "train_end", "val_start", "val_end", "test_start", "test_end",
        }

    切分逻辑：
        train: [data_start, train_end]
        embargo: 跳过 embargo_days 个交易日
        val:   [val_start, val_end] (val_months 个自然月)
        embargo: 跳过 embargo_days 个交易日
        test:  [test_start, test_end] (test_months 个自然月，不超过数据末尾)
    """

    def __init__(
        self,
        train_end: str | pd.Timestamp,
        val_months: int = 6,
        test_months: int = 6,
        embargo_days: int = 20,
    ):
        self.train_end = _normalize_train_end(train_end)
        self.val_months = int(val_months)
        self.test_months = int(test_months)
        self.embargo_days = int(embargo_days)
        if self.val_months <= 0 or self.test_months <= 0:
            raise ValueError("val_months and test_months must be positive")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be non-negative")

    def split(self, labeled_df: pd.DataFrame) -> list[dict[str, Any]]:
        if labeled_df is None or len(labeled_df) == 0:
            return []
        if "date" not in labeled_df.columns:
            raise ValueError("labeled_df must contain a 'date' column")

        dates = sorted(pd.to_datetime(labeled_df["date"]).unique())
        if not dates:
            return []

        frame = labeled_df.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        train_end = (
            _resolve_auto_single_train_end(
                frame=frame,
                dates=dates,
                val_months=self.val_months,
                test_months=self.test_months,
                embargo_days=self.embargo_days,
            )
            if _is_auto_train_end(self.train_end)
            else self.train_end
        )
        if train_end is None:
            return []

        fold = _build_fold(
            frame=frame,
            dates=dates,
            fold_id=0,
            train_end=train_end,
            val_months=self.val_months,
            test_months=self.test_months,
            embargo_days=self.embargo_days,
        )
        return [fold] if fold is not None else []


class WalkForwardSplitter:
    """滚动 walk-forward 多 fold 切分器。

    fold schema 与 ``SingleSplitSplitter`` 完全一致。每个 fold 的 ``train_end``
    相对上一 fold 向后滚动 ``step_months`` 个自然月，直到凑够 ``n_folds``
    或数据用尽。val_months、test_months、embargo_days 语义与单次切分相同。
    """

    def __init__(
        self,
        train_end: str | pd.Timestamp,
        val_months: int = 1,
        test_months: int = 1,
        embargo_days: int = 20,
        n_folds: int = 6,
        step_months: int = 1,
    ):
        self.train_end = _normalize_train_end(train_end)
        self.val_months = int(val_months)
        self.test_months = int(test_months)
        self.embargo_days = int(embargo_days)
        self.n_folds = int(n_folds)
        self.step_months = int(step_months)
        if self.val_months <= 0 or self.test_months <= 0:
            raise ValueError("val_months and test_months must be positive")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be non-negative")
        if self.n_folds <= 0:
            raise ValueError("n_folds must be positive")
        if self.step_months <= 0:
            raise ValueError("step_months must be positive")

    def split(self, labeled_df: pd.DataFrame) -> list[dict[str, Any]]:
        if labeled_df is None or len(labeled_df) == 0:
            return []
        if "date" not in labeled_df.columns:
            raise ValueError("labeled_df must contain a 'date' column")

        dates = sorted(pd.to_datetime(labeled_df["date"]).unique())
        if not dates:
            return []

        frame = labeled_df.copy()
        frame["date"] = pd.to_datetime(frame["date"])

        train_end = (
            _resolve_auto_walk_forward_train_end(
                frame=frame,
                dates=dates,
                val_months=self.val_months,
                test_months=self.test_months,
                embargo_days=self.embargo_days,
                n_folds=self.n_folds,
                step_months=self.step_months,
            )
            if _is_auto_train_end(self.train_end)
            else self.train_end
        )
        if train_end is None:
            return []
        return _build_walk_forward_folds(
            frame=frame,
            dates=dates,
            train_end=train_end,
            val_months=self.val_months,
            test_months=self.test_months,
            embargo_days=self.embargo_days,
            n_folds=self.n_folds,
            step_months=self.step_months,
        )


def _normalize_train_end(train_end: str | pd.Timestamp) -> str | pd.Timestamp:
    if isinstance(train_end, str) and train_end.strip().lower() == "auto":
        return "auto"
    return pd.Timestamp(train_end)


def _is_auto_train_end(train_end: str | pd.Timestamp) -> bool:
    return isinstance(train_end, str) and train_end == "auto"


def _build_walk_forward_folds(
    *,
    frame: pd.DataFrame,
    dates: list,
    train_end: pd.Timestamp,
    val_months: int,
    test_months: int,
    embargo_days: int,
    n_folds: int,
    step_months: int,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for fold_index in range(n_folds):
        fold_train_end = train_end + relativedelta(months=step_months * fold_index)
        fold = _build_fold(
            frame=frame,
            dates=dates,
            fold_id=fold_index,
            train_end=fold_train_end,
            val_months=val_months,
            test_months=test_months,
            embargo_days=embargo_days,
        )
        if fold is None:
            break
        folds.append(fold)
    return folds


def _resolve_auto_single_train_end(
    *,
    frame: pd.DataFrame,
    dates: list,
    val_months: int,
    test_months: int,
    embargo_days: int,
) -> pd.Timestamp | None:
    label_end = _last_complete_label_date(frame, dates)
    if label_end is None:
        return None
    for candidate in reversed([pd.Timestamp(date) for date in dates if pd.Timestamp(date) < label_end]):
        fold = _build_fold(
            frame=frame,
            dates=dates,
            fold_id=0,
            train_end=candidate,
            val_months=val_months,
            test_months=test_months,
            embargo_days=embargo_days,
        )
        if fold is not None and pd.Timestamp(fold["test_end"]) <= label_end:
            return candidate
    return None


def _resolve_auto_walk_forward_train_end(
    *,
    frame: pd.DataFrame,
    dates: list,
    val_months: int,
    test_months: int,
    embargo_days: int,
    n_folds: int,
    step_months: int,
) -> pd.Timestamp | None:
    label_end = _last_complete_label_date(frame, dates)
    if label_end is None:
        return None
    candidate_dates = [pd.Timestamp(date) for date in dates if pd.Timestamp(date) < label_end]
    for candidate in reversed(candidate_dates):
        folds = _build_walk_forward_folds(
            frame=frame,
            dates=dates,
            train_end=candidate,
            val_months=val_months,
            test_months=test_months,
            embargo_days=embargo_days,
            n_folds=n_folds,
            step_months=step_months,
        )
        if len(folds) != n_folds:
            continue
        if all(pd.Timestamp(fold["test_end"]) <= label_end for fold in folds):
            return candidate
    return None


def _last_complete_label_date(frame: pd.DataFrame, dates: list) -> pd.Timestamp | None:
    label_columns = [
        column
        for column in frame.columns
        if column.startswith("label_relative_net_return_")
        or column.startswith("label_return_")
        or column.startswith("label_net_return_")
        or column.startswith("label_volatility_")
    ]
    if not label_columns:
        return pd.Timestamp(dates[-1])

    complete_dates: list[pd.Timestamp] = []
    for date, group in frame.groupby("date", sort=True):
        if all(pd.to_numeric(group[column], errors="coerce").notna().any() for column in label_columns):
            complete_dates.append(pd.Timestamp(date))
    if not complete_dates:
        return None
    return max(complete_dates)


def _build_fold(
    *,
    frame: pd.DataFrame,
    dates: list,
    fold_id: int,
    train_end: pd.Timestamp,
    val_months: int,
    test_months: int,
    embargo_days: int,
) -> dict[str, Any] | None:
    min_date = pd.Timestamp(dates[0])
    max_date = pd.Timestamp(dates[-1])

    train_start = min_date
    if train_end <= train_start or train_end >= max_date:
        return None

    val_start = _shift_by_trading_days(dates, train_end, embargo_days + 1)
    if val_start is None:
        return None
    val_end = val_start + relativedelta(months=val_months) - relativedelta(days=1)
    if val_end > max_date:
        val_end = max_date

    test_start = _shift_by_trading_days(dates, val_end, embargo_days + 1)
    if test_start is None:
        return None
    test_end = test_start + relativedelta(months=test_months) - relativedelta(days=1)
    if test_end > max_date:
        test_end = max_date

    train_df = frame[(frame["date"] >= train_start) & (frame["date"] <= train_end)].reset_index(drop=True)
    val_df = frame[(frame["date"] >= val_start) & (frame["date"] <= val_end)].reset_index(drop=True)
    test_df = frame[(frame["date"] >= test_start) & (frame["date"] <= test_end)].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        return None

    return {
        "fold_id": fold_id,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
    }


def _shift_by_trading_days(
    dates: list,
    anchor: pd.Timestamp,
    shift: int,
) -> pd.Timestamp | None:
    """返回 anchor 之后的第 shift 个交易日（1-indexed）。超出数据返回 None。"""
    anchor_ts = pd.Timestamp(anchor)
    later_dates = [pd.Timestamp(d) for d in dates if pd.Timestamp(d) > anchor_ts]
    if len(later_dates) <= shift - 1:
        return None
    return later_dates[shift - 1]
