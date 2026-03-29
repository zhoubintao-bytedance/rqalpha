# -*- coding: utf-8 -*-

import pandas as pd
from dateutil.relativedelta import relativedelta


class WalkForwardSplitter(object):
    def __init__(self, train_years=3, val_months=6, test_months=6, embargo_days=20):
        self.train_years = int(train_years)
        self.val_months = int(val_months)
        self.test_months = int(test_months)
        self.embargo_days = int(embargo_days)

    def split(self, labeled_df):
        if labeled_df is None or len(labeled_df) == 0:
            return []
        dates = sorted(pd.to_datetime(labeled_df["date"]).unique())
        if not dates:
            return []

        min_date = pd.Timestamp(dates[0])
        max_date = pd.Timestamp(dates[-1])
        train_start = min_date
        train_end = train_start + relativedelta(years=self.train_years) - relativedelta(days=1)
        folds = []

        while True:
            if train_end > max_date:
                break
            val_start = self._shift_by_trading_days(dates, train_end, self.embargo_days + 1)
            if val_start is None:
                break
            val_end = val_start + relativedelta(months=self.val_months) - relativedelta(days=1)
            test_start = self._shift_by_trading_days(dates, val_end, self.embargo_days + 1)
            if test_start is None:
                break
            test_end = test_start + relativedelta(months=self.test_months) - relativedelta(days=1)
            if test_end > max_date:
                break

            train_df = labeled_df[(labeled_df["date"] >= train_start) & (labeled_df["date"] <= train_end)].copy()
            val_df = labeled_df[(labeled_df["date"] >= val_start) & (labeled_df["date"] <= val_end)].copy()
            test_df = labeled_df[(labeled_df["date"] >= test_start) & (labeled_df["date"] <= test_end)].copy()
            if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
                break
            folds.append(
                {
                    "train_df": train_df.reset_index(drop=True),
                    "val_df": val_df.reset_index(drop=True),
                    "test_df": test_df.reset_index(drop=True),
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )
            train_start = train_start + relativedelta(months=self.test_months)
            train_end = train_start + relativedelta(years=self.train_years) - relativedelta(days=1)
            if train_end > max_date:
                break
        return folds

    @staticmethod
    def _shift_by_trading_days(dates, anchor, shift):
        anchor_ts = pd.Timestamp(anchor)
        later_dates = [pd.Timestamp(d) for d in dates if pd.Timestamp(d) > anchor_ts]
        if len(later_dates) <= shift - 1:
            return None
        return later_dates[shift - 1]
