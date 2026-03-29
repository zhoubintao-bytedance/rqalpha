# -*- coding: utf-8 -*-

import pandas as pd


class PortfolioProxy(object):
    def __init__(self, buy_top_k=20, hold_top_k=30):
        self.buy_top_k = int(buy_top_k)
        self.hold_top_k = int(hold_top_k)

    def build(self, prediction_df):
        if prediction_df is None or len(prediction_df) == 0:
            return pd.DataFrame(columns=["date", "order_book_id", "weight", "prediction"])
        rows = []
        previous_holdings = set()
        for date, day_df in prediction_df.groupby("date", sort=True):
            ranked = day_df.sort_values("prediction", ascending=False).reset_index(drop=True)
            buy_candidates = ranked.head(self.buy_top_k)
            hold_candidates = ranked.head(self.hold_top_k)
            buy_set = set(buy_candidates["order_book_id"])
            hold_set = set(hold_candidates["order_book_id"])
            active = sorted(buy_set | (previous_holdings & hold_set))
            if not active:
                previous_holdings = set()
                continue
            active_df = ranked[ranked["order_book_id"].isin(active)].copy()
            active_df["weight"] = 1.0 / len(active_df)
            rows.extend(active_df[["date", "order_book_id", "weight", "prediction"]].to_dict("records"))
            previous_holdings = set(active_df["order_book_id"])
        return pd.DataFrame(rows)
