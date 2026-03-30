# -*- coding: utf-8 -*-

import pandas as pd


class PortfolioProxy(object):
    def __init__(self, buy_top_k=20, hold_top_k=50, rebalance_interval=20, holding_bonus=0.5):
        self.buy_top_k = int(buy_top_k)
        self.hold_top_k = int(hold_top_k)
        self.rebalance_interval = int(rebalance_interval)
        self.holding_bonus = float(holding_bonus)

    def build(self, prediction_df):
        if prediction_df is None or len(prediction_df) == 0:
            return pd.DataFrame(columns=["date", "order_book_id", "weight", "prediction"])
        rows = []
        previous_holdings = set()
        days_since_rebalance = self.rebalance_interval  # force rebalance on first day
        for date, day_df in prediction_df.groupby("date", sort=True):
            ranked = day_df.sort_values("prediction", ascending=False).reset_index(drop=True)
            should_rebalance = (
                days_since_rebalance >= self.rebalance_interval
                or not previous_holdings
            )
            if should_rebalance:
                # Apply holding bonus: existing holdings get a score boost
                if previous_holdings and self.holding_bonus > 0:
                    pred_std = ranked["prediction"].std()
                    bonus = self.holding_bonus * pred_std if pred_std > 0 else 0.0
                    adjusted = ranked.copy()
                    mask = adjusted["order_book_id"].isin(previous_holdings)
                    adjusted.loc[mask, "prediction"] = adjusted.loc[mask, "prediction"] + bonus
                    ranked = adjusted.sort_values("prediction", ascending=False).reset_index(drop=True)

                buy_candidates = ranked.head(self.buy_top_k)
                hold_candidates = ranked.head(self.hold_top_k)
                buy_set = set(buy_candidates["order_book_id"])
                hold_set = set(hold_candidates["order_book_id"])
                active = sorted(buy_set | (previous_holdings & hold_set))
                if not active:
                    previous_holdings = set()
                    days_since_rebalance += 1
                    continue
                active_df = ranked[ranked["order_book_id"].isin(active)].copy()
                active_df["weight"] = 1.0 / len(active_df)
                rows.extend(active_df[["date", "order_book_id", "weight", "prediction"]].to_dict("records"))
                previous_holdings = set(active_df["order_book_id"])
                days_since_rebalance = 0
            else:
                # Carry forward previous holdings with equal weight
                active = sorted(previous_holdings)
                if not active:
                    days_since_rebalance += 1
                    continue
                active_df = ranked[ranked["order_book_id"].isin(active)].copy()
                if active_df.empty:
                    days_since_rebalance += 1
                    continue
                active_df["weight"] = 1.0 / len(active_df)
                rows.extend(active_df[["date", "order_book_id", "weight", "prediction"]].to_dict("records"))
            days_since_rebalance += 1
        return pd.DataFrame(rows)
