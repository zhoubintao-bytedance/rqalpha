# -*- coding: utf-8 -*-

import pandas as pd


class PortfolioProxy(object):
    def __init__(self, buy_top_k=25, hold_top_k=45, rebalance_interval=20, holding_bonus=0.5):
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
            target_weights = self.build_target_weights(
                ranked,
                current_holdings=previous_holdings,
                should_rebalance=should_rebalance,
            )
            if should_rebalance:
                if not target_weights:
                    previous_holdings = set()
                    days_since_rebalance += 1
                    continue
                active_df = ranked[ranked["order_book_id"].isin(target_weights)].copy()
                active_df["weight"] = active_df["order_book_id"].map(target_weights)
                rows.extend(active_df[["date", "order_book_id", "weight", "prediction"]].to_dict("records"))
                previous_holdings = set(active_df["order_book_id"])
                days_since_rebalance = 0
            else:
                # 非调仓日直接沿用当前持仓的目标权重。
                if not target_weights:
                    days_since_rebalance += 1
                    continue
                active_df = ranked[ranked["order_book_id"].isin(target_weights)].copy()
                if active_df.empty:
                    days_since_rebalance += 1
                    continue
                active_df["weight"] = active_df["order_book_id"].map(target_weights)
                rows.extend(active_df[["date", "order_book_id", "weight", "prediction"]].to_dict("records"))
            days_since_rebalance += 1
        return pd.DataFrame(rows)

    def build_target_weights(self, prediction_df, current_holdings=None, should_rebalance=True):
        """基于单日打分、当前持仓和组合规则生成目标权重。"""
        if prediction_df is None or len(prediction_df) == 0:
            return {}

        ranked = prediction_df.sort_values("prediction", ascending=False).reset_index(drop=True)
        previous_holdings = set(current_holdings or [])
        if should_rebalance:
            ranked = self._apply_holding_bonus(ranked, previous_holdings)
            buy_candidates = ranked.head(self.buy_top_k)
            hold_candidates = ranked.head(self.hold_top_k)
            buy_set = set(buy_candidates["order_book_id"])
            hold_set = set(hold_candidates["order_book_id"])
            active = sorted(buy_set | (previous_holdings & hold_set))
        else:
            # 实盘非调仓日需要保持当前持仓权重，不能偷偷改成等权。
            preserved_weights = self._normalize_current_weight_map(current_holdings)
            if preserved_weights:
                return preserved_weights
            active = sorted(previous_holdings)

        if not active:
            return {}
        active_df = ranked[ranked["order_book_id"].isin(active)].copy()
        if active_df.empty:
            return {}
        weight = 1.0 / float(len(active_df))
        return {
            str(order_book_id): weight
            for order_book_id in active_df["order_book_id"]
        }

    @staticmethod
    def _normalize_current_weight_map(current_holdings):
        """把 dict 形式的当前持仓权重归一化后原样返回。"""
        if not isinstance(current_holdings, dict):
            return {}
        weight_map = {
            str(order_book_id): float(weight)
            for order_book_id, weight in current_holdings.items()
            if float(weight) > 0
        }
        total_weight = sum(weight_map.values())
        if total_weight <= 0:
            return {}
        return {
            order_book_id: weight / total_weight
            for order_book_id, weight in weight_map.items()
        }

    def _apply_holding_bonus(self, ranked, previous_holdings):
        """在调仓日给现有持仓加分，保持研究侧与实盘侧口径一致。"""
        if not previous_holdings or self.holding_bonus <= 0:
            return ranked
        pred_std = ranked["prediction"].std()
        bonus = self.holding_bonus * pred_std if pred_std > 0 else 0.0
        adjusted = ranked.copy()
        mask = adjusted["order_book_id"].isin(previous_holdings)
        adjusted.loc[mask, "prediction"] = adjusted.loc[mask, "prediction"] + bonus
        return adjusted.sort_values("prediction", ascending=False).reset_index(drop=True)
