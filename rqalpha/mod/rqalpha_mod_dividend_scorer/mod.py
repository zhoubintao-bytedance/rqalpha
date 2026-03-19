# -*- coding: utf-8 -*-

from collections import deque

from rqalpha.api import export_as_api
from rqalpha.core.events import EVENT
from rqalpha.environment import Environment
from rqalpha.interface import AbstractMod
from rqalpha.utils.logger import user_system_log

from rqalpha.dividend_scorer.main import DividendScorer


@export_as_api
def get_dividend_score(lag=0):
    env = Environment.get_instance()
    if lag < 0:
        raise ValueError("lag must be >= 0")
    history = getattr(env.global_vars, "dividend_score_history", None)
    if history:
        history_list = list(history)
        if lag < len(history_list):
            return history_list[-1 - lag]
        return None
    return getattr(env.global_vars, "dividend_score", None)


class DividendScorerMod(AbstractMod):
    def __init__(self):
        self._env = None
        self._scorer = None

    def start_up(self, env, mod_config):
        self._env = env
        prior_blend = getattr(mod_config, "prior_blend", None)
        self._scorer = DividendScorer(
            db_path=getattr(mod_config, "db_path", None),
            bundle_path=getattr(mod_config, "bundle_path", None),
            prior_blend=1.0 if prior_blend is None else prior_blend,
            dynamic_diagnostic=getattr(mod_config, "dynamic_diagnostic", False),
        )
        env.event_bus.add_listener(EVENT.POST_SYSTEM_INIT, self._bootstrap)
        env.event_bus.add_listener(EVENT.POST_USER_INIT, self._bind_context)
        env.event_bus.add_listener(EVENT.BEFORE_TRADING, self._on_before_trading)

    def tear_down(self, success, exception=None):
        if self._env is not None:
            self._reset_score_store()

    def _bootstrap(self, event):
        self._scorer.data_fetcher.data_proxy = self._env.data_proxy
        self._scorer.precompute(env=self._env)
        self._reset_score_store()

    def _bind_context(self, event):
        self._store_score(getattr(self._env.global_vars, "dividend_score", None))

    def _on_before_trading(self, event):
        score_date = self._env.data_proxy.get_previous_trading_date(self._env.trading_dt).date()
        try:
            score_result = dict(self._scorer.score(score_date))
            score_result["trade_date"] = self._env.trading_dt.date().strftime("%Y-%m-%d")
        except Exception as exc:
            user_system_log.error("dividend scorer failed on {}: {}".format(score_date, exc))
            score_result = {
                "date": score_date.strftime("%Y-%m-%d"),
                "trade_date": self._env.trading_dt.date().strftime("%Y-%m-%d"),
                "error": str(exc),
            }
        self._store_score(score_result)

    def _reset_score_store(self):
        self._env.global_vars.dividend_score = None
        self._env.global_vars.dividend_score_history = deque(maxlen=32)
        if getattr(self._env, "user_strategy", None) is not None:
            self._env.user_strategy.user_context.dividend_score = None
            self._env.user_strategy.user_context.dividend_score_history = self._env.global_vars.dividend_score_history

    def _store_score(self, score_result):
        history = getattr(self._env.global_vars, "dividend_score_history", None)
        if history is None:
            history = deque(maxlen=32)
            self._env.global_vars.dividend_score_history = history

        score_date = score_result.get("date") if isinstance(score_result, dict) else None
        if score_date is not None:
            last_score = history[-1] if history else None
            if not last_score or last_score.get("date") != score_date:
                history.append(score_result)

        self._env.global_vars.dividend_score = score_result
        if getattr(self._env, "user_strategy", None) is not None:
            self._env.user_strategy.user_context.dividend_score = score_result
            self._env.user_strategy.user_context.dividend_score_history = history
