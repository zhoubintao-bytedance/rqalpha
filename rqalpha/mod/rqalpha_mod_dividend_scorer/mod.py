# -*- coding: utf-8 -*-

from rqalpha.api import export_as_api
from rqalpha.core.events import EVENT
from rqalpha.environment import Environment
from rqalpha.interface import AbstractMod
from rqalpha.utils.logger import user_system_log

from rqalpha.dividend_scorer.main import DividendScorer


@export_as_api
def get_dividend_score():
    env = Environment.get_instance()
    return getattr(env.global_vars, "dividend_score", None)


class DividendScorerMod(AbstractMod):
    def __init__(self):
        self._env = None
        self._scorer = None

    def start_up(self, env, mod_config):
        self._env = env
        self._scorer = DividendScorer(
            db_path=getattr(mod_config, "db_path", None),
            bundle_path=getattr(mod_config, "bundle_path", None),
        )
        env.event_bus.add_listener(EVENT.POST_SYSTEM_INIT, self._bootstrap)
        env.event_bus.add_listener(EVENT.POST_USER_INIT, self._bind_context)
        env.event_bus.add_listener(EVENT.BAR, self._on_bar)

    def tear_down(self, success, exception=None):
        if self._env is not None:
            self._store_score(None)

    def _bootstrap(self, event):
        self._scorer.data_fetcher.data_proxy = self._env.data_proxy
        self._scorer.precompute(env=self._env)
        self._store_score(None)

    def _bind_context(self, event):
        self._store_score(getattr(self._env.global_vars, "dividend_score", None))

    def _on_bar(self, event):
        try:
            score_result = self._scorer.score(self._env.trading_dt.date())
        except Exception as exc:
            user_system_log.error("dividend scorer failed on {}: {}".format(self._env.trading_dt.date(), exc))
            score_result = {
                "date": self._env.trading_dt.date().strftime("%Y-%m-%d"),
                "error": str(exc),
            }
        self._store_score(score_result)

    def _store_score(self, score_result):
        self._env.global_vars.dividend_score = score_result
        if getattr(self._env, "user_strategy", None) is not None:
            self._env.user_strategy.user_context.dividend_score = score_result
