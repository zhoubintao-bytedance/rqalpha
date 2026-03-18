# -*- coding: utf-8 -*-

from rqalpha.dividend_scorer.config import CACHE_DB_PATH

__config__ = {
    "enabled": False,
    "lib": "rqalpha.mod.rqalpha_mod_dividend_scorer",
    "priority": 60,
    "db_path": CACHE_DB_PATH,
    "bundle_path": None,
}


def load_mod():
    from rqalpha.mod.rqalpha_mod_dividend_scorer.mod import DividendScorerMod
    return DividendScorerMod()
