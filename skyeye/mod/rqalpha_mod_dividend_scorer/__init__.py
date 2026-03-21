# -*- coding: utf-8 -*-

from skyeye.dividend_scorer.config import CACHE_DB_PATH, WEIGHT_DYNAMIC_DIAGNOSTIC, WEIGHT_PRIOR_BLEND

__config__ = {
    "enabled": False,
    "lib": "skyeye.mod.rqalpha_mod_dividend_scorer",
    "priority": 60,
    "db_path": CACHE_DB_PATH,
    "bundle_path": None,
    "prior_blend": WEIGHT_PRIOR_BLEND,
    "dynamic_diagnostic": WEIGHT_DYNAMIC_DIAGNOSTIC,
}


def load_mod():
    from skyeye.mod.rqalpha_mod_dividend_scorer.mod import DividendScorerMod
    return DividendScorerMod()
