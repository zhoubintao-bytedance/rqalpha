"""Profile-driven parameter loading for the history-aware strategy."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from rqalpha.environment import Environment


PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
DEFAULT_PROFILE = "baseline"


def _extra_strategy_profile():
    try:
        env = Environment.get_instance()
    except Exception:
        return None
    extra = getattr(getattr(env, "config", None), "extra", None)
    if extra is None:
        return None
    return getattr(extra, "strategy_profile", None)


def resolve_profile_name(explicit_name=None):
    if explicit_name:
        return explicit_name
    env_name = os.environ.get("SKYEYE_DIVIDEND_LOW_VOL_PROFILE")
    if env_name:
        return env_name
    extra_name = _extra_strategy_profile()
    if extra_name:
        return extra_name
    return DEFAULT_PROFILE


def load_profile(profile_name=None):
    resolved = resolve_profile_name(profile_name)
    profile_path = PROFILE_DIR / "{}.yaml".format(resolved)
    if not profile_path.exists():
        raise RuntimeError("missing strategy profile: {}".format(profile_path))
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = yaml.safe_load(handle) or {}
    profile.setdefault("profile", resolved)
    return profile


def configure_context(context, profile_name=None):
    profile = load_profile(profile_name)
    for key, value in profile.items():
        if key == "profile":
            continue
        setattr(context, key, value)

    # Runtime-only state is always reset on init.
    context.profile_name = profile["profile"]
    context.heat_cooldown_left = 0
    context.heat_cooldown_cap = None
    context.high_reentry_clear_count = 0
    context.last_rebalance_week = None
    context.precomputed_score_df = None
    context.dividend_scorer = None
    return profile
