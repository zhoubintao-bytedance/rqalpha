"""Profile-driven parameter loading for the TX1 rolling-score strategy."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from rqalpha.environment import Environment


PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
DEFAULT_PROFILE = "smooth"

FROZEN_FIELDS = frozenset({"benchmark", "artifact_line_id"})


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
    env_name = os.environ.get("SKYEYE_TX1_PROFILE")
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
