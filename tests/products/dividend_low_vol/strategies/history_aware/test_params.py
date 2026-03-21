from skyeye.products.dividend_low_vol.strategies.history_aware.params import load_profile
from skyeye.products.dividend_low_vol.strategies.history_aware.params import resolve_profile_name


def test_load_profile_defaults_to_baseline(monkeypatch):
    monkeypatch.delenv("SKYEYE_DIVIDEND_LOW_VOL_PROFILE", raising=False)

    profile = load_profile()

    assert profile["profile"] == "baseline"
    assert profile["etf"] == "512890.XSHG"
    assert profile["slow_window"] == 15


def test_resolve_profile_name_prefers_explicit_env_then_extra(monkeypatch):
    monkeypatch.setenv("SKYEYE_DIVIDEND_LOW_VOL_PROFILE", "aggressive")

    assert resolve_profile_name("conservative") == "conservative"
    assert resolve_profile_name() == "aggressive"

    monkeypatch.delenv("SKYEYE_DIVIDEND_LOW_VOL_PROFILE", raising=False)
    monkeypatch.setattr(
        "skyeye.products.dividend_low_vol.strategies.history_aware.params._extra_strategy_profile",
        lambda: "conservative",
    )

    assert resolve_profile_name() == "conservative"
