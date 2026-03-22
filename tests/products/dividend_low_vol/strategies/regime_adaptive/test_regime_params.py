from skyeye.products.dividend_low_vol.strategies.regime_adaptive.params import load_profile
from skyeye.products.dividend_low_vol.strategies.regime_adaptive.params import resolve_profile_name


def test_load_profile_defaults_to_baseline(monkeypatch):
    monkeypatch.delenv("SKYEYE_DIVIDEND_LOW_VOL_REGIME_PROFILE", raising=False)

    profile = load_profile()

    assert profile["profile"] == "baseline"
    assert profile["etf"] == "512890.XSHG"
    assert profile["trend_long_window"] == 120


def test_resolve_profile_name_prefers_explicit_env_then_extra(monkeypatch):
    monkeypatch.setenv("SKYEYE_DIVIDEND_LOW_VOL_REGIME_PROFILE", "candidate")

    assert resolve_profile_name("baseline") == "baseline"
    assert resolve_profile_name() == "candidate"

    monkeypatch.delenv("SKYEYE_DIVIDEND_LOW_VOL_REGIME_PROFILE", raising=False)
    monkeypatch.setattr(
        "skyeye.products.dividend_low_vol.strategies.regime_adaptive.params._extra_strategy_profile",
        lambda: "baseline",
    )

    assert resolve_profile_name() == "baseline"
