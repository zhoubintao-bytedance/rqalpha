from types import SimpleNamespace

from skyeye.products.dividend_low_vol.mod.rqalpha_mod_dividend_scorer.mod import DividendScorerMod


class DummyEventBus:
    def add_listener(self, event, callback):
        return None


def test_dividend_scorer_mod_passes_auto_sync_to_prepare(monkeypatch):
    prepare_calls = []

    class FakeScorer:
        def prepare(self, env=None, auto_sync=None, sync_progress=None):
            prepare_calls.append(
                {
                    "env": env,
                    "auto_sync": auto_sync,
                    "sync_progress": sync_progress,
                }
            )

    monkeypatch.setattr(
        "skyeye.products.dividend_low_vol.mod.rqalpha_mod_dividend_scorer.mod.DividendScorer",
        lambda **kwargs: FakeScorer(),
    )

    env = SimpleNamespace(event_bus=DummyEventBus(), global_vars=SimpleNamespace())
    mod = DividendScorerMod()
    mod.start_up(env, SimpleNamespace(db_path=None, bundle_path=None, prior_blend=None, dynamic_diagnostic=True, auto_sync=False))
    mod._bootstrap(None)

    assert prepare_calls == [
        {
            "env": env,
            "auto_sync": False,
            "sync_progress": None,
        }
    ]
