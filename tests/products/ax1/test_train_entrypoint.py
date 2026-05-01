# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _profile_config() -> dict:
    return {
        "universe": {
            "layers": {
                "core": {
                    "asset_type": "etf",
                    "exposure_group": "broad_beta",
                    "include": ["510300.XSHG"],
                },
                "industry": {
                    "asset_type": "etf",
                    "exposure_group": "sector",
                    "include": ["512800.XSHG"],
                },
                "style": {
                    "asset_type": "etf",
                    "exposure_group": "style_factor",
                    "include": ["510880.XSHG"],
                },
                "stock_satellite": {
                    "asset_type": "stock",
                    "exposure_group": "stock_alpha",
                    "enabled": False,
                    "include": ["000001.XSHE"],
                },
            }
        }
    }


class FakeAX1DataFacade:
    def __init__(self) -> None:
        self.daily_bar_calls = []

    def get_daily_bars(self, order_book_ids, start_date, end_date, fields=None, adjust_type="pre"):
        self.daily_bar_calls.append(
            {
                "order_book_ids": list(order_book_ids),
                "start_date": str(start_date),
                "end_date": str(end_date),
                "fields": list(fields or []),
                "adjust_type": adjust_type,
            }
        )
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        rows = []
        for asset_idx, order_book_id in enumerate(order_book_ids):
            for day_idx, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "order_book_id": order_book_id,
                        "open": 10.0 + asset_idx + day_idx,
                        "high": 10.5 + asset_idx + day_idx,
                        "low": 9.5 + asset_idx + day_idx,
                        "close": 10.2 + asset_idx + day_idx,
                        "volume": 1_000_000 + asset_idx * 1000,
                        "total_turnover": 10_000_000 + asset_idx * 1000,
                    }
                )
        return pd.DataFrame(rows).set_index("date")

    def all_instruments(self, type=None, date=None):
        return pd.DataFrame(
            {
                "order_book_id": ["510300.XSHG", "512800.XSHG", "510880.XSHG"],
                "listed_date": pd.to_datetime(["2012-01-01", "2013-01-01", "2014-01-01"]),
            }
        )

    def is_st_stock(self, order_book_ids, start_date, end_date):
        return pd.DataFrame()

    def is_suspended(self, order_book_ids, start_date, end_date):
        return pd.DataFrame()


def test_ax1_training_data_builder_uses_data_facade_and_attaches_training_contract():
    from skyeye.products.ax1.data_builder import AX1TrainingDataBuilder, AX1TrainingDataRequest

    facade = FakeAX1DataFacade()
    request = AX1TrainingDataRequest(
        profile_config=_profile_config(),
        start_date="2024-01-02",
        end_date="2024-01-05",
    )

    frame = AX1TrainingDataBuilder(data_facade=facade).build(request)

    assert facade.daily_bar_calls == [
        {
            "order_book_ids": ["510300.XSHG", "512800.XSHG", "510880.XSHG"],
            "start_date": "2024-01-02",
            "end_date": "2024-01-05",
            "fields": ["open", "high", "low", "close", "volume", "total_turnover"],
            "adjust_type": "pre",
        }
    ]
    assert set(frame["order_book_id"]) == {"510300.XSHG", "512800.XSHG", "510880.XSHG"}
    required = {
        "date",
        "order_book_id",
        "close",
        "adjusted_close",
        "volume",
        "asset_type",
        "universe_layer",
        "exposure_group",
        "industry",
        "listed_date",
        "is_st",
        "is_suspended",
        "price_adjustment_status",
    }
    assert required.issubset(frame.columns)
    assert frame["asset_type"].eq("etf").all()
    assert frame.set_index("order_book_id").loc["510300.XSHG", "universe_layer"].iloc[0] == "core"
    assert frame.set_index("order_book_id").loc["512800.XSHG", "exposure_group"].iloc[0] == "sector"
    assert frame.set_index("order_book_id").loc["510300.XSHG", "industry"].iloc[0] == "broad"
    assert frame.set_index("order_book_id").loc["512800.XSHG", "industry"].iloc[0] == "bank"
    assert frame.set_index("order_book_id").loc["510880.XSHG", "industry"].iloc[0] == "dividend"
    assert frame["is_st"].eq(False).all()
    assert frame["is_suspended"].eq(False).all()
    assert frame["price_adjustment_status"].eq("pre_adjusted_via_data_facade").all()
    pd.testing.assert_series_equal(
        frame["adjusted_close"].reset_index(drop=True),
        frame["close"].reset_index(drop=True),
        check_names=False,
    )


def test_ax1_universe_metadata_preserves_and_reapplies_industry_labels():
    from skyeye.products.ax1.run_experiment import _attach_universe_metadata
    from skyeye.products.ax1.universe import DynamicUniverseBuilder

    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "order_book_id": ["510300.XSHG", "512800.XSHG", "510300.XSHG", "512800.XSHG"],
            "asset_type": ["etf", "etf", "etf", "etf"],
            "universe_layer": ["core", "industry", "core", "industry"],
            "industry": ["broad", "bank", "broad", "bank"],
            "close": [10.0, 8.0, 10.1, 8.1],
            "volume": [1_000_000, 2_000_000, 1_100_000, 2_100_000],
        }
    )

    metadata = DynamicUniverseBuilder().build_with_metadata(
        raw,
        as_of_date="2024-01-03",
        config=_profile_config()["universe"],
        data_provider=None,
    )

    assert set(metadata.columns) >= {"order_book_id", "asset_type", "universe_layer", "industry"}
    assert metadata.set_index("order_book_id").loc["510300.XSHG", "industry"] == "broad"
    assert metadata.set_index("order_book_id").loc["512800.XSHG", "industry"] == "bank"

    stripped = raw.drop(columns=["industry"])
    restored = _attach_universe_metadata(stripped, metadata)
    restored = restored.drop_duplicates("order_book_id", keep="last").set_index("order_book_id")
    assert restored.loc["510300.XSHG", "industry"] == "broad"
    assert restored.loc["512800.XSHG", "industry"] == "bank"


def test_ax1_train_cli_builds_data_with_same_facade_and_writes_run_tag_dir(monkeypatch, tmp_path, capsys):
    import skyeye.products.ax1.train as train_module

    created_facades = []
    captured = {}

    class FakeFacadeFactory(FakeAX1DataFacade):
        def __init__(self):
            super().__init__()
            created_facades.append(self)

    def fake_load_profile(path):
        captured["profile_path"] = Path(path)
        return _profile_config()

    def fake_run_experiment(**kwargs):
        captured["run_experiment"] = kwargs
        return {"output_dir": str(kwargs["output_dir"])}

    monkeypatch.setattr(train_module, "DataFacade", FakeFacadeFactory)
    monkeypatch.setattr(train_module, "load_profile", fake_load_profile)
    monkeypatch.setattr(train_module, "run_experiment", fake_run_experiment)

    rc = train_module.main(
        [
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-05",
            "--profile",
            "personal_etf_core",
            "--run-tag",
            "ax1_demo",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert len(created_facades) == 1
    kwargs = captured["run_experiment"]
    assert kwargs["data_provider"] is created_facades[0]
    assert kwargs["raw_df"] is not None
    assert "raw_csv" not in kwargs or kwargs["raw_csv"] is None
    assert Path(kwargs["output_dir"]) == tmp_path / "ax1_demo"
    assert kwargs["experiment_name"] == "ax1_demo"
    assert captured["profile_path"].name == "personal_etf_core.yaml"
    assert str(tmp_path / "ax1_demo") in capsys.readouterr().out


def test_ax1_train_parser_exposes_simple_training_args():
    from skyeye.products.ax1.train import build_parser

    parser = build_parser()

    parsed = parser.parse_args(
        [
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-05",
            "--profile",
            "personal_etf_core",
            "--run-tag",
            "ax1_demo",
        ]
    )

    assert parsed.start_date == "2024-01-02"
    assert parsed.end_date == "2024-01-05"
    assert parsed.profile == "personal_etf_core"
    assert parsed.run_tag == "ax1_demo"
