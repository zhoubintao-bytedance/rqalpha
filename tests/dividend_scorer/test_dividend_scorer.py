# -*- coding: utf-8 -*-

from collections import OrderedDict
import io
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
from requests.exceptions import ConnectionError as RequestsConnectionError

from rqalpha.dividend_scorer.config import DOMAIN_WEIGHTS
from rqalpha.dividend_scorer.config import VALUATION_FEATURES
from rqalpha.dividend_scorer.data_fetcher import DataFetcher
from rqalpha.dividend_scorer.feature_engine import FeatureEngine
from rqalpha.dividend_scorer.main import DividendScorer
from rqalpha.dividend_scorer.main import SyncProgressReporter
from rqalpha.dividend_scorer.main import format_score_report
from rqalpha.dividend_scorer.main import main as dividend_main
from rqalpha.dividend_scorer.score_synthesizer import ScoreSynthesizer
from rqalpha.dividend_scorer.weight_calculator import WeightCalculator
from rqalpha.utils import RqAttrDict


def make_history_df(rows=900):
    index = pd.bdate_range("2020-01-01", periods=rows)
    base = np.linspace(1.0, 1.6, rows)
    seasonal = 0.03 * np.sin(np.arange(rows) / 21.0)
    close = base + seasonal
    close_hfq = close * (1.0 + 0.001 * np.cos(np.arange(rows) / 11.0))
    volume = 1e7 + np.sin(np.arange(rows) / 7.0) * 1e6
    nav = close * (1.0 - 0.002)
    premium = close / nav - 1.0
    dividend = 0.03 + 0.01 * np.sin(np.arange(rows) / 50.0)
    bond = 0.02 + 0.002 * np.cos(np.arange(rows) / 90.0)
    pe = 8.0 + 1.5 * np.sin(np.arange(rows) / 60.0)
    return pd.DataFrame(
        {
            "etf_close": close,
            "etf_close_hfq": close_hfq,
            "etf_volume": volume,
            "etf_nav": nav,
            "pe_ttm": pe,
            "dividend_yield": dividend,
            "bond_10y": bond,
            "premium_rate": premium,
        },
        index=index,
    )


def test_feature_engine_precompute_inverts_dividend_features():
    history_df = make_history_df()
    engine = FeatureEngine()
    normalized = engine.precompute(history_df)
    last_date = history_df.index[-1]
    dividend_snapshot = engine.compute_single(last_date)

    assert not normalized.empty
    assert dividend_snapshot["dividend_yield_pct"]["percentile"] is not None
    assert np.isclose(
        dividend_snapshot["dividend_yield_pct"]["normalized"],
        1.0 - dividend_snapshot["dividend_yield_pct"]["percentile"],
    )
    assert np.isclose(
        dividend_snapshot["price_percentile"]["normalized"],
        dividend_snapshot["price_percentile"]["percentile"],
    )


def test_weight_calculator_falls_back_to_domain_weights_when_signal_is_invalid():
    index = pd.bdate_range("2020-01-01", periods=900)
    feature_matrix = pd.DataFrame(
        {
            "dividend_yield_pct": np.linspace(0.1, 0.9, len(index)),
            "yield_spread": np.linspace(0.1, 0.9, len(index)),
            "pe_percentile": np.linspace(0.1, 0.9, len(index)),
            "ma250_deviation": np.linspace(0.1, 0.9, len(index)),
            "price_percentile": np.linspace(0.1, 0.9, len(index)),
            "rsi_20": np.linspace(0.1, 0.9, len(index)),
            "premium_rate": np.linspace(0.1, 0.9, len(index)),
            "premium_rate_ma20": np.linspace(0.1, 0.9, len(index)),
        },
        index=index,
    )
    # Increasing prices make future returns positively correlated with expensive features.
    price_series = pd.Series(np.linspace(1.0, 2.5, len(index)), index=index)

    calculator = WeightCalculator()
    result = calculator.calculate_ic_ir_weights(feature_matrix, price_series)

    assert result["method"] == "domain_knowledge_fallback"
    assert np.isclose(sum(result["weights"].values()), 1.0)
    assert np.isclose(result["weights"]["dividend_yield_pct"], DOMAIN_WEIGHTS["dividend_yield_pct"])


def test_score_synthesizer_renormalizes_weights_and_applies_confidence_penalty():
    feature_snapshot = {
        "dividend_yield_pct": {
            "raw": 0.05,
            "percentile": 0.8,
            "normalized": 0.2,
            "inverted": True,
            "dimension": "dividend",
            "sample_size": 200,
            "under_sampled": False,
        },
        "yield_spread": {
            "raw": 0.02,
            "percentile": 0.7,
            "normalized": 0.3,
            "inverted": True,
            "dimension": "dividend",
            "sample_size": 200,
            "under_sampled": False,
        },
        "pe_percentile": {
            "raw": 8.5,
            "percentile": 0.4,
            "normalized": 0.4,
            "inverted": False,
            "dimension": "pe",
            "sample_size": 200,
            "under_sampled": False,
        },
        "ma250_deviation": {
            "raw": -0.03,
            "percentile": 0.35,
            "normalized": 0.35,
            "inverted": False,
            "dimension": "price",
            "sample_size": 200,
            "under_sampled": False,
        },
        "price_percentile": {
            "raw": 1.1,
            "percentile": None,
            "normalized": None,
            "inverted": False,
            "dimension": "price",
            "sample_size": 0,
            "under_sampled": True,
        },
        "rsi_20": {
            "raw": 44,
            "percentile": 0.45,
            "normalized": 0.45,
            "inverted": False,
            "dimension": "price",
            "sample_size": 200,
            "under_sampled": False,
        },
        "premium_rate": {
            "raw": -0.001,
            "percentile": 0.3,
            "normalized": 0.3,
            "inverted": False,
            "dimension": "premium",
            "sample_size": 200,
            "under_sampled": False,
        },
        "premium_rate_ma20": {
            "raw": -0.0008,
            "percentile": 0.32,
            "normalized": 0.32,
            "inverted": False,
            "dimension": "premium",
            "sample_size": 200,
            "under_sampled": False,
        },
        "volatility_percentile": {
            "raw": 0.5,
            "percentile": 0.95,
            "normalized": 0.95,
            "dimension": "confidence",
            "sample_size": 200,
            "under_sampled": False,
        },
        "volume_ratio": {
            "raw": 0.9,
            "percentile": 0.5,
            "normalized": 0.5,
            "dimension": "confidence",
            "sample_size": 200,
            "under_sampled": False,
        },
    }
    weight_result = {
        "weights": {
            "dividend_yield_pct": 0.2,
            "yield_spread": 0.2,
            "pe_percentile": 0.2,
            "ma250_deviation": 0.2,
            "price_percentile": 0.1,
            "rsi_20": 0.05,
            "premium_rate": 0.025,
            "premium_rate_ma20": 0.025,
        },
        "method": "ic_ir",
        "fallback_reason": None,
        "test_ic_avg": -0.1,
        "test_ic_ir_avg": 0.5,
    }

    synth = ScoreSynthesizer()
    result = synth.synthesize(feature_snapshot, weight_result)

    assert 0 < result["total_score"] < 10
    assert result["confidence"] == "lowered"
    assert np.isclose(sum(item["weight"] for item in result["features"].values()), 1.0)
    assert result["features"]["price_percentile"]["weight"] == 0.0


def test_format_score_report_displays_chinese_feature_names():
    report = format_score_report(
        {
            "etf": "512890",
            "date": "2026-03-19",
            "total_score": 6.32,
            "score_percentile": 0.709,
            "score_percentile_window": 504,
            "score_percentile_sample_size": 504,
            "confidence": "lowered",
            "model_meta": {"method": "fixed_domain_prior"},
            "features": OrderedDict(
                [
                    (
                        "dividend_yield_pct",
                        {
                            "raw": 0.0321,
                            "percentile": 0.25,
                            "normalized": 0.75,
                            "sub_score": 7.5,
                            "weight": 0.2,
                        },
                    ),
                    (
                        "premium_rate_ma20",
                        {
                            "raw": -0.0012,
                            "percentile": 0.40,
                            "normalized": 0.40,
                            "sub_score": 4.0,
                            "weight": 0.1,
                        },
                    ),
                ]
            ),
            "confidence_modifiers": OrderedDict(),
            "warnings": [],
        }
    )

    assert "红利低波打分器结论" in report
    assert "ETF" in report
    assert "日期" in report
    assert "综合评分" in report
    assert "滚动分位" in report
    assert "置信度" in report
    assert "权重方案" in report
    assert "msg" in report
    assert "当前评分 6.32/10" in report
    assert "处于历史中高位" in report
    assert "│ 指标" in report
    assert "│ 指标英文" in report
    assert "│ 原始值" in report
    assert "│ 分位" in report
    assert "│ 归一值" in report
    assert "│ 子分" in report
    assert "│ 权重" in report
    assert "股息率" in report
    assert "dividend_yield_pct" in report
    assert "20日平均溢价率" in report
    assert "premium_rate_ma20" in report


def test_sync_banner_prints_logo_before_title(tmp_path, monkeypatch):
    class FakeStream(io.StringIO):
        def isatty(self):
            return False

    stream = FakeStream()
    reporter = SyncProgressReporter(stream=stream)
    reporter.banner(
        title="红利低波打分器缓存同步",
        start_date="2020-01-01",
        end_date="2026-03-20",
        db_path="/tmp/cache.db",
    )

    output = stream.getvalue()
    assert output.startswith("红利低波打分器缓存同步\n")
    assert "红利低波打分器缓存同步" in output


def test_main_sync_only_implies_sync(monkeypatch):
    calls = []

    class FakeScorer(object):
        def __init__(self, db_path=None, bundle_path=None):
            self.data_fetcher = SimpleNamespace(db_path="/tmp/cache.db")

        def sync_all(self, start_date, end_date, progress=None):
            calls.append(("sync_all", start_date, end_date, progress is not None))

        def precompute(self, start_date=None, end_date=None):
            calls.append(("precompute", start_date, end_date))

        def score_latest(self, date=None):
            calls.append(("score_latest", date))
            return {}

    class FakeStream(io.StringIO):
        def isatty(self):
            return False

    stdout = io.StringIO()
    stderr = FakeStream()
    monkeypatch.setattr("rqalpha.dividend_scorer.main.DividendScorer", FakeScorer)
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    dividend_main(["--sync-only"])

    assert calls
    assert calls[0][0] == "sync_all"
    assert calls[0][1] == "2020-01-01"
    assert calls[0][3] is True
    assert all(call[0] != "precompute" for call in calls)
    assert "sync success ✅" in stdout.getvalue()


def test_main_scoring_prints_logo_and_runs_cli_spinner(monkeypatch):
    calls = []

    class FakeScorer(object):
        def __init__(self, db_path=None, bundle_path=None):
            self.data_fetcher = SimpleNamespace(db_path="/tmp/cache.db")

        def precompute(self, start_date=None, end_date=None):
            calls.append(("precompute", start_date, end_date))

        def score_latest(self, date=None):
            calls.append(("score_latest", date))
            return {
                "etf": "512890",
                "date": "2026-03-19",
                "total_score": 6.32,
                "score_percentile": 0.709,
                "score_percentile_window": 504,
                "score_percentile_sample_size": 504,
                "buy_percentile_threshold": 0.2,
                "sell_percentile_threshold": 0.8,
                "confidence": "lowered",
                "model_meta": {"method": "fixed_domain_prior"},
                "features": OrderedDict(),
                "confidence_modifiers": OrderedDict(),
                "warnings": [],
            }

    class FakeStream(io.StringIO):
        def isatty(self):
            return True

    class FakeIndicator(object):
        def __init__(self, label, stream=None, enabled=True):
            calls.append(("indicator_init", label, enabled))

        def start(self):
            calls.append(("indicator_start",))
            return self

        def update(self, label):
            calls.append(("indicator_update", label))

        def stop(self):
            calls.append(("indicator_stop",))

    def fake_render_logo(stream, use_color):
        calls.append(("logo", use_color))

    stdout = io.StringIO()
    stderr = FakeStream()
    monkeypatch.setattr("rqalpha.dividend_scorer.main.DividendScorer", FakeScorer)
    monkeypatch.setattr("rqalpha.dividend_scorer.main.CliActivityIndicator", FakeIndicator)
    monkeypatch.setattr("rqalpha.dividend_scorer.main._render_gradient_logo", fake_render_logo)
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    dividend_main([])

    assert ("logo", True) in calls
    assert ("indicator_init", "正在预计算估值特征", True) in calls
    assert ("indicator_start",) in calls
    assert ("indicator_update", "正在生成评分报告") in calls
    assert ("indicator_stop",) in calls
    assert "红利低波打分器结论" in stdout.getvalue()


def test_data_fetcher_load_history_aggregates_dividend_yield(tmp_path):
    db_path = tmp_path / "cache.db"
    fetcher = DataFetcher(db_path=str(db_path))

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            "INSERT INTO etf_daily (date, etf_code, close, close_hfq, volume, amount, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2024-01-02", "512890", 1.0, 1.0, 1000.0, 1000.0, "2024-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO etf_nav (date, etf_code, nav, acc_nav, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("2024-01-02", "512890", 0.99, 0.99, "2024-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO index_daily (date, index_code, pe_ttm, updated_at) VALUES (?, ?, ?, ?)",
            ("2024-01-02", "H30269", 8.2, "2024-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO bond_yield (date, china_10y, updated_at) VALUES (?, ?, ?)",
            ("2024-01-02", 2.5, "2024-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO index_weight (index_code, stock_code, stock_name, weight, snapshot_date, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("H30269", "000001", "A", 60.0, "2024-01-02", "2024-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO index_weight (index_code, stock_code, stock_name, weight, snapshot_date, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("H30269", "000002", "B", 40.0, "2024-01-02", "2024-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO stock_indicator (date, stock_code, dv_ttm, updated_at) VALUES (?, ?, ?, ?)",
            ("2024-01-02", "000001", 5.0, "2024-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO stock_indicator (date, stock_code, dv_ttm, updated_at) VALUES (?, ?, ?, ?)",
            ("2024-01-02", "000002", 3.0, "2024-01-02T00:00:00"),
        )
        conn.commit()

    history = fetcher.load_history("2024-01-02", "2024-01-02")

    assert len(history) == 1
    assert np.isclose(history.iloc[0]["dividend_yield"], 0.042)
    assert np.isclose(history.iloc[0]["bond_10y"], 0.025)
    assert np.isclose(history.iloc[0]["premium_rate"], 1.0 / 0.99 - 1.0)


def test_sync_stock_indicators_uses_nested_safe_transaction(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    fetcher = DataFetcher(db_path=str(db_path))
    monkeypatch.setattr("rqalpha.dividend_scorer.data_fetcher.time.sleep", lambda _: None)

    class FakeAk(object):
        @staticmethod
        def stock_a_indicator_lg(symbol):
            return pd.DataFrame(
                {
                    "trade_date": ["2024-01-02"],
                    "dv_ttm": [5.0],
                    "pe_ttm": [8.1],
                    "pb": [1.2],
                    "total_mv": [100000000.0],
                }
            )

    with fetcher._connect() as conn:
        conn.execute(
            "INSERT INTO data_source_meta (source_name, last_update_date, last_fetch_time, record_count) VALUES (?, ?, ?, ?)",
            ("dummy", "2024-01-01", "2024-01-01T00:00:00", 1),
        )
        fetcher._sync_stock_indicators(conn, FakeAk(), ["000001"], "2024-01-01", "2024-01-31")
        row = conn.execute(
            "SELECT dv_ttm, pe_ttm, pb, total_mv FROM stock_indicator WHERE date = ? AND stock_code = ?",
            ("2024-01-02", "000001"),
        ).fetchone()

    assert row is not None
    assert np.isclose(row[0], 5.0)
    assert np.isclose(row[1], 8.1)


def test_fetch_stock_indicator_supports_old_akshare_alias():
    class FakeAk(object):
        @staticmethod
        def stock_a_lg_indicator(symbol):
            return pd.DataFrame({"trade_date": ["2024-01-02"], "dv_ttm": [5.0]})

    fetcher = DataFetcher(db_path=":memory:")
    df = fetcher._fetch_stock_indicator(FakeAk(), "000001")
    assert not df.empty
    assert "dv_ttm" in df.columns


def test_fetch_stock_indicator_falls_back_to_value_and_dividend_history(tmp_path):
    fetcher = DataFetcher(db_path=str(tmp_path / "cache.db"))

    class FakeAk(object):
        @staticmethod
        def stock_value_em(symbol):
            return pd.DataFrame(
                {
                    "数据日期": ["2024-01-16", "2024-06-16", "2025-01-20"],
                    "总市值": [1000.0, 1000.0, 1000.0],
                    "PE(TTM)": [8.1, 8.2, 8.3],
                    "市净率": [1.2, 1.3, 1.4],
                }
            )

        @staticmethod
        def stock_fhps_detail_em(symbol):
            return pd.DataFrame(
                {
                    "除权除息日": ["2024-01-15", "2024-06-15"],
                    "现金分红-现金分红比例": [2.0, 1.0],
                    "总股本": [100.0, 100.0],
                    "方案进度": ["实施分配", "实施分配"],
                }
            )

    df = fetcher._fetch_stock_indicator(FakeAk(), "000001")

    assert list(df["trade_date"].dt.strftime("%Y-%m-%d")) == ["2024-01-16", "2024-06-16", "2025-01-20"]
    assert np.isclose(df.loc[0, "dv_ttm"], 2.0)
    assert np.isclose(df.loc[1, "dv_ttm"], 3.0)
    assert np.isclose(df.loc[2, "dv_ttm"], 1.0)
    assert np.isclose(df.loc[0, "pe_ttm"], 8.1)
    assert np.isclose(df.loc[2, "pb"], 1.4)


def test_sync_index_daily_uses_recent_pe_overlay_without_overwriting_close(tmp_path):
    fetcher = DataFetcher(db_path=str(tmp_path / "cache.db"))

    class FakeAk(object):
        @staticmethod
        def stock_zh_index_hist_csindex(symbol):
            return pd.DataFrame(
                {
                    "日期": ["2024-06-03"],
                    "收盘": [11138.83],
                    "滚动市盈率": [6.79],
                    "成交量": [3363.19],
                    "成交金额": [239.92],
                }
            )

        @staticmethod
        def stock_zh_index_value_csindex(symbol):
            return pd.DataFrame(
                {
                    "日期": ["2024-06-03", "2026-03-18"],
                    "市盈率1": [6.88, 8.12],
                }
            )

    with fetcher._connect() as conn:
        fetcher._sync_index_daily(conn, FakeAk(), "2024-06-01", "2026-03-19")
        old_row = conn.execute(
            "SELECT close, pe_ttm FROM index_daily WHERE date = ? AND index_code = ?",
            ("2024-06-03", "H30269"),
        ).fetchone()
        new_row = conn.execute(
            "SELECT close, pe_ttm FROM index_daily WHERE date = ? AND index_code = ?",
            ("2026-03-18", "H30269"),
        ).fetchone()

    assert old_row is not None
    assert np.isclose(old_row[0], 11138.83)
    assert np.isclose(old_row[1], 6.88)
    assert new_row is not None
    assert new_row[0] is None
    assert np.isclose(new_row[1], 8.12)


def test_validate_trading_day_coverage_skips_without_real_calendar():
    df = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.to_datetime(["2024-02-09", "2024-02-19"]))
    DataFetcher.validate_trading_day_coverage(df, "2024-02-09", "2024-02-19", data_proxy=None)


def test_sync_all_skips_remote_calls_when_checkpoint_covers_requested_range(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    fetcher = DataFetcher(db_path=str(db_path))

    with fetcher._connect() as conn:
        conn.execute(
            "INSERT INTO index_daily (date, index_code, close, pe_ttm, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("2024-01-02", "H30269", 100.0, 8.0, "2024-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO etf_daily (date, etf_code, close, close_hfq, volume, amount, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2024-01-02", "512890", 1.0, 1.0, 1000.0, 1000.0, "2024-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO etf_nav (date, etf_code, nav, acc_nav, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("2024-01-02", "512890", 0.99, 0.99, "2024-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO bond_yield (date, china_10y, updated_at) VALUES (?, ?, ?)",
            ("2024-01-02", 2.5, "2024-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO index_weight (index_code, stock_code, stock_name, weight, snapshot_date, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("H30269", "000001", "A", 100.0, "2024-01-02", "2024-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO stock_indicator (date, stock_code, dv_ttm, pe_ttm, pb, total_mv, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2024-01-02", "000001", 5.0, 8.0, 1.2, 1000.0, "2024-01-02T00:00:00"),
        )
        for source_name in (
            "index_daily",
            "etf_daily",
            "etf_nav",
            "bond_yield",
            "index_weight",
            "stock_indicator",
        ):
            conn.execute(
                "INSERT INTO sync_checkpoint (source_name, sync_start_date, sync_end_date, updated_at) VALUES (?, ?, ?, ?)",
                (source_name, "2024-01-01", "2024-01-31", "2024-01-31T00:00:00"),
            )
        conn.commit()

    class FakeAk(object):
        def __getattr__(self, name):
            raise AssertionError("unexpected remote call: {}".format(name))

    monkeypatch.setattr(fetcher, "_require_akshare", lambda: FakeAk())

    fetcher.sync_all("2024-01-01", "2024-01-31")


def test_call_akshare_retries_transient_network_errors(tmp_path, monkeypatch):
    fetcher = DataFetcher(db_path=str(tmp_path / "cache.db"))
    calls = []

    def flaky():
        calls.append("call")
        if len(calls) < 2:
            raise RequestsConnectionError("remote closed")
        return {"ok": True}

    monkeypatch.setattr("rqalpha.dividend_scorer.data_fetcher.time.sleep", lambda _: None)

    result = fetcher._call_akshare(flaky)

    assert result == {"ok": True}
    assert len(calls) == 2


def test_dividend_scorer_precompute_reads_end_date_from_rqattrdict_env():
    class DummyFetcher(object):
        def __init__(self):
            self.data_proxy = None
            self.loaded = None

        def get_available_range(self):
            return pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")

        def load_history(self, start_date, end_date):
            self.loaded = (start_date, end_date)
            return pd.DataFrame(
                {"etf_close_hfq": [1.0, 1.1]},
                index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            )

    class DummyFeatureEngine(object):
        def precompute(self, history_df):
            return pd.DataFrame(
                {
                    feature_name: np.linspace(0.1, 0.2, len(history_df))
                    for feature_name in VALUATION_FEATURES
                },
                index=history_df.index,
            )

    class DummyWeightCalculator(object):
        def calculate_ic_ir_weights(self, feature_matrix, price_series):
            return {"weights": {}, "method": "test"}

    scorer = DividendScorer(
        data_fetcher=DummyFetcher(),
        feature_engine=DummyFeatureEngine(),
        weight_calculator=DummyWeightCalculator(),
        score_synthesizer=ScoreSynthesizer(),
    )
    env = SimpleNamespace(
        data_proxy="mock-data-proxy",
        config=SimpleNamespace(base=RqAttrDict({"end_date": pd.Timestamp("2024-01-15")})),
    )

    scorer.precompute(env=env)

    assert scorer.data_fetcher.data_proxy == "mock-data-proxy"
    assert scorer.data_fetcher.loaded == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-15"))
