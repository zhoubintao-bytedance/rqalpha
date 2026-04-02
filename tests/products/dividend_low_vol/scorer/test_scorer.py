# -*- coding: utf-8 -*-

from collections import OrderedDict
import io
import sqlite3
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd

from skyeye.products.dividend_low_vol.scorer.config import DOMAIN_WEIGHTS
from skyeye.products.dividend_low_vol.scorer.config import VALUATION_FEATURES
from skyeye.products.dividend_low_vol.scorer.data_fetcher import DataFetcher
from skyeye.products.dividend_low_vol.scorer.feature_engine import FeatureEngine
from skyeye.products.dividend_low_vol.scorer.main import DividendScorer
from skyeye.products.dividend_low_vol.scorer.main import SyncProgressReporter
from skyeye.products.dividend_low_vol.scorer.main import format_score_report
from skyeye.products.dividend_low_vol.scorer.main import main as dividend_main
from skyeye.products.dividend_low_vol.scorer.score_synthesizer import ScoreSynthesizer
from skyeye.products.dividend_low_vol.scorer.weight_calculator import WeightCalculator
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


def test_feature_engine_precompute_does_not_emit_pct_change_future_warning():
    history_df = make_history_df()
    history_df.loc[history_df.index[10], "etf_close_hfq"] = np.nan
    engine = FeatureEngine()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.precompute(history_df)

    assert not any("pct_change" in str(item.message) for item in caught)


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
    assert "单日分位仅反映当前位置" in report
    assert "直接开平仓信号" in report
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


def test_format_score_report_wraps_long_msg_inside_summary_box(monkeypatch):
    monkeypatch.setattr(
        "skyeye.products.dividend_low_vol.scorer.main.shutil.get_terminal_size",
        lambda fallback=(100, 20): SimpleNamespace(columns=72),
    )

    report = format_score_report(
        {
            "etf": "512890",
            "date": "2026-03-19",
            "total_score": 5.79,
            "score_percentile": 0.406,
            "score_percentile_window": 504,
            "score_percentile_sample_size": 504,
            "buy_percentile_threshold": 0.2,
            "sell_percentile_threshold": 0.8,
            "confidence": "normal",
            "model_meta": {"method": "fixed_domain_prior"},
            "features": OrderedDict(),
            "confidence_modifiers": OrderedDict(),
            "warnings": [],
        }
    )

    lines = report.splitlines()
    msg_line_index = next(idx for idx, line in enumerate(lines) if "│ msg" in line)
    msg_block = []
    for line in lines[msg_line_index:]:
        if line.startswith("└"):
            break
        msg_block.append(line)

    assert len(msg_block) >= 3
    assert "单日分位仅反映当前位置" not in msg_block[0]
    assert any("单日分位仅反映当前位置" in line for line in msg_block[1:])
    assert any("直接开平仓信号" in line for line in msg_block[1:])
    assert any(line.startswith("│           ") for line in msg_block[1:])


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


def test_sync_banner_prints_red_end_date_warning_when_dates_mismatch():
    class FakeStream(io.StringIO):
        def isatty(self):
            return True

    stream = FakeStream()
    reporter = SyncProgressReporter(stream=stream)
    reporter.banner(
        title="红利低波打分器缓存同步",
        start_date="2026-02-05",
        end_date="2026-02-13",
        db_path="/tmp/cache.db",
        end_date_warning="请求的 end-date=2026-02-19，实际计算日期=2026-02-13（原因：该日期处于周末和节假日连休区间，已回退到最近交易日）",
    )

    output = stream.getvalue()
    assert "\033[1;33m" in output
    assert "请求的 end-date=2026-02-19，实际计算日期=2026-02-13" in output
    assert "原因：该日期处于周末和节假日连休区间，已回退到最近交易日" in output


def test_main_sync_only_implies_sync(monkeypatch):
    calls = []

    class FakeScorer(object):
        def __init__(self, db_path=None, bundle_path=None, prior_blend=None):
            self.data_fetcher = SimpleNamespace(db_path="/tmp/cache.db")

        def _resolve_sync_range(self, start_date=None, end_date=None):
            return ("2020-01-01", end_date or "2026-03-20")

        def sync_all(self, start_date, end_date, progress=None):
            calls.append(("sync_all", start_date, end_date, progress is not None))

        def precompute(self, start_date=None, end_date=None, progress=None):
            calls.append(("precompute", start_date, end_date, progress is not None))

        def score_latest(self, date=None):
            calls.append(("score_latest", date))
            return {}

    class FakeStream(io.StringIO):
        def isatty(self):
            return False

    stdout = io.StringIO()
    stderr = FakeStream()
    monkeypatch.setattr("skyeye.products.dividend_low_vol.scorer.main.DividendScorer", FakeScorer)
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    dividend_main(["--sync-only"])

    assert calls
    assert calls[0][0] == "sync_all"
    assert calls[0][1] == "2020-01-01"
    assert calls[0][3] is True
    assert all(call[0] != "precompute" for call in calls)
    assert "sync success ✅" in stdout.getvalue()


def test_main_defaults_to_auto_sync_before_scoring(monkeypatch):
    calls = []

    class FakeScorer(object):
        def __init__(self, db_path=None, bundle_path=None, prior_blend=None):
            self.data_fetcher = SimpleNamespace(db_path="/tmp/cache.db")

        def _resolve_sync_range(self, start_date=None, end_date=None):
            return ("2020-01-01", end_date or "2026-03-20")

        def sync_all(self, start_date, end_date, progress=None):
            calls.append(("sync_all", start_date, end_date, progress is not None))

        def precompute(self, start_date=None, end_date=None, progress=None):
            calls.append(("precompute", start_date, end_date, progress is not None))

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
            return False

    stdout = io.StringIO()
    stderr = FakeStream()
    monkeypatch.setattr("skyeye.products.dividend_low_vol.scorer.main.DividendScorer", FakeScorer)
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    dividend_main([])

    assert calls[0][0] == "sync_all"
    assert calls[0][1] == "2020-01-01"
    assert calls[1][0] == "precompute"
    assert calls[2][0] == "score_latest"
    assert "红利低波打分器结论" in stdout.getvalue()


def test_main_no_sync_uses_local_cache_only(monkeypatch):
    calls = []

    class FakeScorer(object):
        def __init__(self, db_path=None, bundle_path=None, prior_blend=None):
            self.data_fetcher = SimpleNamespace(db_path="/tmp/cache.db")

        def _resolve_sync_range(self, start_date=None, end_date=None):
            return ("2020-01-01", end_date or "2026-03-20")

        def sync_all(self, start_date, end_date, progress=None):
            calls.append(("sync_all", start_date, end_date, progress is not None))

        def precompute(self, start_date=None, end_date=None, progress=None):
            calls.append(("precompute", start_date, end_date, progress is not None))

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
            return False

    stdout = io.StringIO()
    stderr = FakeStream()
    monkeypatch.setattr("skyeye.products.dividend_low_vol.scorer.main.DividendScorer", FakeScorer)
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    dividend_main(["--no-sync"])

    assert all(call[0] != "sync_all" for call in calls)
    assert calls[0][0] == "precompute"
    assert calls[1][0] == "score_latest"
    assert "红利低波打分器结论" in stdout.getvalue()


def test_main_scoring_prints_logo_and_runs_cli_spinner(monkeypatch):
    calls = []

    class FakeScorer(object):
        def __init__(self, db_path=None, bundle_path=None, prior_blend=None):
            self.data_fetcher = SimpleNamespace(db_path="/tmp/cache.db")

        def _resolve_sync_range(self, start_date=None, end_date=None):
            return ("2020-01-01", end_date or "2026-03-20")

        def sync_all(self, start_date, end_date, progress=None):
            calls.append(("sync_all", start_date, end_date, progress is not None))

        def precompute(self, start_date=None, end_date=None, progress=None):
            calls.append(("precompute", start_date, end_date, progress is not None))

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

        def update_progress(self, current=None, total=None, detail=None):
            calls.append(("indicator_progress", current, total, detail))

        def stop(self):
            calls.append(("indicator_stop",))

    def fake_render_logo(stream, use_color):
        calls.append(("logo", use_color))

    stdout = io.StringIO()
    stderr = FakeStream()
    monkeypatch.setattr("skyeye.products.dividend_low_vol.scorer.main.DividendScorer", FakeScorer)
    monkeypatch.setattr("skyeye.products.dividend_low_vol.scorer.main.CliActivityIndicator", FakeIndicator)
    monkeypatch.setattr("skyeye.products.dividend_low_vol.scorer.main._render_gradient_logo", fake_render_logo)
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    dividend_main([])

    assert ("logo", True) in calls
    assert any(call[0] == "sync_all" for call in calls)
    assert ("indicator_init", "正在预计算估值特征", True) in calls
    assert ("indicator_start",) in calls
    assert ("indicator_update", "正在生成评分报告") in calls
    assert ("indicator_stop",) in calls
    assert "红利低波打分器结论" in stdout.getvalue()


def test_main_banner_warns_when_requested_end_date_is_clipped(monkeypatch):
    calls = []

    class FakeScorer(object):
        def __init__(self, db_path=None, bundle_path=None, prior_blend=None):
            self.data_fetcher = SimpleNamespace(db_path="/tmp/cache.db")

        def _resolve_sync_range(self, start_date=None, end_date=None):
            calls.append(("resolve_sync_range", start_date, end_date))
            return ("2026-02-05", "2026-02-13")

        def sync_all(self, start_date, end_date, progress=None):
            calls.append(("sync_all", start_date, end_date, progress is not None))

        def precompute(self, start_date=None, end_date=None, progress=None):
            calls.append(("precompute", start_date, end_date, progress is not None))

        def score_latest(self, date=None):
            calls.append(("score_latest", date))
            return {
                "etf": "512890",
                "date": "2026-02-13",
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

        def update_progress(self, current=None, total=None, detail=None):
            calls.append(("indicator_progress", current, total, detail))

        def stop(self):
            calls.append(("indicator_stop",))

    stdout = io.StringIO()
    stderr = FakeStream()
    monkeypatch.setattr("skyeye.products.dividend_low_vol.scorer.main.DividendScorer", FakeScorer)
    monkeypatch.setattr("skyeye.products.dividend_low_vol.scorer.main.CliActivityIndicator", FakeIndicator)
    monkeypatch.setattr(
        "skyeye.products.dividend_low_vol.scorer.main._render_gradient_logo", lambda stream, use_color: None
    )
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    dividend_main(["--end-date", "20260219"])

    assert ("resolve_sync_range", None, "20260219") in calls
    assert "range : 2026-02-05 -> 2026-02-13" in stderr.getvalue()
    assert "cache : /tmp/cache.db" in stderr.getvalue()
    assert "请求的 end-date=2026-02-19，实际计算日期=2026-02-13" in stderr.getvalue()
    assert "原因：该日期处于周末和节假日连休区间，已回退到最近交易日" in stderr.getvalue()
    assert "\033[1;33m" in stderr.getvalue()
    assert "红利低波打分器结论" in stdout.getvalue()


def test_main_banner_warns_with_weekend_reason(monkeypatch):
    class FakeScorer(object):
        def __init__(self, db_path=None, bundle_path=None, prior_blend=None):
            self.data_fetcher = SimpleNamespace(db_path="/tmp/cache.db")

        def _resolve_sync_range(self, start_date=None, end_date=None):
            return ("2026-03-05", "2026-03-20")

        def sync_all(self, start_date, end_date, progress=None):
            pass

        def precompute(self, start_date=None, end_date=None, progress=None):
            pass

        def score_latest(self, date=None):
            return {
                "etf": "512890",
                "date": "2026-03-20",
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
            pass

        def start(self):
            return self

        def update(self, label):
            pass

        def update_progress(self, current=None, total=None, detail=None):
            pass

        def stop(self):
            pass

    stdout = io.StringIO()
    stderr = FakeStream()
    monkeypatch.setattr("skyeye.products.dividend_low_vol.scorer.main.DividendScorer", FakeScorer)
    monkeypatch.setattr("skyeye.products.dividend_low_vol.scorer.main.CliActivityIndicator", FakeIndicator)
    monkeypatch.setattr(
        "skyeye.products.dividend_low_vol.scorer.main._render_gradient_logo", lambda stream, use_color: None
    )
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    dividend_main(["--end-date", "20260321"])

    assert "原因：该日期落在周末休市，已回退到最近交易日" in stderr.getvalue()


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


def test_validate_trading_day_coverage_skips_without_real_calendar():
    df = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.to_datetime(["2024-02-09", "2024-02-19"]))
    DataFetcher.validate_trading_day_coverage(df, "2024-02-09", "2024-02-19", data_proxy=None)


def test_resolve_source_sync_range_uses_checkpoint_overlap_when_start_is_implicit(tmp_path):
    class FakeDataProxy(object):
        def get_trading_dates(self, start_date, end_date):
            return pd.bdate_range(start=start_date, end=end_date)

    fetcher = DataFetcher(db_path=str(tmp_path / "cache.db"), data_proxy=FakeDataProxy())

    with fetcher._connect() as conn:
        conn.execute(
            "INSERT INTO etf_daily (date, etf_code, close, close_hfq, volume, amount, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2024-01-31", "512890", 1.0, 1.0, 1000.0, 1000.0, "2024-01-31T00:00:00"),
        )
        conn.execute(
            "INSERT INTO sync_checkpoint (source_name, sync_start_date, sync_end_date, updated_at) VALUES (?, ?, ?, ?)",
            ("etf_daily", "2020-01-01", "2024-02-20", "2024-02-20T00:00:00"),
        )
        start_date, end_date = fetcher._resolve_source_sync_range(conn, "etf_daily", None, "2024-02-20")

    assert start_date == "2024-01-17"
    assert end_date == "2024-02-20"


def test_resolve_source_sync_range_clips_weekend_end_date_for_stock_indicator(tmp_path):
    class FakeDataProxy(object):
        def get_trading_dates(self, start_date, end_date):
            return pd.bdate_range(start=start_date, end=end_date)

    fetcher = DataFetcher(db_path=str(tmp_path / "cache.db"), data_proxy=FakeDataProxy())

    with fetcher._connect() as conn:
        conn.execute(
            "INSERT INTO stock_indicator (date, stock_code, dv_ttm, pe_ttm, pb, total_mv, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2026-03-06", "600000", 2.0, 8.0, 0.8, 1000.0, "2026-03-20T00:00:00"),
        )
        conn.execute(
            "INSERT INTO stock_indicator (date, stock_code, dv_ttm, pe_ttm, pb, total_mv, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2026-03-20", "600000", 2.1, 8.1, 0.81, 1010.0, "2026-03-20T00:00:00"),
        )
        conn.execute(
            "INSERT INTO sync_checkpoint (source_name, sync_start_date, sync_end_date, updated_at) VALUES (?, ?, ?, ?)",
            ("stock_indicator", "2020-01-01", "2026-03-20", "2026-03-20T00:00:00"),
        )

        start_date, end_date = fetcher._resolve_source_sync_range(conn, "stock_indicator", None, "2026-03-21")

        assert start_date == "2026-03-06"
        assert end_date == "2026-03-20"
        assert fetcher._should_skip_sync(conn, "stock_indicator", start_date, end_date) is True


def test_update_sync_checkpoint_merges_existing_range_and_uses_actual_source_coverage(tmp_path):
    fetcher = DataFetcher(db_path=str(tmp_path / "cache.db"))

    with fetcher._connect() as conn:
        conn.execute(
            "INSERT INTO etf_daily (date, etf_code, close, close_hfq, volume, amount, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2024-01-02", "512890", 1.0, 1.0, 1000.0, 1000.0, "2024-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO etf_daily (date, etf_code, close, close_hfq, volume, amount, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2024-02-02", "512890", 1.1, 1.1, 1200.0, 1200.0, "2024-02-02T00:00:00"),
        )
        conn.execute(
            "INSERT INTO sync_checkpoint (source_name, sync_start_date, sync_end_date, updated_at) VALUES (?, ?, ?, ?)",
            ("etf_daily", "2020-01-01", "2024-01-31", "2024-01-31T00:00:00"),
        )

        fetcher._update_sync_checkpoint(conn, "etf_daily", "2024-01-20", "2024-02-15")
        row = conn.execute(
            "SELECT sync_start_date, sync_end_date FROM sync_checkpoint WHERE source_name = ?",
            ("etf_daily",),
        ).fetchone()

    assert row["sync_start_date"] == "2024-01-02"
    assert row["sync_end_date"] == "2024-02-02"


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

        def compute_single(self, date):
            return {}

    class DummyWeightCalculator(object):
        def calculate_shrunk_weights(self, feature_matrix, price_series, prior_blend=None, compute_diagnostics=False):
            return {"weights": {}, "method": "test"}

    scorer = DividendScorer(
        data_fetcher=DummyFetcher(),
        feature_engine=DummyFeatureEngine(),
        weight_calculator=DummyWeightCalculator(),
        score_synthesizer=ScoreSynthesizer(),
        prior_blend=1.0,
    )
    env = SimpleNamespace(
        data_proxy="mock-data-proxy",
        config=SimpleNamespace(base=RqAttrDict({"end_date": pd.Timestamp("2024-01-15")})),
    )

    scorer.precompute(env=env)

    assert scorer.data_fetcher.data_proxy == "mock-data-proxy"
    assert scorer.data_fetcher.loaded == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-15"))


def test_dividend_scorer_score_warns_when_actual_score_date_lags_target_date():
    class DummyFetcher(object):
        def __init__(self):
            self.data_proxy = None
            self.loaded = None
            self.freshness_reference_date = None

        def get_available_range(self):
            return "2026-03-01", "2026-03-06"

        def load_history(self, start_date, end_date):
            self.loaded = (start_date, end_date)
            index = pd.to_datetime(["2026-03-05", "2026-03-06"])
            return pd.DataFrame(
                {
                    "etf_close": [1.20, 1.19],
                    "etf_close_hfq": [2.40, 2.38],
                    "etf_volume": [1.0e8, 1.1e8],
                    "etf_nav": [1.19, 1.18],
                    "pe_ttm": [8.0, 8.1],
                    "dividend_yield": [0.031, 0.031],
                    "bond_10y": [0.020, 0.020],
                    "premium_rate": [0.01, 0.01],
                },
                index=index,
            )

        def _latest_trading_day_on_or_before(self, date_value):
            return pd.Timestamp(date_value).strftime("%Y-%m-%d")

        def _count_trading_days(self, start_date, end_date):
            if start_date == "2026-03-06" and end_date == "2026-03-20":
                return 10
            return 0

        def get_data_freshness(self, reference_date=None):
            self.freshness_reference_date = pd.Timestamp(reference_date).strftime("%Y-%m-%d")
            return {
                "etf_daily": {
                    "last_update_date": "2026-03-06",
                    "stale_trading_days": 10,
                    "status": "expired",
                }
            }

    class DummyFeatureEngine(object):
        def precompute(self, history_df):
            return pd.DataFrame(
                {
                    feature_name: np.linspace(0.2, 0.3, len(history_df))
                    for feature_name in VALUATION_FEATURES
                },
                index=history_df.index,
            )

        def compute_single(self, date):
            return {}

    class DummyWeightCalculator(object):
        def calculate_shrunk_weights(self, feature_matrix, price_series, prior_blend=None, compute_diagnostics=False):
            return {"weights": {}, "method": "fixed_domain_prior", "fallback_reason": None}

    class DummyScoreSynthesizer(object):
        def synthesize(self, feature_snapshot, weight_result, freshness=None):
            warnings = []
            if freshness:
                warnings.append("stale_sources: etf_daily")
            return {
                "total_score": 5.79,
                "confidence": "normal",
                "features": OrderedDict(),
                "confidence_modifiers": OrderedDict(),
                "warnings": warnings,
                "model_meta": {
                    "method": "fixed_domain_prior",
                    "fallback_reason": None,
                    "test_ic_avg": None,
                    "test_ic_ir_avg": None,
                    "label_window_n": 60,
                    "subsample_interval": 60,
                    "params_version": "test",
                },
            }

    scorer = DividendScorer(
        data_fetcher=DummyFetcher(),
        feature_engine=DummyFeatureEngine(),
        weight_calculator=DummyWeightCalculator(),
        score_synthesizer=DummyScoreSynthesizer(),
        prior_blend=1.0,
    )

    scorer.precompute(end_date="2026-03-20")
    result = scorer.score_latest()
    report = format_score_report(result)

    assert scorer.loaded_end_date == "2026-03-06"
    assert scorer.data_fetcher.loaded == ("2026-03-01", pd.Timestamp("2026-03-06"))
    assert scorer.data_fetcher.freshness_reference_date == "2026-03-20"
    assert result["requested_date"] == "2026-03-20"
    assert result["date"] == "2026-03-06"
    assert result["score_lag_trading_days"] == 10
    assert result["confidence"] == "lowered"
    assert "stale_sources: etf_daily" in result["warnings"]
    assert any(
        warning == "score_date_lag: requested=2026-03-20 actual=2026-03-06 trading_days=10"
        for warning in result["warnings"]
    )
    assert "目标日期" in report
    assert "评分日期" in report
    assert "滞后 10 个交易日" in report
