# -*- coding: utf-8 -*-

import argparse
import json
import sys
import time
from collections import OrderedDict

import pandas as pd

from rqalpha.dividend_scorer.config import (
    ETF_CODE,
    SCORE_BUY_PERCENTILE,
    SCORE_PERCENTILE_MIN_DATA,
    SCORE_PERCENTILE_WINDOW,
    SCORE_SELL_PERCENTILE,
    VALUATION_FEATURES,
    WEIGHT_DYNAMIC_DIAGNOSTIC,
    WEIGHT_PRIOR_BLEND,
)
from rqalpha.dividend_scorer.data_fetcher import DataFetcher
from rqalpha.dividend_scorer.feature_engine import FeatureEngine
from rqalpha.dividend_scorer.score_synthesizer import ScoreSynthesizer, ScoreUnavailableError
from rqalpha.dividend_scorer.weight_calculator import WeightCalculator


class DividendScorer(object):
    def __init__(
        self,
        db_path=None,
        bundle_path=None,
        data_proxy=None,
        data_fetcher=None,
        feature_engine=None,
        weight_calculator=None,
        score_synthesizer=None,
        prior_blend=WEIGHT_PRIOR_BLEND,
        dynamic_diagnostic=WEIGHT_DYNAMIC_DIAGNOSTIC,
    ):
        self.data_fetcher = data_fetcher or DataFetcher(
            db_path=db_path,
            bundle_path=bundle_path,
            data_proxy=data_proxy,
        )
        self.feature_engine = feature_engine or FeatureEngine()
        self.weight_calculator = weight_calculator or WeightCalculator()
        self.score_synthesizer = score_synthesizer or ScoreSynthesizer()
        self.prior_blend = float(prior_blend)
        self.dynamic_diagnostic = bool(dynamic_diagnostic)
        self.history_df = None
        self.weight_result = None
        self.score_history_df = None
        self.weight_result_by_date = {}

    def sync_all(self, start_date, end_date, force=False, progress=None):
        self.data_fetcher.sync_all(start_date, end_date, force=force, progress=progress)

    def precompute(self, env=None, start_date=None, end_date=None):
        if env is not None and getattr(env, "data_proxy", None) is not None:
            self.data_fetcher.data_proxy = env.data_proxy

        available_start, available_end = self.data_fetcher.get_available_range()
        start_date = start_date or available_start
        end_date = end_date or self._get_env_end_date(env) or available_end
        history_df = self.data_fetcher.load_history(start_date, end_date)
        feature_matrix = self.feature_engine.precompute(history_df)

        self.weight_result = self.weight_calculator.calculate_shrunk_weights(
            feature_matrix.loc[:, list(VALUATION_FEATURES)],
            history_df["etf_close_hfq"],
            prior_blend=self.prior_blend,
            compute_diagnostics=self.dynamic_diagnostic,
        )
        self.history_df = history_df
        self.score_history_df = self._build_score_history()
        return self

    def score(self, date=None):
        if self.history_df is None or self.weight_result is None or self.score_history_df is None:
            self.precompute()
        if date is None:
            date = self.history_df.index[-1]
        resolved_date = self._resolve_date(date)
        score_row = self.score_history_df.loc[resolved_date]
        if bool(score_row.get("error")):
            raise ScoreUnavailableError(score_row["error"])
        feature_snapshot = self.feature_engine.compute_single(resolved_date)
        freshness = self.data_fetcher.get_data_freshness(reference_date=resolved_date.date())
        weight_result = self.weight_result_by_date.get(resolved_date, self.weight_result)
        result = self.score_synthesizer.synthesize(
            feature_snapshot=feature_snapshot,
            weight_result=weight_result,
            freshness=freshness,
        )
        result["date"] = resolved_date.strftime("%Y-%m-%d")
        result["etf"] = ETF_CODE
        result["data_freshness"] = freshness
        result["score_percentile"] = self._maybe_float(score_row.get("score_percentile"))
        result["score_percentile_sample_size"] = self._maybe_int(score_row.get("score_percentile_sample_size"))
        result["score_percentile_window"] = SCORE_PERCENTILE_WINDOW
        result["buy_percentile_threshold"] = SCORE_BUY_PERCENTILE
        result["sell_percentile_threshold"] = SCORE_SELL_PERCENTILE
        return result

    def score_latest(self, date=None):
        return self.score(date=date)

    def _build_score_history(self):
        score_rows = []
        self.weight_result_by_date = {}
        static_weights = self.prior_blend >= 1.0

        for date in self.history_df.index:
            weight_result = self.weight_result
            if not static_weights:
                weight_result = self.weight_calculator.calculate_shrunk_weights(
                    self.feature_engine.normalized_matrix.loc[:date, list(VALUATION_FEATURES)],
                    self.history_df.loc[:date, "etf_close_hfq"],
                    prior_blend=self.prior_blend,
                    compute_diagnostics=False,
                )
            self.weight_result_by_date[date] = weight_result
            try:
                feature_snapshot = self.feature_engine.compute_single(date)
                score_result = self.score_synthesizer.synthesize(
                    feature_snapshot=feature_snapshot,
                    weight_result=weight_result,
                    freshness=None,
                )
                score_rows.append({
                    "date": date,
                    "total_score": float(score_result["total_score"]),
                    "confidence": score_result["confidence"],
                    "method": score_result["model_meta"]["method"],
                    "fallback_reason": score_result["model_meta"]["fallback_reason"],
                    "error": None,
                })
            except ScoreUnavailableError as exc:
                score_rows.append({
                    "date": date,
                    "total_score": None,
                    "confidence": None,
                    "method": weight_result.get("method"),
                    "fallback_reason": weight_result.get("fallback_reason"),
                    "error": str(exc),
                })

        score_df = pd.DataFrame(score_rows).set_index("date")
        percentile_series, count_series = self._rolling_hazen_percentile(score_df["total_score"])
        score_df["score_percentile"] = percentile_series
        score_df["score_percentile_sample_size"] = count_series
        return score_df

    @staticmethod
    def _rolling_hazen_percentile(series):
        counts = series.rolling(window=SCORE_PERCENTILE_WINDOW, min_periods=1).count()
        percentile = series.rolling(window=SCORE_PERCENTILE_WINDOW, min_periods=1).apply(
            DividendScorer._hazen_percentile_of_last,
            raw=False,
        )
        percentile = percentile.where(counts >= SCORE_PERCENTILE_MIN_DATA)
        return percentile, counts

    @staticmethod
    def _hazen_percentile_of_last(window):
        valid = pd.Series(window).dropna()
        if len(valid) == 0:
            return None
        current = valid.iloc[-1]
        ranks = valid.rank(method="average")
        rank = ranks.iloc[-1]
        return (rank - 0.5) / float(len(valid))

    def _resolve_date(self, date):
        ts = pd.Timestamp(date)
        if ts in self.history_df.index:
            return ts
        prior = self.history_df.index[self.history_df.index <= ts]
        if len(prior) == 0:
            raise KeyError("no history available on or before {}".format(ts.strftime("%Y-%m-%d")))
        return prior[-1]

    @staticmethod
    def _get_env_end_date(env):
        base_config = getattr(getattr(env, "config", None), "base", None)
        if base_config is None:
            return None
        if hasattr(base_config, "end_date"):
            return getattr(base_config, "end_date")
        if isinstance(base_config, dict):
            return base_config.get("end_date")
        return None

    @staticmethod
    def _maybe_float(value):
        if value is None or pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _maybe_int(value):
        if value is None or pd.isna(value):
            return None
        return int(value)


def format_score_report(score_result):
    lines = []
    lines.append("红利低波打分器 | {} | {}".format(score_result["etf"], score_result["date"]))
    lines.append("综合评分: {:.2f} / 10".format(score_result["total_score"]))
    if score_result.get("score_percentile") is not None:
        lines.append(
            "滚动分位: {:.1%} (window={}d, n={})".format(
                score_result["score_percentile"],
                score_result.get("score_percentile_window"),
                score_result.get("score_percentile_sample_size"),
            )
        )
    lines.append("置信度: {}".format(score_result["confidence"]))
    lines.append("权重方案: {}".format(score_result["model_meta"]["method"]))
    lines.append("")
    lines.append("估值指标:")
    for feature_name, feature_info in score_result["features"].items():
        raw_value = feature_info.get("raw")
        raw_repr = "-" if raw_value is None else "{:.6g}".format(raw_value)
        percentile = feature_info.get("percentile")
        normalized = feature_info.get("normalized")
        sub_score = feature_info.get("sub_score")
        weight = feature_info.get("weight", 0.0)
        lines.append(
            "  {:<18} raw={:<10} pct={} norm={} sub={} weight={:.1%}".format(
                feature_name,
                raw_repr,
                "-" if percentile is None else "{:.2f}".format(percentile),
                "-" if normalized is None else "{:.2f}".format(normalized),
                "-" if sub_score is None else "{:.2f}".format(sub_score),
                weight,
            )
        )
    lines.append("")
    lines.append("置信度修正因子:")
    for feature_name, feature_info in score_result["confidence_modifiers"].items():
        raw_value = feature_info.get("raw")
        raw_repr = "-" if raw_value is None else "{:.6g}".format(raw_value)
        percentile = feature_info.get("percentile")
        lines.append(
            "  {:<18} raw={:<10} pct={} status={}".format(
                feature_name,
                raw_repr,
                "-" if percentile is None else "{:.2f}".format(percentile),
                feature_info.get("status"),
            )
        )
    if score_result.get("warnings"):
        lines.append("")
        lines.append("警告: {}".format("; ".join(score_result["warnings"])))
    return "\n".join(lines)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Dividend low-volatility scorer")
    parser.add_argument("--date", dest="date", default=None, help="score date, defaults to latest available date")
    parser.add_argument("--start-date", dest="start_date", default=None, help="history start date")
    parser.add_argument("--end-date", dest="end_date", default=None, help="history end date")
    parser.add_argument("--db-path", dest="db_path", default=None, help="SQLite cache path")
    parser.add_argument("--bundle-path", dest="bundle_path", default=None, help="RQAlpha bundle path")
    parser.add_argument("--sync", action="store_true", help="sync data from AKShare before scoring")
    parser.add_argument("--sync-only", action="store_true", help="only sync data, do not score afterwards")
    parser.add_argument("--force-sync", action="store_true", help="force remote sync even if local cache already covers the requested range")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    scorer = DividendScorer(db_path=args.db_path, bundle_path=args.bundle_path)
    if args.sync:
        available_start = args.start_date or "2018-01-01"
        available_end = args.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
        progress = SyncProgressReporter(stream=sys.stderr)
        scorer.sync_all(available_start, available_end, force=args.force_sync, progress=progress)
        if args.sync_only:
            print("sync completed: {} -> {}".format(available_start, available_end))
            return
    scorer.precompute(start_date=args.start_date, end_date=args.end_date)
    score_result = scorer.score_latest(date=args.date)
    if args.json:
        print(json.dumps(score_result, ensure_ascii=False, indent=2, default=_json_default))
        return
    print(format_score_report(score_result))


def _json_default(value):
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    return value


class SyncProgressReporter(object):
    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.total_steps = 0
        self.current_step = 0
        self.current_label = ""
        self.step_started_at = None
        self._line_open = False
        self._last_width = 0

    def start(self, total_steps):
        self.total_steps = total_steps

    def start_step(self, source_name, label):
        self.current_step += 1
        self.current_label = label
        self.step_started_at = time.time()
        self._render(self._format_line("START", "starting"))

    def update_step(self, current=None, total=None, detail=None):
        if current is None or total in (None, 0):
            return
        self._render(self._format_line("RUN", self._progress_bar(current, total, detail)))

    def finish_step(self, status, detail=None):
        elapsed = 0.0 if self.step_started_at is None else max(time.time() - self.step_started_at, 0.0)
        suffix = detail or "-"
        suffix = "{} | {:.1f}s".format(suffix, elapsed)
        self._render(self._format_line(status.upper(), suffix), newline=True)

    def close(self):
        if self._line_open:
            self.stream.write("\n")
            self.stream.flush()
            self._line_open = False

    def _format_line(self, state, message):
        return "[sync {}/{}] {:<8} {:<5} {}".format(
            self.current_step,
            self.total_steps,
            self.current_label,
            state,
            message,
        )

    def _progress_bar(self, current, total, detail):
        width = 24
        filled = int(round(width * float(current) / float(total)))
        bar = "[{}{}]".format("#" * filled, "." * (width - filled))
        message = "{} {}/{}".format(bar, current, total)
        if detail:
            message = "{} {}".format(message, detail)
        return message

    def _render(self, line, newline=False):
        padded = line
        if len(padded) < self._last_width:
            padded += " " * (self._last_width - len(padded))
        if newline:
            self.stream.write("\r" + padded + "\n")
            self.stream.flush()
            self._line_open = False
        else:
            self.stream.write("\r" + padded)
            self.stream.flush()
            self._line_open = True
        self._last_width = len(line)


if __name__ == "__main__":
    main()
