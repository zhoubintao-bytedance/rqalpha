# -*- coding: utf-8 -*-

import argparse
import json
import re
import shutil
import sys
import threading
import time
import unicodedata
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

    def sync_all(self, start_date, end_date, progress=None):
        self.data_fetcher.sync_all(start_date, end_date, progress=progress)

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
    parser.add_argument("--json", action="store_true", help="print JSON output")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    scorer = DividendScorer(db_path=args.db_path, bundle_path=args.bundle_path)
    if args.sync:
        available_start = args.start_date or "2020-01-01"
        available_end = args.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
        progress = SyncProgressReporter(stream=sys.stderr)
        progress.banner(
            title="红利低波打分器缓存同步",
            start_date=available_start,
            end_date=available_end,
            db_path=scorer.data_fetcher.db_path,
        )
        scorer.sync_all(available_start, available_end, progress=progress)
        if args.sync_only:
            print("sync success ✅: {} -> {}".format(available_start, available_end))
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
    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
    ELLIPSIS = "..."
    # SkyEye animation: heartbeat
    SPINNER_FRAMES = (
        "💜",
        "💙",
        "💚",
        "💛",
    )

    SPINNER_INTERVAL = 0.10   # refresh rate for elapsed time
    SPINNER_FRAME_TICKS = 10  # advance heart every N ticks (~1s)

    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.use_color = bool(getattr(self.stream, "isatty", lambda: False)())
        self.total_steps = 0
        self.current_step = 0
        self.current_label = ""
        self.step_started_at = None
        self.sync_started_at = None
        self._current_state = ""
        self._current_message = ""
        self._line_open = False
        self._last_width = 0
        self._spinner_index = 0
        self._spinner_thread = None
        self._spinner_stop = None
        self._lock = threading.Lock()

    def banner(self, title, start_date, end_date, db_path):
        self._emit_line(self._style(title, "header"))
        self._emit_line("  range : {} -> {}".format(start_date, end_date))
        self._emit_line("  cache : {}".format(db_path))
        self._emit_line("")

    def start(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.sync_started_at = time.time()

    def start_step(self, source_name, label):
        self._stop_spinner()
        with self._lock:
            self.current_step += 1
            self.current_label = label
            self.step_started_at = time.time()
            self._current_state = "RUN"
            self._current_message = "preparing"
            self._spinner_index = 0
            line = self._format_line("RUN", "preparing", spinner=self._spinner_frame())
        if self.use_color:
            self._render(line)
            self._start_spinner()
        else:
            self._emit_line(line)

    def update_step(self, current=None, total=None, detail=None, stats=None):
        if current is None or total in (None, 0):
            return
        if not self.use_color:
            return
        with self._lock:
            self._current_state = "RUN"
            self._current_message = self._progress_bar(current, total, detail, stats)
            self._render(self._format_line("RUN", self._current_message, spinner=self._spinner_frame()))

    def finish_step(self, status, detail=None):
        suffix = detail or "-"
        self._stop_spinner()
        with self._lock:
            self._current_state = status.upper()
            self._current_message = suffix
        line = self._format_line(status.upper(), suffix)
        if self.use_color:
            self._render(line, newline=True)
        else:
            self._emit_line(line)

    def close(self):
        self._stop_spinner()
        if self._line_open:
            self.stream.write("\n")
            self.stream.flush()
            self._line_open = False

    def _format_line(self, state, message, spinner=None):
        phase = "[{}/{}]".format(self.current_step, self.total_steps)
        label = self._pad_display(self.current_label, 10)
        spinner_width = max(self._display_width(frame) for frame in self.SPINNER_FRAMES)
        if self.use_color and spinner:
            visible = self.ANSI_RE.sub("", spinner)
            pad = max(spinner_width - self._display_width(visible), 0)
            spinner_token = spinner + (" " * pad)
        else:
            spinner_token = " " * spinner_width
        state_padded = self._pad_display(state, 4)
        elapsed = 0.0 if self.step_started_at is None else max(time.time() - self.step_started_at, 0.0)
        show_elapsed = state.strip() != "RUN" or elapsed >= 0.5
        elapsed_str = self._pad_display(self._format_duration(elapsed) if show_elapsed else "", 6)
        prefix_parts = [
            self._style(phase, "phase"),
            self._style(label, "label"),
            spinner_token,
            self._style(state_padded, self._state_style(state.strip())),
            self._style(elapsed_str, "phase"),
        ]
        prefix = " ".join(prefix_parts)
        terminal_width = self._terminal_width()
        if terminal_width is None:
            return "{} {}".format(prefix, message)
        available = max(terminal_width - self._display_width(prefix) - 1, 8)
        fitted_message = message
        if self._display_width(message) > available:
            fitted_message = self._truncate_display(self.ANSI_RE.sub("", message), available)
        return "{} {}".format(prefix, fitted_message)

    def _progress_bar(self, current, total, detail, stats):
        width = 22
        ratio = max(0.0, min(float(current) / float(total), 1.0))
        filled = int(round(width * ratio))
        percent = "{:>5.1f}%".format(ratio * 100.0)
        filled_bar = self._style("█" * filled, "bar_fill")
        empty_bar = self._style("░" * (width - filled), "bar_empty")
        bar = "{}{}".format(filled_bar, empty_bar)
        message = "{} {}/{} {}".format(bar, current, total, percent)
        elapsed = 0.0 if self.step_started_at is None else max(time.time() - self.step_started_at, 0.0)
        if current > 0 and elapsed >= 0.5:
            rate = current / elapsed
            eta = (total - current) / rate if rate > 0 else None
            if eta is not None:
                message = "{} | 预计剩余 {}".format(message, self._format_duration(eta))
        if detail:
            message = "{} | {}".format(message, detail)
        return message

    def _render(self, line, newline=False):
        fitted = self._fit_line_to_terminal(line)
        visible_width = self._display_width(fitted)
        padded = fitted
        if visible_width < self._last_width:
            padded += " " * (self._last_width - visible_width)
        clear = "\r\033[2K" if self.use_color else "\r"
        if newline:
            self.stream.write(clear + padded + "\n")
            self.stream.flush()
            self._line_open = False
            self._last_width = 0
        else:
            self.stream.write(clear + padded)
            self.stream.flush()
            self._line_open = True
            self._last_width = visible_width

    def _emit_line(self, line):
        if self._line_open:
            clear = "\r\033[2K" if self.use_color else ("\r" + " " * self._last_width + "\r")
            self.stream.write(clear)
            self.stream.flush()
            self._line_open = False
            self._last_width = 0
        self.stream.write(line + "\n")
        self.stream.flush()

    def _style(self, text, kind):
        if not self.use_color:
            return text
        if not text:
            return text
        styles = {
            "header": "\033[1;36m",
            "phase": "\033[1;37m",
            "label": "\033[1;34m",
            "spinner": "\033[1;35m",
            "run": "\033[1;36m",
            "done": "\033[1;32m",
            "skip": "\033[1;33m",
            "fail": "\033[1;31m",
            "bar_fill": "\033[32m",
            "bar_empty": "\033[2;37m",
        }
        prefix = styles.get(kind, "")
        suffix = "\033[0m" if prefix else ""
        return "{}{}{}".format(prefix, text, suffix)

    @staticmethod
    def _state_style(state):
        if state == "DONE":
            return "done"
        if state == "SKIP":
            return "skip"
        if state == "FAIL":
            return "fail"
        return "run"

    @classmethod
    def _display_width(cls, text):
        visible = cls.ANSI_RE.sub("", str(text))
        total = 0
        for ch in visible:
            cp = ord(ch)
            # variation selectors and zero-width joiners take no space
            if cp in (0xFE0E, 0xFE0F, 0x200D):
                continue
            # emoji and other wide Unicode blocks rendered as 2 columns
            if (0x1F300 <= cp <= 0x1FAFF) or (0x2600 <= cp <= 0x27BF):
                total += 2
            elif unicodedata.east_asian_width(ch) in ("W", "F"):
                total += 2
            else:
                total += 1
        return total

    @classmethod
    def _pad_display(cls, text, width):
        text = str(text)
        pad = max(width - cls._display_width(text), 0)
        return text + (" " * pad)

    @staticmethod
    def _format_duration(seconds):
        if seconds is None:
            return "-"
        seconds = max(float(seconds), 0.0)
        if seconds < 60:
            return "{:.1f}s".format(seconds)
        whole = int(round(seconds))
        minutes, secs = divmod(whole, 60)
        if minutes < 60:
            return "{}m{:02d}s".format(minutes, secs)
        hours, minutes = divmod(minutes, 60)
        return "{}h{:02d}m".format(hours, minutes)

    @staticmethod
    def _format_stats(stats):
        parts = []
        for key in ("empty", "out", "rows"):
            if key not in stats:
                continue
            value = stats[key]
            if value is None:
                continue
            if key == "rows":
                parts.append("rows={:,}".format(int(value)))
            else:
                parts.append("{}={}".format(key, int(value)))
        return " ".join(parts)

    def _terminal_width(self):
        if not self.use_color:
            return None
        try:
            return shutil.get_terminal_size(fallback=(100, 20)).columns
        except Exception:
            return 100

    def _fit_line_to_terminal(self, line):
        width = self._terminal_width()
        if width is None:
            return line
        max_width = max(width - 1, 20)
        if self._display_width(line) <= max_width:
            return line
        plain = self.ANSI_RE.sub("", line)
        return self._truncate_display(plain, max_width)

    @classmethod
    def _truncate_display(cls, text, max_width):
        text = str(text)
        if cls._display_width(text) <= max_width:
            return text
        if max_width <= cls._display_width(cls.ELLIPSIS):
            return cls.ELLIPSIS[:max_width]
        target = max_width - cls._display_width(cls.ELLIPSIS)
        out = []
        used = 0
        for ch in text:
            ch_width = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if used + ch_width > target:
                break
            out.append(ch)
            used += ch_width
        return "".join(out) + cls.ELLIPSIS

    def _spinner_frame(self):
        idx = self._spinner_index % len(self.SPINNER_FRAMES)
        return self.SPINNER_FRAMES[idx]

    def _start_spinner(self):
        if not self.use_color:
            return
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._spinner_loop,
            args=(stop_event,),
            daemon=True,
            name="dividend-scorer-sync-spinner",
        )
        with self._lock:
            self._spinner_stop = stop_event
            self._spinner_thread = thread
        thread.start()

    def _stop_spinner(self):
        thread = None
        stop_event = None
        with self._lock:
            stop_event = self._spinner_stop
            thread = self._spinner_thread
            self._spinner_stop = None
            self._spinner_thread = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.3)

    def _spinner_loop(self, stop_event):
        tick = 0
        while not stop_event.wait(self.SPINNER_INTERVAL):
            with self._lock:
                if self._current_state != "RUN":
                    continue
                tick += 1
                if tick % self.SPINNER_FRAME_TICKS == 0:
                    self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_FRAMES)
                self._render(
                    self._format_line(
                        self._current_state,
                        self._current_message,
                        spinner=self._spinner_frame(),
                    )
                )


if __name__ == "__main__":
    main()
