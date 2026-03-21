# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from collections import OrderedDict

import pandas as pd

from skyeye.products.dividend_low_vol.scorer.config import (
    ETF_CODE,
    FEATURE_DISPLAY_NAMES,
    SCORE_BUY_PERCENTILE,
    SCORE_PERCENTILE_MIN_DATA,
    SCORE_PERCENTILE_WINDOW,
    SCORE_SELL_PERCENTILE,
    VALUATION_FEATURES,
    WEIGHT_DYNAMIC_DIAGNOSTIC,
    WEIGHT_PRIOR_BLEND,
)
from skyeye.products.dividend_low_vol.scorer.data_fetcher import DataFetcher
from skyeye.products.dividend_low_vol.scorer.feature_engine import FeatureEngine
from skyeye.products.dividend_low_vol.scorer.score_synthesizer import ScoreSynthesizer, ScoreUnavailableError
from skyeye.products.dividend_low_vol.scorer.weight_calculator import WeightCalculator

HEARTBEAT_FRAMES = (
    "💜",
    "💙",
    "💚",
    "💛",
)
HEARTBEAT_INTERVAL = 0.10
HEARTBEAT_FRAME_TICKS = 10


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
        self.target_end_date = None
        self.loaded_end_date = None

    def sync_all(self, start_date, end_date, progress=None):
        self.data_fetcher.sync_all(start_date, end_date, progress=progress)

    def prepare(self, env=None, start_date=None, end_date=None, auto_sync=True, sync_progress=None):
        if env is not None and getattr(env, "data_proxy", None) is not None:
            self.data_fetcher.data_proxy = env.data_proxy

        if auto_sync:
            sync_start, sync_end = self._resolve_sync_range(
                env=env,
                start_date=start_date,
                end_date=end_date,
            )
            self.sync_all(sync_start, sync_end, progress=sync_progress)

        return self.precompute(env=env, start_date=start_date, end_date=end_date)

    def precompute(self, env=None, start_date=None, end_date=None, progress=None):
        if env is not None and getattr(env, "data_proxy", None) is not None:
            self.data_fetcher.data_proxy = env.data_proxy

        self._progress_update(progress, "pre-computing | 读取历史数据")
        available_start, available_end = self.data_fetcher.get_available_range()
        start_date = start_date or available_start
        requested_end_date = end_date or self._get_env_end_date(env) or pd.Timestamp.today().strftime("%Y-%m-%d")
        self.target_end_date = self._resolve_reference_score_date(requested_end_date)
        actual_end_ts = min(pd.Timestamp(self.target_end_date), pd.Timestamp(available_end))
        self.loaded_end_date = actual_end_ts.strftime("%Y-%m-%d")
        history_df = self.data_fetcher.load_history(start_date, actual_end_ts)
        self._progress_update(progress, "pre-computing | 计算估值特征")
        feature_matrix = self.feature_engine.precompute(history_df)

        self._progress_update(progress, "pre-computing | 计算收缩权重")
        self.weight_result = self.weight_calculator.calculate_shrunk_weights(
            feature_matrix.loc[:, list(VALUATION_FEATURES)],
            history_df["etf_close_hfq"],
            prior_blend=self.prior_blend,
            compute_diagnostics=self.dynamic_diagnostic,
        )
        self.history_df = history_df
        self._progress_update(progress, "pre-computing | 计算历史分数")
        self.score_history_df = self._build_score_history(progress=progress)
        return self

    def score(self, date=None):
        if self.history_df is None or self.weight_result is None or self.score_history_df is None:
            self.precompute()
        target_date = date or self.target_end_date or self.history_df.index[-1]
        target_date = pd.Timestamp(self._resolve_reference_score_date(target_date))
        resolved_date = self._resolve_date(target_date)
        score_row = self.score_history_df.loc[resolved_date]
        if bool(score_row.get("error")):
            raise ScoreUnavailableError(score_row["error"])
        feature_snapshot = self.feature_engine.compute_single(resolved_date)
        freshness = self.data_fetcher.get_data_freshness(reference_date=target_date.date())
        weight_result = self.weight_result_by_date.get(resolved_date, self.weight_result)
        result = self.score_synthesizer.synthesize(
            feature_snapshot=feature_snapshot,
            weight_result=weight_result,
            freshness=freshness,
        )
        result["date"] = resolved_date.strftime("%Y-%m-%d")
        result["requested_date"] = target_date.strftime("%Y-%m-%d")
        result["etf"] = ETF_CODE
        result["data_freshness"] = freshness
        result["score_percentile"] = self._maybe_float(score_row.get("score_percentile"))
        result["score_percentile_sample_size"] = self._maybe_int(score_row.get("score_percentile_sample_size"))
        result["score_percentile_window"] = SCORE_PERCENTILE_WINDOW
        result["buy_percentile_threshold"] = SCORE_BUY_PERCENTILE
        result["sell_percentile_threshold"] = SCORE_SELL_PERCENTILE
        lag_days = self._count_score_lag_trading_days(result["date"], result["requested_date"])
        result["score_lag_trading_days"] = lag_days
        if lag_days:
            warnings = list(result.get("warnings") or [])
            warnings.append(
                "score_date_lag: requested={} actual={} trading_days={}".format(
                    result["requested_date"],
                    result["date"],
                    lag_days,
                )
            )
            result["warnings"] = warnings
            if result.get("confidence") == "normal":
                result["confidence"] = "lowered"
        return result

    def score_latest(self, date=None):
        return self.score(date=date)

    def _build_score_history(self, progress=None):
        score_rows = []
        self.weight_result_by_date = {}
        static_weights = self.prior_blend >= 1.0
        total_dates = len(self.history_df.index)

        for idx, date in enumerate(self.history_df.index, start=1):
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
            if progress is not None and (idx == 1 or idx == total_dates or idx % 20 == 0):
                self._progress_update_step(
                    progress,
                    current=idx,
                    total=total_dates,
                    detail="截止 {}".format(date.strftime("%Y-%m-%d")),
                )

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

    def _resolve_sync_range(self, env=None, start_date=None, end_date=None):
        return self.data_fetcher.suggest_sync_range(
            start_date=start_date,
            end_date=end_date or self._get_env_end_date(env) or pd.Timestamp.today().strftime("%Y-%m-%d"),
        )

    def _resolve_reference_score_date(self, date_value):
        resolver = getattr(self.data_fetcher, "_latest_trading_day_on_or_before", None)
        if callable(resolver):
            return resolver(date_value)
        return pd.Timestamp(date_value).strftime("%Y-%m-%d")

    def _count_score_lag_trading_days(self, actual_date, target_date):
        counter = getattr(self.data_fetcher, "_count_trading_days", None)
        if callable(counter):
            return counter(actual_date, target_date)
        if pd.Timestamp(actual_date) >= pd.Timestamp(target_date):
            return 0
        return max(len(pd.bdate_range(start=actual_date, end=target_date)) - 1, 0)

    @staticmethod
    def _progress_update(progress, label):
        if progress is None or not hasattr(progress, "update"):
            return
        progress.update(label)

    @staticmethod
    def _progress_update_step(progress, current, total, detail=None):
        if progress is None or not hasattr(progress, "update_progress"):
            return
        progress.update_progress(current=current, total=total, detail=detail)

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
    lines = ["", ""]
    requested_date = score_result.get("requested_date")
    actual_date = score_result["date"]
    date_label = "评分日期" if requested_date and requested_date != actual_date else "日期"
    summary_items = [
        ("ETF", score_result["etf"]),
        (date_label, actual_date),
        ("综合评分", "{:.2f} / 10".format(score_result["total_score"])),
        ("滚动分位", _format_percentile_summary(score_result)),
        ("置信度", _format_confidence_display(score_result.get("confidence"))),
        ("权重方案", _format_weight_method_display(score_result["model_meta"]["method"])),
        ("msg", _build_score_summary_message(score_result)),
    ]
    if requested_date and requested_date != actual_date:
        summary_items.insert(1, ("目标日期", requested_date))
    lines.extend(_render_text_box(_format_key_value_lines(summary_items), title="红利低波打分器结论"))
    lines.append("")
    lines.append("估值指标:")
    feature_rows = []
    for feature_name, feature_info in score_result["features"].items():
        feature_rows.append([
            FEATURE_DISPLAY_NAMES.get(feature_name, feature_name),
            feature_name,
            _format_metric_value(feature_info.get("raw"), "{:.6g}"),
            _format_metric_value(feature_info.get("percentile"), "{:.2f}"),
            _format_metric_value(feature_info.get("normalized"), "{:.2f}"),
            _format_metric_value(feature_info.get("sub_score"), "{:.2f}"),
            _format_metric_value(feature_info.get("weight"), "{:.1%}"),
        ])
    lines.extend(
        _render_text_table(
            headers=("指标", "指标英文", "原始值", "分位", "归一值", "子分", "权重"),
            rows=feature_rows,
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


def _format_metric_value(value, pattern):
    if value is None or pd.isna(value):
        return "-"
    return pattern.format(value)


def _format_percentile_summary(score_result):
    percentile = score_result.get("score_percentile")
    if percentile is None:
        return "-"
    return "{:.1%} (window={}d, n={})".format(
        percentile,
        score_result.get("score_percentile_window"),
        score_result.get("score_percentile_sample_size"),
    )


def _format_confidence_display(confidence):
    label_map = {
        "normal": "normal（正常）",
        "lowered": "lowered（下调）",
        "low": "low（较低）",
    }
    return label_map.get(confidence, confidence or "-")


def _format_weight_method_display(method):
    label_map = {
        "fixed_domain_prior": "fixed_domain_prior（固定先验权重）",
        "shrunk_ic_ir": "shrunk_ic_ir（先验+动态收缩）",
        "ic_ir": "ic_ir（动态IC/IR权重）",
        "domain_knowledge_fallback": "domain_knowledge_fallback（领域先验回退）",
    }
    return label_map.get(method, method or "-")


def _build_score_summary_message(score_result):
    total_score = score_result.get("total_score")
    percentile = score_result.get("score_percentile")
    percentile_window = score_result.get("score_percentile_window")
    buy_threshold = score_result.get("buy_percentile_threshold")
    sell_threshold = score_result.get("sell_percentile_threshold")
    confidence = score_result.get("confidence")
    requested_date = score_result.get("requested_date")
    actual_date = score_result.get("date")
    lag_days = score_result.get("score_lag_trading_days")
    warnings = score_result.get("warnings") or []

    parts = []
    if total_score is not None:
        summary = "当前评分 {:.2f}/10".format(total_score)
        if percentile is not None:
            summary = "{}，位于近{}日{:.1%}分位（单日信号），{}".format(
                summary,
                percentile_window,
                percentile,
                _describe_percentile_zone(percentile, buy_threshold, sell_threshold),
            )
        parts.append(summary)
    elif percentile is not None:
        parts.append(
            "当前位于近{}日{:.1%}分位（单日信号），{}".format(
                percentile_window,
                percentile,
                _describe_percentile_zone(percentile, buy_threshold, sell_threshold),
            )
        )

    confidence_map = {
        "normal": "置信度正常",
        "lowered": "置信度已下调",
        "low": "置信度较低",
    }
    parts.append(confidence_map.get(confidence, "置信度={}".format(confidence)))
    if lag_days:
        parts.append(
            "目标日期 {}，但当前只能计算到 {}，滞后 {} 个交易日".format(
                requested_date,
                actual_date,
                lag_days,
            )
        )
    if percentile is not None:
        parts.append(
            "单日分位仅反映当前位置，不是历史感知策略的直接开平仓信号，实际调仓仍会结合近5/15期分位均值、趋势、热度与置信度"
        )

    warning_hint = _summarize_warnings(warnings)
    if warning_hint:
        parts.append(warning_hint)
    return "；".join(parts) + "。"


def _describe_percentile_zone(percentile, buy_threshold, sell_threshold):
    buy_threshold = 0.2 if buy_threshold is None else float(buy_threshold)
    sell_threshold = 0.8 if sell_threshold is None else float(sell_threshold)
    if percentile <= buy_threshold:
        return "接近历史低位，历史感知策略的主仓位通常会偏高"
    if percentile >= sell_threshold:
        return "接近历史高位，历史感知策略的主仓位通常会偏低"
    if percentile < 0.5:
        return "处于历史中低位，历史感知策略的主仓位通常偏中高"
    return "处于历史中高位，历史感知策略的主仓位通常偏中性偏低"


def _summarize_warnings(warnings):
    if not warnings:
        return ""
    if any(str(item).startswith("stale_sources:") for item in warnings):
        return "存在数据新鲜度告警"
    if "under_sampled_percentile_window" in warnings:
        return "分位样本窗口仍偏短"
    return "存在额外告警"


def _format_key_value_lines(items):
    key_width = max(_display_width(key) for key, _ in items)
    max_width = _preferred_text_box_content_width()
    prefix_width = key_width + 2
    value_width = max(max_width - prefix_width, 20)
    lines = []
    for key, value in items:
        prefix = "{}  ".format(_pad_display(key, key_width))
        wrapped_values = _wrap_display_text(value, value_width)
        lines.append("{}{}".format(prefix, wrapped_values[0]))
        continuation_prefix = " " * _display_width(prefix)
        for wrapped in wrapped_values[1:]:
            lines.append("{}{}".format(continuation_prefix, wrapped))
    return lines


def _render_text_box(lines, title=None):
    width = max(_display_width(line) for line in lines) if lines else 0
    if title:
        title_width = _display_width(title)
        width = max(width, title_width + 4)

    output = []
    if title:
        pad_total = width - _display_width(title) - 2
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        output.append("┌─" + ("─" * pad_left) + " " + title + " " + ("─" * pad_right) + "─┐")
    else:
        output.append("┌" + ("─" * (width + 2)) + "┐")
    for line in lines:
        output.append("│ " + line + (" " * max(width - _display_width(line), 0)) + " │")
    output.append("└" + ("─" * (width + 2)) + "┘")
    return output


def _preferred_text_box_content_width():
    try:
        columns = int(shutil.get_terminal_size(fallback=(100, 20)).columns)
    except Exception:
        columns = 100
    return max(min(columns - 4, 96), 48)


def _wrap_display_text(text, max_width):
    text = str(text)
    if max_width is None or max_width <= 0:
        return text.splitlines() or [text]

    paragraphs = text.splitlines() or [text]
    lines = []
    for paragraph in paragraphs:
        if paragraph == "":
            lines.append("")
            continue
        lines.extend(_wrap_display_paragraph(paragraph, max_width))
    return lines or [""]


def _wrap_display_paragraph(text, max_width):
    if _display_width(text) <= max_width:
        return [text]

    chars = list(str(text))
    lines = []
    start = 0
    total = len(chars)
    while start < total:
        used = 0
        end = start
        preferred_break = None
        while end < total:
            ch = chars[end]
            ch_width = _display_char_width(ch)
            if end > start and used + ch_width > max_width:
                break
            used += ch_width
            end += 1
            if _is_display_wrap_boundary(ch):
                preferred_break = end
            if used >= max_width:
                break

        if end >= total:
            lines.append("".join(chars[start:total]).rstrip())
            break

        if preferred_break is not None and preferred_break > start:
            end = preferred_break
        elif end == start:
            end = start + 1

        lines.append("".join(chars[start:end]).rstrip())
        start = end
        while start < total and chars[start] == " ":
            start += 1
    return lines or [""]


def _render_text_table(headers, rows):
    headers = [str(item) for item in headers]
    normalized_rows = [[str(cell) for cell in row] for row in rows]
    columns = len(headers)
    widths = []
    for idx in range(columns):
        cells = [headers[idx]]
        cells.extend(row[idx] for row in normalized_rows)
        widths.append(max(_display_width(cell) for cell in cells))

    top = "┌" + "┬".join("─" * (width + 2) for width in widths) + "┐"
    sep = "├" + "┼".join("─" * (width + 2) for width in widths) + "┤"
    bottom = "└" + "┴".join("─" * (width + 2) for width in widths) + "┘"

    output = [top, _render_table_row(headers, widths), sep]
    if normalized_rows:
        for row in normalized_rows:
            output.append(_render_table_row(row, widths))
    else:
        empty = ["-"] + [""] * (columns - 1)
        output.append(_render_table_row(empty, widths))
    output.append(bottom)
    return output


def _render_table_row(cells, widths):
    padded = [
        " " + _pad_display(cell, width) + " "
        for cell, width in zip(cells, widths)
    ]
    return "│" + "│".join(padded) + "│"


def _display_width(text):
    visible = str(text)
    total = 0
    for ch in visible:
        total += _display_char_width(ch)
    return total


def _pad_display(text, width):
    text = str(text)
    return text + (" " * max(width - _display_width(text), 0))


def _display_char_width(ch):
    cp = ord(ch)
    if cp in (0xFE0E, 0xFE0F, 0x200D):
        return 0
    if (0x1F300 <= cp <= 0x1FAFF) or (0x2600 <= cp <= 0x27BF):
        return 2
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 2
    return 1


def _is_display_wrap_boundary(ch):
    return ch.isspace() or ch in ",.;:!?)]}%，。；：！？、）》」』】"


def _candidate_logo_paths():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return (
        os.path.join(repo_root, "logo.txt"),
        os.path.join(repo_root, "RFC", "logo.txt"),
    )


def _resolve_logo_path():
    for path in _candidate_logo_paths():
        if os.path.isfile(path):
            return path
    return None


def _render_gradient_logo(stream, use_color):
    logo_path = _resolve_logo_path()
    if logo_path is None:
        return

    with open(logo_path, "r") as handle:
        lines = [line.rstrip("\n") for line in handle.readlines()]

    colors = [
        (17, 95, 179),
        (27, 113, 179),
        (37, 130, 178),
        (47, 148, 178),
        (56, 166, 178),
        (66, 183, 177),
        (76, 201, 177),
    ]

    for line in lines:
        if not use_color:
            stream.write(line + "\n")
            continue
        if not line:
            stream.write("\n")
            continue
        size = len(line)
        painted = []
        for index, ch in enumerate(line):
            if ch == " ":
                painted.append(ch)
                continue
            t = index / max(size - 1, 1)
            palette_index = t * (len(colors) - 1)
            left = min(int(palette_index), len(colors) - 2)
            frac = palette_index - left
            red = int(colors[left][0] + (colors[left + 1][0] - colors[left][0]) * frac)
            green = int(colors[left][1] + (colors[left + 1][1] - colors[left][1]) * frac)
            blue = int(colors[left][2] + (colors[left + 1][2] - colors[left][2]) * frac)
            painted.append("\033[38;2;{};{};{}m{}\033[0m".format(red, green, blue, ch))
        stream.write("".join(painted) + "\n")
    stream.write("\n")
    stream.flush()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Dividend low-volatility scorer")
    parser.add_argument("--date", dest="date", default=None, help="score date, defaults to latest available date")
    parser.add_argument("--start-date", dest="start_date", default=None, help="history start date")
    parser.add_argument("--end-date", dest="end_date", default=None, help="history end date")
    parser.add_argument("--db-path", dest="db_path", default=None, help="SQLite cache path")
    parser.add_argument("--bundle-path", dest="bundle_path", default=None, help="RQAlpha bundle path")
    parser.add_argument("--sync", action="store_true", help="explicitly enable auto sync before scoring (default)")
    parser.add_argument("--no-sync", action="store_true", help="skip auto sync and score from local cache only")
    parser.add_argument("--sync-only", action="store_true", help="only sync data and exit, implies --sync")
    parser.add_argument(
        "--prior-blend",
        type=float,
        default=WEIGHT_PRIOR_BLEND,
        help="blend ratio for domain prior weights, within [0, 1]",
    )
    parser.add_argument("--json", action="store_true", help="print JSON output")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.prior_blend < 0.0 or args.prior_blend > 1.0:
        parser.error("--prior-blend must be within [0, 1]")
    if args.no_sync and (args.sync or args.sync_only):
        parser.error("--no-sync cannot be combined with --sync or --sync-only")
    ui_stream = sys.stderr
    show_cli_ui = _should_show_cli_ui(args, stream=ui_stream)
    do_sync = not args.no_sync

    if show_cli_ui:
        _render_gradient_logo(ui_stream, use_color=True)

    scorer = DividendScorer(
        db_path=args.db_path,
        bundle_path=args.bundle_path,
        prior_blend=args.prior_blend,
    )
    sync_start = None
    sync_end = None
    end_date_warning = None
    if do_sync:
        sync_start, sync_end = scorer._resolve_sync_range(
            start_date=args.start_date,
            end_date=args.end_date,
        )
        end_date_warning = _build_end_date_warning(args.end_date, sync_end)
        progress = SyncProgressReporter(stream=ui_stream)
        if show_cli_ui:
            progress.banner(
                title="红利低波打分器缓存同步",
                start_date=sync_start,
                end_date=sync_end,
                db_path=scorer.data_fetcher.db_path,
                end_date_warning=end_date_warning,
            )
        scorer.sync_all(sync_start, sync_end, progress=progress)
        if args.sync_only:
            print("sync success ✅: {} -> {}".format(sync_start, sync_end))
            return
    indicator = CliActivityIndicator(
        label="正在预计算估值特征",
        stream=ui_stream,
        enabled=show_cli_ui,
    )
    indicator.start()
    try:
        scorer.precompute(start_date=args.start_date, end_date=args.end_date, progress=indicator)
        indicator.update("正在生成评分报告")
        score_result = scorer.score_latest(date=args.date)
    finally:
        indicator.stop()
    if args.json:
        print(json.dumps(score_result, ensure_ascii=False, indent=2, default=_json_default))
        return
    print(format_score_report(score_result))


def _json_default(value):
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    return value


def _should_show_cli_ui(args, stream=None):
    if getattr(args, "json", False):
        return False
    target = stream or sys.stderr
    return bool(getattr(target, "isatty", lambda: False)())


def _build_end_date_warning(requested_end_date, actual_end_date):
    requested_end_date = _normalize_cli_date(requested_end_date)
    actual_end_date = _normalize_cli_date(actual_end_date)
    if requested_end_date is None or actual_end_date is None:
        return None
    if requested_end_date == actual_end_date:
        return None
    reason = _describe_end_date_roll_back_reason(requested_end_date, actual_end_date)
    if reason:
        return "请求的 end-date={}，实际计算日期={}（原因：{}）".format(
            requested_end_date,
            actual_end_date,
            reason,
        )
    return "请求的 end-date={}，实际计算日期={}".format(requested_end_date, actual_end_date)


def _normalize_cli_date(date_value):
    if date_value in (None, ""):
        return None
    return pd.Timestamp(date_value).strftime("%Y-%m-%d")


def _describe_end_date_roll_back_reason(requested_end_date, actual_end_date):
    requested_ts = pd.Timestamp(requested_end_date).normalize()
    actual_ts = pd.Timestamp(actual_end_date).normalize()
    if requested_ts <= actual_ts:
        return ""

    closure_days = pd.date_range(actual_ts + pd.Timedelta(days=1), requested_ts, freq="D")
    if len(closure_days) == 0:
        return ""

    weekend_days = sum(1 for day in closure_days if day.weekday() >= 5)
    holiday_days = len(closure_days) - weekend_days
    if holiday_days == 0:
        return "该日期落在周末休市，已回退到最近交易日"
    if weekend_days == 0:
        return "该日期处于节假日休市区间，已回退到最近交易日"
    return "该日期处于周末和节假日连休区间，已回退到最近交易日"


class CliActivityIndicator(object):
    def __init__(self, label, stream=None, enabled=True):
        self.stream = stream or sys.stderr
        self.enabled = bool(enabled and getattr(self.stream, "isatty", lambda: False)())
        self.label = label
        self._line_open = False
        self._last_width = 0
        self._spinner_index = 0
        self._spinner_thread = None
        self._spinner_stop = None
        self._lock = threading.Lock()
        self._stage_started_at = None
        self._progress_current = None
        self._progress_total = None
        self._progress_detail = None

    def start(self):
        if not self.enabled:
            return self
        with self._lock:
            self._spinner_index = 0
            self._stage_started_at = time.time()
            self._progress_current = None
            self._progress_total = None
            self._progress_detail = None
        self._render_current()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._spinner_loop,
            args=(stop_event,),
            daemon=True,
            name="dividend-scorer-cli-spinner",
        )
        with self._lock:
            self._spinner_stop = stop_event
            self._spinner_thread = thread
        thread.start()
        return self

    def update(self, label):
        if not self.enabled:
            return
        with self._lock:
            self.label = label
            self._stage_started_at = time.time()
            self._progress_current = None
            self._progress_total = None
            self._progress_detail = None
        self._render_current()

    def update_progress(self, current=None, total=None, detail=None):
        if not self.enabled or current is None or total in (None, 0):
            return
        with self._lock:
            self._progress_current = int(current)
            self._progress_total = int(total)
            self._progress_detail = detail
        self._render_current()

    def stop(self):
        if not self.enabled:
            return
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
        with self._lock:
            if self._line_open:
                self.stream.write("\r\033[2K")
                self.stream.flush()
                self._line_open = False
                self._last_width = 0

    def _spinner_loop(self, stop_event):
        tick = 0
        while not stop_event.wait(HEARTBEAT_INTERVAL):
            with self._lock:
                tick += 1
                if tick % HEARTBEAT_FRAME_TICKS == 0:
                    self._spinner_index = (self._spinner_index + 1) % len(HEARTBEAT_FRAMES)
            self._render_current()

    def _render_current(self):
        if not self.enabled:
            return
        with self._lock:
            frame = HEARTBEAT_FRAMES[self._spinner_index % len(HEARTBEAT_FRAMES)]
            line = "{} {}".format(frame, self._style(self.label, "run"))
            if self._progress_total not in (None, 0) and self._progress_current is not None:
                line = "{} {}".format(
                    line,
                    self._progress_bar(
                        self._progress_current,
                        self._progress_total,
                        detail=self._progress_detail,
                    ),
                )
            visible_width = _display_width(line)
            padded = line
            if visible_width < self._last_width:
                padded += " " * (self._last_width - visible_width)
            self.stream.write("\r\033[2K" + padded)
            self.stream.flush()
            self._line_open = True
            self._last_width = visible_width

    @staticmethod
    def _style(text, kind):
        if not text:
            return text
        if kind == "run":
            return "\033[1;36m{}\033[0m".format(text)
        if kind == "bar_fill":
            return "\033[1;36m{}\033[0m".format(text)
        if kind == "bar_empty":
            return "\033[2;37m{}\033[0m".format(text)
        if kind == "detail":
            return "\033[2;37m{}\033[0m".format(text)
        return text

    def _progress_bar(self, current, total, detail=None):
        width = 18
        ratio = max(0.0, min(float(current) / float(total), 1.0))
        filled = int(round(width * ratio))
        percent = "{:>5.1f}%".format(ratio * 100.0)
        filled_bar = self._style("█" * filled, "bar_fill")
        empty_bar = self._style("░" * (width - filled), "bar_empty")
        message = "{}{} {}/{} {}".format(filled_bar, empty_bar, current, total, percent)
        elapsed = 0.0 if self._stage_started_at is None else max(time.time() - self._stage_started_at, 0.0)
        if current > 0 and elapsed >= 0.5:
            rate = current / elapsed
            eta = (total - current) / rate if rate > 0 else None
            if eta is not None:
                message = "{} | 预计剩余 {}".format(message, self._format_duration(eta))
        if detail:
            message = "{} | {}".format(message, self._style(detail, "detail"))
        return message

    @staticmethod
    def _format_duration(seconds):
        seconds = max(int(round(seconds)), 0)
        if seconds >= 3600:
            hours, remainder = divmod(seconds, 3600)
            minutes = remainder // 60
            return "{}h{:02d}m".format(hours, minutes)
        if seconds >= 60:
            minutes, remainder = divmod(seconds, 60)
            return "{}m{:02d}s".format(minutes, remainder)
        return "{:.1f}s".format(float(seconds))


class SyncProgressReporter(object):
    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
    ELLIPSIS = "..."
    SPINNER_FRAMES = HEARTBEAT_FRAMES
    SPINNER_INTERVAL = HEARTBEAT_INTERVAL
    SPINNER_FRAME_TICKS = HEARTBEAT_FRAME_TICKS

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

    def banner(self, title, start_date, end_date, db_path, end_date_warning=None):
        self._emit_line(self._style(title, "header"))
        self._emit_line("  range : {} -> {}".format(start_date, end_date))
        self._emit_line("  cache : {}".format(db_path))
        if end_date_warning:
            self._emit_line(self._style("  warn  : {}".format(end_date_warning), "warn"))
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
            message = "{} | {}".format(message, self._style(detail, "skip"))
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
            "warn": "\033[1;33m",
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
            line = None
            with self._lock:
                if self._current_state != "RUN":
                    continue
                tick += 1
                if tick % self.SPINNER_FRAME_TICKS == 0:
                    self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_FRAMES)
                line = self._format_line(
                    self._current_state,
                    self._current_message,
                    spinner=self._spinner_frame(),
                )
            if line is not None:
                self._render(line)


if __name__ == "__main__":
    main()
