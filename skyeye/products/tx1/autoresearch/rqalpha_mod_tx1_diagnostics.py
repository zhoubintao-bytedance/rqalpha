# -*- coding: utf-8 -*-

"""TX1 autoresearch 用的轻量回测诊断 mod。"""

from __future__ import annotations

from typing import Any

from rqalpha.interface import AbstractMod


__config__ = {}


def load_mod():
    """返回 TX1 诊断 mod 实例。"""
    return Tx1DiagnosticsMod()


class Tx1DiagnosticsMod(AbstractMod):
    """在每个回测窗口结束时导出 TX1 策略上下文诊断。"""

    def __init__(self):
        """初始化环境引用。"""
        self._env = None

    def start_up(self, env, mod_config):
        """保存环境对象，供 tear_down 阶段读取用户上下文。"""
        del mod_config
        self._env = env

    def tear_down(self, code, exception=None):
        """在环境销毁前提取 TX1 执行层诊断。"""
        if self._env is None or getattr(self._env, "user_strategy", None) is None:
            return {
                "strategy_diagnostics": {},
                "runtime": {},
                "last_signal_date": None,
                "pending_turnover": 0.0,
                "had_pending_target": False,
                "exception": str(exception) if exception else None,
                "exit_code": str(code),
            }

        user_context = self._env.user_strategy.user_context
        runtime = dict(getattr(user_context, "tx1_runtime", {}) or {})
        profile = dict(runtime.get("profile") or {})
        signal_book = dict(getattr(user_context, "tx1_signal_book", {}) or {})
        diagnostics = dict(getattr(user_context, "tx1_diagnostics", {}) or {})

        # 只导出 autoresearch 需要的最小诊断集，避免把运行时对象整包写进结果。
        return {
            "strategy_diagnostics": diagnostics,
            "runtime": {
                "strategy_id": runtime.get("strategy_id"),
                "artifact_line_id": runtime.get("artifact_line_id"),
                "benchmark": runtime.get("benchmark"),
                "profile": profile.get("profile"),
                "signal_count": int(len(signal_book)),
            },
            "last_signal_date": getattr(user_context, "tx1_pending_signal_date", None),
            "pending_turnover": float(
                getattr(user_context, "tx1_pending_turnover", 0.0) or 0.0
            ),
            "had_pending_target": bool(
                getattr(user_context, "tx1_pending_target", None)
            ),
            "exception": str(exception) if exception else None,
            "exit_code": str(code),
        }
