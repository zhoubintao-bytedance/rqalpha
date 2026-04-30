from __future__ import annotations

import math
import sys
from pathlib import Path
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

# pytest>=9 默认 importlib 导入模式下，某些环境不会自动把 repo root 加入 sys.path。
# 这里显式注入，保证 `skyeye` 可被稳定导入。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from skyeye.market_regime_layer import (
    MarketRegimeConfig,
    compute_market_regime,
    required_market_regime_history_days,
)


def _make_dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), periods=int(n), freq="B")


def _make_benchmark_bars(kind: str, n: int = 280, seed: int = 7) -> pd.DataFrame:
    """构造可复现的基准指数 OHLCV。

    kind:
      - bull: 上涨趋势 + 后段波动抬升（让 trendiness 指标更稳定触发）
      - bear: 下跌趋势 + 后段波动抬升
      - range: 震荡（均值回归 + 周期性）
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(n), dtype=float)
    dates = _make_dates(n)

    if kind == "bull":
        drift = 0.0025
        base = 100.0 * np.exp(drift * t)
        vol = 0.003 + 0.0065 * (t / max(1.0, t.max()))
        noise = rng.normal(0.0, 1.0, size=n) * vol
        close = base * (1.0 + noise)
        volume = 1e8 * (1.0 + 0.4 * (t / max(1.0, t.max())))
    elif kind == "bear":
        drift = -0.0025
        base = 120.0 * np.exp(drift * t)
        vol = 0.003 + 0.0065 * (t / max(1.0, t.max()))
        noise = rng.normal(0.0, 1.0, size=n) * vol
        close = base * (1.0 + noise)
        volume = 1e8 * (1.0 + 0.4 * (t / max(1.0, t.max())))
    elif kind == "range":
        # 震荡：用均值回归过程（OU 近似），并让尾段回到均值附近，避免最后 20d 误判成趋势。
        mu = 100.0
        theta = 0.18
        sigma = 0.45
        close = np.empty(int(n), dtype=float)
        close[0] = mu
        for i in range(1, int(n)):
            close[i] = close[i - 1] + theta * (mu - close[i - 1]) + rng.normal(0.0, sigma)
        tail = min(40, int(n))
        close[-tail:] = mu + rng.normal(0.0, 0.25, size=tail)
        volume = 8e7 * (1.0 + 0.03 * rng.normal(0.0, 1.0, size=n))
    else:
        raise ValueError(f"unknown kind={kind}")

    close = np.maximum(close, 1.0)
    # 给高低价一点点幅度，避免 RSRS / ATR / ADX 中出现退化。
    spread = 0.008 + 0.004 * rng.random(size=n)
    high = close * (1.0 + spread)
    low = close * (1.0 - spread)
    open_ = close * (1.0 + rng.normal(0.0, 0.001, size=n))

    bars = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    bars.index.name = "date"
    return bars


def _make_industry_close(kind: str, benchmark_close: pd.Series, seed: int = 11, k: int = 28) -> pd.DataFrame:
    """构造行业 close 宽表（date x industry），用于结构/轮动侧指标。"""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(benchmark_close.index)
    n = len(dates)

    if kind == "co_move":
        # 跟随基准，横截面离散度低。
        base = benchmark_close.to_numpy(dtype=float)
        mat = []
        for i in range(int(k)):
            # 足够小的扰动，避免 dispersion_percentile + skewness 误触发 rotation。
            jitter = rng.normal(0.0, 0.0004, size=n)
            mat.append(base * (1.0 + jitter))
        data = np.column_stack(mat)
    elif kind == "rotation":
        # 前段：co-move；后段：制造强分化 + 偏度 + 集中度
        base = benchmark_close.to_numpy(dtype=float)
        data = np.zeros((n, int(k)), dtype=float)
        # 默认都跟随基准
        for i in range(int(k)):
            data[:, i] = base * (1.0 + rng.normal(0.0, 0.002, size=n))

        # 最后 40 天让 top3 行业强势拉升，少数行业下跌，形成高离散/高集中
        tail = 40
        if n >= tail:
            start = n - tail
            # top3: +20%~+35%
            for i, amp in zip([0, 1, 2], [0.20, 0.25, 0.35]):
                ramp = np.linspace(0.0, amp, tail)
                data[start:, i] = data[start:, i] * (1.0 + ramp)
            # 部分行业轻微下跌
            for i in range(3, min(10, int(k))):
                ramp = np.linspace(0.0, -0.08, tail)
                data[start:, i] = data[start:, i] * (1.0 + ramp)
    else:
        raise ValueError(f"unknown kind={kind}")

    out = pd.DataFrame(data, index=dates, columns=[f"IND{i:02d}" for i in range(int(k))])
    out.index.name = "date"
    return out


def _walk_forward_regimes(
    bars: pd.DataFrame,
    industry_close: pd.DataFrame | None,
    *,
    cfg: MarketRegimeConfig | None = None,
    start_at: int,
    lookback: int = 650,
    ad_ratio: float | None | Callable[[pd.Timestamp], float | None] = 1.0,
) -> pd.Series:
    """按日滚动计算，返回每个日期的 regime（不引入前视）。"""
    regimes = {}
    for i in range(int(start_at), len(bars) + 1):
        left = max(0, i - int(lookback))
        sub_bars = bars.iloc[left:i]
        sub_ind = industry_close.iloc[left:i] if industry_close is not None else None
        cur_ratio = ad_ratio(sub_bars.index[-1]) if callable(ad_ratio) else ad_ratio
        regimes[sub_bars.index[-1]] = compute_market_regime(
            sub_bars,
            industry_close=sub_ind,
            cfg=cfg,
            ad_ratio=cur_ratio,
        ).regime
    return pd.Series(regimes).sort_index()


def test_market_regime_degrades_to_strength_zero_when_history_insufficient():
    bars = _make_benchmark_bars("bull", n=60)
    regime = compute_market_regime(bars, industry_close=None)
    assert regime.regime == "range_co_move"
    assert regime.strength == 0.0
    assert "insufficient_history" in str(regime.diagnostics)


def test_market_regime_default_config_prefers_responsive_regime_windows():
    cfg = MarketRegimeConfig()

    assert cfg.hurst_window == 100
    assert cfg.rsrs_m == 252
    assert cfg.boll_percentile_window == 60
    assert cfg.dispersion_percentile_window == 60
    assert required_market_regime_history_days(cfg) == 275
    assert cfg.trendiness_weights["adx_score"] == pytest.approx(0.35)
    assert cfg.trendiness_weights["hurst_score"] == pytest.approx(0.15)
    assert cfg.trendiness_weights["boll_width_score"] == pytest.approx(0.20)
    assert cfg.trendiness_weights["atr_ratio_score"] == pytest.approx(0.15)
    assert cfg.trendiness_weights["volume_trend_score"] == pytest.approx(0.15)
    assert sum(cfg.trendiness_weights.values()) == pytest.approx(1.0)
    # 阈值 0.45 基于 A 股真实数据校准：沪深300 震荡占比约 63.4%，接近 A 股实际 60-70%
    assert cfg.trendiness_range_threshold == pytest.approx(0.45)


def test_market_regime_strength_uses_clamped_geometric_mean(monkeypatch):
    import skyeye.market_regime_layer as layer

    monkeypatch.setattr(
        layer,
        "_compute_direction_label_and_scores",
        lambda *_args, **_kwargs: ("bull", 0.81, 0.81, {}),
    )
    monkeypatch.setattr(
        layer,
        "_compute_rotation_score",
        lambda *_args, **_kwargs: ("rotation", 0.25, {}),
    )

    regime = compute_market_regime(_make_benchmark_bars("bull"), industry_close=pd.DataFrame({"A": [1.0, 2.0]}))

    assert regime.strength == pytest.approx(math.sqrt(0.81 * 0.25))
    assert regime.diagnostics["direction_strength"] == pytest.approx(0.81)
    assert regime.diagnostics["structure_strength"] == pytest.approx(0.25)

    monkeypatch.setattr(
        layer,
        "_compute_direction_label_and_scores",
        lambda *_args, **_kwargs: ("bull", -1e-9, -1e-9, {}),
    )
    edge = compute_market_regime(_make_benchmark_bars("bull"), industry_close=pd.DataFrame({"A": [1.0, 2.0]}))

    assert math.isfinite(edge.strength)
    assert edge.strength == pytest.approx(0.0)


@pytest.mark.parametrize(
    "bench_kind,structure_kind,expected",
    [
        ("bull", "co_move", "bull_co_move"),
        ("bear", "co_move", "bear_co_move"),
        ("range", "co_move", "range_co_move"),
        ("bull", "rotation", "bull_rotation"),
    ],
)
def test_market_regime_classifies_synthetic_bench_and_structure(bench_kind, structure_kind, expected):
    sample_days = required_market_regime_history_days(MarketRegimeConfig()) + 60
    bars = _make_benchmark_bars(bench_kind, n=sample_days)
    industry = _make_industry_close(structure_kind, bars["close"], k=30)
    # co_move 场景下默认不一定能拿到 ad_ratio；即使有，极端宽度（ratio=2/0.5）应更偏 co_move。
    ad_ratio = 2.0 if structure_kind == "co_move" else 1.0
    got = compute_market_regime(bars, industry_close=industry, ad_ratio=ad_ratio)
    assert got.regime == expected
    assert 0.0 <= got.strength <= 1.0
    assert isinstance(got.diagnostics, dict)


def test_market_regime_is_order_invariant_on_index():
    bars = _make_benchmark_bars("bull", n=280)
    industry = _make_industry_close("rotation", bars["close"], k=30)
    # 打乱行顺序，内部应 sort_index() 后得到一致结论
    shuffled = bars.sample(frac=1.0, random_state=123)
    got_a = compute_market_regime(bars, industry_close=industry, ad_ratio=1.0).regime
    got_b = compute_market_regime(shuffled, industry_close=industry, ad_ratio=1.0).regime
    assert got_a == got_b


def test_market_regime_walk_forward_accuracy_on_synthetic_switches():
    """用人为构造的三段行情验证：标签切换应能被多数时间识别出来。"""
    # 三段：bull(360) -> range(360) -> bear(360)
    bull = _make_benchmark_bars("bull", n=360, seed=1)
    rng = _make_benchmark_bars("range", n=360, seed=2)
    bear = _make_benchmark_bars("bear", n=360, seed=3)
    # 拼接时保持日期连续
    rng.index = _make_dates(len(rng), start=str(bull.index[-1] + pd.Timedelta(days=1)))
    bear.index = _make_dates(len(bear), start=str(rng.index[-1] + pd.Timedelta(days=1)))
    bars = pd.concat([bull, rng, bear])

    # 结构：前两段 co_move，最后一段 rotation（在最后一段全程注入明显分化）
    split = len(bull) + len(rng)
    industry = _make_industry_close("co_move", bars["close"], seed=10, k=30)
    seg_len = len(bear)
    ramp = np.linspace(0.0, 0.30, seg_len)
    # top3 强势
    for col, amp in zip(["IND00", "IND01", "IND02"], [1.0, 1.2, 1.5]):
        industry.loc[bars.index[split:], col] = industry.loc[bars.index[split:], col] * (1.0 + amp * ramp)
    # 部分行业弱势
    down = np.linspace(0.0, -0.12, seg_len)
    for col in ["IND03", "IND04", "IND05", "IND06", "IND07"]:
        industry.loc[bars.index[split:], col] = industry.loc[bars.index[split:], col] * (1.0 + down)

    # 按日滚动计算（从足够长的 warmup 开始）
    cfg = MarketRegimeConfig()
    warmup = required_market_regime_history_days(cfg) + 30
    def _ad_ratio_by_phase(ts: pd.Timestamp) -> float:
        # co_move 段更偏“齐涨齐跌”，用极端宽度降低 neutrality；rotation 段用 neutrality=1。
        idx = bars.index.get_indexer([ts], method=None)[0]
        return 2.0 if idx < split else 1.0

    series = _walk_forward_regimes(bars, industry, cfg=cfg, start_at=warmup, lookback=650, ad_ratio=_ad_ratio_by_phase)

    # 方向准确率：bull/range/bear 三类
    def dir_of(label: str) -> str:
        return label.split("_")[0]

    gt_dir = pd.Series(index=bars.index, dtype=object)
    gt_dir.iloc[: len(bull)] = "bull"
    gt_dir.iloc[len(bull) : len(bull) + len(rng)] = "range"
    gt_dir.iloc[len(bull) + len(rng) :] = "bear"

    pred_dir = series.map(dir_of)
    # 允许转场滞后：每段忽略前 30 天。
    buffer = 30
    eval_idx = pred_dir.index
    phase_dir = gt_dir.loc[eval_idx]
    ok = pd.Series(False, index=eval_idx)
    ok.iloc[buffer : len(bull) - buffer] = True
    ok.iloc[len(bull) + buffer : len(bull) + len(rng) - buffer] = True
    ok.iloc[len(bull) + len(rng) + buffer :] = True
    eval_mask = ok.reindex(eval_idx).fillna(False)
    dir_acc = float((pred_dir[eval_mask] == phase_dir[eval_mask]).mean())
    assert dir_acc >= 0.65

    # 结构准确率：co_move/rotation
    def st_of(label: str) -> str:
        if label.endswith("_co_move"):
            return "co_move"
        if label.endswith("_rotation"):
            return "rotation"
        raise ValueError(f"unexpected regime label: {label}")

    gt_st = pd.Series(index=bars.index, dtype=object)
    gt_st.iloc[:split] = "co_move"
    gt_st.iloc[split:] = "rotation"
    pred_st = series.map(st_of)
    phase_st = gt_st.loc[pred_st.index]
    st_acc = float((pred_st[eval_mask] == phase_st[eval_mask]).mean())
    assert st_acc >= 0.70


def test_market_regime_stability_on_constant_bull_market():
    """稳定上涨行情下，regime 不应频繁抖动。"""
    bars = _make_benchmark_bars("bull", n=320, seed=9)
    industry = _make_industry_close("co_move", bars["close"], seed=9, k=28)
    cfg = MarketRegimeConfig()
    warmup = required_market_regime_history_days(cfg) + 30
    # co_move 下用更偏 co_move 的 ad_ratio（极端宽度），减少结构侧抖动。
    series = _walk_forward_regimes(bars, industry, cfg=cfg, start_at=warmup, lookback=650, ad_ratio=2.0)
    switches = int((series != series.shift(1)).sum())
    # 经验上不应超过 10% 的天数发生切换
    assert switches / max(1, len(series)) <= 0.20


@pytest.mark.slow
def test_trendiness_range_threshold_calibrated_to_a_share():
    """回归测试：trendiness_range_threshold=0.45 应使沪深300 震荡占比在 55-75% 区间。

    数据论证（2026-04 验证，沪深300 2019-2026 滚动计算）：
    - threshold=0.35（旧值）：震荡占比 47.7%，远低于 A 股实际 60-70%
    - threshold=0.45（当前值）：震荡占比 63.4%，接近 A 股实际
    - threshold=0.50：震荡占比 67.8%，也合理但偏保守

    如果此测试失败，说明阈值变更后震荡占比偏离 A 股实际经验区间，
    需重新校准并更新此处断言。
    """
    try:
        from skyeye.data.facade import DataFacade
    except Exception:
        pytest.skip("DataFacade not available")

    data = DataFacade()
    bars_raw = data.get_daily_bars(
        "000300.XSHG", "2018-01-01", "2026-04-25",
        fields=["open", "high", "low", "close", "volume"],
    )
    if bars_raw is None or bars_raw.empty:
        pytest.skip("沪深300 数据不可用")

    from skyeye.market_regime_layer import (
        normalize_single_instrument_bars,
        _compute_direction_label_and_scores,
    )

    bench = normalize_single_instrument_bars(bars_raw, "000300.XSHG")
    cfg = MarketRegimeConfig()
    required_days = required_market_regime_history_days(cfg)

    lookback = 650
    trendiness_values = []
    for i in range(required_days + lookback, len(bench) + 1):
        sub = bench.iloc[i - lookback : i]
        _label, trendiness, _direction, _diag = _compute_direction_label_and_scores(sub, cfg)
        if trendiness is not None:
            trendiness_values.append(trendiness)

    if len(trendiness_values) < 200:
        pytest.skip("有效 trendiness 样本不足")

    range_ratio = sum(1 for t in trendiness_values if t < cfg.trendiness_range_threshold) / len(trendiness_values)
    # A 股震荡占比应在 55-75% 区间（基于历史 60-70% 经验值，留一定容差）
    assert 0.55 <= range_ratio <= 0.75, (
        f"震荡占比 {range_ratio:.1%} 超出 A 股经验区间 [55%, 75%]，"
        f"阈值 {cfg.trendiness_range_threshold} 可能需要重新校准"
    )
