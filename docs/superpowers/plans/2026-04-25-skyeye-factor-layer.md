# SkyEye 因子层 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改动现有 `market_regime_layer.py` 的前提下，实现独立的 `skyeye.factor_layer`，支持按 `MarketRegime` 输出两类核心因子及其历史分位。

**Architecture:** 新增 `skyeye/factor_layer/` 子包，按 `config / result / utils / indicators / core` 拆分职责。`compute_factors(...)` 负责纯函数计算和 regime 映射，`compute_factors_from_data_facade(...)` 负责取数和按需复用现有 market regime facade，单指标统一先生成完整历史序列，再由工具函数提取 `value` 与 `percentile`。

**Tech Stack:** Python 3、pandas、numpy、pytest、现有 `skyeye.data.facade` 与 `skyeye.market_regime_layer`

---

## File Map

- Create: `skyeye/factor_layer/__init__.py`
  - 责任：暴露 `FactorLayerConfig`、`FactorLayerResult`、`compute_factors`、`compute_factors_from_data_facade`
- Create: `skyeye/factor_layer/config.py`
  - 责任：定义配置 dataclass
- Create: `skyeye/factor_layer/result.py`
  - 责任：定义 `FactorValue` 与 `FactorLayerResult`
- Create: `skyeye/factor_layer/utils.py`
  - 责任：bars 规范化、分位计算、因子构造包装、共享数值函数
- Create: `skyeye/factor_layer/core.py`
  - 责任：regime 映射、类别选择、入口函数、facade 协作
- Create: `skyeye/factor_layer/indicators/__init__.py`
- Create: `skyeye/factor_layer/indicators/trend.py`
  - 责任：`MACD`、`ADX`、`RSRS`、`MA_*`、`EMA_*`
- Create: `skyeye/factor_layer/indicators/momentum.py`
  - 责任：`MOM`、`ROC`、`BIAS`
- Create: `skyeye/factor_layer/indicators/oscillator.py`
  - 责任：`RSI`、`KDJ_*`、`CCI`
- Create: `skyeye/factor_layer/indicators/volatility.py`
  - 责任：`ATR`、`BBANDS_*`、`DC_*`
- Create: `skyeye/factor_layer/indicators/volume.py`
  - 责任：`OBV`、`OBV_MA`、`MFI`
- Test: `tests/unittest/test_skyeye_factor_layer.py`
  - 责任：结果结构、regime 映射、降级、percentile 范围、facade 行为

## Task 1: 建立包结构和结果骨架

**Files:**
- Create: `skyeye/factor_layer/__init__.py`
- Create: `skyeye/factor_layer/config.py`
- Create: `skyeye/factor_layer/result.py`
- Create: `skyeye/factor_layer/core.py`
- Test: `tests/unittest/test_skyeye_factor_layer.py`

- [ ] **Step 1: 写失败测试，锁住 regime 映射和空结果结构**

```python
from skyeye.market_regime_layer import MarketRegime
from skyeye.factor_layer.core import _select_factor_categories, _empty_result


def test_select_factor_categories_matches_design_doc():
    assert _select_factor_categories("bull_co_move") == ("trend", "volume")
    assert _select_factor_categories("bull_rotation") == ("momentum", "trend")
    assert _select_factor_categories("range_co_move") == ("oscillator", "volatility")
    assert _select_factor_categories("range_rotation") == ("momentum", "oscillator")
    assert _select_factor_categories("bear_co_move") == ("volatility", "volume")
    assert _select_factor_categories("bear_rotation") == ("momentum", "volatility")


def test_empty_result_preserves_regime_and_empty_buckets():
    regime = MarketRegime(regime="range_co_move", strength=0.4)
    result = _empty_result(regime)

    assert result.regime is regime
    assert result.trend_factors == {}
    assert result.momentum_factors == {}
    assert result.oscillator_factors == {}
    assert result.volatility_factors == {}
    assert result.volume_factors == {}
```

- [ ] **Step 2: 跑测试确认当前失败**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/unittest/test_skyeye_factor_layer.py::test_select_factor_categories_matches_design_doc -q
```

Expected: FAIL，提示 `skyeye.factor_layer` 模块不存在。

- [ ] **Step 3: 创建配置、结果对象和最小 core 骨架**

```python
# skyeye/factor_layer/config.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorLayerConfig:
    percentile_window: int = 252
    rsi_period: int = 14
    kdj_n: int = 9
    kdj_m1: int = 3
    kdj_m2: int = 3
    cci_period: int = 20
    mom_period: int = 10
    roc_period: int = 10
    bias_period: int = 6
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ma_periods: tuple[int, ...] = (5, 10, 20, 60)
    ema_periods: tuple[int, ...] = (12, 26)
    boll_period: int = 20
    boll_std: float = 2.0
    dc_period: int = 20
    atr_period: int = 14
    obv_ma_period: int = 20
    mfi_period: int = 14
```

```python
# skyeye/factor_layer/result.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from skyeye.market_regime_layer import MarketRegime


FactorValue: TypeAlias = dict[str, float]


@dataclass(frozen=True)
class FactorLayerResult:
    regime: MarketRegime
    trend_factors: dict[str, FactorValue] = field(default_factory=dict)
    momentum_factors: dict[str, FactorValue] = field(default_factory=dict)
    oscillator_factors: dict[str, FactorValue] = field(default_factory=dict)
    volatility_factors: dict[str, FactorValue] = field(default_factory=dict)
    volume_factors: dict[str, FactorValue] = field(default_factory=dict)
```

```python
# skyeye/factor_layer/core.py
from __future__ import annotations

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.result import FactorLayerResult
from skyeye.market_regime_layer import MarketRegime

REGIME_FACTOR_CATEGORIES = {
    "bull_co_move": ("trend", "volume"),
    "bull_rotation": ("momentum", "trend"),
    "range_co_move": ("oscillator", "volatility"),
    "range_rotation": ("momentum", "oscillator"),
    "bear_co_move": ("volatility", "volume"),
    "bear_rotation": ("momentum", "volatility"),
}


def _select_factor_categories(regime_label: str) -> tuple[str, str]:
    return REGIME_FACTOR_CATEGORIES[str(regime_label)]


def _empty_result(regime: MarketRegime) -> FactorLayerResult:
    return FactorLayerResult(regime=regime)


def compute_factors(benchmark_bars, regime: MarketRegime, cfg: FactorLayerConfig | None = None) -> FactorLayerResult:
    return _empty_result(regime)
```

```python
# skyeye/factor_layer/__init__.py
from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.core import compute_factors, compute_factors_from_data_facade
from skyeye.factor_layer.result import FactorLayerResult

__all__ = [
    "FactorLayerConfig",
    "FactorLayerResult",
    "compute_factors",
    "compute_factors_from_data_facade",
]
```

- [ ] **Step 4: 在 core 中补最小 facade 占位入口，保证包可导入**

```python
def compute_factors_from_data_facade(
    end_date,
    benchmark_id: str = "000300.XSHG",
    regime: MarketRegime | None = None,
    cfg: FactorLayerConfig | None = None,
):
    raise NotImplementedError("implemented in Task 3")
```

- [ ] **Step 5: 跑定向测试确认骨架通过**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/unittest/test_skyeye_factor_layer.py::test_empty_result_preserves_regime_and_empty_buckets -q
```

Expected: PASS。

- [ ] **Step 6: 提交骨架任务**

```bash
git add skyeye/factor_layer/__init__.py skyeye/factor_layer/config.py skyeye/factor_layer/result.py skyeye/factor_layer/core.py tests/unittest/test_skyeye_factor_layer.py
git commit -m "feat: add skyeye factor layer skeleton"
```

## Task 2: 实现公共工具与核心指标类别函数

**Files:**
- Create: `skyeye/factor_layer/utils.py`
- Create: `skyeye/factor_layer/indicators/__init__.py`
- Create: `skyeye/factor_layer/indicators/trend.py`
- Create: `skyeye/factor_layer/indicators/momentum.py`
- Create: `skyeye/factor_layer/indicators/oscillator.py`
- Create: `skyeye/factor_layer/indicators/volatility.py`
- Create: `skyeye/factor_layer/indicators/volume.py`
- Test: `tests/unittest/test_skyeye_factor_layer.py`

- [ ] **Step 1: 写失败测试，锁住 value/percentile 包装与基础动量指标**

```python
import math
import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.indicators.momentum import compute_momentum_factors
from skyeye.factor_layer.utils import build_factor_value


def test_build_factor_value_returns_latest_value_and_percentile():
    history = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2024-01-01", periods=4, freq="B"))
    payload = build_factor_value(history, window=4)

    assert payload["value"] == 4.0
    assert 0.0 <= payload["percentile"] <= 1.0


def test_compute_momentum_factors_returns_mom_roc_bias():
    bars = pd.DataFrame(
        {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]},
        index=pd.date_range("2024-01-01", periods=12, freq="B"),
    )
    factors = compute_momentum_factors(
        bars,
        FactorLayerConfig(mom_period=3, roc_period=3, bias_period=3, percentile_window=8),
    )

    assert set(factors) == {"MOM", "ROC", "BIAS"}
    assert factors["MOM"]["value"] > 0
    assert factors["ROC"]["value"] > 0
    assert not math.isnan(factors["BIAS"]["value"])
```

- [ ] **Step 2: 跑定向测试确认失败**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/unittest/test_skyeye_factor_layer.py::test_compute_momentum_factors_returns_mom_roc_bias -q
```

Expected: FAIL，提示 `utils.py` 或 `compute_momentum_factors` 不存在。

- [ ] **Step 3: 实现 `utils.py` 的规范化、分位和共享数值函数**

```python
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def normalize_benchmark_bars(bars: pd.DataFrame | None) -> pd.DataFrame:
    if bars is None or getattr(bars, "empty", True):
        return pd.DataFrame()
    out = bars.copy()
    out.columns = [str(col).lower() for col in out.columns]
    if not isinstance(out.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.index.name = "date"
    return out


def ema(series: pd.Series, span: int) -> pd.Series:
    return pd.Series(series, dtype=float).ewm(span=int(span), adjust=False).mean()


def wilder_smooth(values: pd.Series | np.ndarray, window: int) -> pd.Series:
    x = pd.Series(values, dtype=float)
    return x.ewm(alpha=1.0 / float(window), adjust=False, min_periods=int(window)).mean()


def percentile_rank(window_values: pd.Series, current: float) -> float:
    valid = pd.Series(window_values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return math.nan
    return float((valid <= float(current)).mean())


def build_factor_value(history: pd.Series, window: int) -> dict[str, float] | None:
    valid = pd.Series(history, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return None
    current = float(valid.iloc[-1])
    lookback = valid.iloc[-int(window):]
    percentile = math.nan if len(lookback) < min(int(window), 20) else percentile_rank(lookback, current)
    return {"value": current, "percentile": percentile}


def add_factor_from_series(target: dict[str, dict[str, float]], name: str, history: pd.Series, window: int) -> None:
    payload = build_factor_value(history, window=window)
    if payload is not None:
        target[name] = payload
```

- [ ] **Step 4: 实现五个指标模块的最小可用版本**

```python
# skyeye/factor_layer/indicators/momentum.py
from __future__ import annotations

import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series


def compute_momentum_factors(bars: pd.DataFrame, cfg: FactorLayerConfig) -> dict[str, dict[str, float]]:
    close = pd.Series(bars["close"], dtype=float)
    factors: dict[str, dict[str, float]] = {}
    add_factor_from_series(factors, "MOM", close - close.shift(cfg.mom_period), cfg.percentile_window)
    add_factor_from_series(factors, "ROC", close / close.shift(cfg.roc_period) - 1.0, cfg.percentile_window)
    add_factor_from_series(
        factors,
        "BIAS",
        close / close.rolling(cfg.bias_period).mean() - 1.0,
        cfg.percentile_window,
    )
    return factors
```

```python
# skyeye/factor_layer/indicators/trend.py
from __future__ import annotations

import numpy as np
import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series, ema, wilder_smooth


def compute_trend_factors(bars: pd.DataFrame, cfg: FactorLayerConfig) -> dict[str, dict[str, float]]:
    close = pd.Series(bars["close"], dtype=float)
    factors: dict[str, dict[str, float]] = {}
    macd = ema(close, cfg.macd_fast) - ema(close, cfg.macd_slow)
    signal = ema(macd, cfg.macd_signal)
    add_factor_from_series(factors, "MACD", macd - signal, cfg.percentile_window)
    for period in cfg.ma_periods:
        add_factor_from_series(factors, f"MA_{period}", close.rolling(period).mean(), cfg.percentile_window)
    for period in cfg.ema_periods:
        add_factor_from_series(factors, f"EMA_{period}", ema(close, period), cfg.percentile_window)
    if {"high", "low"}.issubset(bars.columns):
        high = pd.Series(bars["high"], dtype=float)
        low = pd.Series(bars["low"], dtype=float)
        prev_close = close.shift(1)
        tr = pd.Series(np.maximum.reduce([(high - low), (high - prev_close).abs(), (low - prev_close).abs()]))
        plus_dm = (high.diff()).where(lambda x: x > 0, 0.0)
        minus_dm = (-low.diff()).where(lambda x: x > 0, 0.0)
        atr = wilder_smooth(tr, cfg.atr_period)
        plus_di = 100.0 * wilder_smooth(plus_dm, cfg.atr_period) / atr
        minus_di = 100.0 * wilder_smooth(minus_dm, cfg.atr_period) / atr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        add_factor_from_series(factors, "ADX", wilder_smooth(dx, cfg.atr_period), cfg.percentile_window)
        rsrs = (high.rolling(18).mean() / low.rolling(18).mean()).replace([np.inf, -np.inf], np.nan)
        add_factor_from_series(factors, "RSRS", (rsrs - rsrs.rolling(60).mean()) / rsrs.rolling(60).std(ddof=0), cfg.percentile_window)
    return factors
```

```python
# skyeye/factor_layer/indicators/oscillator.py
from __future__ import annotations

import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series, wilder_smooth


def compute_oscillator_factors(bars: pd.DataFrame, cfg: FactorLayerConfig) -> dict[str, dict[str, float]]:
    close = pd.Series(bars["close"], dtype=float)
    factors: dict[str, dict[str, float]] = {}
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    rs = wilder_smooth(gain, cfg.rsi_period) / wilder_smooth(loss, cfg.rsi_period)
    add_factor_from_series(factors, "RSI", 100.0 - 100.0 / (1.0 + rs), cfg.percentile_window)
    if {"high", "low"}.issubset(bars.columns):
        high = pd.Series(bars["high"], dtype=float)
        low = pd.Series(bars["low"], dtype=float)
        rsv = (close - low.rolling(cfg.kdj_n).min()) / (high.rolling(cfg.kdj_n).max() - low.rolling(cfg.kdj_n).min()) * 100.0
        k = rsv.ewm(alpha=1.0 / cfg.kdj_m1, adjust=False).mean()
        d = k.ewm(alpha=1.0 / cfg.kdj_m2, adjust=False).mean()
        add_factor_from_series(factors, "KDJ_K", k, cfg.percentile_window)
        add_factor_from_series(factors, "KDJ_D", d, cfg.percentile_window)
        add_factor_from_series(factors, "KDJ_J", 3.0 * k - 2.0 * d, cfg.percentile_window)
        tp = (high + low + close) / 3.0
        ma = tp.rolling(cfg.cci_period).mean()
        md = (tp - ma).abs().rolling(cfg.cci_period).mean()
        add_factor_from_series(factors, "CCI", (tp - ma) / (0.015 * md), cfg.percentile_window)
    return factors
```

```python
# skyeye/factor_layer/indicators/volatility.py
from __future__ import annotations

import numpy as np
import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series, wilder_smooth


def compute_volatility_factors(bars: pd.DataFrame, cfg: FactorLayerConfig) -> dict[str, dict[str, float]]:
    close = pd.Series(bars["close"], dtype=float)
    factors: dict[str, dict[str, float]] = {}
    mean = close.rolling(cfg.boll_period).mean()
    std = close.rolling(cfg.boll_period).std(ddof=0)
    upper = mean + cfg.boll_std * std
    lower = mean - cfg.boll_std * std
    add_factor_from_series(factors, "BBANDS_UPPER", upper, cfg.percentile_window)
    add_factor_from_series(factors, "BBANDS_MIDDLE", mean, cfg.percentile_window)
    add_factor_from_series(factors, "BBANDS_LOWER", lower, cfg.percentile_window)
    add_factor_from_series(factors, "BBANDS_WIDTH", (upper - lower) / mean, cfg.percentile_window)
    if {"high", "low"}.issubset(bars.columns):
        high = pd.Series(bars["high"], dtype=float)
        low = pd.Series(bars["low"], dtype=float)
        prev_close = close.shift(1)
        tr = pd.Series(np.maximum.reduce([(high - low), (high - prev_close).abs(), (low - prev_close).abs()]))
        add_factor_from_series(factors, "ATR", wilder_smooth(tr, cfg.atr_period), cfg.percentile_window)
        dc_upper = high.rolling(cfg.dc_period).max()
        dc_lower = low.rolling(cfg.dc_period).min()
        add_factor_from_series(factors, "DC_UPPER", dc_upper, cfg.percentile_window)
        add_factor_from_series(factors, "DC_LOWER", dc_lower, cfg.percentile_window)
        add_factor_from_series(factors, "DC_MIDDLE", (dc_upper + dc_lower) / 2.0, cfg.percentile_window)
    return factors
```

```python
# skyeye/factor_layer/indicators/volume.py
from __future__ import annotations

import numpy as np
import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series


def compute_volume_factors(bars: pd.DataFrame, cfg: FactorLayerConfig) -> dict[str, dict[str, float]]:
    if "volume" not in bars.columns:
        return {}
    close = pd.Series(bars["close"], dtype=float)
    volume = pd.Series(bars["volume"], dtype=float)
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    factors: dict[str, dict[str, float]] = {}
    add_factor_from_series(factors, "OBV", obv, cfg.percentile_window)
    add_factor_from_series(factors, "OBV_MA", obv.rolling(cfg.obv_ma_period).mean(), cfg.percentile_window)
    if {"high", "low"}.issubset(bars.columns):
        high = pd.Series(bars["high"], dtype=float)
        low = pd.Series(bars["low"], dtype=float)
        tp = (high + low + close) / 3.0
        raw_flow = tp * volume
        pos_flow = raw_flow.where(tp.diff() > 0, 0.0).rolling(cfg.mfi_period).sum()
        neg_flow = raw_flow.where(tp.diff() < 0, 0.0).rolling(cfg.mfi_period).sum().abs()
        add_factor_from_series(factors, "MFI", 100.0 - 100.0 / (1.0 + pos_flow / neg_flow), cfg.percentile_window)
    return factors
```

- [ ] **Step 5: 跑类别级定向测试并修正缺口**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/unittest/test_skyeye_factor_layer.py::test_build_factor_value_returns_latest_value_and_percentile -q
PYTHONPATH="$PWD" pytest tests/unittest/test_skyeye_factor_layer.py::test_compute_momentum_factors_returns_mom_roc_bias -q
```

Expected: PASS。

- [ ] **Step 6: 补回归测试，锁住缺量与缺高低价时的降级**

```python
from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.indicators.volume import compute_volume_factors
from skyeye.factor_layer.indicators.volatility import compute_volatility_factors


def test_volume_factors_return_empty_when_volume_missing():
    bars = pd.DataFrame({"close": [1, 2, 3, 4]}, index=pd.date_range("2024-01-01", periods=4, freq="B"))
    assert compute_volume_factors(bars, FactorLayerConfig()) == {}


def test_volatility_factors_still_return_bbands_without_high_low():
    bars = pd.DataFrame({"close": range(1, 40)}, index=pd.date_range("2024-01-01", periods=39, freq="B"))
    factors = compute_volatility_factors(bars, FactorLayerConfig(percentile_window=20))
    assert "BBANDS_WIDTH" in factors
    assert "ATR" not in factors
```

- [ ] **Step 7: 跑测试文件到当前通过**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/unittest/test_skyeye_factor_layer.py -q
```

Expected: PASS。

- [ ] **Step 8: 提交指标与工具实现**

```bash
git add skyeye/factor_layer/utils.py skyeye/factor_layer/indicators/__init__.py skyeye/factor_layer/indicators/trend.py skyeye/factor_layer/indicators/momentum.py skyeye/factor_layer/indicators/oscillator.py skyeye/factor_layer/indicators/volatility.py skyeye/factor_layer/indicators/volume.py tests/unittest/test_skyeye_factor_layer.py
git commit -m "feat: add skyeye factor layer indicators"
```

## Task 3: 接通 core 入口和 DataFacade 便捷入口

**Files:**
- Modify: `skyeye/factor_layer/core.py`
- Modify: `skyeye/factor_layer/__init__.py`
- Test: `tests/unittest/test_skyeye_factor_layer.py`

- [ ] **Step 1: 写失败测试，锁住只输出命中类别与 facade 复用行为**

```python
import pandas as pd

from skyeye.factor_layer.core import compute_factors, compute_factors_from_data_facade
from skyeye.market_regime_layer import MarketRegime


def test_compute_factors_only_populates_categories_for_regime():
    bars = pd.DataFrame(
        {
            "open": range(1, 80),
            "high": [x + 1 for x in range(1, 80)],
            "low": [x - 1 for x in range(1, 80)],
            "close": range(1, 80),
            "volume": [1000] * 79,
        },
        index=pd.date_range("2024-01-01", periods=79, freq="B"),
    )
    regime = MarketRegime(regime="bull_co_move", strength=0.8)

    result = compute_factors(bars, regime)

    assert result.trend_factors
    assert result.volume_factors
    assert result.momentum_factors == {}
    assert result.oscillator_factors == {}
    assert result.volatility_factors == {}


def test_compute_factors_from_data_facade_reuses_market_regime_when_missing(monkeypatch):
    calls = {"market": 0, "data": 0}

    class DummyFacade:
        def get_daily_bars(self, *args, **kwargs):
            calls["data"] += 1
            return pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=80, freq="B"),
                    "order_book_id": ["000300.XSHG"] * 80,
                    "open": range(1, 81),
                    "high": [x + 1 for x in range(1, 81)],
                    "low": [x - 1 for x in range(1, 81)],
                    "close": range(1, 81),
                    "volume": [1000] * 80,
                }
            ).set_index("date")

    def _fake_market_regime_from_data_facade(**kwargs):
        calls["market"] += 1
        return MarketRegime(regime="bull_co_move", strength=0.7)

    monkeypatch.setattr("skyeye.factor_layer.core.DataFacade", DummyFacade)
    monkeypatch.setattr("skyeye.factor_layer.core.compute_market_regime_from_data_facade", _fake_market_regime_from_data_facade)

    result = compute_factors_from_data_facade(end_date="2024-05-31")

    assert calls["data"] == 1
    assert calls["market"] == 1
    assert result.regime.regime == "bull_co_move"
```

- [ ] **Step 2: 跑定向测试确认失败**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/unittest/test_skyeye_factor_layer.py::test_compute_factors_only_populates_categories_for_regime -q
```

Expected: FAIL，提示 `compute_factors` 仍返回空结果。

- [ ] **Step 3: 在 `core.py` 接入类别函数和空表降级**

```python
from __future__ import annotations

import pandas as pd

from skyeye.data.facade import DataFacade
from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.indicators.momentum import compute_momentum_factors
from skyeye.factor_layer.indicators.oscillator import compute_oscillator_factors
from skyeye.factor_layer.indicators.trend import compute_trend_factors
from skyeye.factor_layer.indicators.volatility import compute_volatility_factors
from skyeye.factor_layer.indicators.volume import compute_volume_factors
from skyeye.factor_layer.result import FactorLayerResult
from skyeye.factor_layer.utils import normalize_benchmark_bars
from skyeye.market_regime_layer import (
    MarketRegime,
    compute_market_regime_from_data_facade,
    normalize_single_instrument_bars,
)

CATEGORY_COMPUTERS = {
    "trend": ("trend_factors", compute_trend_factors),
    "momentum": ("momentum_factors", compute_momentum_factors),
    "oscillator": ("oscillator_factors", compute_oscillator_factors),
    "volatility": ("volatility_factors", compute_volatility_factors),
    "volume": ("volume_factors", compute_volume_factors),
}


def compute_factors(benchmark_bars: pd.DataFrame, regime: MarketRegime, cfg: FactorLayerConfig | None = None) -> FactorLayerResult:
    cfg = cfg or FactorLayerConfig()
    bars = normalize_benchmark_bars(benchmark_bars)
    if bars.empty or "close" not in bars.columns:
        return _empty_result(regime)

    selected = _select_factor_categories(regime.regime)
    payload = {
        "trend_factors": {},
        "momentum_factors": {},
        "oscillator_factors": {},
        "volatility_factors": {},
        "volume_factors": {},
    }
    for category in selected:
        field_name, computer = CATEGORY_COMPUTERS[category]
        payload[field_name] = computer(bars, cfg)
    return FactorLayerResult(regime=regime, **payload)
```

- [ ] **Step 4: 实现 `compute_factors_from_data_facade(...)`**

```python
def compute_factors_from_data_facade(
    end_date: str | pd.Timestamp,
    benchmark_id: str = "000300.XSHG",
    regime: MarketRegime | None = None,
    cfg: FactorLayerConfig | None = None,
) -> FactorLayerResult:
    end_ts = pd.to_datetime(end_date)
    start_ts = end_ts - pd.Timedelta(days=1200)
    data = DataFacade()
    raw = data.get_daily_bars(
        benchmark_id,
        start_ts.strftime("%Y-%m-%d"),
        end_ts.strftime("%Y-%m-%d"),
        fields=["open", "high", "low", "close", "volume"],
        adjust_type="none",
    )
    if raw is None or getattr(raw, "empty", True):
        resolved_regime = regime or MarketRegime(regime="range_co_move", strength=0.0, diagnostics={"reason": "benchmark_bars_empty"})
        return _empty_result(resolved_regime)

    bars = normalize_single_instrument_bars(raw, benchmark_id)
    resolved_regime = regime or compute_market_regime_from_data_facade(
        end_date=end_date,
        benchmark_id=benchmark_id,
    )
    return compute_factors(bars, resolved_regime, cfg=cfg)
```

- [ ] **Step 5: 跑 facade 和结果结构测试**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/unittest/test_skyeye_factor_layer.py::test_compute_factors_only_populates_categories_for_regime -q
PYTHONPATH="$PWD" pytest tests/unittest/test_skyeye_factor_layer.py::test_compute_factors_from_data_facade_reuses_market_regime_when_missing -q
```

Expected: PASS。

- [ ] **Step 6: 提交入口接线任务**

```bash
git add skyeye/factor_layer/core.py skyeye/factor_layer/__init__.py tests/unittest/test_skyeye_factor_layer.py
git commit -m "feat: wire skyeye factor layer entrypoints"
```

## Task 4: 补齐覆盖面、验证与整理导出

**Files:**
- Modify: `tests/unittest/test_skyeye_factor_layer.py`
- Modify: `skyeye/factor_layer/indicators/trend.py`
- Modify: `skyeye/factor_layer/indicators/volatility.py`
- Modify: `skyeye/factor_layer/indicators/volume.py`
- Modify: `skyeye/factor_layer/result.py`

- [ ] **Step 1: 写失败测试，锁住 percentile 范围与 `nan` 语义**

```python
import math

from skyeye.factor_layer.core import compute_factors
from skyeye.market_regime_layer import MarketRegime


def test_compute_factors_percentiles_are_nan_or_between_zero_and_one():
    bars = pd.DataFrame(
        {
            "open": range(1, 320),
            "high": [x + 1 for x in range(1, 320)],
            "low": [x - 1 for x in range(1, 320)],
            "close": range(1, 320),
            "volume": [1000] * 319,
        },
        index=pd.date_range("2023-01-02", periods=319, freq="B"),
    )
    result = compute_factors(bars, MarketRegime(regime="bear_rotation", strength=0.9))

    for bucket in (result.momentum_factors, result.volatility_factors):
        for payload in bucket.values():
            percentile = payload["percentile"]
            assert math.isnan(percentile) or 0.0 <= percentile <= 1.0
```

- [ ] **Step 2: 跑测试确认可能暴露除零或 `inf` 问题**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/unittest/test_skyeye_factor_layer.py::test_compute_factors_percentiles_are_nan_or_between_zero_and_one -q
```

Expected: 若存在除零、`inf` 或未过滤值则 FAIL。

- [ ] **Step 3: 收口数值稳定性和类型细节**

```python
# skyeye/factor_layer/result.py
from typing import Literal, TypeAlias

FactorValue: TypeAlias = dict[Literal["value", "percentile"], float]
```

```python
# skyeye/factor_layer/utils.py
def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)
```

```python
# skyeye/factor_layer/indicators/volatility.py
add_factor_from_series(factors, "BBANDS_WIDTH", safe_ratio(upper - lower, mean), cfg.percentile_window)
```

```python
# skyeye/factor_layer/indicators/volume.py
money_ratio = safe_ratio(pos_flow, neg_flow)
add_factor_from_series(factors, "MFI", 100.0 - 100.0 / (1.0 + money_ratio), cfg.percentile_window)
```

```python
# skyeye/factor_layer/indicators/trend.py
dx = 100.0 * safe_ratio((plus_di - minus_di).abs(), plus_di + minus_di)
```

- [ ] **Step 4: 跑完整单测和诊断**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/unittest/test_skyeye_factor_layer.py -q
python -m compileall skyeye/factor_layer
```

Then run diagnostics for:

```text
skyeye/factor_layer/core.py
skyeye/factor_layer/utils.py
skyeye/factor_layer/indicators/trend.py
skyeye/factor_layer/indicators/momentum.py
skyeye/factor_layer/indicators/oscillator.py
skyeye/factor_layer/indicators/volatility.py
skyeye/factor_layer/indicators/volume.py
tests/unittest/test_skyeye_factor_layer.py
```

Expected: pytest PASS，`compileall` 无错误，诊断为空或仅保留无法快速消除的非阻塞提示。

- [ ] **Step 5: 提交收尾任务**

```bash
git add skyeye/factor_layer/result.py skyeye/factor_layer/utils.py skyeye/factor_layer/indicators/trend.py skyeye/factor_layer/indicators/volatility.py skyeye/factor_layer/indicators/volume.py tests/unittest/test_skyeye_factor_layer.py
git commit -m "test: harden skyeye factor layer outputs"
```

## Self-Review Checklist

- Spec coverage:
  - 独立 `factor_layer/` 包：Task 1
  - `FactorLayerConfig` / `FactorLayerResult`：Task 1
  - 五大类指标与统一 `value/percentile`：Task 2
  - `regime -> 两类输出`：Task 1、Task 3
  - `compute_factors(...)`：Task 3
  - `compute_factors_from_data_facade(...)`：Task 3
  - 数据不足与单指标失败降级：Task 2、Task 3、Task 4
  - 百分位范围与 `nan` 语义：Task 4
  - 单测覆盖：四个任务均包含
- Placeholder scan:
  - 未保留 `TBD`、`TODO`、`implement later`、`similar to Task N`
- Type consistency:
  - 统一使用 `FactorLayerConfig`、`FactorLayerResult`、`compute_factors`、`compute_factors_from_data_facade`
  - `FactorValue` 最终收口为 `{"value": float, "percentile": float}`

