# 红利低波打分器使用说明

本文档说明如何在 RQAlpha 里使用 `512890` 红利低波打分器，包括：

- 同步打分器所需缓存到 SQLite
- 用 CLI 直接查看指定日期评分
- 在策略中调用 `get_dividend_score()`
- 画出 `512890` 走势与打分曲线的对比图

## 1. 能力概览

打分器目前有两种使用方式：

- CLI：`python -m rqalpha.dividend_scorer.main`
- RQAlpha Mod：策略里直接调用 `get_dividend_score()`

仓库里已经有两份可直接运行的示例策略：

- [dividend_low_vol_score_strategy.py](rqalpha/examples/dividend_low_vol_score_strategy.py)
  固定阈值交易示例：`score < 3.5` 买入，`score > 6.5` 清仓
- [dividend_score_vs_price_plot_strategy.py](rqalpha/examples/dividend_score_vs_price_plot_strategy.py)
  只负责画图，对比 `512890` 走势与 `dividend_score`

## 2. 环境准备

### 2.1 推荐用 `uv` 统一环境

建议统一用 `uv`，原因很简单：

- 避免把依赖装到错误解释器
- 把 Python 版本和 `akshare` 版本锁到项目本地环境
- 后续执行 `python -m ...` 和 `rqalpha run ...` 都走同一个环境

当前项目已经提供可选依赖组：

- `dividend-scorer`：固定 `akshare==1.18.38`

首次初始化：

```bash
uv venv --python 3.13
uv sync --python 3.13 --extra dividend-scorer
```

后续统一使用：

```bash
uv run python -m rqalpha.dividend_scorer.main --help
```

如果你不打算用 `uv`，至少也要保证当前解释器里已经安装：

```bash
pip install akshare==1.18.38
```

### 2.2 默认路径

- SQLite 缓存默认路径：`~/.rqalpha/dividend_scorer/cache.db`
- bundle 默认路径：`~/.rqalpha/bundle`

## 3. 第一步：同步打分器缓存

首次使用建议先全量同步一次：

```bash
uv run python -m rqalpha.dividend_scorer.main \
  --sync \
  --sync-only \
  --start-date 2018-01-01 \
  --end-date 2026-03-19
```

如果你想显式指定缓存路径：

```bash
uv run python -m rqalpha.dividend_scorer.main \
  --sync \
  --sync-only \
  --db-path ~/.rqalpha/dividend_scorer/cache.db \
  --start-date 2018-01-01 \
  --end-date 2026-03-19
```

如果你想无视本地 checkpoint，强制重新拉远端：

```bash
uv run python -m rqalpha.dividend_scorer.main \
  --sync \
  --sync-only \
  --force-sync \
  --db-path ~/.rqalpha/dividend_scorer/cache.db \
  --start-date 2018-01-01 \
  --end-date 2026-03-19
```

同步说明：

- `--sync` 会执行 `AKShare -> SQLite`
- 首次同步时间会比较长，尤其是成分股股息率部分
- 终端会显示统一的分步进度，不再直接暴露 AKShare 自己的碎进度条
- 如果同一个 `db_path + 时间区间` 已经成功同步过，再次执行会命中 checkpoint 并快速跳过
- 如需完全重拉，用 `--force-sync`
- 当 `akshare` 缺失旧版 `stock_a_indicator_*` 接口时，会自动退化为 `stock_value_em + stock_fhps_detail_em` 重建历史股息率
- `stock_zh_index_hist_csindex` 无法完整覆盖最新阶段的指数 PE，当前实现会额外叠加 `stock_zh_index_value_csindex`；中间若仍有空窗，评分时会自动降级为“不参与”

## 4. 第二步：直接看打分结果

同步完成后，可以直接用 CLI 查看评分：

```bash
uv run python -m rqalpha.dividend_scorer.main --date 2025-03-18
```

输出 JSON：

```bash
uv run python -m rqalpha.dividend_scorer.main --date 2025-03-18 --json
```

如果不指定 `--date`，默认取缓存里的最新日期。

常用返回字段：

- `total_score`
- `confidence`
- `features`
- `confidence_modifiers`
- `model_meta`
- `data_freshness`

## 5. 第三步：在策略中使用

### 5.1 运行固定阈值示例策略

推荐先用命令行临时开启 Mod，不要急着改全局配置：

```bash
uv run rqalpha run \
  -f rqalpha/examples/dividend_low_vol_score_strategy.py \
  -s 2024-01-01 \
  -e 2025-12-31 \
  --account stock 100000 \
  -fq 1d \
  -d ~/.rqalpha/bundle \
  -bm 000300.XSHG \
  -mc dividend_scorer.enabled true \
  -mc dividend_scorer.db_path ~/.rqalpha/dividend_scorer/cache.db
```

如果你还想看 RQAlpha 自带图形输出：

```bash
uv run rqalpha run \
  -f rqalpha/examples/dividend_low_vol_score_strategy.py \
  -s 2024-01-01 \
  -e 2025-12-31 \
  --account stock 100000 \
  -fq 1d \
  -d ~/.rqalpha/bundle \
  -bm 000300.XSHG \
  --plot \
  -mc dividend_scorer.enabled true \
  -mc dividend_scorer.db_path ~/.rqalpha/dividend_scorer/cache.db
```

### 5.2 在你自己的策略里调用

只要启用 `dividend_scorer` mod，策略里就可以直接调用：

```python
from rqalpha.apis import *


def handle_bar(context, bar_dict):
    score = get_dividend_score()
    if not score or score.get("error"):
        return

    if score["total_score"] < 3.5:
        order_target_percent("512890.XSHG", 0.95)
    elif score["total_score"] > 6.5:
        order_target_percent("512890.XSHG", 0)
```

## 6. 画出 `512890` 真实走势 vs 打分曲线

如果你现在主要想验证：

- ETF 处于相对低位时，打分是否也偏低
- ETF 处于相对高位时，打分是否也偏高

直接运行：

```bash
uv run rqalpha run \
  -f rqalpha/examples/dividend_score_vs_price_plot_strategy.py \
  -s 2019-01-18 \
  -e 2026-03-06 \
  --account stock 100000 \
  -fq 1d \
  -d ~/.rqalpha/bundle \
  -bm null \
  --plot-save /tmp/dividend_score_vs_price.png \
  -mc dividend_scorer.enabled true \
  -mc dividend_scorer.db_path ~/.rqalpha/dividend_scorer/cache.db
```

说明：

- 这份策略不下单，只负责画图
- `-bm null` 是故意的，这样不会额外要求 benchmark 也覆盖整段区间
- `dividend_score` 就是打分器输出的 `total_score`
- `512890_close_norm` 不是实际价格，而是把 ETF 的收盘价映射到 `[0, 10]` 后得到的比较曲线
- `512890_close_norm` 使用的是“固定到回测结束日的前复权收盘价”，然后在整个回测区间做 min-max 归一化
- 两条线会出现在最下面的自定义 subplot 里；上半部分仍然是 RQAlpha 默认收益面板
- 如果你需要图形窗口，把 `--plot-save` 改成 `--plot`

建议重点观察：

- 两条线的阶段性顶部、底部是否大致同向
- 明显风格切换阶段，比如 `2021`、`2024`，打分拐点是否跟上
- 如果价格创新高而打分没有同步抬升，通常说明估值口径或底层数据还值得继续检查

说明一下日期区间：

- 上面的 `2019-01-18 -> 2026-03-06` 是我本地 bundle 上 `512890.XSHG` 实际验证通过的可用区间
- 如果你的 bundle 更完整，可以按你本地实际可用区间调整 `-s` 和 `-e`

## 7. 常开方式

如果你想长期测试，可以把 `~/.rqalpha/mod_config.yml` 改成：

```yaml
mod:
  dividend_scorer:
    enabled: true
    lib: rqalpha.mod.rqalpha_mod_dividend_scorer
    db_path: ~/.rqalpha/dividend_scorer/cache.db
```

这样以后运行策略就不用每次再写：

- `-mc dividend_scorer.enabled true`
- `-mc dividend_scorer.db_path ~/.rqalpha/dividend_scorer/cache.db`

## 8. 常见问题

### 8.1 报错 `akshare is required for sync_all`

原因：

- 你执行了 `--sync`
- 但当前环境没有安装 `akshare`

处理：

```bash
uv sync --python 3.13 --extra dividend-scorer
```

### 8.2 走代理时报 `ProxyError`

原因：

- 某些代理环境下，`Eastmoney` 链路会被中途断开
- 常见表现是 `fund_etf_hist_em` 或 `stock_value_em` 报 `ProxyError`

处理：

- 优先直接跑，不要给同步命令套代理
- 如果你开了类似 `fanqiang` 的环境变量，先取消再执行：

```bash
unset http_proxy https_proxy no_proxy
uv run python -m rqalpha.dividend_scorer.main \
  --sync \
  --sync-only \
  --start-date 2018-01-01 \
  --end-date 2026-03-19
```

### 8.3 报错 `etf_daily cache is empty, run sync_all first`

原因：

- SQLite 缓存还没建立
- 或 `db_path` 指错了

处理：

1. 先执行一次 `--sync`
2. 确认策略和 CLI 用的是同一个 `db_path`

### 8.4 打分器在策略里返回 `error`

常见原因：

- SQLite 数据缺列或缺日期
- 交易日覆盖率不够
- 股息率维度不可用
- 可用维度不足 3 个

建议先单独跑：

```bash
uv run python -m rqalpha.dividend_scorer.main --json
```

先确认评分本身能正常算出来，再进策略。

### 8.5 画图时报 benchmark 或日期覆盖错误

常见原因：

- 你把 benchmark 设成了一个本地 bundle 覆盖不完整的标的
- 或者 `512890.XSHG` 在你的 bundle 里起止日期比命令里更短

处理：

- 对比图策略优先用 `-bm null`
- 根据你本地 bundle 的实际覆盖范围调整 `-s` 和 `-e`
