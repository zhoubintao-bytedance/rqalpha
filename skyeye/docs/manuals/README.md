# SkyEye Manuals

如果你第一次使用 `skyeye/`，推荐按下面顺序走一遍。默认都在仓库根目录执行。

## 1. 建环境

```bash
uv venv --python 3.13
uv sync --python 3.13 --extra dividend-scorer
```

如果你只想给普通 RQAlpha 策略做回测打分，不联动红利低波打分器，也可以只看：

- [策略回测打分器.md](./策略回测打分器.md)

## 2. 下载 bundle

```bash
uv run rqalpha download-bundle
```

默认路径：

- bundle：`~/.rqalpha/bundle`
- 红利低波缓存库：`~/.rqalpha/dividend_scorer/cache.db`

## 3. 同步红利低波打分器缓存

```bash
uv run python -m skyeye.dividend_scorer.main \
  --sync-only \
  --end-date "$(date +%F)"
```

这条命令只做同步，不输出评分。
首次同步会从 `2020-01-01` 全量建库；后续默认按“上次成功日期往前回溯 10 个交易日 -> 今天”做增量同步。

## 4. 直接看最新评分

```bash
uv run python -m skyeye.dividend_scorer.main --json
```

默认命令会先自动检查今天这一天是否需要同步；如果缓存已覆盖，请求会直接 `skip`，然后输出当前最新可用日期的评分。
这一步先确认评分本身能正常算出来，再进入策略和回测。

## 5. 用 RQAlpha 原生回测跑单区间并保存图片

```bash
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run rqalpha run \
  -f skyeye/examples/dividend_low_vol_score_strategy_history_aware.py \
  -s 2024-05-03 \
  -e 2025-04-20 \
  --account stock 100000 \
  -fq 1d \
  -d ~/.rqalpha/bundle \
  -bm 512890.XSHG \
  --plot-save dividend_price_2405_2504.png \
  -mc dividend_scorer.enabled true \
  -mc dividend_scorer.db_path ~/.rqalpha/dividend_scorer/cache.db
```

这条路径适合做“单区间策略迭代”：

- 直接对比策略净值和 `512890.XSHG` 买入持有
- 把回测图片落到本地，方便横向比较不同参数版本
- 一边看收益/回撤，一边看日志里的仓位变化是否符合预期

如果你是在本地桌面环境运行，也可以去掉前面的 `UV_CACHE_DIR` 和 `MPLCONFIGDIR`。

## 6. 给策略做滚动窗口打分

```bash
uv run python skyeye/strategy_scorer.py \
  skyeye/examples/dividend_low_vol_score_strategy_history_aware.py \
  -w 37 \
  --mod dividend_scorer \
  -mc dividend_scorer.db_path ~/.rqalpha/dividend_scorer/cache.db \
  --log high
```

如果窗口 37 没问题，再逐步扩大到 `33-37` 或完整窗口。

这条路径和上面的原生回测是互补关系：

- `rqalpha run --plot-save`：看某一个具体区间的真实净值、基准对比和图片
- `skyeye/strategy_scorer.py`：看多窗口稳定性，避免策略只在单一区间表现好

## 7. 接下来读什么

- [红利低波打分器使用说明.md](./红利低波打分器使用说明.md)
  适合继续看缓存同步、CLI 输出、策略调用、参数审计和诊断工具画图
- [策略回测打分器.md](./策略回测打分器.md)
  适合继续看滚动窗口打分、日志解释、稀疏窗口和 `--mod/-mc` 用法
- [../rfc/dividend_scorer_iteration_directions.md](../rfc/dividend_scorer_iteration_directions.md)
  适合看后续设计讨论和迭代优先级，不是使用手册
