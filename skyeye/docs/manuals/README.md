# SkyEye Manuals

如果你第一次使用 `skyeye/`，推荐按下面顺序走。默认都在仓库根目录执行。

## 1. 建环境

```bash
uv venv --python 3.13
uv sync --python 3.13 --extra dividend-scorer
```

如果仓库根目录已经有 `.venv/` 且依赖已经装好，通常不需要重复执行 `uv venv`。
如果只是补齐或更新依赖，直接执行 `uv sync --python 3.13 --extra dividend-scorer` 即可。
如果你确实要重建环境，再手动删除 `.venv/` 或使用 `uv venv --clear --python 3.13`。

## 2. 下载 bundle

```bash
uv run rqalpha download-bundle
```

默认路径：

- bundle：`~/.rqalpha/bundle`
- 红利低波缓存库：`~/.rqalpha/dividend_scorer/cache.db`

## 3. 一键打分

[方式1] 对红利低波一键打分，会输出当前最新可用日期的评分报告。
```bash
uv run python -m skyeye.products.dividend_low_vol.scorer.main
```

[方式2] 对 `2026-03-19` 当天价格的红利低波进行评分，如果输入的日期不是交易日，会自动调整到最近的交易日。
```bash
uv run python -m skyeye.products.dividend_low_vol.scorer.main --end-date 2026-03-19
```

## 4. 跑单区间原生回测并生成图片

```bash
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run rqalpha run \
  -f skyeye/products/dividend_low_vol/strategies/history_aware/strategy.py \
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

这条路径用来回答单一区间里策略有没有跑赢 `512890.XSHG` 买入持有，以及仓位变化是否合理。

## 5. 跑滚动窗口打分

```bash
uv run python -m skyeye.evaluation.rolling_score.cli \
  skyeye/products/dividend_low_vol/strategies/history_aware/strategy.py \
  -w 37 \
  --mod dividend_scorer \
  -mc dividend_scorer.db_path ~/.rqalpha/dividend_scorer/cache.db \
  --log high
```

这条路径用来看跨年份稳定性，而不是只看单一区间。

## 6. 跑基础单元测试

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH="$PWD" MPLCONFIGDIR=/tmp/mplconfig uv run pytest -q \
  tests/products/dividend_low_vol/scorer \
  tests/unittest \
  tests/evaluation \
  tests/products
```

这条路径用来验证评分器、滚动打分器、产品线逻辑和基础单元测试没有被改坏。
如果你要跑 `tests/` 全量集成测试，需要先补齐 `TA-Lib`，否则 `tests/integration_tests/test_backtest_results/` 会在收集阶段失败。

## 7. 当前结构的基本约定

- 红利低波产品线在 `skyeye/products/dividend_low_vol/`
- 策略目录必须带 `spec.yaml`、`profiles/` 和薄 `strategy.py`
- 如果后续需要独立诊断脚本，放在 `skyeye/products/dividend_low_vol/tools/`，不要伪装成策略
- 滚动窗口打分器主入口是 `python -m skyeye.evaluation.rolling_score.cli`

## 8. 接下来读什么

- [红利低波打分器使用说明.md](./红利低波打分器使用说明.md)
  适合看评分器、单区间回测、单元测试和策略迭代约定。
- [策略回测打分器.md](./策略回测打分器.md)
  适合看滚动窗口打分、策略卡片、窗口选择和 `--mod/-mc`。
- [../rfc/dividend_scorer_iteration_directions.md](../rfc/dividend_scorer_iteration_directions.md)
  适合看设计讨论和迭代优先级，不是使用手册。

## 9. 如果你在看 TX1

TX1 当前不是 `dividend_scorer` 那条产品线，它的默认入口在：

- [../../products/tx1/README.md](../../products/tx1/README.md)
- [../../products/tx1/PLAYBOOK.md](../../products/tx1/PLAYBOOK.md)
- [../../products/tx1/strategies/rolling_score/README.md](../../products/tx1/strategies/rolling_score/README.md)
- [../rfc/tianyan_tx1_strategy_v1.md](../rfc/tianyan_tx1_strategy_v1.md)
- [../rfc/tianyan_tx1_research_design_v1_1.md](../rfc/tianyan_tx1_research_design_v1_1.md)

TX1 最常用的两条命令：

```bash
uv run python -m skyeye.products.tx1.run_baseline_experiment \
  --model lgbm \
  --output-dir skyeye/artifacts/experiments/tx1
```

```bash
uv run python -m skyeye.evaluation.rolling_score.cli \
  skyeye/products/tx1/strategies/rolling_score/strategy.py
```
