# 线上高风险链路修复

- [x] 补充数据覆盖区间错误标记的回归测试
- [x] 补充 `trade_date` 选股池前视偏差的回归测试
- [x] 补充非调仓日权重漂移与 snapshot coverage gate 的回归测试
- [x] 修复 `skyeye/data/facade.py` 的空结果 coverage 标记逻辑
- [x] 修复 `run_baseline_experiment.py` 的 live universe 截止日传递
- [x] 修复 `portfolio_proxy.py` / `service.py` 的非调仓日权重保留逻辑
- [x] 修复 `runtime_gates.py` 的 universe coverage 阈值逻辑
- [x] 运行定向测试并记录证据

## Review

- 已新增 5 条回归测试，分别锁住：`daily_bars` 空结果不应冻结缺口、`factor` 空结果不应冻结缺口、`build_live_raw_df(trade_date=...)` 必须把截止日传进 universe 选择、非调仓日必须保留当前权重、候选池总量低于 coverage 下限时必须 stop-serve。
- `skyeye/data/facade.py` 现改为仅对在线请求里真正返回到本地的数据日期写 coverage；空表或部分返回只记 audit，不会把整段缺口永久封死。
- `skyeye/products/tx1/run_baseline_experiment.py` 已给 live universe 选择链路补上 `data_end` 透传，历史日期 snapshot 不再默认按“最新数据”选股。
- `skyeye/products/tx1/portfolio_proxy.py` 已在非调仓日优先保留 dict 形式的当前持仓权重，不再偷偷改成等权。
- `skyeye/products/tx1/live_advisor/runtime_gates.py` 已把 coverage 阈值恢复成 spec 定义的 `max(300, history_median * 0.8)`，候选池总量不足时会正确 stop-serve。
- 验证证据：
  - 定向回归：`env PYTHONPATH=$PWD pytest tests/unittest/test_data/test_skyeye_data_facade.py tests/products/tx1/test_run_baseline_experiment.py tests/products/tx1/test_live_advisor_service.py tests/products/tx1/test_live_advisor_runtime_gates.py -q`
  - 结果：`24 passed`
- 扩大回归：`env PYTHONPATH=$PWD pytest tests/unittest/test_data/test_skyeye_data_facade.py tests/products/tx1/test_baseline_models.py tests/products/tx1/test_preprocessor.py tests/products/tx1/test_run_baseline_experiment.py tests/products/tx1/test_run_feature_experiment.py tests/products/tx1/test_live_advisor_calibration.py tests/products/tx1/test_live_advisor_holdings_io.py tests/products/tx1/test_live_advisor_package_io.py tests/products/tx1/test_live_advisor_promotion.py tests/products/tx1/test_live_advisor_runtime_gates.py tests/products/tx1/test_live_advisor_service.py tests/products/tx1/test_live_advisor_snapshot.py tests/products/tx1/test_run_live_advisor.py -q`
  - 结果：`63 passed`

# TX1 实盘建议器当前可用性评估

- [x] 梳理 TX1 当前 live package、实验产物和最近任务记录
- [x] 核对 live advisor 关键代码与测试，确认 freshness / stop-serve / 组合建议语义
- [x] 补充当前运行证据，判断是否适合作为“每天实盘指导”

## Review

- 截至 `2026-04-13`，TX1 live advisor 还不能算“已经完善好、可每天指导实盘”。
- 当前最新实盘包 `skyeye/artifacts/packages/tx1/tx1_canary_live_tx1_refresh_20260331_lgbm_20260303/manifest.json` 仍是 `canary_live`；`stability_score=14.2857 < 50`、`cv=0.8854 > 0.6`，没达到 `default_live` 门槛。
- 真实链路验证：
  - `env PYTHONPATH=$PWD python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260303 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-03-31 --top-k 5 --format json`
  - 结果：`status=ok`，说明在本地最新可用数据日上，系统能输出 top-k 排名、历史 bucket 统计、recent canary 统计和 `portfolio_advice`。
  - `env PYTHONPATH=$PWD python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260303 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-04-10 --top-k 3 --format json`
  - `env PYTHONPATH=$PWD python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260303 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-04-13 --top-k 3 --format json`
  - 结果：两次都 `status=stopped`，`latest_available_trade_date=2026-03-31`，说明今天并没有“当日可用”的实盘建议。
- 当前代码里的 freshness / stop-serve 设计已经比较完整，但磁盘上的现有 package manifest 仍缺 `evidence_end_date` / `freshness_policy` / `data_dependency_summary`，说明产包没有按最新契约重新生成；运行时 `evidence_freshness` 现在还是 `unknown`。
- 定向测试验证：
  - `env PYTHONPATH=$PWD pytest tests/products/tx1/test_live_advisor_calibration.py tests/products/tx1/test_live_advisor_holdings_io.py tests/products/tx1/test_live_advisor_package_io.py tests/products/tx1/test_live_advisor_promotion.py tests/products/tx1/test_live_advisor_runtime_gates.py tests/products/tx1/test_live_advisor_service.py tests/products/tx1/test_live_advisor_snapshot.py tests/products/tx1/test_live_advisor_universe.py tests/products/tx1/test_run_live_advisor.py -q`
  - 结果：`31 passed`
- 近端证据仍偏弱：`2026-03-31` 实跑 top bucket 的历史 OOS `mean_return=+1.08%`，但 recent canary `2025-11-28 ~ 2026-03-03` 的同 bucket `mean_return=-0.23%`、`median_return=-1.28%`、`win_rate=42.56%`，更像“可参考的排序器”，还不是“可放心每天拿来指导实盘”的状态。

# TX1 数据更新与 live package 重建

- [x] 梳理 TX1 数据更新入口、当前 bundle 截止日与 promote 命令链路
- [x] 补齐最新可用数据并确认 bundle / raw data 截止日
- [x] 基于最新实验重新 promote live package
- [x] 复核当天 live advisor 结果是否可用并记录证据

## Review

- 先验证了官方 `python -m rqalpha update-bundle -d /home/tiger/.rqalpha -c 4` 的失败根因：不是网络，而是 `rqdatac.share.errors.QuotaExceeded: login machine num exceeds`。原因在于官方 bundle 更新链路会在主进程和 worker 进程重复 `rqdatac.init()`，超出当前账号登录机数限制。
- 随后改用单进程增量补数方式，只更新 TX1 实际依赖的 bundle 组件：`instruments.pk`、`trading_dates.npy`、`indexes.h5`、`stocks.h5`。执行后本地 bundle 已从 `2026-03-31` 更新到 `2026-04-13`：
  - `indexes.h5 000300.XSHG 2026-04-13`
  - `stocks.h5 000001.XSHE 2026-04-13`
  - `resolve_data_end=2026-04-13`
- 真实数据补齐证据：
  - 单进程补数脚本完成后输出：`indexes.h5 done: ... latest=2026-04-13`、`stocks.h5 done: ... latest=2026-04-13`
  - 轻量核对命令：`python - <<'PY' ... print(resolve_data_end()) ... PY`
  - 结果：`resolve_data_end 2026-04-13 00:00:00`
- 旧 package 在数据更新后已恢复“当天可用”：
  - `env MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=$PWD python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260303 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-04-13 --top-k 3 --universe-cache-root /tmp/tx1_live_universe_cache --format json`
  - 结果：`status=ok`，`score_date=2026-04-13`，但仍有 `model_freshness_warning`，因为旧 package 的 `fit_end_date=2026-03-03`。
- 已重新 promote 出新 package：
  - 路径：`skyeye/artifacts/packages/tx1/tx1_canary_live_tx1_refresh_20260331_lgbm_20260313`
  - `package_id=tx1_canary_live_tx1_refresh_20260331_lgbm_20260313`
  - `fit_end_date=2026-03-13`
  - `label_end_date=2026-03-13`
  - `evidence_end_date=2026-03-13`
  - `data_end_date=2026-04-13`
  - `freshness_policy={'snapshot_max_delay_days': 1, 'model_warning_days': 20, 'model_stop_days': 40, 'evidence_warning_days': 20, 'evidence_stop_days': 40}`
  - `canary_reason=['stability_score', 'cv']`
- 新 package 的当天复核：
  - `env MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=$PWD python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260313 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-04-13 --top-k 3 --universe-cache-root /tmp/tx1_live_universe_cache --format json`
  - 结果：`status=ok`，`requested_trade_date=2026-04-13`，`latest_available_trade_date=2026-04-13`，`score_date=2026-04-13`，说明当天建议已可正常产出。
  - 但仍有轻微 freshness warning：`model/evidence gap_days=21`，刚好略高于 `warning_days=20`；没有 stop-serve。
  - top-3 为：`000793.XSHE`、`002024.XSHE`、`600690.XSHG`
  - 新 package 的 recent canary 已明显改善：窗口 `2025-12-10 ~ 2026-03-13`，top bucket `win_rate=53.11%`、`mean_return=+1.26%`、`median_return=+0.50%`，相比旧 package 最近 canary 的负均值/负中位数更健康。

# 本地改动并行 Review

- [x] 梳理工作区改动范围并按子系统分组
- [x] 并行审查 `skyeye/data` 数据访问、bundle/cache 与依赖调用改动
- [x] 并行审查 `skyeye/products/tx1` 主实验链路与序列化改动
- [x] 并行审查 `skyeye/products/tx1/live_advisor` 新增链路与 CLI 暴露
- [x] 主线程交叉复核共享接口、回归风险与测试缺口
- [x] 汇总 findings 并在 review 小节记录结论

## Review

- 已用并行 agent + 主线程完成工作区本地改动审查，重点覆盖 `skyeye/data`、`skyeye/products/tx1` 与新增 `live_advisor`。
- 发现高优先级问题集中在三类：数据覆盖区间被过早标记导致永久缺口、live runtime gate 对 universe 崩塌放行、非调仓日组合建议仍会强制改成等权。
- 主线程额外确认了一个链路缺口：`build_live_raw_df(trade_date=...)` 在未显式传入 universe 时不会把 `trade_date` 传进 universe 构建逻辑，历史日期评分会混入“按最新数据选股”的截面。
- 验证证据：
  - `env PYTHONPATH=$PWD pytest tests/unittest/test_data/test_skyeye_data_facade.py tests/products/tx1/test_baseline_models.py tests/products/tx1/test_preprocessor.py tests/products/tx1/test_run_baseline_experiment.py tests/products/tx1/test_run_feature_experiment.py tests/products/tx1/test_live_advisor_calibration.py tests/products/tx1/test_live_advisor_holdings_io.py tests/products/tx1/test_live_advisor_package_io.py tests/products/tx1/test_live_advisor_promotion.py tests/products/tx1/test_live_advisor_runtime_gates.py tests/products/tx1/test_live_advisor_service.py tests/products/tx1/test_live_advisor_snapshot.py tests/products/tx1/test_run_live_advisor.py -q`
  - 结果：`58 passed`，说明当前问题主要是测试覆盖空洞而不是现有测试失败。

# TX1 ROE 增量验证

- [x] 梳理 TX1 当前默认线、研究链路和现有产物
- [x] 确认本次目标是验证 `baseline_5f + return_on_equity_ttm`
- [x] 输出 Spec 第 1 段：现状分析
- [x] 输出 Spec 第 2 段：功能点与实验设计
- [x] 输出 Spec 第 3 段：风险与决策
- [x] 等待用户 HARD-GATE 确认
- [x] 实现最小改动
- [x] 运行验证并记录证据

## Review

- 已新增 `baseline_5f_roe` 候选，并补充 report / CSV 的 spread 与稳定性 delta 导出。
- 测试验证：`PYTHONPATH="$PWD" pytest tests/products/tx1/test_run_feature_experiment.py tests/products/tx1/test_dataset_builder.py tests/products/tx1/test_label_builder.py -q` 通过，`26 passed`。
- 实验验证：`PYTHONPATH="$PWD" MPLCONFIGDIR=/tmp/mplconfig python -m skyeye.products.tx1.run_feature_experiment --variant baseline_5f baseline_5f_roe --output-dir skyeye/artifacts/experiments/tx1_feature_roe_min`
- 结论：`baseline_5f_roe` 未满足推进条件，`rank_ic_mean` 与 `top_bucket_spread_mean` 均下降，`net_mean_return` 仅有可忽略的微弱提升，且 `fold_net_return_std` 恶化。
- 已更新 `skyeye/products/tx1/PLAYBOOK.md`，补充 `ROE` 增量实验结果、无效实验判定口径和后续研究排坑约定。

# TX1 PLAYBOOK 优先级修订

- [x] 将“前置闸门”和“候选方向优先级”从同一条建议中拆开
- [x] 更新 `skyeye/products/tx1/PLAYBOOK.md` 的下一轮资源投入建议
- [x] 在 review 中记录这次修订结论

## Review

- 已将 `PLAYBOOK.md` 中混合表述拆成两层：先写基本面增量实验的前置闸门，再写“下一轮最该投资源的候选方向”。
- 已把“组合层小范围微调”上调为当前第一优先级，因为现有 `combo_h40_bonus1` / `combo_h45_bonus1` 相对 `combo_b25_h45` 仍有小幅正向收益增量且无新增 warnings。
- 已明确把 `ep_ratio_ttm` / `return_on_equity_ttm` 的单因子直加路线降级，不再作为前排候选方向。

# TX1 4 月缺失数据补齐与实盘建议验证

- [x] 检查本地 bundle、TX1 live package 与 4 月日期覆盖现状
- [x] 按 bundle 优先原则补齐 4 月缺失数据
- [x] 运行 TX1 live advisor，验证 4 月可用交易日的建议输出
- [x] 记录执行证据与结论

## Review

- 本地 bundle 基线仍停在 `2026-03-31`，最新可用 promoted package 为 `skyeye/artifacts/packages/tx1/tx1_canary_live_tx1_refresh_20260331_lgbm_20260303/manifest.json`，`gate_level=canary_live`，`fit_end_date=2026-03-03`。
- 直接按现有 `get_liquid_universe()` 跑 live advisor 首次补 4 月缺口过慢，最终改为在 `/tmp/tx1_live_april_20260410.py` 中直接读取 `~/.rqalpha/bundle/stocks.h5`，按 `2015-01-01 ~ 2026-03-31` 的成交量中位数计算 local liquid top 300，再只为这 300 只股票补 4 月缺失数据并评分。
- 运行证据：
  - `tmux` 任务：`env PYTHONPATH=$PWD /home/tiger/miniconda3/bin/python -u /tmp/tx1_live_april_20260410.py`
  - 日志：`/tmp/tx1_live_april_20260410.log`
  - 结果：`raw_df shape: (769362, 12)`，`date range: 2015-01-05 – 2026-04-10`，`raw_data_end_date=2026-04-10`，说明 4 月缺失数据已成功补到 `2026-04-10`。
- `2026-04-10` 实盘建议验证结果：
  - `status=ok`
  - `requested_trade_date=2026-04-10`
  - `latest_available_trade_date=2026-04-10`
  - `score_date=2026-04-10`
  - `top10`: `600115.XSHG`、`000793.XSHE`、`600029.XSHG`、`601212.XSHG`、`002024.XSHE`、`000980.XSHE`、`300287.XSHE`、`601728.XSHG`、`600690.XSHG`、`600015.XSHG`
  - top bucket 历史 OOS 统计：`win_rate=0.4812`，`mean_return=0.01078`，`median_return=-0.00322`，属于右偏收益分布，不是高胜率信号。
  - recent canary 窗口 `2025-11-28 ~ 2026-03-03`：`win_rate=0.4256`，`mean_return=-0.00229`，`median_return=-0.01284`，近端证据偏弱。
  - `portfolio_advice.rebalance_due=true`，`estimated_turnover=0.50`，默认会给出 25 只等权 `4%` 的调仓目标。
- 对 `2026-04-13` 的追跑验证显示：`requested_trade_date=2026-04-13`，但 `latest_available_trade_date=2026-04-10`，`score_date=2026-04-10`，说明当前环境下 4 月数据已经补到 `2026-04-10`，尚未拿到 `2026-04-13` 的可评分收盘数据。

# TX1 run_live_advisor 正式程序性能改造

- [x] 输出 Spec 第 1 段：现状分析与根因边界
- [x] 输出 Spec 第 2 段：方案选项与推荐
- [x] 输出 Spec 第 3 段：风险、接口与验证口径
- [x] 等待用户 HARD-GATE 确认
- [x] 先写失败测试，锁定 runtime fast path / universe cache / CLI 默认行为
- [x] 新增 runtime universe resolver 与本地 cache 快照
- [x] 接入 `build_live_raw_df` / `build_live_snapshot` / `LiveAdvisorService` / CLI 默认路径
- [x] 运行定向测试与必要的真实链路验证
- [x] 在 review 记录性能收益、行为一致性与剩余风险

## Review

- 已新增 `skyeye/products/tx1/live_advisor/universe.py`，提供 runtime fast path：
  - 直接从 bundle 读取 `stocks.h5` / `indexes.h5`
  - 按历史 `volume` 中位数计算 liquid top 300
  - 以 `data_end_date + universe_size` 为 key 写入 `~/.rqalpha/tx1_live_advisor/universe_cache/`
- 已将 runtime 默认调用链切到 fast path：
  - `run_live_advisor.py` 新增 `--universe-source` / `--universe-cache-root`
  - `LiveAdvisorService.get_recommendations()` 默认 `universe_source="runtime_fast"`
  - `build_live_snapshot()` 默认透传 `runtime_fast`
  - `build_live_raw_df()` 新增 `universe_source` / `universe_cache_root`，仅 live runtime 默认走 fast path；研究侧旧路径仍保留
- 已修正一个边界问题：
  - 初版 fast path 误把 `trading_dates.npy` 的未来交易日历当成实际 bundle 数据终点
  - 现改为优先探测 `indexes.h5:000300.XSHG` 的真实最新行情日，不再虚报 `data_end_date`
- 测试验证：
  - 定向红绿：`env PYTHONPATH=$PWD pytest tests/products/tx1/test_live_advisor_universe.py tests/products/tx1/test_live_advisor_snapshot.py tests/products/tx1/test_run_baseline_experiment.py tests/products/tx1/test_run_live_advisor.py -q`
  - 结果：`18 passed`
  - 扩大回归：`env PYTHONPATH=$PWD pytest tests/products/tx1/test_live_advisor_*.py tests/products/tx1/test_run_baseline_experiment.py tests/products/tx1/test_run_live_advisor.py -q`
  - 结果：`35 passed`
- 真实链路验证：
  - 命令：`env PYTHONPATH=$PWD /home/tiger/miniconda3/bin/python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260303 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-04-10 --top-k 1 --format json`
  - 结果：`status=ok`，`score_date=2026-04-10`，`raw_data_end_date=2026-04-10`，top-1 仍为 `600115.XSHG`
  - runtime universe cache 已落盘：`~/.rqalpha/tx1_live_advisor/universe_cache/liquid_top_300_20260331.json`
- 行为与风险：
  - daily runtime 默认不再先卡在研究侧 `get_liquid_universe()` 的全市场长历史扫描；现在会直接进入 300 股票批量加载阶段
  - fast path 当前只保证“纯 liquid top 300”语义；若显式传入 `market_cap_floor_quantile` / `market_cap_column`，仍会回退到旧研究路径
  - 之前临时脚本留下的 `liquid_top_300_20260410.json` 旧 cache 仍在用户主目录，但正式代码现在按真实 bundle 数据终点使用 `20260331` key，不再继续写错日期

# TX1 最新数据补跑

- [x] 梳理 TX1 数据截止日、训练窗口和 artifact 过期原因
- [x] 确认补跑链路依赖是否齐全（bundle / rqdatac / 因子 / 输出目录）
- [x] 视情况修正数据截止逻辑或执行补跑命令
- [x] 验证新 artifact 的日期覆盖和 rolling-score 可执行性
- [x] 记录结论与原因说明

## Review

- 代码修正：
  - `skyeye/data/facade.py` 新增 `RQALPHA_BUNDLE_PATH` / `SKYEYE_BUNDLE_PATH` 覆盖，允许临时切换 bundle 根目录。
  - `skyeye/products/tx1/run_baseline_experiment.py` 新增 `resolve_data_end()`，不再依赖手写 `DATA_END`，改为按基准真实可用行情自动收敛；仍保留 `DEFAULT_DATA_END` 作为兜底。
- 测试验证：`PYTHONPATH="$PWD" pytest tests/products/tx1/test_run_baseline_experiment.py -q` 通过，`5 passed`。
- 数据验证：
  - `~/.rqalpha/bundle/indexes.h5` 的 `000300.XSHG` 最新日期已更新到 `2026-03-31`。
  - 轻量链路验证显示：`resolve_data_end=2026-03-31`，`raw_df` / `dataset` 终点均为 `2026-03-31`，`labeled_df` 终点为 `2026-03-03`。
  - 在当前 `train=3y / val=6m / test=6m / embargo=20d / horizon=20d` 设计下，fold 数仍为 `14`，最后一个 `test_end` 仍是 `2025-11-18`。
- 全量补跑验证：
- 已生成新实验 `skyeye/artifacts/experiments/tx1/tx1_refresh_20260331_lgbm/experiment.json`。
- 新实验关键指标：`rank_ic_mean=0.0554`，`top_bucket_spread_mean=0.0115`，`net_mean_return=0.000330`，`max_drawdown=0.0559`。
- `build_runtime('tx1.rolling_score', artifact_line='refresh_20260331_lgbm', load_signal_book=True)` 可正常加载，但 `signal_end` 仍为 `2025-11-18`。

# SkyEye 接入 autoresearch 方案设计

- [x] 梳理 `skyeye` / `autoresearch` 现状与约束证据
- [x] 输出 Spec 第 1 段：现状分析与问题边界
- [x] 等待用户确认 Spec 第 1 段
- [x] 输出 Spec 第 2 段：方案选项与推荐架构
- [x] 记录用户决策：第一期改用方案 A（原版 autoresearch 风格，直接改 TX1 研究源码）
- [x] 等待用户确认 Spec 第 2 段
- [x] 输出 Spec 第 3 段：实施计划、风险与验证口径
- [x] 等待用户 HARD-GATE 确认

## Implementation Plan

### Task 1: 搭建 TX1 autoresearch 运行骨架

- [x] 新建 `skyeye/products/tx1/autoresearch/__init__.py`
- [x] 新建 `skyeye/products/tx1/autoresearch/state.py`，定义 `state.json` / `results.tsv` / run root` 的读写逻辑
- [x] 新建 `skyeye/products/tx1/autoresearch/judge.py`，实现 `keep/discard/crash/frontier/champion` 判定
- [x] 新建 `skyeye/products/tx1/autoresearch/git_ops.py`，封装 baseline commit、实验 commit、discard 回滚、keep 推进
- [x] 新建 `skyeye/products/tx1/autoresearch/runner.py`，封装 baseline run / candidate run / experiment artifact 采集
- [x] 新建 `skyeye/products/tx1/autoresearch/loop.py`，实现主循环与阶段切换
- [x] 新建 `skyeye/products/tx1/run_autoresearch.py`，提供 CLI 入口，解析 `run_tag` / `max_experiments` / `smoke_max_folds` / `full_max_folds`

### Task 2: 接入现有 TX1 研究链路

- [ ] 在 `runner.py` 中默认复用 `run_feature_experiments(...)` 与 `ExperimentRunner.run(...)`，避免重写研究执行协议
- [x] 统一解析 `experiment.json`、fold metrics、robustness 指标，产出 judge 可消费的标准摘要
- [ ] 落盘 run 级目录到 `skyeye/artifacts/experiments/tx1_autoresearch/<run_tag>/`
- [ ] 保持单次实验产物写入独立子目录，禁止覆盖已有实验目录

### Task 3: 实现原版 autoresearch 风格的 keep/discard/rollback 循环

- [ ] 记录每轮实验起点 commit，并在 patch 后创建实验 commit
- [ ] 实验失败或 judge 判 `discard/crash` 时，只回滚本轮实验 commit，不碰用户已有改动
- [ ] 实验通过并被 judge 判 `keep/champion` 时，推进当前分支 head
- [ ] `results.tsv` 保持未追踪，不写入 git 历史
- [ ] 命中只读路径（`skyeye/products/tx1/live_advisor/**`、`skyeye/products/tx1/strategies/rolling_score/runtime.py`）时，直接判 invalid 并回滚

### Task 4: 落地“稳健优先、收益最大化” judge

# TX1 2026-04-17 实盘回测器建议

- [x] 确认 TX1 live advisor 的可执行入口、可用 package 与 `2026-04-17` 数据可达性
- [x] 运行 TX1 live advisor，获取 `2026-04-17` 的推荐结果与组合建议
- [x] 整理基于模型输出的买入建议、风险提示与验证证据

## Review

- 运行入口：
  - `skyeye/products/tx1/run_live_advisor.py` 的 `main()` 会把 `--trade-date / --top-k / --holdings-file` 透传给 `LiveAdvisorService.get_recommendations(...)`。
- stop-serve 规则：
  - `skyeye/products/tx1/live_advisor/runtime_gates.py` 的 `evaluate_snapshot_runtime_gates(...)` 会在 `requested_trade_date` 超过 `latest_available_trade_date` 且超出 `snapshot_max_delay_days` 时返回 `requested_trade_date_stale`，禁止把旧快照伪装成当天建议。
- 组合建议规则：
  - `skyeye/products/tx1/live_advisor/service.py` 的 `_build_portfolio_advice(...)` 会把 `portfolio_policy.buy_top_k` 默认成 `25`。
  - `skyeye/products/tx1/portfolio_proxy.py` 的 `build_target_weights(...)` 在空仓假设下会按 `buy_top_k` 生成等权目标组合。
- 本地只读核对：
  - `env PYTHONPATH=$PWD python -c 'from skyeye.products.tx1.run_baseline_experiment import resolve_data_end; print(resolve_data_end())'`
  - 结果：`2026-04-13 00:00:00`
- 沙箱内首次运行：
  - `env PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mplconfig python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260313 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-04-17 --top-k 25 --universe-cache-root /tmp/tx1_live_universe_cache --format json`
  - 结果：`status=stopped`，`latest_available_trade_date=2026-04-13`，`score_date=2026-04-13`
- 放开网络后重跑：
  - `env PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mplconfig /home/tiger/miniconda3/bin/python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260313 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-04-17 --top-k 25 --universe-cache-root /tmp/tx1_live_universe_cache --format json`
  - 结果：`status=ok`
  - `requested_trade_date=2026-04-17`
  - `latest_available_trade_date=2026-04-17`
  - `score_date=2026-04-17`
  - `raw_data_end_date=2026-04-17`
  - `raw_df shape=(733470, 7)`，`date range: 2015-01-05 – 2026-04-17`
- Top 10 建议：
  - `300433.XSHE 蓝思科技`
  - `601615.XSHG 明阳智能`
  - `002024.XSHE ST易购`
  - `600115.XSHG 中国东航`
  - `600690.XSHG 海尔智家`
  - `601728.XSHG 中国电信`
  - `300182.XSHE 捷成股份`
  - `601006.XSHG 大秦铁路`
  - `600029.XSHG 南方航空`
  - `300253.XSHE 卫宁健康`
- 风险提示：
  - `model_end_date=2026-03-13`，相对 `trade_date=2026-04-17` 落后 `25` 个交易日，处于 `warning`
  - `evidence_end_date=2026-03-13`，相对 `trade_date=2026-04-17` 落后 `25` 个交易日，处于 `warning`
  - top bucket 历史 OOS：`win_rate=48.1%`、`mean_return=+1.08%`、`median_return=-0.32%`
  - 近端 canary 窗口 `2025-12-10 ~ 2026-03-13`：`win_rate=53.1%`、`mean_return=+1.26%`、`median_return=+0.50%`
- 组合层结论：
  - `portfolio_advice.rebalance_due=true`
  - 默认空仓假设下给出 `25` 只股票等权 `4%` 的目标组合
  - `estimated_turnover=50%`

- [x] 以 `stability_score`、`cv`、`positive_ratio`、`max_drawdown`、`mean_turnover`、`flag_ic_decay`、`flag_spread_decay`、`flag_val_dominant` 作为第一层硬门槛
- [x] 在过线候选中用 `net_mean_return`、`top_bucket_spread_mean`、`rank_ic_mean` 做第二层排序
- [x] 输出 `keep/discard/crash/frontier/champion` 原因，避免黑盒决策
- [ ] 支持 baseline、current champion、frontier 候选的三方比较

### Task 5: 落地“两阶段评估”降低实验成本

- [ ] 第一阶段 smoke 默认走 `max_folds=1~2`，快速淘汰明显差解
- [ ] 第二阶段 full 仅对 smoke 过线候选运行完整 fold 评估
- [ ] judge 必须区分 smoke 结果与 full 结果，禁止仅凭 smoke 直接晋升 champion

### Task 6: 补测试并做回归验证

- [x] 新增 `tests/products/tx1/test_autoresearch_state.py`
- [x] 新增 `tests/products/tx1/test_autoresearch_judge.py`
- [x] 新增 `tests/products/tx1/test_autoresearch_git_ops.py`
- [x] 新增 `tests/products/tx1/test_run_autoresearch.py`
- [ ] 回归运行：`tests/products/tx1/test_dataset_builder.py`
- [ ] 回归运行：`tests/products/tx1/test_label_builder.py`
- [ ] 回归运行：`tests/products/tx1/test_baseline_models.py`
- [ ] 回归运行：`tests/products/tx1/test_run_feature_experiment.py`
- [ ] 回归运行：`tests/products/tx1/test_run_baseline_experiment.py`
- [ ] 回归运行：`tests/products/tx1/test_robustness.py`
- [ ] 回归运行：`tests/products/tx1/test_persistence.py`

### Task 7: 真实 smoke loop 验证

- [ ] 跑一次 baseline experiment，确认 `state.json` / `results.tsv` / baseline summary 正常生成
- [ ] 人工构造一次劣化 patch，验证 `discard` 与 git 回滚路径
- [ ] 人工构造一次提升 patch，验证 `keep/champion` 与分支推进路径
- [ ] 用 `compare_experiments.py` 输出对比报告，作为最终验收证据

## Review

- 已在隔离 worktree `.worktrees/tx1-autoresearch` 内完成 autoresearch 第一层骨架：
  - `skyeye/products/tx1/autoresearch/__init__.py`
  - `skyeye/products/tx1/autoresearch/state.py`
  - `skyeye/products/tx1/autoresearch/judge.py`
  - `skyeye/products/tx1/autoresearch/git_ops.py`
  - `skyeye/products/tx1/autoresearch/runner.py`
  - `skyeye/products/tx1/autoresearch/loop.py`
  - `skyeye/products/tx1/run_autoresearch.py`
- 已新增 5 组定向测试：
  - `tests/products/tx1/test_autoresearch_state.py`
  - `tests/products/tx1/test_autoresearch_judge.py`
  - `tests/products/tx1/test_autoresearch_git_ops.py`
  - `tests/products/tx1/test_autoresearch_runner.py`
  - `tests/products/tx1/test_run_autoresearch.py`
- 当前已落地能力：
  - `state.json` / `results.tsv` 初始化与状态推进
  - “稳健优先、收益最大化”的基础 judge
  - git 当前分支 / commit / 改动路径 / 提交 / 回滚封装
  - 实验摘要标准化加载
  - `run_autoresearch` CLI 和 loop 初始化
- 验证证据：
  - worktree 基线：`env PYTHONPATH=$PWD pytest tests/products/tx1/test_config.py tests/products/tx1/test_run_feature_experiment.py tests/products/tx1/test_run_baseline_experiment.py -q`
  - 结果：`29 passed`
  - autoresearch 定向：`env PYTHONPATH=$PWD pytest tests/products/tx1/test_autoresearch_state.py tests/products/tx1/test_autoresearch_judge.py tests/products/tx1/test_autoresearch_git_ops.py tests/products/tx1/test_autoresearch_runner.py tests/products/tx1/test_run_autoresearch.py -q`
  - 结果：`13 passed`
  - 扩大回归：`env PYTHONPATH=$PWD pytest tests/products/tx1/test_autoresearch_state.py tests/products/tx1/test_autoresearch_judge.py tests/products/tx1/test_autoresearch_git_ops.py tests/products/tx1/test_autoresearch_runner.py tests/products/tx1/test_run_autoresearch.py tests/products/tx1/test_config.py tests/products/tx1/test_persistence.py tests/products/tx1/test_run_feature_experiment.py tests/products/tx1/test_run_baseline_experiment.py -q`
  - 结果：`44 passed`
- 当前仍未完成：
  - 真实实验执行器还未接入 `run_feature_experiments(...)` / `ExperimentRunner.run(...)`
  - 主循环还未实现实际 candidate run、discard rollback、keep/champion 推进
  - smoke/full 两阶段评估还未真正跑通

# TX1 2026-04-15 实盘建议复核

- [x] 核对 TX1 live advisor 当前可用 package 与本地数据终点
- [x] 运行 `2026-04-15` 交易日的 live advisor，获取最新建议结果
- [x] 记录命令、输出证据与结论

## Review

- 使用 package：`skyeye/artifacts/packages/tx1/tx1_canary_live_tx1_refresh_20260331_lgbm_20260313/manifest.json`
  - `package_id=tx1_canary_live_tx1_refresh_20260331_lgbm_20260313`
  - `gate_level=canary_live`
  - `data_end_date=2026-04-13`
- 本地只读核对：
  - `env PYTHONPATH=$PWD python - <<'PY' ... print(resolve_data_end()) ... PY`
  - 结果：`resolve_data_end=2026-04-13 00:00:00`
- 首次在沙箱内直接运行：
  - `env MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=$PWD /home/tiger/miniconda3/bin/python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260313 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-04-15 --top-k 10 --universe-cache-root /tmp/tx1_live_universe_cache --format json`
  - 结果：`status=stopped`，`latest_available_trade_date=2026-04-13`，`score_date=2026-04-13`
  - 根因：当前沙箱内 `rqdatac` 无法做 DNS 解析，`DataFacade.provider` 初始化失败，只能退回本地 bundle
- 放开外网访问后重跑同一条命令，成功补齐到 `2026-04-15`：
  - `status=ok`
  - `requested_trade_date=2026-04-15`
  - `latest_available_trade_date=2026-04-15`
  - `score_date=2026-04-15`
  - `raw_data_end_date=2026-04-15`
  - `raw_df shape=(732870, 7)`，`date range: 2015-01-05 – 2026-04-15`
- Top 10 建议：
  - `600115.XSHG`、`002024.XSHE`、`601615.XSHG`、`600029.XSHG`、`601728.XSHG`、`601006.XSHG`、`601212.XSHG`、`300253.XSHE`、`600036.XSHG`、`300027.XSHE`
- 风险提示：
  - `model_end_date=2026-03-13`，相对 `trade_date=2026-04-15` 落后 `23` 个交易日，处于 `warning`
  - `evidence_end_date=2026-03-13`，相对 `trade_date=2026-04-15` 落后 `23` 个交易日，处于 `warning`
- 结果解读：
  - top bucket 历史 OOS：`win_rate=48.1%`、`mean_return=+1.08%`、`median_return=-0.32%`，仍是右偏收益排序器，不是高胜率信号
  - 近端 canary 窗口 `2025-12-10 ~ 2026-03-13`：`win_rate=53.1%`、`mean_return=+1.26%`、`median_return=+0.50%`
  - `portfolio_advice.rebalance_due=true`，默认给出 `25` 只股票等权 `4%` 的目标组合，`estimated_turnover=50%`
- 结论：
  - 之前“数据不是最新”有两层原因：一是本地 bundle 价格数据确实停在旧日期；二是即使补到 `2026-03-31`，现有 TX1 replay 设计也还不足以产出 `2026-04` 的新信号。
  - 按当前切分规则，要出现第 15 个 fold，`labeled_df` 至少要到 `2026-05-04`，对应 `raw_df` 约需要到 `2026-06-01` 左右。

# TX1 实盘辅助双轨重构设计

- [x] 输出 Spec 第 1 段：现状分析与双轨必要性
- [x] 输出 Spec 第 2 段：高目标能力范围与系统边界
- [x] 输出 Spec 第 3 段：核心架构与产物设计
- [x] 输出 Spec 第 4 段：风险、校准口径与上线闸门
- [x] 等待用户 HARD-GATE 确认
- [x] 输出实施计划：文件边界、阶段任务、验证口径
- [x] Task 1：建立 live package 契约、产物目录和 resolver
- [x] Task 2：产出 OOS calibration bundle，并落实 promotion gate
- [x] Task 3：实现 live model final-fit / model bundle / preprocessor bundle
- [x] Task 4：实现当日 snapshot 构建、特征完备性检查和 runtime gate
- [x] Task 5：实现 advisor service / CLI 输出 / 排名与校准解释
- [x] Task 6：补齐单测、集成验证和 canary 验收证据

## 实施计划

### 文件边界

修改文件：
- `skyeye/products/tx1/experiment_runner.py`
  责任：稳定导出 calibration 需要的 `test_df` 预测列和标签列，提供 promotion 侧可复用的 OOS 明细输入。
- `skyeye/products/tx1/persistence.py`
  责任：在不破坏旧实验加载的前提下，补充 live package 需要的 metadata / schema version / 额外引用字段。
- `skyeye/products/tx1/artifacts.py`
  责任：保留 replay artifact 解析，同时新增 live package 的解析与加载入口，避免把 replay signal book 和 live package 混在一起。
- `skyeye/products/tx1/baseline_models.py`
  责任：为 `linear / tree / lgbm` 提供统一的序列化、反序列化和预测接口，支持 promoted package 离线保存与线上加载。
- `skyeye/products/tx1/preprocessor.py`
  责任：暴露 live package 必需的预处理配置摘要、required columns 和配置 hash，保证训练与实盘评分口径一致。
- `skyeye/products/tx1/run_baseline_experiment.py`
  责任：抽出可复用的数据装载、universe 构建和 `raw_df` 构建能力，供 promotion / live snapshot 共用，避免第二套数据链路。

新增文件：
- `skyeye/products/tx1/live_advisor/__init__.py`
  责任：live advisor 子系统入口。
- `skyeye/products/tx1/live_advisor/schema.py`
  责任：定义 manifest、feature schema、gate summary、advisor output schema。
- `skyeye/products/tx1/live_advisor/calibration.py`
  责任：从研究侧 OOS fold 生成 bucket 校准表，并提供 runtime lookup。
- `skyeye/products/tx1/live_advisor/package_io.py`
  责任：读写 promoted package 六件套：`manifest / feature_schema / preprocessor_bundle / model_bundle / calibration_bundle / portfolio_policy`。
- `skyeye/products/tx1/live_advisor/promotion.py`
  责任：把研究实验提升为 live package，并执行 promotion gate。
- `skyeye/products/tx1/live_advisor/snapshot.py`
  责任：构建指定交易日的 live universe 与特征快照。
- `skyeye/products/tx1/live_advisor/runtime_gates.py`
  责任：实现 freshness、feature completeness、universe coverage、score sanity 的 stop-serve 规则。
- `skyeye/products/tx1/live_advisor/service.py`
  责任：串起 package 加载、实时评分、bucket 映射、推荐卡片生成。
- `skyeye/products/tx1/run_live_advisor.py`
  责任：提供日频实盘辅助 CLI。

测试文件：
- `tests/products/tx1/test_live_advisor_calibration.py`
- `tests/products/tx1/test_live_advisor_promotion.py`
- `tests/products/tx1/test_live_advisor_package_io.py`
- `tests/products/tx1/test_live_advisor_snapshot.py`
- `tests/products/tx1/test_live_advisor_runtime_gates.py`
- `tests/products/tx1/test_live_advisor_service.py`
- `tests/products/tx1/test_baseline_models.py`
- `tests/products/tx1/test_persistence.py`
- `tests/products/tx1/test_artifacts.py`
- `tests/products/tx1/test_run_baseline_experiment.py`

### Task 1：建立 live package 契约、产物目录和 resolver

目标：
- 在现有 replay artifact 体系外，新增一层独立的 live package 标准，保证未来因子、模型、算法升级时，advisor 只依赖标准产包，不依赖具体实验实现。

实施点：
- 在 `live_advisor/schema.py` 定义 `package_id / source_experiment / package_type / horizon / fit_end_date / universe_id / required_features / hashes / gate_summary / created_at` 等 manifest 字段。
- 在 `live_advisor/package_io.py` 定义 promoted package 目录规范，根目录固定为 `skyeye/artifacts/packages/tx1/<package_id>/`。
- 在 `artifacts.py` 增加 live package resolver，要求缺失任一关键文件时直接报错，不允许 silent fallback。
- 在 `persistence.py` 保持对旧 `experiment.json` 的兼容，同时允许记录 package 引用与 schema version。

验证命令：
- `PYTHONPATH="$PWD" pytest tests/products/tx1/test_live_advisor_package_io.py tests/products/tx1/test_artifacts.py tests/products/tx1/test_persistence.py -q`

完成标准：
- 可以从磁盘加载 promoted package 元数据。
- replay artifact 与 live package 两套加载逻辑边界清楚，互不污染。
- 缺 manifest / calibration / model bundle 时，测试明确失败。

### Task 2：产出 OOS calibration bundle，并落实 promotion gate

目标：
- 把研究回测的 `test_df` 预测结果压成 live advisor 可直接消费的历史校准表，并把 spec 中定义的 promotion gate 做成机器可判定规则。

实施点：
- 在 `live_advisor/calibration.py` 实现：
  - 仅消费 `ExperimentRunner.run` 产出的 `fold_results[*].predictions_df`，不读取 `train_df / val_df`。
  - 先对每个交易日计算 `score_rank_pct`，再映射到固定 bucket。
  - 为每个 bucket 计算 `sample_count / win_rate / mean_return / median_return / return_quantiles / volatility_quantiles / max_drawdown_quantiles`。
- 在 `experiment_runner.py` 确保 `predictions_df` 持续包含 `prediction / prediction_ret / label_return_raw / label_volatility_horizon / label_max_drawdown_horizon / reliability_score` 等校准必需列。
- 在 `live_advisor/promotion.py` 实现 promotion gate：
  - `num_folds >= 12`
  - 每个对外 bucket 样本数 `>= 300`
  - `rank_ic_mean >= 0.03`
  - `top_bucket_spread_mean >= 0.005`
  - `flag_ic_decay = false`
  - `flag_spread_decay = false`
  - `flag_val_dominant = false`
  - `canary_live` 额外要求 `positive_ratio >= 0.70`
  - `default_live` 额外要求 `cv <= 0.60` 且 `stability_score >= 50`
- promotion 结果必须落到 `manifest.gate_summary`，供 runtime 和人工审计共用。

验证命令：
- `PYTHONPATH="$PWD" pytest tests/products/tx1/test_live_advisor_calibration.py tests/products/tx1/test_live_advisor_promotion.py tests/products/tx1/test_experiment_runner.py -q`

完成标准：
- 校准只使用 OOS test 样本。
- bucket 统计可直接回答“胜率、预期收益、风险区间”。
- 当前 `tx1_refresh_20260331_lgbm` 会被判定为 `canary_live` 而非 `default_live`。

### Task 3：实现 live model final-fit / model bundle / preprocessor bundle

目标：
- 在研究实验通过 gate 后，生成一个真正可用于实时评分的 final-fit 模型包，而不是继续依赖历史 fold 的回放结果。

实施点：
- 在 `baseline_models.py` 为三类模型补齐 `dump / load / predict` 统一接口，保证序列化前后预测一致。
- 在 `live_advisor/promotion.py` 增加 final-fit 流程：
  - 读取实验 config。
  - 复用 `DatasetBuilder / LabelBuilder / FeaturePreprocessor`。
  - 用当前可得标签数据重训一份 live model，`fit_end_date` 记录到 manifest。
  - 将 `model_bundle`、`preprocessor_bundle`、`feature_schema.json` 一起写入 package。
- 在 `preprocessor.py` 暴露 required feature 列和配置摘要，避免 runtime 只凭列名猜口径。
- 在 `run_baseline_experiment.py` 抽出 live promotion 可复用的数据装载接口，禁止复制一套 raw data 逻辑。

验证命令：
- `PYTHONPATH="$PWD" pytest tests/products/tx1/test_baseline_models.py tests/products/tx1/test_live_advisor_promotion.py tests/products/tx1/test_preprocessor.py tests/products/tx1/test_run_baseline_experiment.py -q`

完成标准：
- promoted package 内的模型可独立加载并对 snapshot 打分。
- 同一批输入下，序列化前后预测误差在测试中严格对齐。
- manifest 明确记录 `fit_end_date / horizon / model_kind / feature_schema_hash`。

### Task 4：实现当日 snapshot 构建、特征完备性检查和 runtime gate

目标：
- 对指定交易日生成 live universe 的完整评分输入，并在输入不合格时拒绝出分。

实施点：
- 在 `live_advisor/snapshot.py` 复用 `run_baseline_experiment.py` 的 universe / raw_df / dataset 构建逻辑，新增按单日或截止某日构建 snapshot 的入口。
- snapshot 输出必须包含：
  - `trade_date`
  - `eligible_universe`
  - `dropped_reasons`
  - `feature_coverage_summary`
  - `snapshot_features`
- 在 `live_advisor/runtime_gates.py` 实现 stop-serve：
  - 数据 freshness 超过 1 个交易日拒绝出分。
  - required feature 缺失拒绝整体或剔除个股。
  - eligible universe 小于 `max(300, 近 60 个交易日中位数的 80%)` 时拒绝整体出分。
  - score 分布塌缩、几乎常数或 top spread 畸低时拒绝整体出分。
- 对个股输出结构化剔除原因，避免用户看到“今天没推荐”却不知道为什么。

验证命令：
- `PYTHONPATH="$PWD" pytest tests/products/tx1/test_live_advisor_snapshot.py tests/products/tx1/test_live_advisor_runtime_gates.py tests/products/tx1/test_run_baseline_experiment.py -q`

完成标准：
- 可以稳定构建指定日期 snapshot。

# TX1 实盘建议器指标同值排查

- [x] 复现 `run_live_advisor` 在不同 `trade_date` 下输出一致的现象
- [x] 沿 `skyeye.products.tx1.run_live_advisor` 调用链定位胜率/均值收益指标来源
- [x] 确认这些指标按股票、按日期是否本就设计为包级静态校准值，还是错误地忽略了日期/股票维度
- [x] 输出根因结论，并给出是否属于 bug 的判断依据

## Review

- 本地复现：
  - `--trade-date 2026-02-01` 实际解析到 `score_date=2026-01-30`
  - `--trade-date 2026-03-01` 实际解析到 `score_date=2026-02-27`
  - 两次 `top_k=25` 的推荐股票列表不同，但 `win_rate / mean_return / median_return / sample_count` 完全一致。
- 根因：
  - `skyeye/products/tx1/live_advisor/service.py` 先按当日分数排序，再把 `rank` 转成 `percentile`，随后仅用这个分位去 `lookup_calibration_bucket(...)`，把 bucket 级历史统计回填到每只股票。
  - `skyeye/products/tx1/live_advisor/calibration.py` 的 `build_calibration_bundle(...)` 也是按历史 OOS `score_rank_pct` 分桶后，聚合得到每个 bucket 的 `win_rate / mean_return / median_return / quantiles / sample_count`。
  - 当前 package 的 `calibration_bundle.json` 里 top bucket `b09` 的统计固定为：`win_rate=0.4812`、`mean_return=0.0108`、`median_return=-0.0032`、`sample_count=49952`。在 300 只股票 universe 下，`top_k=25` 的分位区间约为 `92%~100%`，全部落在 `b09`，所以不同股票显示同一组指标是设计结果，不是计算 bug。
- 额外发现：
  - `snapshot.py` 会把非交易日请求回退到“不晚于请求日的最近可用交易日”。

# TX1 QuotaExceeded 显式报错保护

- [x] 梳理 `DataFacade` 当前 provider 异常吞掉与 bundle 回退路径
- [x] 先补测试，覆盖 `QuotaExceeded` 时显式报错且不回退 bundle
- [x] 实现最小修复
- [x] 运行相关测试并记录证据

## Review

- 代码修正：
  - `skyeye/data/facade.py` 新增 `_raise_if_quota_exceeded(...)`，统一识别 `rqdatac.share.errors.QuotaExceeded`。
  - `get_daily_bars / get_trading_dates / all_instruments / index_components / index_weights / get_factor` 遇到 `QuotaExceeded` 时改为显式抛错，不再静默吞掉。
  - 其中 `get_daily_bars` 与 `all_instruments` 不再在配额耗尽时回退到本地 bundle，避免把旧数据误当在线最新数据。
- 测试验证：
  - 新增 3 个测试，覆盖 `get_daily_bars / all_instruments / get_factor` 的 `QuotaExceeded` 行为。
  - `PYTHONPATH="$PWD" pytest tests/products/tx1/test_run_baseline_experiment.py -q` 通过，`8 passed`。
  - 当前 runtime gate 没有阻止“用 `fit_end_date=2026-03-03` 的 live package 去回看 `2026-02-01` / `2026-03-01` 这类更早日期”，所以如果目的是做严格历史对比，这条链路本身不合适，应该走 `research / replay`。

# TX1 bundle 优先补缺口设计

- [x] 输出 Spec 第 1 段：现状分析
- [x] 输出 Spec 第 2 段：功能点与数据访问策略
- [x] 输出 Spec 第 3 段：风险与决策
- [x] 等待用户 HARD-GATE 确认
- [x] 先补失败测试，覆盖 bundle -> SQLite -> rqdatac 的读取顺序、缺口补数与配额报错
- [x] 实现 `skyeye/data` 本地 SQLite 缓存层与 bundle 读取层
- [x] 重构 `DataFacade` 接入三层读取与在线回写缓存逻辑
- [x] 运行定向测试与必要回归
- [x] 在 review 中记录行为边界、验证证据与后续迁移注意项

## Review

- 数据层重构：
  - 新增 `skyeye/data/bundle_reader.py`，把 bundle 读取从 `DataFacade` 私有分支中抽离出来，统一负责 `trading_dates / instruments / daily_bars` 的只读访问。
  - 新增 `skyeye/data/cache_store.py`，用 SQLite 持久化 `daily_bars / factor_values / trading_dates / snapshot_cache / coverage_checkpoint / fetch_audit / cache_meta`。
  - `skyeye/data/facade.py` 已改成固定三层顺序：`bundle -> SQLite -> rqdatac`；仅对本地缺口触发在线补数，在线成功后立即回写 SQLite。
- 行为边界：
  - `get_daily_bars` 以“证券 + 复权方式 + 字段”粒度维护覆盖区间；bundle 与 SQLite 对同一日期冲突时，仍以 bundle 为先，SQLite 只补 bundle 缺字段或缺日期。
  - `get_trading_dates` 以 bundle 覆盖区间 + SQLite coverage checkpoint 判定是否需要在线补日历，避免因为周末/节假日误判为缺口并重复消耗流量。
  - `all_instruments / index_components / index_weights` 已支持按 snapshot key 缓存；`date=None` 的 instruments 仍优先读 bundle，带日期查询则走 `SQLite -> rqdatac`。
  - `get_factor` 以“证券 + 因子名”粒度缓存与判定覆盖；缺口在线补齐后，后续重复运行同一区间不会再次消耗流量。
  - 只要本地存在缺口且在线补数遇到 `QuotaExceeded`，仍会显式抛错，不会静默伪装成“本地已有最新数据”。
- 兼容性补充：
  - `skyeye/products/dividend_low_vol/scorer/data_fetcher.py` 对 `RQDataProvider()` 初始化失败做了延迟失败处理；离线/单测场景不再因为 DNS 或配额问题在构造阶段直接炸掉，真正调用在线接口时才会明确报错。
  - 默认 SQLite 路径优先使用 `~/.rqalpha/skyeye_data_cache.sqlite3`；在当前受限执行环境里若该路径不可写，会回退到 `/tmp/skyeye_data_cache.sqlite3`，避免测试阶段因沙箱权限阻断。
- 测试验证：
  - 新增 `tests/unittest/test_data/test_skyeye_data_facade.py`，覆盖：
    - bundle 已完整覆盖时不触网
    - bundle 缺口仅在线补一次并持久化到 SQLite
    - trading dates / instruments / factors 的在线补数与二次命中缓存
    - 本地缺口遇到 `QuotaExceeded` 时显式报错
  - `PYTHONPATH="$PWD" pytest tests/unittest/test_data/test_skyeye_data_facade.py -q` 通过，`7 passed`
  - `PYTHONPATH="$PWD" pytest tests/products/tx1/test_run_baseline_experiment.py -q` 通过，`8 passed`
  - `PYTHONPATH="$PWD" pytest tests/unittest/test_data/test_skyeye_data_facade.py tests/products/dividend_low_vol/scorer/test_scorer.py tests/products/tx1/test_run_baseline_experiment.py -q` 通过，`36 passed`
  - `PYTHONPATH="$PWD" pytest tests/unittest/test_data/test_skyeye_data_facade.py tests/products/tx1 -q` 通过，`171 passed`

# TX1 实盘使用建议梳理

- [x] 梳理 TX1 当前实盘辅助入口、产物依赖和运行时闸门
- [x] 核对当前 package / artifact 状态与最近实验结论
- [x] 基于代码现状输出可执行的实盘使用建议与风险边界

## Review

- 当前 TX1 的实时入口是 `skyeye/products/tx1/run_live_advisor.py`，它调用 `LiveAdvisorService.get_recommendations()` 读取 promoted package、构建 snapshot、过 runtime gate 后输出推荐卡片；这条链路不直接下单。
- `skyeye/products/tx1/strategies/rolling_score/strategy.py` 仍然是 replay frozen signal book 的执行桥，`before_trading()` 只取上一交易日的历史 signal；结合 `skyeye/artifacts/experiments/tx1/tx1_refresh_20260331_lgbm/experiment.json` 最后一折 `test_end=2025-11-18`，说明它不适合作为 2026-04 的实时信号源。
- 当前可用 live package 是 `skyeye/artifacts/packages/tx1/tx1_live_canary_refresh_20260331_lgbm/manifest.json`，`package_type=canary_live`，`default_live_passed=false`，失败点是 `stability_score=14.29 < 50`、`cv=0.885 > 0.60`，所以更适合影子盘 / 小仓试运行。
- package 的 `portfolio_policy.json` 仍然对应 `Top25 买入 / Top45 持有 / 20 交易日调仓 / holding_bonus=0.5`；但 live advisor 当前只返回推荐卡片，不会自动把该 policy 落成目标持仓，实盘需要人工或额外执行层来维护持仓与调仓。
- 校准包显示顶层 bucket（`0.9~1.0` 分位）20 日 horizon 下 `mean_return≈1.08%`、`win_rate≈48.1%`、`median_return≈-0.32%`、`max_drawdown p75≈10.7%`，说明它更像“正期望分组排序器”，适合分散持有，不适合重仓押单票。

# TX1 live advisor 表格输出优化

- [x] 梳理 CLI 表格输出和股票名数据来源
- [x] 先补失败测试，覆盖股票名列和中文指标说明
- [x] 实现表格渲染优化并完成验证

## Review

- 本次仅修改 `skyeye/products/tx1/run_live_advisor.py` 的表格渲染层，不改 `LiveAdvisorService` 的返回契约。
- 表格输出新增 `股票名` 列，名称通过 `DataFacade.all_instruments(type="CS")` 建立 `order_book_id -> symbol` 映射；名称缺失时保留 `-` 回退。
- 表格头部新增中文“指标说明”，明确 `pct / win_rate / median_ret / p25~p75 / samples` 的含义，便于直接阅读实盘输出。
- 先新增 `tests/products/tx1/test_run_live_advisor.py` 让需求失败，再补实现；验证命令：`PYTHONPATH="$PWD" ./.venv/bin/pytest tests/products/tx1/test_run_live_advisor.py tests/products/tx1/test_live_advisor_service.py -q`，结果 `2 passed`。

# TX1 live advisor 可读性优化

- [x] 梳理当前表格痛点与可解释性缺口
- [x] 先补失败测试，覆盖中文列名、易读数值、颜色和结果解读
- [x] 实现终端表格可读性优化并完成验证

## Review

- 本次仍只修改 `skyeye/products/tx1/run_live_advisor.py` 的 CLI 渲染层，不改 `LiveAdvisorService` 的输出契约与校准逻辑。
- 输出新增 `结果解读` 区块：当 `mean_return > 0` 且 `median_return < 0` 时，直接解释为“右偏收益排序器，不是高胜率信号”，并把胜率、均值收益、中位收益按百分比打印出来。
- 表格已改成中文列名与易读格式：`代码 / 股票名 / 排名 / 分位 / 胜率 / 均值收益 / 中位收益 / P25~P75 / 样本数`；百分比统一用 `%`，收益保留 2 位小数，样本数加千分位。
- 已加入终端颜色输出支持：`status/gate_level`、胜率、均值收益、中位收益会在支持 ANSI 的终端里着色；同时用显示宽度感知对齐解决中文列错位。
- TDD 验证：
  - 先让 `tests/products/tx1/test_run_live_advisor.py` 因缺少 `_supports_color_output` 与新格式失败，再补实现。
  - 回归命令：`PYTHONPATH="$PWD" ./.venv/bin/pytest tests/products/tx1/test_run_live_advisor.py tests/products/tx1/test_live_advisor_service.py -q`
  - 结果：`3 passed`
- 真实命令验证：
  - `MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH="$PWD" /home/tiger/miniconda3/bin/python -m skyeye.products.tx1.run_live_advisor --package-id tx1_live_canary_refresh_20260331_lgbm --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-03-31 --top-k 20 --format table`
  - 结果：`status=ok`，解释区块与新表格已生效。

# TX1 文档入口与 PLAYBOOK 同步

- [x] 梳理 TX1 文档入口与启动时可见的索引位置
- [x] 更新 `PLAYBOOK.md` 中 live advisor 的命令和输出说明
- [x] 更新 `README.md` 与 `CLAUDE.md` 的 TX1 文档入口
- [x] 验证索引路径和命令说明可达

## Review

- 已更新 `skyeye/products/tx1/PLAYBOOK.md`：
  - 在顶部入口列表中明确实盘辅助任务直接看“TX1 实盘辅助使用说明”。
  - live advisor 的 table 命令示例改为 `--top-k 25`，并补充“使用依赖完整的 Python 环境”说明。
  - “输出怎么读”已同步到新版 CLI 口径：先看 `status / score_date / gate_level`，再看 `结果解读`，最后看 `代码 / 股票名 / 排名 / 分位 / 胜率 / 均值收益 / 中位收益 / P25~P75 / 样本数`。
  - 已补“右偏收益排序器，不是高胜率信号”的解释，避免把 `win_rate < 50%` 和 `mean_return > 0` 当成矛盾。
- 已更新 `skyeye/products/tx1/README.md`：
  - 新增“阅读顺序”，把 `PLAYBOOK.md -> strategies/rolling_score/README.md -> 历史 RFC` 固化成默认阅读路径。
  - 明确如果任务涉及 `run_live_advisor` 或实盘辅助，优先看 `PLAYBOOK.md` 的 live advisor 章节。
  - 当前状态描述已改为“默认 replay 执行线 + canary 级 live advisor 辅助链路”。
- 已更新根级 `CLAUDE.md`：
  - 新增 `TX1 Quick Entry`，要求涉及 TX1 任务时先读 `skyeye/products/tx1/README.md`、`PLAYBOOK.md`、`strategies/rolling_score/README.md`。
  - 明确 `PLAYBOOK.md` 是 TX1 操作层单一真相源，并补了 `run_live_advisor` / replay 策略两条常用入口命令。
- 验证方式：
  - 通过 `nl -ba` 检查 3 个文件对应段落已落盘。
  - 通过 `rg -n "TX1 Quick Entry|阅读顺序|run_live_advisor|结果解读|右偏收益排序器|PLAYBOOK.md"` 确认索引关键词和命令入口可检索。
  - 本次是文档修改，没有额外运行测试用例。
- 任一 mandatory input 缺失时，系统明确 stop-serve，而不是偷偷回退旧数据。
- 个股剔除原因可追踪。

### Task 5：实现 advisor service / CLI 输出 / 排名与校准解释

目标：
- 把 promoted package 和当日 snapshot 组合成你真正能看的“今日建议卡片”，并且只展示校准后的口径。

实施点：
- 在 `live_advisor/service.py` 实现主流程：
  - 加载 promoted package。
  - 对 snapshot 运行 preprocessor 和 model。
  - 计算 live universe 内的 `rank / percentile`。
  - 根据 `score_rank_pct` 映射 calibration bucket。
  - 生成推荐结果：`order_book_id / rank / percentile / win_rate / mean_return / median_return / return_quantile_range / mdd_quantile_range / sample_count / warnings`。
- 严禁对外暴露原始 `prediction` 数值；CLI 只显示校准后的解释字段。
- 在 `run_live_advisor.py` 提供：
  - `--package-id`
  - `--trade-date`
  - `--top-k`
  - `--format table|json`
  - `--include-dropped`
- 输出中必须明确 `package_id / fit_end_date / score_date / gate_level`。

验证命令：
- `PYTHONPATH="$PWD" pytest tests/products/tx1/test_live_advisor_service.py tests/products/tx1/test_live_advisor_package_io.py -q`

完成标准：
- CLI 能输出当日 top-k 推荐及解释。
- 用户能看到排名、胜率、预期收益、风险区间、样本数。
- 原始 score 不会出现在对外输出中。

### Task 6：补齐单测、集成验证和 canary 验收证据

目标：
- 在开始长期使用前，给出研究侧、promotion 侧、runtime 侧三层证据，确保双轨系统不是纸面设计。

实施点：
- 单测层：跑完 `tests/products/tx1/` 下新增与受影响用例。
- 集成层：
  - 以 `skyeye/artifacts/experiments/tx1/tx1_refresh_20260331_lgbm` 为输入，构建一次 promoted package。
  - 对最新可用交易日运行一次 `run_live_advisor.py`，确认能输出 top-k 卡片或正确 stop-serve。
- 回归层：
  - 确认 `strategies/rolling_score/strategy.py` 与 `strategies/rolling_score/runtime.py` 的 replay 行为未被破坏。
  - 旧 artifact 加载、旧测试继续通过。
- canary 验收：
  - package manifest 正确标记当前包仅为 `canary_live`。
  - 推荐输出包含 `package_id / gate_level / sample_count / warnings`。

验证命令：
- `PYTHONPATH="$PWD" pytest tests/products/tx1 -q`
- `PYTHONPATH="$PWD" python -m skyeye.products.tx1.run_live_advisor --package-id <promoted_package_id> --trade-date 2026-03-31 --top-k 20 --format table`

完成标准：
- 单测与集成验证都有证据。
- replay 研究链路不回归。
- live advisor 至少能形成 canary 级可审计输出。

## Review

- 已完成 `TX1 live_advisor` 子系统首版落地，新增：
  - `live_advisor/package_io.py`、`live_advisor/schema.py`、`live_advisor/calibration.py`
  - `live_advisor/promotion.py`、`live_advisor/snapshot.py`、`live_advisor/runtime_gates.py`
  - `live_advisor/service.py`
  - `run_live_advisor.py`
- 已扩展底层能力：
  - `artifacts.py` 新增 live package resolver。
  - `baseline_models.py` 新增模型 bundle dump/load，支持单头与多头。
  - `preprocessor.py` 新增 bundle 导出与恢复。
  - `run_baseline_experiment.py` 新增 `build_live_raw_df()`，供 live snapshot 复用真实数据链路。
- 测试验证：
  - `PYTHONPATH="$PWD" pytest tests/products/tx1 -q`
  - 结果：`153 passed`
- 真实链路 smoke test：
  - 用真实实验 `skyeye/artifacts/experiments/tx1/tx1_refresh_20260331_lgbm` + 真数据 `universe_size=10` 成功提升 package：
    - `skyeye/artifacts/packages/tx1/tx1_live_canary_refresh_20260331_lgbm_u10`
    - `gate_level=canary_live`
    - `fit_end_date=2026-03-03`
    - `data_end_date=2026-03-31`
  - advisor CLI 验证命令：
    - `PYTHONPATH="$PWD" python -m skyeye.products.tx1.run_live_advisor --package-id tx1_live_canary_refresh_20260331_lgbm_u10 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-03-31 --top-k 5 --universe-size 10 --format table`
- 验证结果：
  - `status=ok`
  - 成功输出 `2026-03-31` 的 top-5 推荐、排名、胜率、中位收益、样本数。

# TX1 live advisor 4月实盘有效性分析

- [x] 核对 `run_live_advisor` / `LiveAdvisorService` / promotion / snapshot 的数据链路
- [x] 核对当前 live package 与 refresh artifact 的日期边界
- [x] 复现默认 CLI 的包路径行为并确认失败原因
- [x] 整理“为什么对 2026-04 实盘指导意义弱”的代码级结论

## Review

- 默认 CLI 会去 `/home/tiger/rqalpha/artifacts/packages/tx1` 找包，而当前实际 package 在 `/home/tiger/rqalpha/skyeye/artifacts/packages/tx1`，不传 `--packages-root` 时直接 `FileNotFoundError`。
- 当前 package `tx1_live_canary_refresh_20260331_lgbm` 的 `data_end_date=2026-03-31`，但 `fit_end_date=2026-03-03`；也就是 2026-04 运行时最多只能拿到 3 月底特征，模型监督标签只到 3 月 3 日。
- 当前 package 仅是 `canary_live`，不是 `default_live`；`stability_score=14.2857`、`cv=0.8854` 均未过默认实盘闸门。
- refresh 实验虽然把行情补到 `2026-03-31`，但 walk-forward 最后一个 fold 的 `test_end` 仍停在 `2025-11-18`，所以 live advisor 的校准解释并不覆盖 `2025-11-19` 之后的新 OOS 区间。
- `LiveAdvisorService` 当前只输出 top-k 排名和历史 bucket 统计，不消费 `portfolio_policy` 做持仓约束、调仓建议或仓位建议，因此它更像“排序参考”而不是“2026-04 实盘执行助手”。

# TX1 live advisor 实盘化增强

- [x] 梳理 freshness / canary 证据 / 持仓建议 三项需求的现状与边界
- [x] 输出 Spec 第 1 段：现状分析
- [x] 输出 Spec 第 2 段：方案选项与推荐
- [x] 输出 Spec 第 3 段：风险、接口与验证口径
- [x] 等待用户 HARD-GATE 确认
- [x] 先写失败测试并验证失败
- [x] 实现 freshness gate、近端 canary 证据与持仓调仓输出
- [x] 运行验证并记录证据

## Review

- freshness：
  - `build_live_snapshot()` 现在会显式返回 `requested_trade_date / latest_available_trade_date / requested_vs_available_trading_gap`。
  - `evaluate_snapshot_runtime_gates()` 改为优先用“请求日 vs 最新可用快照日”的交易日差判定 stale；超过 1 个交易日直接 stop-serve。
  - stop-serve 结果会带结构化 `warnings`，`run_live_advisor.render_table()` 顶部会用红色 ANSI warning 打印请求日与最新可用快照日，禁止把旧快照伪装成当日建议。
- 近端 canary 证据：
  - promotion 阶段新增 `recent_canary_bundle`，由 shadow canary model 在近端已标注窗口上生成单独证据，不再只依赖历史 OOS bucket。
  - package IO 改成“必需组件 + 可选组件”模式，支持 `recent_canary_bundle.json` 向后兼容加载。
  - service 对每条推荐会额外返回 `recent_canary_evidence`，CLI 表格会补一段 `近端Canary` 摘要。
- 组合建议：
  - 新增 `live_advisor/holdings_io.py`，支持 `--holdings-file` 读取 CSV / JSON 持仓并归一化。
  - `LiveAdvisorService.get_recommendations()` 现在支持 `current_holdings`、`last_rebalance_date`，会结合 `portfolio_policy` 输出 `portfolio_advice`：
    - `rebalance_due`
    - `current_weights`
    - `target_weights`
    - `weight_deltas`
    - `estimated_turnover`
    - `actions`
  - `PortfolioProxy` 抽出单日 `build_target_weights()`，保证 live advisor 和研究侧组合规则一致。
- 测试验证：
  - 定向红绿测试：
    - `PYTHONPATH="$PWD" ./.venv/bin/pytest tests/products/tx1/test_live_advisor_snapshot.py tests/products/tx1/test_live_advisor_runtime_gates.py tests/products/tx1/test_run_live_advisor.py tests/products/tx1/test_live_advisor_package_io.py tests/products/tx1/test_live_advisor_promotion.py tests/products/tx1/test_live_advisor_service.py tests/products/tx1/test_live_advisor_holdings_io.py -q`
    - 结果：`18 passed`
- 全量 TX1 回归：
  - `PYTHONPATH="$PWD" ./.venv/bin/pytest tests/products/tx1 -q`
  - 结果：`161 passed`

# TX1 实盘建议器完成度盘点

- [x] 梳理 `run_live_advisor`、`LiveAdvisorService`、promotion/package、snapshot/runtime gate 的当前实现边界
- [x] 核对测试覆盖、最新 package 产物与最近一次真实链路验证证据
- [x] 基于代码与产物证据评估完成度，并整理高优先级优化项

## Review

- 当前 `TX1` 实盘建议器已经不是“原型”，而是完成了可运行的 `package -> snapshot -> gate -> score -> calibration -> portfolio_advice -> CLI` 闭环；但仍停在 `canary_live`，未达到 `default_live`。
- 当前最强 package 为 `skyeye/artifacts/packages/tx1/tx1_canary_live_tx1_refresh_20260331_lgbm_20260303/manifest.json`：
  - `fit_end_date=2026-03-03`
  - `data_end_date=2026-03-31`
  - `rank_ic_mean=0.0554`
  - `top_bucket_spread_mean=0.0115`
  - `stability_score=14.29` / `cv=0.8854`，因此只能 `canary_live`，不能 `default_live`
- 当前验证证据：
  - `env PYTHONPATH=$PWD ./.venv/bin/pytest tests/products/tx1/test_live_advisor_*.py tests/products/tx1/test_run_live_advisor.py -q`
  - 结果：`25 passed`
  - `env PYTHONPATH=$PWD /home/tiger/miniconda3/bin/python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260303 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-04-10 --top-k 3 --format json`
  - 结果：`status=stopped`，freshness gate 正常拦截 `2026-04-10` 请求，不会拿 `2026-03-31` 旧快照冒充当天建议
  - `env MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=$PWD /home/tiger/miniconda3/bin/python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260303 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-03-31 --top-k 3 --format json`
  - 结果：`status=ok`，可输出 top-k、historical OOS bucket、recent canary evidence、portfolio_advice
- 高优先级优化结论：
  - 第一优先级是把研究侧稳定性做上来，解决 `default_live` 闸门不过的问题；现在最大短板不是“能不能跑”，而是“稳不稳定”。
  - 第二优先级是缩短 `fit_end_date` 与 `trade_date` 的时间断层；当前 3 月 31 日出分时，监督标签只到 3 月 3 日，近端证据窗口也只到 3 月 3 日。
- 第三优先级是补强在线/离线数据源健壮性；真实 CLI 运行时 `northbound` 与 `fundamental factors` 在当前环境里会降级为 unavailable，现阶段默认包还能跑是因为所需特征只依赖价格/流动性因子。

# TX1 实盘建议器四项问题设计与修复

- [x] 输出 Spec 第 1 段：现状分析与四项问题范围确认
- [x] 输出 Spec 第 2 段：功能点 / 方案选项 / 推荐方案
- [x] 输出 Spec 第 3 段：风险、接口、验证口径与实施边界
- [x] 等待用户 HARD-GATE 确认
- [x] 写实现计划并更新任务拆分
- [ ] 按确认后的方案实现四项修复
- [ ] 运行验证并记录证据

### 实现计划

- [x] Task 1：先写并跑失败测试，锁住 package 契约升级
  - 目标：覆盖 `label_end_date / evidence_end_date / canary_reason / data_dependency_summary / freshness_policy`
  - 文件：`tests/products/tx1/test_live_advisor_package_io.py`、`tests/products/tx1/test_live_advisor_promotion.py`
- [x] Task 2：先写并跑失败测试，锁住 runtime freshness 分层 gate
  - 目标：覆盖 `snapshot/model/evidence` 三层 freshness 的 `warn/stop` 语义
  - 文件：`tests/products/tx1/test_live_advisor_runtime_gates.py`、`tests/products/tx1/test_live_advisor_service.py`
- [x] Task 3：先写并跑失败测试，锁住 live raw panel 的依赖裁剪与 source diagnostics
  - 目标：覆盖按 `required_features` 跳过非必要 `northbound/fundamental`，并返回 source summary
  - 文件：`tests/products/tx1/test_run_baseline_experiment.py`、必要时补 `test_live_advisor_snapshot.py`
- [x] Task 4：先写并跑失败测试，锁住 execution-aware portfolio advice
  - 目标：覆盖 `preflight_checks / execution_blockers / advice_level`
  - 文件：`tests/products/tx1/test_live_advisor_service.py`、`tests/products/tx1/test_run_live_advisor.py`
- [x] Task 5：按最小实现补 schema / promotion / package_io / runtime_gates / service / CLI / raw_df
- [x] Task 6：跑定向回归与真实链路验证，记录证据

## Review

- 已按方案 B 完成四项修复，落地范围：
  - package 契约：promotion 新写入 `label_end_date / evidence_end_date / canary_reason / data_dependency_summary / freshness_policy`
  - runtime gate：`evaluate_snapshot_runtime_gates()` 现支持 `snapshot/model/evidence` 三层 freshness，并返回 `warnings / diagnostics / metrics`
  - live raw panel：`build_live_raw_df()` / `build_raw_df()` 新增 `required_features` 驱动的依赖裁剪，baseline_5f 这类 runtime 不再无条件拉 `northbound/fundamental`
  - execution advice：`portfolio_advice` 新增 `cash_buffer / preflight_checks / execution_blockers / advice_level`，CLI 也会展示 `label_end_date / evidence_end_date / 执行阻塞`
- 关键实现文件：
  - `skyeye/products/tx1/run_baseline_experiment.py`
  - `skyeye/products/tx1/live_advisor/promotion.py`
  - `skyeye/products/tx1/live_advisor/runtime_gates.py`
  - `skyeye/products/tx1/live_advisor/service.py`
  - `skyeye/products/tx1/live_advisor/snapshot.py`
  - `skyeye/products/tx1/live_advisor/schema.py`
  - `skyeye/products/tx1/run_live_advisor.py`
- 新增回归覆盖：
  - `tests/products/tx1/test_live_advisor_promotion.py`
  - `tests/products/tx1/test_live_advisor_runtime_gates.py`
  - `tests/products/tx1/test_live_advisor_service.py`
  - `tests/products/tx1/test_run_baseline_experiment.py`
  - `tests/products/tx1/test_run_live_advisor.py`
- 验证证据：
  - 定向红绿：`env PYTHONPATH=$PWD ./.venv/bin/pytest tests/products/tx1/test_live_advisor_promotion.py tests/products/tx1/test_live_advisor_runtime_gates.py tests/products/tx1/test_live_advisor_service.py tests/products/tx1/test_run_baseline_experiment.py tests/products/tx1/test_run_live_advisor.py -q`
  - 结果：`31 passed`
  - 全量 TX1：`env PYTHONPATH=$PWD ./.venv/bin/pytest tests/products/tx1 -q`
  - 结果：`179 passed`
  - 真实 CLI smoke：`env MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=$PWD /home/tiger/miniconda3/bin/python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260303 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-03-31 --top-k 1 --format json`
  - 结果：`status=ok`，并返回 `label_end_date / evidence_end_date / data_source_summary / gate_diagnostics / portfolio_advice.preflight_checks`

# TX1 2026-04-14 实盘建议查看

- [x] 确认可用 live package、bundle 数据截止日与运行参数
- [x] 运行 TX1 live advisor，查看 `2026-04-14` 的建议输出
- [x] 记录命令证据与结果摘要

## Review

- 本次使用 package：`skyeye/artifacts/packages/tx1/tx1_canary_live_tx1_refresh_20260331_lgbm_20260313/manifest.json`
  - `package_id=tx1_canary_live_tx1_refresh_20260331_lgbm_20260313`
  - `gate_level=canary_live`
  - `data_end_date=2026-04-13`
- 实际运行命令：
  - `env MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=$PWD /home/tiger/miniconda3/bin/python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260313 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-04-14 --top-k 10 --universe-cache-root /tmp/tx1_live_universe_cache --format json`
  - `env MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=$PWD /home/tiger/miniconda3/bin/python -m skyeye.products.tx1.run_live_advisor --package-id tx1_canary_live_tx1_refresh_20260331_lgbm_20260313 --packages-root skyeye/artifacts/packages/tx1 --trade-date 2026-04-14 --top-k 10 --universe-cache-root /tmp/tx1_live_universe_cache --format table`
- 运行结果：
  - `status=ok`
  - `requested_trade_date=2026-04-14`
  - `latest_available_trade_date=2026-04-13`
  - `score_date=2026-04-13`
  - `raw_data_end_date=2026-04-13`
  - `raw_df shape=(732270, 7)`，`date range: 2015-01-05 – 2026-04-13`
- Top 10 建议：
  - `000793.XSHE`、`002024.XSHE`、`600690.XSHG`、`001979.XSHE`、`600029.XSHG`、`600115.XSHG`、`601728.XSHG`、`601615.XSHG`、`601006.XSHG`、`300027.XSHE`
- CLI 风险提示：
  - `model_end_date=2026-03-13` 距 `trade_date=2026-04-14` 落后 `21` 个交易日，处于 `warning` 模式
  - `evidence_end_date=2026-03-13` 距 `trade_date=2026-04-14` 落后 `21` 个交易日，处于 `warning` 模式
- 结果解读：
  - top 桶历史 OOS：`win_rate=48.1%`、`mean_return=+1.08%`、`median_return=-0.32%`，更像右偏收益排序器，不是高胜率信号
  - 近端 canary 窗口 `2025-12-10 ~ 2026-03-13`：`win_rate=53.1%`、`mean_return=+1.26%`、`median_return=+0.50%`
  - `portfolio_advice.rebalance_due=true`，默认给出 `25` 只股票等权 `4%` 的建仓目标，`estimated_turnover=50%`
