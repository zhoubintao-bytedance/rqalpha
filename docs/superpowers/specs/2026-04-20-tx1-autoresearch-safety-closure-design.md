# TX1 Autoresearch Safety Closure Design

## 背景

`tx1-autoresearch` 已经具备最小骨架：运行目录、`state.json` / `results.tsv`、基础 judge、候选评估入口和 CLI 均已存在。但当前主路径仍停在“能调用”的阶段，还没有把“安全可回滚、可推进、可审计”的闭环真正做实。

从现状看，最主要的缺口不是研究指标计算本身，而是自治运行时的安全语义：

- 运行边界没有被明确收紧，当前实现默认可在任意工作区启动。
- git 回退仍是硬回滚语义，和“不碰用户已有改动”的目标冲突。
- `keep / discard / invalid / crash / champion` 虽有初步实现，但状态推进和分支推进语义仍不够清晰。
- 候选实验目录已经独立，但启动前置检查、异常落盘和单轮安全闭环还不完整。

这一轮不追求把 autoresearch 一次做成完整多轮自治搜索器，而是先把“单轮 candidate 的安全闭环”补齐，确保它能在隔离 worktree 中安全运行。

## 目标

本轮目标只覆盖 `tx1-autoresearch` 的安全闭环：

1. 明确 autoresearch 只能在干净的独立 git worktree 中运行。
2. 补齐单轮 `candidate commit -> smoke/full -> discard/keep/champion/crash -> 状态落盘` 主路径。
3. 保证回退只影响本轮 candidate，不误伤 run 开始前的基线状态。
4. 保留实验目录和运行账本，支持事后审计。

## 非目标

本轮明确不做以下事项：

- 不兼容“主工作区带脏改动直接运行”。
- 不实现完整多轮自治搜索策略。
- 不实现 baseline / champion / frontier 三方比较。
- 不重写 TX1 研究执行协议，继续复用现有 `run_feature_experiments(...)`。
- 不扩展到 live/runtime 子系统或只读路径之外的更广权限模型。

## 运行边界

`run_autoresearch` 默认只允许在独立 worktree 中启动，并要求工作区处于干净状态。

启动前必须通过以下检查：

- 当前目录位于 git 仓库内。
- 当前目录是独立 worktree，而不是主工作区根目录。
- 工作区没有未提交改动。
- 工作区没有未跟踪文件。

若任一检查失败，CLI 直接返回非零退出码；`loop` 返回结构化 `invalid` 结果，并给出明确 `reason_code`。系统不尝试自动修复工作区状态。

这样做的目的，是把 autoresearch 的安全边界前置成“隔离运行”，而不是依赖复杂的本地改动保护逻辑。用户如果要评估候选补丁，应先在隔离 worktree 中准备好本轮改动，再运行 autoresearch。

## Git 状态流转

### 基线状态

run 启动时记录：

- `baseline_commit`
- `current_commit`
- `best_commit`
- `branch_name`

初始时：

- `baseline_commit == current_commit == best_commit`

同时记录初始 `baseline_summary`，若 baseline 尚未执行，则为空摘要。

### 候选提交

当 worktree 中存在允许范围内的改动时，系统为本轮 candidate 创建一次实验提交：

- 一个 candidate 对应一个独立 commit。
- commit message 带上 `run_tag` 和 `experiment_index`，便于定位。
- `results.tsv` 始终不纳入 git 历史。

### 只读路径约束

若本轮改动触及下列只读路径，直接判 `invalid`：

- `skyeye/products/tx1/live_advisor/**`
- `skyeye/products/tx1/strategies/rolling_score/runtime.py`

命中只读路径时：

- 不执行实验。
- 恢复到 `start_commit`。
- 写入结构化 `invalid` 结果和命中的路径列表。

### 判定与推进

单轮 candidate 流程如下：

1. 审计 changed paths。
2. 创建 candidate commit。
3. 运行 smoke 评估。
4. smoke 过线后再运行 full 评估。
5. 根据结果执行分支推进或 candidate 回退。
6. 写回 `state.json` 和 `results.tsv`。

判定语义：

- `invalid`
  - 启动前置检查失败，或命中只读路径。
  - 恢复到 `start_commit`。
  - 不推进 `current_commit` / `best_commit`。

- `discard`
  - smoke 或 full 判定失败。
  - 恢复到 `start_commit`。
  - 不推进 `current_commit` / `best_commit`。

- `crash`
  - 评估过程中抛出异常。
  - 恢复到 `start_commit`。
  - 记录 `last_error` 与失败阶段。

- `keep`
  - full 过线，但未超过当前最优。
  - `current_commit` 推进到 candidate commit。
  - `best_commit` 保持不变。

- `champion`
  - full 过线且优于当前最优。
  - `current_commit` 与 `best_commit` 一起推进到 candidate commit。
  - `best_summary` 更新为 candidate full summary。

关键约束是：系统只允许对“本轮新建的 candidate commit”做回退，不对 run 开始前的历史基线做 destructive 操作。

## 状态模型

`state.json` 需要稳定承载以下字段：

- `run_tag`
- `branch_name`
- `baseline_commit`
- `current_commit`
- `best_commit`
- `baseline_summary`
- `best_summary`
- `last_status`
- `last_reason_code`
- `last_experiment_path`
- `experiment_count`
- `last_error`

状态推进规则：

- baseline 成功后，初始化 `baseline_summary` 与 `best_summary`。
- `discard / invalid / crash` 只更新“最近一次结果”字段，不改变 `best_commit`。
- `keep` 更新 `current_commit` 和最近一次结果字段，但不改 `best_commit`。
- `champion` 同时更新 `current_commit / best_commit / best_summary`。

`results.tsv` 继续作为只追加账本，记录：

- `commit`
- `status`
- 核心指标摘要
- `reason_code`
- `experiment_path`

它用于审计，不参与 git 历史。

## Runner 约束

本轮不重写研究执行器，只在现有执行链路上补安全闭环。

`runner.py` 继续承担两件事：

- 使用 `run_feature_experiments(...)` 执行候选实验。
- 把结果统一抽成 judge 可消费的标准摘要。

同时保留独立实验目录约束：

- 每次实验固定落到 `.../<run_tag>/experiments/exp_0000`、`exp_0001` 等独立目录。
- 禁止覆盖已有实验目录。
- 即使 candidate 被 `discard` 或 `crash`，实验目录也保留，供事后复盘。

## 错误处理

主循环中所有候选执行异常都统一收口成结构化 `crash`：

- 记录失败阶段，例如 `precheck`、`smoke`、`full`、`state_update`。
- 记录异常字符串到 `last_error`。
- 恢复到 `start_commit`。
- 追加一条 `results.tsv` 记录。

这样可以保证 CLI 和调用方拿到的是“有语义的失败”，而不是裸异常。

## CLI 行为

`run_autoresearch.py` 保持轻量，但要把运行约束说清楚：

- 帮助信息里明确说明只支持隔离 worktree。
- 前置检查失败时返回非零退出码。
- 成功时返回 0。

这一轮不额外扩展更多 CLI 参数，优先把现有参数对应的安全语义补齐。

## 测试策略

本轮测试重点覆盖安全闭环，而不是扩大研究指标测试面。

### 定向测试

- `tests/products/tx1/test_autoresearch_git_ops.py`
  - worktree 检查
  - 干净目录检查
  - candidate 提交与回退语义
  - 不再默认接受“任意 `reset --hard` 就算正确”

- `tests/products/tx1/test_autoresearch_runner.py`
  - 独立实验目录
  - `run_feature_experiments(...)` 仍是执行入口
  - 结果摘要统一抽取

- `tests/products/tx1/test_run_autoresearch.py`
  - 前置检查失败时非零退出
  - `invalid / discard / keep / champion / crash` 的状态推进
  - `keep` 不更新 `best_commit`
  - `champion` 更新 `best_commit`
  - 异常路径写入 `last_error`

### 相关回归

- `tests/products/tx1/test_run_feature_experiment.py`
- `tests/products/tx1/test_run_baseline_experiment.py`

目标是确认 autoresearch 安全闭环补强后，没有破坏现有研究执行入口。

## 验收标准

当以下条件同时满足时，本轮视为完成：

- autoresearch 只能在干净的独立 worktree 中启动。
- 命中只读路径时直接拒绝，并恢复到 `start_commit`。
- `discard / crash` 不推进分支。
- `keep` 只推进 `current_commit`。
- `champion` 同时推进 `current_commit` 与 `best_commit`。
- 每轮实验产物目录独立存在，不覆盖旧目录。
- `state.json` 与 `results.tsv` 能完整反映最近一次候选结果。
- 定向测试和相关回归测试通过。

## 风险与取舍

- 把运行边界限定为独立 worktree，会牺牲一部分使用便利性，但能显著降低误伤主工作区的风险。
- 本轮不兼容脏工作区直跑，意味着用户要先接受更严格的使用纪律。
- 本轮不实现多轮搜索器，意味着“能安全跑一轮”优先于“跑得更聪明”。

这个取舍是有意的：先把安全边界做对，再继续扩充自治能力。
