# OpenSpec Task Runner

自动执行 OpenSpec 变更提案中的任务列表，基于 [continuous-claude](https://github.com/AnandChowdhary/continuous-claude) 的设计理念。

## 目录

- [核心特性](#核心特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [命令行选项](#命令行选项)
- [使用示例](#使用示例)
- [工作流程](#工作流程)
- [任务格式要求](#任务格式要求)
- [错误处理](#错误处理)
- [故障排除](#故障排除)
- [与 continuous-claude 对比](#与-continuous-claude-的对比)

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **任务驱动** | 从 `tasks.md` 文件解析任务，逐个执行 |
| **OpenSpec 集成** | 自动读取 proposal、design、specs 等上下文 |
| **进度跟踪** | 自动更新 tasks.md 中的 checkbox 状态 |
| **自动提交** | 每完成一个任务自动 git commit |
| **限制控制** | 支持 max-runs、max-cost、max-duration |
| **实时输出** | 显示 Claude Code 执行过程 |
| **失败保护** | 连续 3 次失败自动停止 |
| **会话持久化** | 通过 OPENSPEC_NOTES.md 保持跨迭代上下文 |

---

## 安装

### 前置依赖
1. **Claude Code CLI**
2. **jq** - JSON 解析工具
   ```bash
   # macOS
   brew install jq

   # Ubuntu/Debian
   apt-get install jq
   ```
3. **OpenSpec** - 必须

### 脚本权限

```bash
chmod +x scripts/openspec-runner.sh
```

---

## 快速开始

```bash
# 1. 进入项目目录
cd /path/to/your/project

# 3. 创建 scripts 目录
mkdir scripts

# 2. 拷贝脚本
cp /path/to/openspec-runner.sh ./scripts

# 2. 查看帮助
./scripts/openspec-runner.sh --help

# 3. 执行变更任务
./scripts/openspec-runner.sh --change add-student-management-system # add-student-management-system 是 openspec change id
```

---

## 命令行选项

### 必需参数

| 选项 | 说明 |
|------|------|
| `-c, --change <id>` | OpenSpec 变更 ID（如 `add-student-management-system`） |

### 可选参数

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `-t, --tasks <file>` | 自动推断 | tasks.md 文件路径 |
| `-m, --max-runs <n>` | 无限制 | 最大成功任务数 |
| `--max-duration <duration>` | 无限制 | 最大执行时间 |
| `--notes-file <file>` | `OPENSPEC_NOTES.md` | 笔记文件路径 |
| `--dry-run` | false | 模拟执行，不做实际修改 |
| `-v, --verbose` | false | 详细输出（显示提示词） |
| `-h, --help` | - | 显示帮助信息 |

### 时间格式

`--max-duration` 支持以下格式：
- `2h` - 2 小时
- `30m` - 30 分钟
- `1h30m` - 1 小时 30 分钟
- `90s` - 90 秒

---

## 使用示例

### 基本用法

```bash
# 执行所有任务（无限制）
./scripts/openspec-runner.sh --change add-student-management-system

# 限制最多执行 5 个任务
./scripts/openspec-runner.sh --change add-student-management-system --max-runs 5
```

### 时间控制

```bash
# 执行 2 小时
./scripts/openspec-runner.sh --change add-student-management-system --max-duration 2h

# 执行 30 分钟
./scripts/openspec-runner.sh --change add-student-management-system --max-duration 30m
```

### 调试模式

```bash
# 干跑模式 - 不实际执行
./scripts/openspec-runner.sh --change add-student-management-system --dry-run --max-runs 3

# 详细模式 - 显示发送给 Claude 的提示词
./scripts/openspec-runner.sh --change add-student-management-system --max-runs 1 -v
```

### 自定义配置

```bash
# 使用自定义 tasks 文件
./scripts/openspec-runner.sh --change my-change --tasks ./custom/tasks.md

# 使用自定义笔记文件
./scripts/openspec-runner.sh --change my-change --notes-file ./MY_NOTES.md
```

---

## 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenSpec Task Runner                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. 解析 tasks.md                                          │   │
│  │    - 提取所有 checkbox 任务                                │   │
│  │    - 找到第一个未完成任务 (- [ ])                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 2. 构建提示词                                             │   │
│  │    - 工作流上下文                                          │   │
│  │    - 当前任务描述                                          │   │
│  │    - OpenSpec 上下文 (proposal/design/specs)               │   │
│  │    - 会话笔记                                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 3. 调用 Claude Code                                       │   │
│  │    - 实时显示执行过程                                      │   │
│  │    - 执行任务实现                                          │   │
│  │    - 自动 git commit 提交改动                              │   │
│  │    - 捕获 JSON 结果                                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 4. 检测完成信号                                           │   │
│  │    ├─ 包含 "TASK_COMPLETE" → 标记任务完成                  │   │
│  │    └─ 不包含 → 增加失败计数，准备重试                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 5. 检查限制条件                                           │   │
│  │    - max-runs: 成功任务数                                  │   │
│  │    - max-cost: 累计成本                                    │   │
│  │    - max-duration: 执行时间                                │   │
│  │    - 连续失败次数 (默认 3 次)                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│          ┌─────────────────┴─────────────────┐                  │
│          ↓                                   ↓                   │
│    [未达限制]                           [达到限制]               │
│    返回步骤 1                           结束执行                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 任务格式要求

### tasks.md 结构

```markdown
# Tasks: 项目名称

> **开发模式**: TDD（测试驱动开发）

## 1. Project Setup
- [ ] 1.1 初始化项目结构
- [ ] 1.2 配置开发环境
- [x] 1.3 已完成的任务（会被跳过）

## 2. Feature Module (TDD)
- [ ] 2.1 编写测试 → 实现功能 → 重构
- [ ] 2.2 另一个 TDD 任务
```

### 格式说明

| 格式 | 含义 |
|------|------|
| `- [ ]` | 待完成任务 |
| `- [x]` | 已完成任务（脚本会跳过） |
| `→` | TDD 流程阶段分隔符 |

### 任务完成标记

当 Claude Code 完成任务后，会在响应中包含 `TASK_COMPLETE`，脚本会自动将对应任务更新为：

```markdown
- [x] 1.1 初始化项目结构
```

---

## 错误处理

### 连续失败保护

| 行为 | 说明 |
|------|------|
| 默认阈值 | 3 次连续失败 |
| 触发条件 | Claude 未输出 "TASK_COMPLETE" |
| 处理方式 | 自动停止执行，显示错误信息 |

### 失败计数规则

- 任务成功 → 重置连续失败计数为 0
- 任务失败 → 连续失败计数 +1
- 达到阈值 → 退出脚本

### 成本保护

使用 `--max-cost` 参数避免意外高成本：

```bash
./scripts/openspec-runner.sh --change my-change --max-cost 5.00
```

---

## 故障排除

### 问题：脚本卡住无输出

**可能原因**：Claude Code 正在等待用户交互（如 AskUserQuestion）

**解决方案**：
1. 等待连续失败 3 次后自动退出
2. 使用 `Ctrl+C` 中断
3. 检查任务是否需要预先决策（如技术栈选择）

### 问题：任务一直不标记完成

**可能原因**：Claude 输出未包含 "TASK_COMPLETE"

**解决方案**：
1. 使用 `-v` 参数查看提示词内容
2. 检查任务描述是否清晰
3. 检查 Claude Code 输出的 JSON 结果

### 问题：找不到 tasks.md

**错误信息**：`Tasks file not found`

**解决方案**：
```bash
# 检查文件是否存在
ls openspec/changes/<change-id>/tasks.md

# 或手动指定路径
./scripts/openspec-runner.sh --change my-change --tasks ./path/to/tasks.md
```

### 问题：jq 未安装

**错误信息**：`jq is required for JSON parsing`

**解决方案**：
```bash
# macOS
brew install jq

# Ubuntu/Debian
sudo apt-get install jq
```

---

## 示例输出

```
ℹ️  Starting OpenSpec Task Runner
ℹ️  Change ID: add-student-management-system
ℹ️  Tasks file: openspec/changes/add-student-management-system/tasks.md
ℹ️  Initial progress: 0/39

ℹ️  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ️  Iteration 1 | Progress: 0/39
ℹ️  Current Task: 1.1 初始化项目结构（选择技术栈后执行）
ℹ️  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ️  Running Claude Code...
═══════════════════════════════════════════════════════════════

[Claude Code 执行输出...]

═══════════════════════════════════════════════════════════════
💰 Cost: $0.315 (Total: $0.315)
✅ Task completed!

ℹ️  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ️  Session Complete
ℹ️  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ️  Tasks completed: 1
ℹ️  Tasks failed: 0
ℹ️  Final progress: 1/39
ℹ️  Duration: 1m15s
💰 Total cost: $0.315
```

---

## 与 continuous-claude 的对比

| 特性 | continuous-claude | openspec-runner |
|------|------------------|-----------------|
| **设计目标** | 通用持续开发 | OpenSpec 任务执行 |
| **任务来源** | 自由提示词 | tasks.md 文件 |
| **进度跟踪** | SHARED_TASK_NOTES.md | tasks.md checkbox |
| **上下文** | 单一提示词 | proposal + design + specs |
| **完成检测** | 连续信号计数 | TASK_COMPLETE 信号 |
| **Git 集成** | PR 自动创建/合并 | 无（专注任务执行） |
| **并行执行** | 支持 worktree | 不支持 |
| **成本跟踪** | 支持 | 支持 |
| **时间限制** | 支持 | 支持 |

---

## 扩展建议

1. **Git 集成** - 添加类似 continuous-claude 的 PR 自动创建功能
2. **并行执行** - 支持多 worktree 同时执行不同任务
3. **Webhook 通知** - 任务完成时发送通知
4. **断点续传** - 支持从中断处恢复执行
5. **Web UI** - 可视化任务进度和成本

---

## 相关文件

- `scripts/openspec-runner.sh` - 主脚本
- `scripts/openspec-prompts.md` - 提示词模板文档
- `openspec/AGENTS.md` - OpenSpec 工作流说明
