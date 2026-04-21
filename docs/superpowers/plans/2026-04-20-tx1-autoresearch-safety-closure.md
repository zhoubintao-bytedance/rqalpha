# TX1 Autoresearch Safety Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `tx1-autoresearch` 只能在干净的独立 worktree 中安全运行，并把单轮 candidate 的 `commit -> smoke/full -> discard/keep/champion/crash -> 状态落盘` 主路径补齐。

**Architecture:** 在 `git_ops.py` 增加 worktree/洁净度前置检查与 candidate 级回退语义；在 `loop.py` 统一收口 precheck、异常处理和决策推进；在 `state.py` 补强状态字段与更新规则；在 `run_autoresearch.py` 让 CLI 根据结构化结果返回正确退出码。研究执行链继续复用 `run_feature_experiments(...)`，不新建第二套执行器。

**Tech Stack:** Python 3, pytest, git CLI, `pathlib`, 现有 TX1 autoresearch 模块

---

## File Map

- Modify: `skyeye/products/tx1/autoresearch/git_ops.py`
  - 责任：worktree 识别、洁净度检查、candidate commit 创建、candidate 级回退
- Modify: `skyeye/products/tx1/autoresearch/state.py`
  - 责任：扩展 `state.json` 字段和决策后的状态推进
- Modify: `skyeye/products/tx1/autoresearch/loop.py`
  - 责任：precheck、candidate 生命周期、异常转 `crash`、统一记账
- Modify: `skyeye/products/tx1/run_autoresearch.py`
  - 责任：CLI 返回码、错误输出、帮助文本
- Test: `tests/products/tx1/test_autoresearch_git_ops.py`
- Test: `tests/products/tx1/test_autoresearch_state.py`
- Test: `tests/products/tx1/test_run_autoresearch.py`
- Regression: `tests/products/tx1/test_autoresearch_runner.py`
- Regression: `tests/products/tx1/test_run_feature_experiment.py`
- Regression: `tests/products/tx1/test_run_baseline_experiment.py`

### Task 1: 补 git worktree 前置检查与 candidate 回退语义

**Files:**
- Modify: `skyeye/products/tx1/autoresearch/git_ops.py`
- Test: `tests/products/tx1/test_autoresearch_git_ops.py`

- [ ] **Step 1: 写失败测试，锁住 worktree / 洁净度前置检查**

```python
from skyeye.products.tx1.autoresearch import git_ops


def test_git_ops_collects_workspace_safety_checks(monkeypatch, tmp_path):
    responses = {
        ("rev-parse", "--show-toplevel"): str(tmp_path) + "\n",
        ("rev-parse", "--git-common-dir"): str(tmp_path / ".git" / "worktrees" / "tx1") + "\n",
        ("status", "--porcelain"): "",
    }

    def _fake_run_git_command(*, workdir, args):
        return responses[tuple(args)]

    monkeypatch.setattr(git_ops, "_run_git_command", _fake_run_git_command)

    checks = git_ops.collect_workspace_safety_checks(tmp_path)

    assert checks["is_git_repo"] is True
    assert checks["is_worktree"] is True
    assert checks["is_clean"] is True
    assert checks["has_untracked_files"] is False
    assert checks["reason_code"] is None
```

- [ ] **Step 2: 跑单测确认当前实现失败**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/products/tx1/test_autoresearch_git_ops.py::test_git_ops_collects_workspace_safety_checks -q
```

Expected: FAIL，提示 `collect_workspace_safety_checks` 不存在或字段不匹配。

- [ ] **Step 3: 实现最小 git 安全检查接口**

```python
def collect_workspace_safety_checks(workdir: str | Path) -> dict[str, object]:
    repo_root = resolve_repo_root(workdir)
    checks = {
        "is_git_repo": False,
        "is_worktree": False,
        "is_clean": False,
        "has_untracked_files": False,
        "reason_code": None,
    }
    try:
        _run_git_command(workdir=repo_root, args=["rev-parse", "--show-toplevel"])
        checks["is_git_repo"] = True
        common_dir = _run_git_command(workdir=repo_root, args=["rev-parse", "--git-common-dir"]).strip()
        checks["is_worktree"] = "/worktrees/" in common_dir.replace("\\", "/")
        porcelain = _run_git_command(workdir=repo_root, args=["status", "--porcelain"])
        lines = [line for line in porcelain.splitlines() if line.strip()]
        checks["has_untracked_files"] = any(line.startswith("?? ") for line in lines)
        checks["is_clean"] = not lines
    except subprocess.CalledProcessError:
        checks["reason_code"] = "not_git_repo"
        return checks

    if not checks["is_worktree"]:
        checks["reason_code"] = "not_worktree"
    elif checks["has_untracked_files"]:
        checks["reason_code"] = "worktree_not_clean"
    elif not checks["is_clean"]:
        checks["reason_code"] = "worktree_not_clean"
    return checks
```

- [ ] **Step 4: 写失败测试，锁住 candidate 回退只回到指定 commit**

```python
def test_git_ops_rolls_back_candidate_commit_to_start_commit(monkeypatch, tmp_path):
    calls = []

    def _fake_run_git_command(*, workdir, args):
        calls.append(tuple(args))
        return ""

    monkeypatch.setattr(git_ops, "_run_git_command", _fake_run_git_command)

    git_ops.rollback_candidate_commit(tmp_path, "abc1234")

    assert calls == [("reset", "--hard", "abc1234")]
```

- [ ] **Step 5: 跑 git_ops 定向测试确认第二个测试也先失败**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/products/tx1/test_autoresearch_git_ops.py -q
```

Expected: 至少 1 个 FAIL，提示 `rollback_candidate_commit` 尚未实现。

- [ ] **Step 6: 实现 candidate 级回退接口并保留旧名兼容**

```python
def rollback_candidate_commit(workdir: str | Path, commit: str) -> None:
    _run_git_command(
        workdir=resolve_repo_root(workdir),
        args=["reset", "--hard", str(commit)],
    )


def rollback_to_commit(workdir: str | Path, commit: str) -> None:
    rollback_candidate_commit(workdir, commit)
```

- [ ] **Step 7: 跑 git_ops 全量定向测试**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/products/tx1/test_autoresearch_git_ops.py -q
```

Expected: PASS。

- [ ] **Step 8: 提交本任务**

```bash
git add tests/products/tx1/test_autoresearch_git_ops.py skyeye/products/tx1/autoresearch/git_ops.py
git commit -m "test: add autoresearch git safety checks"
```

### Task 2: 扩展状态模型，补 last_reason_code / last_error / last_experiment_path

**Files:**
- Modify: `skyeye/products/tx1/autoresearch/state.py`
- Test: `tests/products/tx1/test_autoresearch_state.py`

- [ ] **Step 1: 写失败测试，锁住初始化状态字段**

```python
from skyeye.products.tx1.autoresearch.state import AutoresearchStateStore


def test_state_store_initializes_safety_closure_fields(tmp_path):
    store = AutoresearchStateStore(tmp_path / "run")

    state = store.initialize(
        run_tag="demo",
        baseline_commit="abc1234",
        branch_name="tx1-autoresearch",
        baseline_summary={},
    )

    assert state["last_reason_code"] is None
    assert state["last_experiment_path"] == ""
    assert state["last_error"] is None
```

- [ ] **Step 2: 跑定向测试确认失败**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/products/tx1/test_autoresearch_state.py::test_state_store_initializes_safety_closure_fields -q
```

Expected: FAIL，提示字段不存在。

- [ ] **Step 3: 实现初始化字段**

```python
state = {
    "run_tag": str(run_tag),
    "branch_name": str(branch_name),
    "baseline_commit": str(baseline_commit),
    "current_commit": str(baseline_commit),
    "best_commit": str(baseline_commit),
    "baseline_summary": dict(baseline_summary or {}),
    "best_summary": dict(baseline_summary or {}),
    "experiment_count": 0,
    "last_status": "initialized",
    "last_reason_code": None,
    "last_experiment_path": "",
    "last_error": None,
    "created_at": datetime.now().isoformat(),
    "updated_at": datetime.now().isoformat(),
}
```

- [ ] **Step 4: 写失败测试，锁住 keep / champion / discard / crash 的状态推进**

```python
def test_state_store_updates_reason_and_error_fields(tmp_path):
    store = AutoresearchStateStore(tmp_path / "run")
    store.initialize(
        run_tag="demo",
        baseline_commit="abc1234",
        branch_name="tx1-autoresearch",
        baseline_summary={"portfolio": {"net_mean_return": 0.001}},
    )

    state = store.update_after_decision(
        decision_status="crash",
        commit="def5678",
        candidate_summary={"experiment_path": "/tmp/exp_0001"},
        reason_code="smoke_crashed",
        error_message="boom",
    )

    assert state["last_status"] == "crash"
    assert state["last_reason_code"] == "smoke_crashed"
    assert state["last_experiment_path"] == "/tmp/exp_0001"
    assert state["last_error"] == "boom"
    assert state["best_commit"] == "abc1234"
```

- [ ] **Step 5: 跑状态文件测试确认失败**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/products/tx1/test_autoresearch_state.py -q
```

Expected: FAIL，提示 `update_after_decision()` 签名和状态字段不匹配。

- [ ] **Step 6: 扩展状态推进实现**

```python
def update_after_decision(
    self,
    *,
    decision_status: str,
    commit: str,
    candidate_summary: dict[str, Any] | None,
    reason_code: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    state = self.load()
    state["experiment_count"] = int(state.get("experiment_count", 0)) + 1
    state["last_status"] = str(decision_status)
    state["last_reason_code"] = reason_code
    state["last_experiment_path"] = str((candidate_summary or {}).get("experiment_path", ""))
    state["last_error"] = error_message
    if decision_status in {"keep", "champion"}:
        state["current_commit"] = str(commit)
    if decision_status == "champion":
        state["best_commit"] = str(commit)
        state["best_summary"] = dict(candidate_summary or {})
    self.save(state)
    return state
```

- [ ] **Step 7: 跑状态定向测试**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/products/tx1/test_autoresearch_state.py -q
```

Expected: PASS。

- [ ] **Step 8: 提交本任务**

```bash
git add tests/products/tx1/test_autoresearch_state.py skyeye/products/tx1/autoresearch/state.py
git commit -m "feat: enrich autoresearch safety state"
```

### Task 3: 在 loop 中落地 precheck、crash 收口和结果推进

**Files:**
- Modify: `skyeye/products/tx1/autoresearch/loop.py`
- Test: `tests/products/tx1/test_run_autoresearch.py`

- [ ] **Step 1: 写失败测试，锁住 precheck 失败直接返回 invalid**

```python
from skyeye.products.tx1.autoresearch import loop as autoresearch_loop


def test_run_loop_returns_invalid_when_workspace_precheck_fails(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    monkeypatch.setattr(
        autoresearch_loop,
        "collect_workspace_safety_checks",
        lambda workdir: {
            "is_git_repo": True,
            "is_worktree": False,
            "is_clean": True,
            "has_untracked_files": False,
            "reason_code": "not_worktree",
        },
    )

    result = autoresearch_loop.run_loop(
        run_tag="demo",
        runs_root=tmp_path / "runs",
        workdir=repo_root,
    )

    assert result["status"] == "invalid"
    assert result["reason_code"] == "not_worktree"
    assert result["state"]["last_status"] == "invalid"
```

- [ ] **Step 2: 跑单测确认当前失败**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/products/tx1/test_run_autoresearch.py::test_run_loop_returns_invalid_when_workspace_precheck_fails -q
```

Expected: FAIL，因为 `run_loop()` 还没有前置检查分支。

- [ ] **Step 3: 在 loop 顶部接入 precheck**

```python
checks = collect_workspace_safety_checks(repo_root)
if checks.get("reason_code") is not None:
    state = store.initialize(
        run_tag=run_tag,
        baseline_commit="",
        branch_name="",
        baseline_summary={},
    )
    store.update_after_decision(
        decision_status="invalid",
        commit="",
        candidate_summary=None,
        reason_code=str(checks["reason_code"]),
        error_message=None,
    )
    return {
        "run_tag": run_tag,
        "run_root": str(run_root),
        "workdir": str(repo_root),
        "state": store.load(),
        "status": "invalid",
        "reason_code": str(checks["reason_code"]),
    }
```

- [ ] **Step 4: 写失败测试，锁住 candidate 评估异常转 crash**

```python
def test_run_loop_records_crash_when_candidate_raises(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    monkeypatch.setattr(autoresearch_loop, "collect_workspace_safety_checks", lambda workdir: {
        "is_git_repo": True,
        "is_worktree": True,
        "is_clean": True,
        "has_untracked_files": False,
        "reason_code": None,
    })
    monkeypatch.setattr(autoresearch_loop, "get_current_commit", lambda workdir: "abc1234")
    monkeypatch.setattr(autoresearch_loop, "get_current_branch", lambda workdir: "tx1-autoresearch")
    monkeypatch.setattr(autoresearch_loop, "evaluate_current_candidate", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    result = autoresearch_loop.run_loop(
        run_tag="demo",
        runs_root=tmp_path / "runs",
        workdir=repo_root,
        evaluate_current=True,
        raw_df=object(),
    )

    assert result["status"] == "crash"
    assert result["state"]["last_status"] == "crash"
    assert result["state"]["last_reason_code"] == "candidate_crash"
    assert result["state"]["last_error"] == "boom"
```

- [ ] **Step 5: 跑 `test_run_autoresearch.py`，确认 crash 用例也先失败**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/products/tx1/test_run_autoresearch.py -q
```

Expected: 至少 1 个 FAIL，当前裸异常会直接中断。

- [ ] **Step 6: 实现 crash 收口与统一记账**

```python
try:
    candidate_result = evaluate_current_candidate(...)
except Exception as exc:
    candidate_result = {
        "status": "crash",
        "reason_code": "candidate_crash",
        "commit": baseline_commit,
        "error_message": str(exc),
        "smoke_summary": {},
    }

state = _record_candidate_result(
    state_store=store,
    baseline_commit=baseline_commit,
    candidate_result=candidate_result,
)
```

```python
return state_store.update_after_decision(
    decision_status=status,
    commit=commit,
    candidate_summary=candidate_summary,
    reason_code=reason_code,
    error_message=candidate_result.get("error_message"),
)
```

- [ ] **Step 7: 写失败测试，锁住 `keep` 不更新 `best_commit`、`champion` 更新**

```python
def test_run_loop_keeps_best_commit_unchanged_for_keep(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    monkeypatch.setattr(autoresearch_loop, "collect_workspace_safety_checks", lambda workdir: {
        "is_git_repo": True,
        "is_worktree": True,
        "is_clean": True,
        "has_untracked_files": False,
        "reason_code": None,
    })
    monkeypatch.setattr(autoresearch_loop, "get_current_commit", lambda workdir: "abc1234")
    monkeypatch.setattr(autoresearch_loop, "get_current_branch", lambda workdir: "tx1-autoresearch")
    monkeypatch.setattr(autoresearch_loop, "run_feature_trial", lambda **kwargs: {
        "portfolio": {"net_mean_return": 0.002, "max_drawdown": 0.08},
        "robustness": {"stability": {"stability_score": 61.0}},
        "experiment_path": str(tmp_path / "exp_0000"),
    })
    monkeypatch.setattr(autoresearch_loop, "evaluate_current_candidate", lambda **kwargs: {
        "status": "keep",
        "reason_code": "full_pass_not_best",
        "commit": "def5678",
        "full_summary": {
            "portfolio": {"net_mean_return": 0.0021, "max_drawdown": 0.08},
            "robustness": {"stability": {"stability_score": 61.5}},
            "experiment_path": str(tmp_path / "exp_0001"),
        },
    })

    result = autoresearch_loop.run_loop(
        run_tag="demo",
        runs_root=tmp_path / "runs",
        workdir=repo_root,
        raw_df=object(),
        evaluate_current=True,
    )

    assert result["state"]["current_commit"] == "def5678"
    assert result["state"]["best_commit"] == "abc1234"
```

- [ ] **Step 8: 跑 `test_run_autoresearch.py` 全量并修正实现直到通过**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/products/tx1/test_run_autoresearch.py -q
```

Expected: PASS。

- [ ] **Step 9: 提交本任务**

```bash
git add tests/products/tx1/test_run_autoresearch.py skyeye/products/tx1/autoresearch/loop.py
git commit -m "feat: close autoresearch candidate safety loop"
```

### Task 4: 调整 CLI 返回码与帮助文本

**Files:**
- Modify: `skyeye/products/tx1/run_autoresearch.py`
- Test: `tests/products/tx1/test_run_autoresearch.py`

- [ ] **Step 1: 写失败测试，锁住 CLI 对 invalid 返回非零**

```python
from skyeye.products.tx1 import run_autoresearch


def test_run_autoresearch_main_returns_nonzero_for_invalid(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_autoresearch,
        "run_loop",
        lambda **kwargs: {"status": "invalid", "reason_code": "not_worktree"},
    )

    rc = run_autoresearch.main(["--run-tag", "demo", "--runs-root", str(tmp_path)])

    assert rc == 2
```

- [ ] **Step 2: 跑单测确认失败**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/products/tx1/test_run_autoresearch.py::test_run_autoresearch_main_returns_nonzero_for_invalid -q
```

Expected: FAIL，当前 `main()` 总是返回 0。

- [ ] **Step 3: 实现 CLI 返回码和帮助文本**

```python
parser = argparse.ArgumentParser(
    description="Run TX1 autoresearch loop (requires a clean dedicated git worktree)"
)
```

```python
result = run_loop(...)
status = str(result.get("status") or "ok")
if status in {"invalid", "crash"}:
    return 2
return 0
```

- [ ] **Step 4: 跑 CLI 相关定向测试**

Run:

```bash
PYTHONPATH="$PWD" pytest tests/products/tx1/test_run_autoresearch.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交本任务**

```bash
git add tests/products/tx1/test_run_autoresearch.py skyeye/products/tx1/run_autoresearch.py
git commit -m "feat: surface autoresearch safety status in cli"
```

### Task 5: 回归验证与交付整理

**Files:**
- Regression: `tests/products/tx1/test_autoresearch_git_ops.py`
- Regression: `tests/products/tx1/test_autoresearch_state.py`
- Regression: `tests/products/tx1/test_autoresearch_runner.py`
- Regression: `tests/products/tx1/test_run_autoresearch.py`
- Regression: `tests/products/tx1/test_run_feature_experiment.py`
- Regression: `tests/products/tx1/test_run_baseline_experiment.py`

- [ ] **Step 1: 跑 autoresearch 定向回归**

Run:

```bash
PYTHONPATH="$PWD" pytest \
  tests/products/tx1/test_autoresearch_git_ops.py \
  tests/products/tx1/test_autoresearch_state.py \
  tests/products/tx1/test_autoresearch_runner.py \
  tests/products/tx1/test_run_autoresearch.py -q
```

Expected: PASS。

- [ ] **Step 2: 跑研究入口相关回归**

Run:

```bash
PYTHONPATH="$PWD" pytest \
  tests/products/tx1/test_run_feature_experiment.py \
  tests/products/tx1/test_run_baseline_experiment.py -q
```

Expected: PASS。

- [ ] **Step 3: 跑最近改动文件诊断**

Run tool:

```text
GetDiagnostics for:
- skyeye/products/tx1/autoresearch/git_ops.py
- skyeye/products/tx1/autoresearch/state.py
- skyeye/products/tx1/autoresearch/loop.py
- skyeye/products/tx1/run_autoresearch.py
```

Expected: 无新增诊断错误。

- [ ] **Step 4: 整理最终变更并提交**

```bash
git add \
  skyeye/products/tx1/autoresearch/git_ops.py \
  skyeye/products/tx1/autoresearch/state.py \
  skyeye/products/tx1/autoresearch/loop.py \
  skyeye/products/tx1/run_autoresearch.py \
  tests/products/tx1/test_autoresearch_git_ops.py \
  tests/products/tx1/test_autoresearch_state.py \
  tests/products/tx1/test_run_autoresearch.py
git commit -m "feat: harden tx1 autoresearch safety closure"
```

## Self-Review

- Spec coverage:
  - worktree 边界：Task 1、Task 3、Task 4
  - candidate 生命周期：Task 3
  - 状态模型：Task 2、Task 3
  - CLI 非零退出：Task 4
  - 回归验证：Task 5
- Placeholder scan:
  - 无 `TODO` / `TBD`
  - 每个代码步骤都给了明确片段和命令
- Type consistency:
  - 前置检查统一使用 `collect_workspace_safety_checks`
  - 回退统一使用 `rollback_candidate_commit`
  - 状态推进统一使用 `reason_code` / `error_message`
