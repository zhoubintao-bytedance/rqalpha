from skyeye.products.tx1.autoresearch import git_ops


def test_git_ops_parses_current_commit_and_branch(monkeypatch, tmp_path):
    calls = []

    def _fake_run_git_command(*, workdir, args):
        calls.append((workdir, tuple(args)))
        if args == ["rev-parse", "--short", "HEAD"]:
            return "abc1234\n"
        if args == ["branch", "--show-current"]:
            return "tx1-autoresearch\n"
        raise AssertionError(args)

    monkeypatch.setattr(git_ops, "_run_git_command", _fake_run_git_command)

    assert git_ops.get_current_commit(tmp_path) == "abc1234"
    assert git_ops.get_current_branch(tmp_path) == "tx1-autoresearch"
    assert calls == [
        (tmp_path, ("rev-parse", "--short", "HEAD")),
        (tmp_path, ("branch", "--show-current")),
    ]


def test_git_ops_lists_changed_paths_and_commits(monkeypatch, tmp_path):
    calls = []

    def _fake_run_git_command(*, workdir, args):
        calls.append((workdir, tuple(args)))
        if args == ["diff", "--name-only", "--relative", "HEAD"]:
            return "skyeye/products/tx1/dataset_builder.py\nskyeye/products/tx1/baseline_models.py\n"
        if args == ["add", "-A"]:
            return ""
        if args == ["commit", "-m", "exp: tune turnover"]:
            return "[tx1-autoresearch deadbee] exp: tune turnover\n"
        if args == ["rev-parse", "--short", "HEAD"]:
            return "deadbee\n"
        raise AssertionError(args)

    monkeypatch.setattr(git_ops, "_run_git_command", _fake_run_git_command)

    changed = git_ops.list_changed_paths(tmp_path)
    commit = git_ops.create_experiment_commit(tmp_path, "exp: tune turnover")

    assert changed == [
        "skyeye/products/tx1/dataset_builder.py",
        "skyeye/products/tx1/baseline_models.py",
    ]
    assert commit == "deadbee"
    assert calls[-3:] == [
        (tmp_path, ("add", "-A")),
        (tmp_path, ("commit", "-m", "exp: tune turnover")),
        (tmp_path, ("rev-parse", "--short", "HEAD")),
    ]


def test_git_ops_rolls_back_to_start_commit(monkeypatch, tmp_path):
    calls = []

    def _fake_run_git_command(*, workdir, args):
        calls.append((workdir, tuple(args)))
        return ""

    monkeypatch.setattr(git_ops, "_run_git_command", _fake_run_git_command)

    git_ops.rollback_to_commit(tmp_path, "abc1234")

    assert calls == [
        (tmp_path, ("reset", "--hard", "abc1234")),
    ]


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


def test_git_ops_rolls_back_candidate_commit_to_start_commit(monkeypatch, tmp_path):
    calls = []

    def _fake_run_git_command(*, workdir, args):
        calls.append(tuple(args))
        return ""

    monkeypatch.setattr(git_ops, "_run_git_command", _fake_run_git_command)

    git_ops.rollback_candidate_commit(tmp_path, "abc1234")

    assert calls == [("reset", "--hard", "abc1234")]
