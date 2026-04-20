from skyeye.products.tx1 import run_autoresearch
from skyeye.products.tx1.autoresearch import loop as autoresearch_loop
from skyeye.products.tx1.autoresearch.state import AutoresearchStateStore


def _allow_workspace_checks(monkeypatch):
    monkeypatch.setattr(
        autoresearch_loop,
        "collect_workspace_safety_checks",
        lambda workdir: {
            "is_git_repo": True,
            "is_worktree": True,
            "is_clean": True,
            "has_untracked_files": False,
            "reason_code": None,
        },
    )


def test_run_autoresearch_main_passes_arguments_to_loop(monkeypatch, tmp_path):
    captured = {}

    def _fake_run_loop(**kwargs):
        captured.update(kwargs)
        return {
            "run_tag": kwargs["run_tag"],
            "status": "ok",
        }

    monkeypatch.setattr(run_autoresearch, "run_loop", _fake_run_loop)

    rc = run_autoresearch.main(
        [
            "--run-tag",
            "demo",
            "--runs-root",
            str(tmp_path),
            "--max-experiments",
            "3",
            "--smoke-max-folds",
            "1",
            "--full-max-folds",
            "4",
        ]
    )

    assert rc == 0
    assert captured["run_tag"] == "demo"
    assert captured["runs_root"] == tmp_path
    assert captured["max_experiments"] == 3
    assert captured["smoke_max_folds"] == 1
    assert captured["full_max_folds"] == 4


def test_run_autoresearch_main_passes_execution_flags_to_loop(monkeypatch, tmp_path):
    captured = {}

    def _fake_run_loop(**kwargs):
        captured.update(kwargs)
        return {
            "run_tag": kwargs["run_tag"],
            "status": "ok",
        }

    monkeypatch.setattr(run_autoresearch, "run_loop", _fake_run_loop)

    rc = run_autoresearch.main(
        [
            "--run-tag",
            "demo",
            "--runs-root",
            str(tmp_path),
            "--build-raw-df",
            "--evaluate-current",
            "--universe-size",
            "123",
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2024-12-31",
            "--variant-name",
            "baseline_5f",
            "--model-kind",
            "linear",
            "--label-transform",
            "rank",
            "--horizon-days",
            "20",
        ]
    )

    assert rc == 0
    assert captured["build_raw_df_for_run"] is True
    assert captured["evaluate_current"] is True
    assert captured["universe_size"] == 123
    assert captured["start_date"] == "2020-01-01"
    assert captured["end_date"] == "2024-12-31"
    assert captured["variant_name"] == "baseline_5f"
    assert captured["model_kind"] == "linear"
    assert captured["label_transform"] == "rank"
    assert captured["horizon_days"] == 20


def test_run_autoresearch_main_returns_nonzero_for_invalid(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_autoresearch,
        "run_loop",
        lambda **kwargs: {"status": "invalid", "reason_code": "not_worktree"},
    )

    rc = run_autoresearch.main(["--run-tag", "demo", "--runs-root", str(tmp_path)])

    assert rc == 2


def test_run_loop_initializes_state_with_git_metadata(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    _allow_workspace_checks(monkeypatch)
    monkeypatch.setattr(autoresearch_loop, "get_current_commit", lambda workdir: "abc1234")
    monkeypatch.setattr(autoresearch_loop, "get_current_branch", lambda workdir: "tx1-autoresearch")

    result = autoresearch_loop.run_loop(
        run_tag="demo",
        runs_root=tmp_path / "runs",
        workdir=repo_root,
        max_experiments=0,
        smoke_max_folds=1,
        full_max_folds=4,
    )

    state = result["state"]
    assert state["baseline_commit"] == "abc1234"
    assert state["branch_name"] == "tx1-autoresearch"
    assert result["workdir"] == str(repo_root)


def test_run_loop_executes_baseline_trial_and_records_summary(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    _allow_workspace_checks(monkeypatch)
    monkeypatch.setattr(autoresearch_loop, "get_current_commit", lambda workdir: "abc1234")
    monkeypatch.setattr(autoresearch_loop, "get_current_branch", lambda workdir: "tx1-autoresearch")
    monkeypatch.setattr(
        autoresearch_loop,
        "run_baseline_trial",
        lambda **kwargs: {
            "prediction": {"rank_ic_mean": 0.05, "top_bucket_spread_mean": 0.01},
            "portfolio": {"net_mean_return": 0.002, "max_drawdown": 0.08, "mean_turnover": 0.16},
            "robustness": {
                "stability": {"stability_score": 61.0, "cv": 0.5},
                "overfit_flags": {
                    "flag_ic_decay": False,
                    "flag_spread_decay": False,
                    "flag_val_dominant": False,
                },
                "regime_scores": {"metric_consistency": {"positive_ratio": 0.82}},
            },
            "experiment_path": str(tmp_path / "runs" / "demo" / "experiments" / "exp_0000"),
        },
    )

    result = autoresearch_loop.run_loop(
        run_tag="demo",
        runs_root=tmp_path / "runs",
        workdir=repo_root,
        raw_df=object(),
        variant_name="baseline_5f",
        model_kind="linear",
        max_experiments=0,
        smoke_max_folds=1,
        full_max_folds=4,
    )

    state = result["state_store"].load()
    assert state["baseline_summary"]["portfolio"]["net_mean_return"] == 0.002
    assert state["best_summary"]["portfolio"]["max_drawdown"] == 0.08

    results_lines = (tmp_path / "runs" / "demo" / "results.tsv").read_text(encoding="utf-8").splitlines()
    assert len(results_lines) == 2
    assert results_lines[1].startswith("abc1234\tkeep\t0.002000\t0.080000\t61.000000\tbaseline\t")


def test_run_loop_builds_raw_df_and_records_candidate_result(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    captured = {}

    _allow_workspace_checks(monkeypatch)
    monkeypatch.setattr(autoresearch_loop, "get_current_commit", lambda workdir: "abc1234")
    monkeypatch.setattr(autoresearch_loop, "get_current_branch", lambda workdir: "tx1-autoresearch")

    def _fake_build_research_raw_df(**kwargs):
        captured["build_research_raw_df"] = dict(kwargs)
        return {"raw_df": "ok"}

    def _fake_run_baseline_trial(**kwargs):
        captured["baseline_raw_df"] = kwargs["raw_df"]
        return {
            "prediction": {"rank_ic_mean": 0.05, "top_bucket_spread_mean": 0.01},
            "portfolio": {"net_mean_return": 0.002, "max_drawdown": 0.08, "mean_turnover": 0.16},
            "robustness": {
                "stability": {"stability_score": 61.0, "cv": 0.5},
                "overfit_flags": {
                    "flag_ic_decay": False,
                    "flag_spread_decay": False,
                    "flag_val_dominant": False,
                },
                "regime_scores": {"metric_consistency": {"positive_ratio": 0.82}},
            },
            "experiment_path": str(tmp_path / "runs" / "demo" / "experiments" / "exp_0000"),
        }

    def _fake_evaluate_current_candidate(**kwargs):
        captured["candidate_raw_df"] = kwargs["raw_df"]
        captured["candidate_baseline_summary"] = kwargs["baseline_summary"]
        return {
            "status": "champion",
            "reason_code": "full_improved",
            "commit": "def5678",
            "full_summary": {
                "prediction": {"rank_ic_mean": 0.06, "top_bucket_spread_mean": 0.012},
                "portfolio": {"net_mean_return": 0.003, "max_drawdown": 0.07, "mean_turnover": 0.15},
                "robustness": {
                    "stability": {"stability_score": 65.0, "cv": 0.45},
                    "overfit_flags": {
                        "flag_ic_decay": False,
                        "flag_spread_decay": False,
                        "flag_val_dominant": False,
                    },
                    "regime_scores": {"metric_consistency": {"positive_ratio": 0.86}},
                },
                "experiment_path": str(tmp_path / "runs" / "demo" / "experiments" / "exp_0001"),
            },
        }

    monkeypatch.setattr(autoresearch_loop, "build_research_raw_df", _fake_build_research_raw_df)
    monkeypatch.setattr(autoresearch_loop, "run_baseline_trial", _fake_run_baseline_trial)
    monkeypatch.setattr(autoresearch_loop, "evaluate_current_candidate", _fake_evaluate_current_candidate)
    monkeypatch.setattr(
        autoresearch_loop,
        "detect_workspace_patch",
        lambda workdir: {"changed_paths": ["skyeye/products/tx1/dataset_builder.py"]},
        raising=False,
    )

    result = autoresearch_loop.run_loop(
        run_tag="demo",
        runs_root=tmp_path / "runs",
        workdir=repo_root,
        build_raw_df_for_run=True,
        evaluate_current=True,
        universe_size=123,
        start_date="2020-01-01",
        end_date="2024-12-31",
        smoke_max_folds=1,
        full_max_folds=4,
    )

    assert captured["build_research_raw_df"] == {
        "universe_size": 123,
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
    }
    assert captured["baseline_raw_df"] == {"raw_df": "ok"}
    assert captured["candidate_raw_df"] == {"raw_df": "ok"}
    assert captured["candidate_baseline_summary"]["portfolio"]["net_mean_return"] == 0.002
    assert result["candidate_result"]["status"] == "champion"

    state = result["state_store"].load()
    assert state["last_status"] == "champion"
    assert state["current_commit"] == "def5678"
    assert state["best_commit"] == "def5678"

    results_lines = (tmp_path / "runs" / "demo" / "results.tsv").read_text(encoding="utf-8").splitlines()
    assert len(results_lines) == 3
    assert results_lines[2].startswith("def5678\tchampion\t0.003000\t0.070000\t65.000000\tfull_improved\t")


def test_run_loop_keeps_best_commit_when_candidate_is_discarded(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    _allow_workspace_checks(monkeypatch)
    monkeypatch.setattr(autoresearch_loop, "get_current_commit", lambda workdir: "abc1234")
    monkeypatch.setattr(autoresearch_loop, "get_current_branch", lambda workdir: "tx1-autoresearch")
    monkeypatch.setattr(autoresearch_loop, "build_research_raw_df", lambda **kwargs: {"raw_df": "ok"})
    monkeypatch.setattr(
        autoresearch_loop,
        "run_baseline_trial",
        lambda **kwargs: {
            "prediction": {"rank_ic_mean": 0.05, "top_bucket_spread_mean": 0.01},
            "portfolio": {"net_mean_return": 0.002, "max_drawdown": 0.08, "mean_turnover": 0.16},
            "robustness": {
                "stability": {"stability_score": 61.0, "cv": 0.5},
                "overfit_flags": {
                    "flag_ic_decay": False,
                    "flag_spread_decay": False,
                    "flag_val_dominant": False,
                },
                "regime_scores": {"metric_consistency": {"positive_ratio": 0.82}},
            },
            "experiment_path": str(tmp_path / "runs" / "demo" / "experiments" / "exp_0000"),
        },
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "evaluate_current_candidate",
        lambda **kwargs: {
            "status": "discard",
            "reason_code": "guardrail_failed",
            "commit": "def5678",
            "smoke_summary": {
                "prediction": {"rank_ic_mean": 0.01, "top_bucket_spread_mean": 0.001},
                "portfolio": {"net_mean_return": -0.001, "max_drawdown": 0.20, "mean_turnover": 0.21},
                "robustness": {
                    "stability": {"stability_score": 10.0, "cv": 1.5},
                    "overfit_flags": {
                        "flag_ic_decay": True,
                        "flag_spread_decay": False,
                        "flag_val_dominant": False,
                    },
                    "regime_scores": {"metric_consistency": {"positive_ratio": 0.2}},
                },
                "experiment_path": str(tmp_path / "runs" / "demo" / "experiments" / "exp_0001"),
            },
        },
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "detect_workspace_patch",
        lambda workdir: {"changed_paths": ["skyeye/products/tx1/dataset_builder.py"]},
        raising=False,
    )

    result = autoresearch_loop.run_loop(
        run_tag="demo",
        runs_root=tmp_path / "runs",
        workdir=repo_root,
        build_raw_df_for_run=True,
        evaluate_current=True,
        smoke_max_folds=1,
        full_max_folds=4,
    )

    assert result["candidate_result"]["status"] == "discard"
    state = result["state_store"].load()
    assert state["last_status"] == "discard"
    assert state["current_commit"] == "abc1234"
    assert state["best_commit"] == "abc1234"
    assert state["best_summary"]["portfolio"]["net_mean_return"] == 0.002

    results_lines = (tmp_path / "runs" / "demo" / "results.tsv").read_text(encoding="utf-8").splitlines()
    assert len(results_lines) == 3
    assert results_lines[2].startswith("def5678\tdiscard\t-0.001000\t0.200000\t10.000000\tguardrail_failed\t")


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
    assert result["state"]["last_reason_code"] == "not_worktree"


def test_run_loop_records_crash_when_candidate_raises(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    _allow_workspace_checks(monkeypatch)
    monkeypatch.setattr(autoresearch_loop, "get_current_commit", lambda workdir: "abc1234")
    monkeypatch.setattr(autoresearch_loop, "get_current_branch", lambda workdir: "tx1-autoresearch")
    monkeypatch.setattr(
        autoresearch_loop,
        "run_baseline_trial",
        lambda **kwargs: {
            "portfolio": {"net_mean_return": 0.002, "max_drawdown": 0.08},
            "robustness": {"stability": {"stability_score": 61.0}},
            "experiment_path": str(tmp_path / "exp_0000"),
        },
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "evaluate_current_candidate",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "detect_workspace_patch",
        lambda workdir: {"changed_paths": ["skyeye/products/tx1/dataset_builder.py"]},
        raising=False,
    )

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


def test_run_loop_keeps_best_commit_unchanged_for_keep(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    _allow_workspace_checks(monkeypatch)
    monkeypatch.setattr(autoresearch_loop, "get_current_commit", lambda workdir: "abc1234")
    monkeypatch.setattr(autoresearch_loop, "get_current_branch", lambda workdir: "tx1-autoresearch")
    monkeypatch.setattr(
        autoresearch_loop,
        "run_baseline_trial",
        lambda **kwargs: {
            "portfolio": {"net_mean_return": 0.002, "max_drawdown": 0.08},
            "robustness": {"stability": {"stability_score": 61.0}},
            "experiment_path": str(tmp_path / "exp_0000"),
        },
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "evaluate_current_candidate",
        lambda **kwargs: {
            "status": "keep",
            "reason_code": "full_pass_not_best",
            "commit": "def5678",
            "full_summary": {
                "portfolio": {"net_mean_return": 0.0021, "max_drawdown": 0.08},
                "robustness": {"stability": {"stability_score": 61.5}},
                "experiment_path": str(tmp_path / "exp_0001"),
            },
        },
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "detect_workspace_patch",
        lambda workdir: {"changed_paths": ["skyeye/products/tx1/dataset_builder.py"]},
        raising=False,
    )

    result = autoresearch_loop.run_loop(
        run_tag="demo",
        runs_root=tmp_path / "runs",
        workdir=repo_root,
        raw_df=object(),
        evaluate_current=True,
    )

    assert result["state"]["current_commit"] == "def5678"
    assert result["state"]["best_commit"] == "abc1234"


def test_run_loop_returns_waiting_for_patch_when_workspace_has_no_candidate(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    _allow_workspace_checks(monkeypatch)
    monkeypatch.setattr(autoresearch_loop, "get_current_commit", lambda workdir: "abc1234")
    monkeypatch.setattr(autoresearch_loop, "get_current_branch", lambda workdir: "tx1-autoresearch")
    monkeypatch.setattr(
        autoresearch_loop,
        "run_baseline_trial",
        lambda **kwargs: {
            "prediction": {"rank_ic_mean": 0.05, "top_bucket_spread_mean": 0.01},
            "portfolio": {"net_mean_return": 0.002, "max_drawdown": 0.08, "mean_turnover": 0.16},
            "robustness": {
                "stability": {"stability_score": 61.0, "cv": 0.5},
                "overfit_flags": {
                    "flag_ic_decay": False,
                    "flag_spread_decay": False,
                    "flag_val_dominant": False,
                },
                "regime_scores": {"metric_consistency": {"positive_ratio": 0.82}},
            },
            "experiment_path": str(tmp_path / "runs" / "demo" / "experiments" / "exp_0000"),
        },
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "detect_workspace_patch",
        lambda workdir: None,
        raising=False,
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "evaluate_current_candidate",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not evaluate candidate without patch")),
    )

    result = autoresearch_loop.run_loop(
        run_tag="demo",
        runs_root=tmp_path / "runs",
        workdir=repo_root,
        raw_df=object(),
        max_experiments=2,
    )

    assert result["status"] == "waiting_for_patch"
    assert result["state"]["last_status"] == "waiting_for_patch"
    assert result["state"]["current_commit"] == "abc1234"


def test_run_loop_resumes_from_current_commit_without_rerunning_baseline(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_root = tmp_path / "runs" / "demo"
    store = AutoresearchStateStore(run_root)
    state = store.initialize(
        run_tag="demo",
        baseline_commit="abc1234",
        branch_name="tx1-autoresearch",
        baseline_summary={
            "prediction": {"rank_ic_mean": 0.05, "top_bucket_spread_mean": 0.01},
            "portfolio": {"net_mean_return": 0.002, "max_drawdown": 0.08, "mean_turnover": 0.16},
            "robustness": {"stability": {"stability_score": 61.0, "cv": 0.5}},
            "experiment_path": str(run_root / "experiments" / "exp_0000"),
        },
        budget={"max_experiments": 2, "smoke_max_folds": 1, "full_max_folds": 4},
        raw_df_spec={"universe_size": 300, "start_date": "2020-01-01", "end_date": "2024-12-31"},
        allowed_write_roots=["skyeye/products/tx1/dataset_builder.py"],
        read_only_roots=["skyeye/products/tx1/live_advisor/"],
    )
    state["current_commit"] = "keep1234"
    state["best_commit"] = "keep1234"
    state["best_summary"] = dict(state["baseline_summary"])
    state["next_experiment_index"] = 3
    store.save(state)

    calls = []
    _allow_workspace_checks(monkeypatch)
    monkeypatch.setattr(autoresearch_loop, "get_current_commit", lambda workdir: "abc1234")
    monkeypatch.setattr(autoresearch_loop, "get_current_branch", lambda workdir: "tx1-autoresearch")
    monkeypatch.setattr(
        autoresearch_loop,
        "checkout_commit",
        lambda workdir, commit: calls.append((workdir, commit)),
        raising=False,
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "run_baseline_trial",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("resume should reuse existing baseline summary")),
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "detect_workspace_patch",
        lambda workdir: None,
        raising=False,
    )

    result = autoresearch_loop.run_loop(
        run_tag="demo",
        runs_root=tmp_path / "runs",
        workdir=repo_root,
        max_experiments=2,
    )

    assert calls == [(repo_root, "keep1234")]
    assert result["status"] == "waiting_for_patch"
    assert result["state"]["current_commit"] == "keep1234"
    assert result["state"]["best_commit"] == "keep1234"


def test_candidate_cycle_rejects_read_only_path_and_rolls_back(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        autoresearch_loop,
        "list_changed_paths",
        lambda workdir, include_untracked=True: ["skyeye/products/tx1/live_advisor/service.py"],
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "ensure_read_only_paths_untouched",
        lambda changed_paths, read_only_roots: list(changed_paths),
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "rollback_candidate_commit",
        lambda workdir, commit: calls.append((workdir, commit)),
    )

    result = autoresearch_loop.evaluate_current_candidate(
        workdir=tmp_path,
        run_root=tmp_path / "runs" / "demo",
        start_commit="abc1234",
        experiment_index=1,
        raw_df=object(),
    )

    assert result["status"] == "invalid"
    assert result["reason_code"] == "read_only_path_touched"
    assert result["changed_paths"] == ["skyeye/products/tx1/live_advisor/service.py"]
    assert calls == [(tmp_path, "abc1234")]


def test_candidate_cycle_rejects_path_outside_allowed_write_roots(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        autoresearch_loop,
        "list_changed_paths",
        lambda workdir, include_untracked=True: ["tasks/todo.md"],
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "ensure_read_only_paths_untouched",
        lambda changed_paths, read_only_roots: [],
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "assert_only_allowed_paths_changed",
        lambda changed_paths, allowed_write_roots, read_only_roots: list(changed_paths),
        raising=False,
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "rollback_candidate_commit",
        lambda workdir, commit: calls.append((workdir, commit)),
    )

    result = autoresearch_loop.evaluate_current_candidate(
        workdir=tmp_path,
        run_root=tmp_path / "runs" / "demo",
        start_commit="keep1234",
        experiment_index=3,
        raw_df=object(),
        allowed_write_roots=["skyeye/products/tx1/dataset_builder.py"],
    )

    assert result["status"] == "invalid"
    assert result["reason_code"] == "path_outside_allowed_write_roots"
    assert result["changed_paths"] == ["tasks/todo.md"]
    assert calls == [(tmp_path, "keep1234")]


def test_candidate_cycle_discards_after_smoke_failure(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        autoresearch_loop,
        "list_changed_paths",
        lambda workdir, include_untracked=True: ["skyeye/products/tx1/dataset_builder.py"],
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "ensure_read_only_paths_untouched",
        lambda changed_paths, read_only_roots: [],
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "create_experiment_commit",
        lambda workdir, message, allowed_paths=None: "def5678",
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "run_candidate_trial",
        lambda **kwargs: {
            "prediction": {"rank_ic_mean": 0.01, "top_bucket_spread_mean": 0.001},
            "portfolio": {"net_mean_return": -0.001, "max_drawdown": 0.20, "mean_turnover": 0.21},
            "robustness": {
                "stability": {"stability_score": 10.0, "cv": 1.5},
                "overfit_flags": {
                    "flag_ic_decay": True,
                    "flag_spread_decay": False,
                    "flag_val_dominant": False,
                },
                "regime_scores": {"metric_consistency": {"positive_ratio": 0.2}},
            },
            "experiment_path": str(tmp_path / "runs" / "demo" / "experiments" / "exp_0001"),
        },
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "judge_candidate",
        lambda candidate_summary, baseline_summary, best_summary=None, stage="full": {
            "status": "discard",
            "reason_code": "guardrail_failed",
            "failed_guards": ["max_drawdown"],
            "score_delta": {"net_mean_return": -0.002},
        },
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "rollback_candidate_commit",
        lambda workdir, commit: calls.append((workdir, commit)),
    )

    result = autoresearch_loop.evaluate_current_candidate(
        workdir=tmp_path,
        run_root=tmp_path / "runs" / "demo",
        start_commit="abc1234",
        experiment_index=1,
        raw_df=object(),
        baseline_summary={"portfolio": {"net_mean_return": 0.001}},
    )

    assert result["status"] == "discard"
    assert result["commit"] == "def5678"
    assert result["smoke_decision"]["reason_code"] == "guardrail_failed"
    assert calls == [(tmp_path, "abc1234")]


def test_candidate_cycle_runs_full_trial_after_smoke_keep(monkeypatch, tmp_path):
    smoke_call = {"count": 0}
    rollback_calls = []

    monkeypatch.setattr(
        autoresearch_loop,
        "list_changed_paths",
        lambda workdir, include_untracked=True: ["skyeye/products/tx1/dataset_builder.py"],
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "ensure_read_only_paths_untouched",
        lambda changed_paths, read_only_roots: [],
    )
    monkeypatch.setattr(
        autoresearch_loop,
        "create_experiment_commit",
        lambda workdir, message, allowed_paths=None: "def5678",
    )

    def _fake_run_candidate_trial(**kwargs):
        smoke_call["count"] += 1
        if kwargs["max_folds"] == 1:
            return {
                "prediction": {"rank_ic_mean": 0.06, "top_bucket_spread_mean": 0.011},
                "portfolio": {"net_mean_return": 0.002, "max_drawdown": 0.08, "mean_turnover": 0.16},
                "robustness": {
                    "stability": {"stability_score": 60.0, "cv": 0.5},
                    "overfit_flags": {
                        "flag_ic_decay": False,
                        "flag_spread_decay": False,
                        "flag_val_dominant": False,
                    },
                    "regime_scores": {"metric_consistency": {"positive_ratio": 0.8}},
                },
                "experiment_path": str(tmp_path / "runs" / "demo" / "experiments" / "exp_0001"),
            }
        return {
            "prediction": {"rank_ic_mean": 0.07, "top_bucket_spread_mean": 0.012},
            "portfolio": {"net_mean_return": 0.0025, "max_drawdown": 0.08, "mean_turnover": 0.16},
            "robustness": {
                "stability": {"stability_score": 64.0, "cv": 0.48},
                "overfit_flags": {
                    "flag_ic_decay": False,
                    "flag_spread_decay": False,
                    "flag_val_dominant": False,
                },
                "regime_scores": {"metric_consistency": {"positive_ratio": 0.85}},
            },
            "experiment_path": str(tmp_path / "runs" / "demo" / "experiments" / "exp_0001"),
            }

    monkeypatch.setattr(autoresearch_loop, "run_candidate_trial", _fake_run_candidate_trial)

    def _fake_judge(candidate_summary, baseline_summary, best_summary=None, stage="full"):
        if stage == "smoke":
            return {"status": "keep", "reason_code": "smoke_pass", "failed_guards": [], "score_delta": {}}
        return {"status": "champion", "reason_code": "full_improved", "failed_guards": [], "score_delta": {}}

    monkeypatch.setattr(autoresearch_loop, "judge_candidate", _fake_judge)
    monkeypatch.setattr(
        autoresearch_loop,
        "rollback_candidate_commit",
        lambda workdir, commit: rollback_calls.append((workdir, commit)),
    )

    result = autoresearch_loop.evaluate_current_candidate(
        workdir=tmp_path,
        run_root=tmp_path / "runs" / "demo",
        start_commit="abc1234",
        experiment_index=1,
        raw_df=object(),
        baseline_summary={"portfolio": {"net_mean_return": 0.001}},
        smoke_max_folds=1,
        full_max_folds=4,
    )

    assert result["status"] == "champion"
    assert result["full_decision"]["reason_code"] == "full_improved"
    assert smoke_call["count"] == 2
    assert rollback_calls == []
