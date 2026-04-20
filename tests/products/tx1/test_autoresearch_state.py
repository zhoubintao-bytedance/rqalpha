import json

from skyeye.products.tx1.autoresearch.state import (
    RESULTS_TSV_HEADER,
    AutoresearchStateStore,
)


def test_state_store_initializes_run_root_and_files(tmp_path):
    run_root = tmp_path / "tx1_autoresearch" / "demo"

    store = AutoresearchStateStore(run_root)
    state = store.initialize(
        run_tag="demo",
        baseline_commit="abc1234",
        branch_name="tx1-autoresearch",
        baseline_summary={"rank_ic_mean": 0.05},
        budget={"max_experiments": 3, "smoke_max_folds": 1, "full_max_folds": 4},
        raw_df_spec={"universe_size": 300, "start_date": "2020-01-01", "end_date": "2024-12-31"},
        allowed_write_roots=["skyeye/products/tx1/dataset_builder.py"],
        read_only_roots=["skyeye/products/tx1/live_advisor/"],
    )

    assert run_root.exists()
    assert (run_root / "state.json").exists()
    assert (run_root / "results.tsv").exists()
    assert state["run_tag"] == "demo"
    assert state["baseline_commit"] == "abc1234"
    assert state["current_commit"] == "abc1234"
    assert state["branch_name"] == "tx1-autoresearch"
    assert state["frontier_commits"] == []
    assert state["next_experiment_index"] == 0
    assert state["budget"] == {"max_experiments": 3, "smoke_max_folds": 1, "full_max_folds": 4}
    assert state["raw_df_spec"] == {
        "universe_size": 300,
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
    }
    assert state["allowed_write_roots"] == ["skyeye/products/tx1/dataset_builder.py"]
    assert state["read_only_roots"] == ["skyeye/products/tx1/live_advisor/"]

    persisted = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    assert persisted["baseline_summary"] == {"rank_ic_mean": 0.05}
    assert (run_root / "results.tsv").read_text(encoding="utf-8").splitlines() == [
        RESULTS_TSV_HEADER
    ]


def test_state_store_appends_tsv_row_and_updates_state(tmp_path):
    run_root = tmp_path / "tx1_autoresearch" / "demo"
    store = AutoresearchStateStore(run_root)
    store.initialize(
        run_tag="demo",
        baseline_commit="abc1234",
        branch_name="tx1-autoresearch",
        baseline_summary={"net_mean_return": 0.001},
        budget={"max_experiments": 3, "smoke_max_folds": 1, "full_max_folds": 4},
        raw_df_spec={"universe_size": 300, "start_date": "2020-01-01", "end_date": "2024-12-31"},
        allowed_write_roots=["skyeye/products/tx1/dataset_builder.py"],
        read_only_roots=["skyeye/products/tx1/live_advisor/"],
    )

    store.append_result(
        experiment_index=1,
        parent_commit="abc1234",
        commit="def5678",
        status="keep",
        stage_reached="full",
        metrics={
            "net_mean_return": 0.002,
            "max_drawdown": 0.08,
            "stability_score": 62.0,
        },
        reason_code="full_pass_not_best",
        experiment_path="/tmp/demo_exp",
    )
    store.update_after_decision(
        experiment_index=1,
        decision_status="keep",
        parent_commit="abc1234",
        commit="def5678",
        candidate_summary={"net_mean_return": 0.002},
        stage_reached="full",
        reason_code="full_pass_not_best",
    )

    lines = (run_root / "results.tsv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[1].split("\t") == [
        "def5678",
        "keep",
        "0.002000",
        "0.080000",
        "62.000000",
        "full_pass_not_best",
        "/tmp/demo_exp",
        "1",
        "abc1234",
        "full",
    ]

    state = store.load()
    assert state["current_commit"] == "def5678"
    assert state["best_commit"] == "abc1234"
    assert state["best_summary"] == {"net_mean_return": 0.001}
    assert state["next_experiment_index"] == 2
    assert state["last_attempt"]["parent_commit"] == "abc1234"
    assert state["last_attempt"]["candidate_commit"] == "def5678"
    assert state["last_attempt"]["stage_reached"] == "full"


def test_state_store_only_promotes_best_commit_on_champion(tmp_path):
    run_root = tmp_path / "tx1_autoresearch" / "demo"
    store = AutoresearchStateStore(run_root)
    store.initialize(
        run_tag="demo",
        baseline_commit="abc1234",
        branch_name="tx1-autoresearch",
        baseline_summary={"net_mean_return": 0.001},
        budget={"max_experiments": 3, "smoke_max_folds": 1, "full_max_folds": 4},
        raw_df_spec={"universe_size": 300, "start_date": "2020-01-01", "end_date": "2024-12-31"},
        allowed_write_roots=["skyeye/products/tx1/dataset_builder.py"],
        read_only_roots=["skyeye/products/tx1/live_advisor/"],
    )

    store.update_after_decision(
        experiment_index=1,
        decision_status="keep",
        parent_commit="abc1234",
        commit="def5678",
        candidate_summary={"net_mean_return": 0.002},
        stage_reached="full",
        reason_code="full_pass_not_best",
    )
    keep_state = store.load()
    assert keep_state["current_commit"] == "def5678"
    assert keep_state["best_commit"] == "abc1234"
    assert keep_state["best_summary"] == {"net_mean_return": 0.001}

    store.update_after_decision(
        experiment_index=2,
        decision_status="champion",
        parent_commit="def5678",
        commit="fedcba9",
        candidate_summary={"net_mean_return": 0.003},
        stage_reached="full",
        reason_code="full_improved",
    )
    champion_state = store.load()
    assert champion_state["current_commit"] == "fedcba9"
    assert champion_state["best_commit"] == "fedcba9"
    assert champion_state["best_summary"] == {"net_mean_return": 0.003}


def test_state_store_initializes_safety_closure_fields(tmp_path):
    store = AutoresearchStateStore(tmp_path / "run")

    state = store.initialize(
        run_tag="demo",
        baseline_commit="abc1234",
        branch_name="tx1-autoresearch",
        baseline_summary={},
        budget={"max_experiments": 3, "smoke_max_folds": 1, "full_max_folds": 4},
        raw_df_spec={"universe_size": 300, "start_date": "2020-01-01", "end_date": "2024-12-31"},
        allowed_write_roots=["skyeye/products/tx1/dataset_builder.py"],
        read_only_roots=["skyeye/products/tx1/live_advisor/"],
    )

    assert state["last_reason_code"] is None
    assert state["last_experiment_path"] == ""
    assert state["last_error"] is None
    assert state["frontier_commits"] == []
    assert state["next_experiment_index"] == 0
    assert state["last_attempt"] == {
        "experiment_index": None,
        "parent_commit": None,
        "candidate_commit": None,
        "status": None,
        "reason_code": None,
        "stage_reached": "",
        "experiment_path": "",
    }


def test_state_store_updates_reason_and_error_fields(tmp_path):
    store = AutoresearchStateStore(tmp_path / "run")
    store.initialize(
        run_tag="demo",
        baseline_commit="abc1234",
        branch_name="tx1-autoresearch",
        baseline_summary={"portfolio": {"net_mean_return": 0.001}},
        budget={"max_experiments": 3, "smoke_max_folds": 1, "full_max_folds": 4},
        raw_df_spec={"universe_size": 300, "start_date": "2020-01-01", "end_date": "2024-12-31"},
        allowed_write_roots=["skyeye/products/tx1/dataset_builder.py"],
        read_only_roots=["skyeye/products/tx1/live_advisor/"],
    )

    state = store.update_after_decision(
        experiment_index=1,
        decision_status="crash",
        parent_commit="abc1234",
        commit="def5678",
        candidate_summary={"experiment_path": "/tmp/exp_0001"},
        stage_reached="smoke",
        reason_code="smoke_crashed",
        error_message="boom",
    )

    assert state["last_status"] == "crash"
    assert state["last_reason_code"] == "smoke_crashed"
    assert state["last_experiment_path"] == "/tmp/exp_0001"
    assert state["last_error"] == "boom"
    assert state["best_commit"] == "abc1234"
    assert state["last_attempt"] == {
        "experiment_index": 1,
        "parent_commit": "abc1234",
        "candidate_commit": "def5678",
        "status": "crash",
        "reason_code": "smoke_crashed",
        "stage_reached": "smoke",
        "experiment_path": "/tmp/exp_0001",
    }
