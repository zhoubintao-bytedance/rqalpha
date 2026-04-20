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
    )

    assert run_root.exists()
    assert (run_root / "state.json").exists()
    assert (run_root / "results.tsv").exists()
    assert state["run_tag"] == "demo"
    assert state["baseline_commit"] == "abc1234"
    assert state["current_commit"] == "abc1234"
    assert state["branch_name"] == "tx1-autoresearch"

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
    )

    store.append_result(
        commit="def5678",
        status="keep",
        metrics={
            "net_mean_return": 0.002,
            "max_drawdown": 0.08,
            "stability_score": 62.0,
        },
        description="raise holding bonus",
        experiment_path="/tmp/demo_exp",
    )
    store.update_after_decision(
        decision_status="keep",
        commit="def5678",
        candidate_summary={"net_mean_return": 0.002},
    )

    lines = (run_root / "results.tsv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[1].split("\t") == [
        "def5678",
        "keep",
        "0.002000",
        "0.080000",
        "62.000000",
        "raise holding bonus",
        "/tmp/demo_exp",
    ]

    state = store.load()
    assert state["current_commit"] == "def5678"
    assert state["best_commit"] == "abc1234"
    assert state["best_summary"] == {"net_mean_return": 0.001}


def test_state_store_only_promotes_best_commit_on_champion(tmp_path):
    run_root = tmp_path / "tx1_autoresearch" / "demo"
    store = AutoresearchStateStore(run_root)
    store.initialize(
        run_tag="demo",
        baseline_commit="abc1234",
        branch_name="tx1-autoresearch",
        baseline_summary={"net_mean_return": 0.001},
    )

    store.update_after_decision(
        decision_status="keep",
        commit="def5678",
        candidate_summary={"net_mean_return": 0.002},
    )
    keep_state = store.load()
    assert keep_state["current_commit"] == "def5678"
    assert keep_state["best_commit"] == "abc1234"
    assert keep_state["best_summary"] == {"net_mean_return": 0.001}

    store.update_after_decision(
        decision_status="champion",
        commit="fedcba9",
        candidate_summary={"net_mean_return": 0.003},
    )
    champion_state = store.load()
    assert champion_state["current_commit"] == "fedcba9"
    assert champion_state["best_commit"] == "fedcba9"
    assert champion_state["best_summary"] == {"net_mean_return": 0.003}
