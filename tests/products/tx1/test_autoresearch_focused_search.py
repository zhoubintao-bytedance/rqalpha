import json

import pandas as pd

from skyeye.products.tx1.autoresearch import focused_search


def _proxy_summary(
    *,
    net_mean_return,
    max_drawdown,
    stability_score,
    positive_ratio,
    experiment_path,
    flag_ic_decay=False,
    flag_spread_decay=False,
    flag_val_dominant=False,
):
    """构造最小可用的 proxy summary。"""
    return {
        "prediction": {
            "rank_ic_mean": 0.05,
            "top_bucket_spread_mean": 0.01,
        },
        "portfolio": {
            "net_mean_return": net_mean_return,
            "max_drawdown": max_drawdown,
            "mean_turnover": 0.02,
        },
        "robustness": {
            "stability": {
                "stability_score": stability_score,
                "cv": 0.20,
            },
            "overfit_flags": {
                "flag_ic_decay": flag_ic_decay,
                "flag_spread_decay": flag_spread_decay,
                "flag_val_dominant": flag_val_dominant,
            },
            "regime_scores": {
                "metric_consistency": {
                    "positive_ratio": positive_ratio,
                }
            },
        },
        "experiment_path": experiment_path,
    }


def _replay_summary(
    *,
    composite_score,
    annualized_returns,
    max_drawdown,
    stability_score,
    experiment_path,
    risk_codes=None,
):
    """构造最小可用的 replay summary。"""
    risk_codes = list(risk_codes or [])
    return {
        "prediction": {
            "rank_ic_mean": composite_score,
            "top_bucket_spread_mean": composite_score,
        },
        "portfolio": {
            "net_mean_return": annualized_returns,
            "max_drawdown": max_drawdown,
            "win_rate": 0.56,
            "mean_turnover": 0.0,
        },
        "robustness": {
            "stability": {
                "stability_score": stability_score,
                "cv": 0.25,
            },
            "regime_scores": {
                "metric_consistency": {
                    "positive_ratio": 0.80,
                }
            },
            "risk_codes": risk_codes,
        },
        "replay": {
            "composite_score": composite_score,
            "stability_score": stability_score,
            "risk_codes": risk_codes,
            "num_windows": 5,
        },
        "experiment_path": experiment_path,
    }


def test_build_liquidity_focus_candidates_targets_stabilization_grid():
    candidates = focused_search.build_liquidity_focus_candidates()
    anchor_signature = focused_search._candidate_signature(
        focused_search._build_proxy_anchor_candidate(model_kind="lgbm")
    )

    assert len(candidates) == 2
    assert {candidate["phase"] for candidate in candidates} == {"phase1"}
    assert {candidate["evaluation_mode"] for candidate in candidates} == {"proxy"}
    assert {
        str(candidate.get("reg_profile")) for candidate in candidates
    } == {"heavy_reg", "ultra_reg"}
    assert {
        (candidate.get("model") or {}).get("kind") for candidate in candidates
    } == {"lgbm"}
    assert all(
        not (candidate.get("preprocessing") or {}).get("enabled", False)
        for candidate in candidates
    )
    assert all(
        focused_search._candidate_signature(candidate) != anchor_signature
        for candidate in candidates
    )
    assert any((candidate.get("model") or {}).get("params") for candidate in candidates)


def test_build_replay_search_candidates_targets_combo_neighborhood():
    candidates = focused_search.build_replay_search_candidates()

    assert len(candidates) == 5
    assert {candidate["phase"] for candidate in candidates} == {"phase2"}
    assert {candidate["evaluation_mode"] for candidate in candidates} == {"replay"}
    assert {
        "combo_h40_bonus1",
        "combo_h45_bonus1",
        "combo_b25_h45",
    } <= {candidate["artifact_line_id"] for candidate in candidates}
    assert {candidate["strategy_profile"] for candidate in candidates} == {"smooth"}
    assert any(candidate.get("tx1_profile_overrides") for candidate in candidates)
    assert not any(
        candidate.get("tx1_profile_overrides") == {"turnover_threshold": 0.20}
        for candidate in candidates
    )


def test_build_replay_probe_neighbors_skips_anchor_equivalent_noop_override():
    neighbors = focused_search._build_replay_probe_neighbors(
        round_index=1,
        source_candidate={
            "artifact_line_id": "combo_b25_h45",
            "strategy_profile": "smooth",
            "tx1_profile_overrides": {"turnover_threshold": 0.25},
        },
    )

    assert all(candidate.get("tx1_profile_overrides") for candidate in neighbors)
    assert not any(
        candidate.get("tx1_profile_overrides") == {"turnover_threshold": 0.20}
        for candidate in neighbors
    )


def test_build_liquidity_focus_candidates_uses_frontier_entries_for_backfill():
    candidates = focused_search.build_liquidity_focus_candidates(
        round_index=3,
        frontier_entries=[
            {
                "candidate": {
                    "reg_profile": "default",
                    "model": {"kind": "lgbm"},
                    "preprocessing": None,
                }
            }
        ],
    )

    assert any(
        (candidate.get("preprocessing") or {}).get("enabled", False)
        for candidate in candidates
    )
    assert {"default", "heavy_reg", "slow_lr"} <= {
        str(candidate.get("reg_profile"))
        for candidate in candidates
    }


def test_build_liquidity_focus_candidates_backfill_skips_proxy_anchor_equivalent_neighbor():
    candidates = focused_search.build_liquidity_focus_candidates(
        round_index=3,
        frontier_entries=[
            {
                "candidate": {
                    "reg_profile": "heavy_reg",
                    "model": {"kind": "lgbm"},
                    "preprocessing": None,
                }
            }
        ],
    )

    anchor_signature = focused_search._candidate_signature(
        focused_search._build_proxy_anchor_candidate(model_kind="lgbm")
    )
    assert all(
        focused_search._candidate_signature(candidate) != anchor_signature
        for candidate in candidates
    )
    assert {
        str(candidate.get("reg_profile")) for candidate in candidates
    } == {"heavy_reg", "ultra_reg", "leaf_guard"}


def test_build_replay_profile_seed_candidates_prefers_neighbor_profiles():
    candidates = focused_search._build_replay_profile_seed_candidates(
        round_index=1,
        replay_entries=[
            {
                "candidate": {
                    "artifact_line_id": "combo_b25_h45",
                    "strategy_profile": "smooth",
                },
                "entry": {"status": "keep"},
                "summary": {"replay": {"num_windows": 5}},
            }
        ],
    )

    assert {"baseline", "soft_sticky"} <= {
        candidate["strategy_profile"] for candidate in candidates
    }
    assert not any(candidate["strategy_profile"] == "smooth" for candidate in candidates)


def test_update_axis_state_freezes_unproductive_phase2_axis():
    axis_key = "phase2:combo_b25_h45:smooth:turnover_threshold"
    axis_states = {}
    entry = {
        "phase": "phase2",
        "status": "discard",
        "reason_code": "replay_no_material_improvement",
        "search_axis": axis_key,
        "score_delta": {},
        "metrics": {
            "composite_score": 50.0,
            "net_mean_return": 0.10,
            "max_drawdown": 0.08,
            "stability_score": 3.0,
        },
        "risk_codes": [],
    }

    for _ in range(3):
        focused_search._update_axis_state(axis_states, entry)

    assert axis_states[axis_key]["frozen"] is True
    assert axis_states[axis_key]["no_improvement_streak"] == 3


def test_resolve_round_cap_expands_for_longer_budgets():
    assert (
        focused_search._resolve_round_cap(
            requested_rounds=4,
            max_runtime_hours=8.0,
            phase="phase2",
        )
        > 4
    )
    assert (
        focused_search._resolve_round_cap(
            requested_rounds=3,
            max_runtime_hours=8.0,
            phase="phase1",
        )
        > 3
    )


def test_register_candidates_skips_blocked_proxy_anchor_signature():
    discovered_signatures = set()
    dedupe_stats = {"duplicate_skips": 0}
    proxy_anchor = focused_search._build_proxy_anchor_candidate(model_kind="lgbm")
    duplicate_candidate = focused_search._build_proxy_candidate(
        round_index=0,
        model_kind="lgbm",
        reg_profile="default",
        use_preprocessing=False,
    )

    fresh_candidates = focused_search._register_candidates(
        [duplicate_candidate],
        discovered_signatures,
        dedupe_stats,
        blocked_signatures={
            focused_search._candidate_signature(proxy_anchor),
        },
    )

    assert fresh_candidates == []
    assert discovered_signatures == set()
    assert dedupe_stats["duplicate_skips"] == 1


def test_apply_cross_section_filter_keeps_only_rows_above_per_date_median():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "order_book_id": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "ep_ratio_ttm": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
            "return_on_equity_ttm": [1.0, 2.0, 4.0, 3.0, 1.0, 4.0, 3.0, 2.0],
        }
    )

    filtered = focused_search.apply_cross_section_filter(
        frame,
        {
            "ep_ratio_ttm": "above_median",
            "return_on_equity_ttm": "above_median",
        },
    )

    assert list(filtered["order_book_id"]) == ["C", "D", "B"]


def test_run_replay_candidate_trial_writes_summary_and_routes_profile_overrides(
    monkeypatch,
    tmp_path,
):
    candidate = focused_search.build_replay_search_candidates()[0]
    candidate = {
        **candidate,
        "artifact_line_id": "combo_b25_h45",
        "strategy_profile": "smooth",
        "tx1_profile_overrides": {"single_stock_cap": 0.08, "ema_halflife": 8},
    }
    captured = {}

    def _fake_run_rolling_backtests(
        strategy_file,
        cash,
        selected_indices=None,
        extra_mods=None,
        mod_configs=None,
        extra_config=None,
        benchmark_id=None,
        return_details=False,
    ):
        captured["strategy_file"] = strategy_file
        captured["cash"] = cash
        captured["selected_indices"] = selected_indices
        captured["extra_mods"] = extra_mods
        captured["mod_configs"] = mod_configs
        captured["extra_config"] = extra_config
        captured["return_details"] = return_details
        return {
            "windows": [
                {
                    "idx": 1,
                    "start": pd.Timestamp("2024-01-01").date(),
                    "end": pd.Timestamp("2024-03-31").date(),
                    "score": 52.0,
                    "summary": {
                        "annualized_returns": 0.12,
                        "max_drawdown": -0.08,
                        "sharpe": 1.10,
                        "win_rate": 0.57,
                    },
                    "sample_diagnostics": {"active_days": 40, "sparse": False},
                    "mod_results": {
                        focused_search.TX1_DIAGNOSTIC_MOD_NAME: {
                            "strategy_diagnostics": {
                                "rebalance_checks": 50,
                                "executed_rebalances": 3,
                                "missing_signal_days": 0,
                                "turnover_skips": 1,
                            },
                            "runtime": {"profile": "smooth"},
                            "last_signal_date": "2024-03-29",
                            "pending_turnover": 0.12,
                        }
                    },
                },
                {
                    "idx": 2,
                    "start": pd.Timestamp("2024-04-01").date(),
                    "end": pd.Timestamp("2024-06-30").date(),
                    "score": 49.0,
                    "summary": {
                        "annualized_returns": 0.10,
                        "max_drawdown": -0.09,
                        "sharpe": 1.00,
                        "win_rate": 0.55,
                    },
                    "sample_diagnostics": {"active_days": 35, "sparse": False},
                    "mod_results": {
                        focused_search.TX1_DIAGNOSTIC_MOD_NAME: {
                            "strategy_diagnostics": {
                                "rebalance_checks": 45,
                                "executed_rebalances": 2,
                                "missing_signal_days": 0,
                                "turnover_skips": 2,
                            },
                            "runtime": {"profile": "smooth"},
                            "last_signal_date": "2024-06-28",
                            "pending_turnover": 0.10,
                        }
                    },
                },
            ],
            "failed_windows": [],
            "total_windows": 2,
            "successful_windows": 2,
        }

    monkeypatch.setattr(
        focused_search,
        "run_rolling_backtests",
        _fake_run_rolling_backtests,
    )

    summary = focused_search.run_replay_candidate_trial(
        run_root=tmp_path / "runs" / "demo",
        experiment_index=3,
        candidate=candidate,
        cash=500000,
        selected_indices=[1, 2],
    )

    assert captured["cash"] == 500000
    assert captured["selected_indices"] == [1, 2]
    assert captured["extra_mods"] == [focused_search.TX1_DIAGNOSTIC_MOD_NAME]
    assert captured["mod_configs"] == {
        focused_search.TX1_DIAGNOSTIC_MOD_NAME: {
            "lib": focused_search.TX1_DIAGNOSTIC_MOD_LIB
        }
    }
    assert captured["return_details"] is True
    assert captured["extra_config"] == {
        "strategy_profile": "smooth",
        "tx1_artifact_line": "combo_b25_h45",
        "tx1_profile_overrides": {"single_stock_cap": 0.08, "ema_halflife": 8},
    }
    assert summary["replay"]["artifact_line_id"] == "combo_b25_h45"
    assert summary["replay"]["strategy_profile"] == "smooth"
    assert summary["replay"]["profile_overrides"] == {
        "single_stock_cap": 0.08,
        "ema_halflife": 8,
    }
    assert summary["replay"]["num_windows"] == 2
    assert summary["replay"]["diagnostics"]["health_code"] == "ok"
    assert summary["replay"]["diagnostics"]["executed_rebalances_total"] == 5
    assert (tmp_path / "runs" / "demo" / "experiments" / "exp_0003" / "replay" / "replay_summary.json").exists()


def test_judge_replay_candidate_marks_empty_replay_as_infra_failure():
    summary = _replay_summary(
        composite_score=0.0,
        annualized_returns=0.0,
        max_drawdown=0.0,
        stability_score=0.0,
        experiment_path="/tmp/replay_empty",
    )
    summary["replay"]["num_windows"] = 0
    summary["replay"]["diagnostics"] = {"health_code": "window_exception"}

    decision = focused_search.judge_replay_candidate(
        summary,
        baseline_summary=_replay_summary(
            composite_score=50.0,
            annualized_returns=0.10,
            max_drawdown=0.08,
            stability_score=40.0,
            experiment_path="/tmp/replay_base",
        ),
        best_summary=None,
    )

    assert decision["status"] == "crash"
    assert decision["reason_code"] == "replay_infra_failure"
    assert "window_exception" in decision["failed_guards"]


def test_run_liquidity_focus_search_runs_two_phase_budget_loop(monkeypatch, tmp_path):
    labeled_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "order_book_id": ["A", "B"],
            "mom_40d": [0.1, 0.2],
            "volume_ratio_20d": [1.1, 1.2],
            "target_label": [0.1, 0.2],
            "label_return_raw": [0.1, 0.2],
        }
    )
    phase_calls = []

    phase1_round0 = [
        {
            "id": "phase1_a",
            "phase": "phase1",
            "evaluation_mode": "proxy",
            "family": "stabilization",
            "description": "phase1 round0",
            "search_round": 0,
            "features": ["mom_40d", "volume_ratio_20d"],
            "model": {"kind": "linear"},
        }
    ]
    phase1_round1 = [
        {
            "id": "phase1_b",
            "phase": "phase1",
            "evaluation_mode": "proxy",
            "family": "stabilization",
            "description": "phase1 round1",
            "search_round": 1,
            "features": ["mom_40d", "volume_ratio_20d"],
            "model": {"kind": "tree"},
        }
    ]
    phase2_round0 = [
        {
            "id": "replay_a",
            "phase": "phase2",
            "evaluation_mode": "replay",
            "family": "rolling_score",
            "description": "phase2 round0",
            "search_round": 0,
            "artifact_line_id": "combo_h40_bonus1",
            "strategy_profile": "smooth",
            "tx1_profile_overrides": {},
        }
    ]
    phase2_round1 = [
        {
            "id": "replay_b",
            "phase": "phase2",
            "evaluation_mode": "replay",
            "family": "rolling_score",
            "description": "phase2 round1",
            "search_round": 1,
            "artifact_line_id": "combo_b25_h45",
            "strategy_profile": "smooth",
            "tx1_profile_overrides": {"single_stock_cap": 0.08},
        }
    ]

    def _fake_build_liquidity_focus_candidates(*, round_index=0, frontier_entries=None, tried_signatures=None):
        del frontier_entries, tried_signatures
        if round_index == 0:
            return list(phase1_round0)
        if round_index == 1:
            return list(phase1_round1)
        return []

    def _fake_build_replay_search_candidates(*, round_index=0, replay_entries=None, tried_signatures=None):
        del replay_entries, tried_signatures
        if round_index == 0:
            return list(phase2_round0)
        if round_index == 1:
            return list(phase2_round1)
        return []

    def _fake_run_focused_candidate_trial(
        *,
        run_root,
        experiment_index,
        labeled_df,
        config,
        stage=None,
        max_folds=None,
    ):
        del run_root, experiment_index, labeled_df, max_folds
        phase_calls.append(("proxy", config["experiment_name"], stage))
        experiment_name = config["experiment_name"]
        if experiment_name == "tx1_autoresearch_liquidity_plus_anchor":
            return _proxy_summary(
                net_mean_return=0.00020,
                max_drawdown=0.050,
                stability_score=12.0,
                positive_ratio=0.70,
                experiment_path=str(tmp_path / "proxy_anchor"),
                flag_ic_decay=True,
            )
        if experiment_name == "tx1_autoresearch_phase1_a" and stage == "smoke":
            return _proxy_summary(
                net_mean_return=0.00024,
                max_drawdown=0.048,
                stability_score=12.5,
                positive_ratio=0.75,
                experiment_path=str(tmp_path / "phase1_a_smoke"),
            )
        if experiment_name == "tx1_autoresearch_phase1_a" and stage == "full":
            return _proxy_summary(
                net_mean_return=0.00025,
                max_drawdown=0.047,
                stability_score=14.0,
                positive_ratio=0.78,
                experiment_path=str(tmp_path / "phase1_a_full"),
            )
        return _proxy_summary(
            net_mean_return=0.00018,
            max_drawdown=0.051,
            stability_score=11.0,
            positive_ratio=0.68,
            experiment_path=str(tmp_path / "phase1_b_full"),
            flag_ic_decay=True,
        )

    def _fake_smoke_judge(summary, *, baseline_summary=None, best_summary=None, stage="full", guardrails=None):
        del summary, baseline_summary, best_summary, guardrails
        if stage == "smoke":
            return {
                "status": "keep",
                "reason_code": "smoke_pass",
                "failed_guards": [],
                "score_delta": {},
                "best_score_delta": {},
            }
        raise AssertionError("full stage should not call base smoke judge in this test")

    def _fake_judge_phase1_candidate(candidate_summary, *, anchor_summary=None):
        del anchor_summary
        if candidate_summary["experiment_path"].endswith("phase1_a_full"):
            return {
                "status": "frontier_seed",
                "reason_code": "flags_cleared",
                "failed_guards": [],
                "score_delta": {"net_mean_return": 0.00005},
                "best_score_delta": {"net_mean_return": 0.00005},
            }
        return {
            "status": "discard",
            "reason_code": "proxy_rejected",
            "failed_guards": ["flag_ic_decay"],
            "score_delta": {},
            "best_score_delta": {},
        }

    def _fake_run_replay_candidate_trial(
        *,
        run_root,
        experiment_index,
        candidate,
        cash=1000000,
        selected_indices=None,
    ):
        del run_root, experiment_index, cash, selected_indices
        phase_calls.append(("replay", candidate["id"], candidate.get("artifact_line_id")))
        if candidate["id"].startswith("replay_r0_combo_b25_h45_smooth"):
            return _replay_summary(
                composite_score=50.5,
                annualized_returns=0.11,
                max_drawdown=0.080,
                stability_score=48.0,
                experiment_path=str(tmp_path / "replay_anchor"),
            )
        if candidate["id"] == "replay_a":
            return _replay_summary(
                composite_score=51.0,
                annualized_returns=0.112,
                max_drawdown=0.079,
                stability_score=49.0,
                experiment_path=str(tmp_path / "replay_a"),
            )
        return _replay_summary(
            composite_score=52.0,
            annualized_returns=0.115,
            max_drawdown=0.077,
            stability_score=50.5,
            experiment_path=str(tmp_path / "replay_b"),
        )

    def _fake_judge_replay_candidate(candidate_summary, *, baseline_summary=None, best_summary=None):
        del baseline_summary, best_summary
        if candidate_summary["experiment_path"].endswith("replay_a"):
            return {
                "status": "keep",
                "reason_code": "replay_pass_not_best",
                "failed_guards": [],
                "score_delta": {"composite_score": 0.5},
                "best_score_delta": {"composite_score": 0.5},
            }
        return {
            "status": "champion",
            "reason_code": "replay_improved",
            "failed_guards": [],
            "score_delta": {"composite_score": 1.5},
            "best_score_delta": {"composite_score": 1.0},
        }

    monkeypatch.setattr(
        focused_search,
        "build_research_raw_df",
        lambda **kwargs: pd.DataFrame({"raw": [1]}),
    )
    monkeypatch.setattr(
        focused_search,
        "build_labeled_panel",
        lambda **kwargs: labeled_df.copy(),
    )
    monkeypatch.setattr(
        focused_search,
        "build_liquidity_focus_candidates",
        _fake_build_liquidity_focus_candidates,
    )
    monkeypatch.setattr(
        focused_search,
        "build_replay_search_candidates",
        _fake_build_replay_search_candidates,
    )
    monkeypatch.setattr(
        focused_search,
        "_build_replay_profile_seed_candidates",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        focused_search,
        "_build_replay_artifact_backfill_candidates",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        focused_search,
        "run_focused_candidate_trial",
        _fake_run_focused_candidate_trial,
    )
    monkeypatch.setattr(focused_search, "judge_candidate", _fake_smoke_judge)
    monkeypatch.setattr(
        focused_search,
        "judge_phase1_candidate",
        _fake_judge_phase1_candidate,
    )
    monkeypatch.setattr(
        focused_search,
        "run_replay_candidate_trial",
        _fake_run_replay_candidate_trial,
    )
    monkeypatch.setattr(
        focused_search,
        "judge_replay_candidate",
        _fake_judge_replay_candidate,
    )

    result = focused_search.run_liquidity_focus_search(
        run_tag="demo",
        runs_root=tmp_path / "runs",
        max_runtime_hours=0.01,
        smoke_max_folds=1,
        full_max_folds=2,
        max_stabilization_rounds=2,
        max_replay_rounds=2,
    )

    assert result["status"] == "value_pool_exhausted"
    assert result["stop_reason_detail"] == "all_value_queues_exhausted"
    assert result["phase_progress"]["stabilization_rounds_started"] == 2
    assert result["phase_progress"]["replay_rounds_started"] == 2
    assert result["candidates_total"] == 4
    assert result["candidates_evaluated"] == 4
    assert result["baselines"]["phase1_proxy"]["candidate_id"] == "liquidity_plus_anchor"
    assert result["baselines"]["phase2_replay"]["candidate_id"].startswith("replay_r0_combo_b25_h45_smooth")
    assert result["champion"]["candidate_id"] == "replay_b"
    assert result["frontier"][0]["candidate_id"] == "phase1_a"
    assert result["search_diagnostics"]["axes_total"] >= 2

    run_root = tmp_path / "runs" / "demo"
    assert (run_root / "focused_results.json").exists()
    assert (run_root / "focused_leaderboard.tsv").exists()

    payload = json.loads((run_root / "focused_results.json").read_text(encoding="utf-8"))
    assert payload["leaderboard"][0]["candidate_id"] == "replay_b"
    assert {entry["evaluation_mode"] for entry in payload["leaderboard"]} == {"proxy", "replay"}
    assert payload["status"] == "value_pool_exhausted"
    assert payload["search_diagnostics"]["high_value_queue_size"] == 0
    assert phase_calls[:4] == [
        ("proxy", "tx1_autoresearch_liquidity_plus_anchor", "proxy_anchor"),
        ("proxy", "tx1_autoresearch_phase1_a", "smoke"),
        ("proxy", "tx1_autoresearch_phase1_a", "full"),
        ("replay", "replay_r0_combo_b25_h45_smooth", "combo_b25_h45"),
    ]
