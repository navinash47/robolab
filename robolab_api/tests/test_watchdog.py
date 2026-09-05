"""Unit tests for Phase 3 watchdog kill policy (no live RunPod calls)."""

from __future__ import annotations

from robolab_api.watchdog import decide_kill


def test_unknown_run_id_killed() -> None:
    d = decide_kill(
        run_id="abc",
        run_known=False,
        heartbeat_age_sec=None,
        runtime_min=1.0,
        max_runtime_min=120.0,
        accrued_usd=0.0,
        budget_usd=1.0,
    )
    assert d.should_kill
    assert "unknown" in d.reason


def test_missing_run_id_killed() -> None:
    d = decide_kill(
        run_id=None,
        run_known=False,
        heartbeat_age_sec=None,
        runtime_min=0.0,
        max_runtime_min=120.0,
        accrued_usd=0.0,
        budget_usd=0.0,
    )
    assert d.should_kill


def test_stale_heartbeat_killed() -> None:
    d = decide_kill(
        run_id="r1",
        run_known=True,
        heartbeat_age_sec=11 * 60,
        runtime_min=5.0,
        max_runtime_min=120.0,
        accrued_usd=0.01,
        budget_usd=5.0,
    )
    assert d.should_kill
    assert "stale" in d.reason


def test_runtime_cap_killed() -> None:
    d = decide_kill(
        run_id="r1",
        run_known=True,
        heartbeat_age_sec=30.0,
        runtime_min=121.0,
        max_runtime_min=120.0,
        accrued_usd=0.01,
        budget_usd=5.0,
    )
    assert d.should_kill
    assert "MAX_RUNTIME" in d.reason


def test_budget_usd_killed() -> None:
    d = decide_kill(
        run_id="r1",
        run_known=True,
        heartbeat_age_sec=30.0,
        runtime_min=5.0,
        max_runtime_min=120.0,
        accrued_usd=0.06,
        budget_usd=0.05,
    )
    assert d.should_kill
    assert "budget_usd" in d.reason


def test_budget_zero_means_no_per_run_cap() -> None:
    d = decide_kill(
        run_id="r1",
        run_known=True,
        heartbeat_age_sec=30.0,
        runtime_min=5.0,
        max_runtime_min=120.0,
        accrued_usd=50.0,
        budget_usd=0.0,
    )
    assert not d.should_kill


def test_healthy_run_survives() -> None:
    d = decide_kill(
        run_id="r1",
        run_known=True,
        heartbeat_age_sec=60.0,
        runtime_min=10.0,
        max_runtime_min=120.0,
        accrued_usd=0.02,
        budget_usd=1.0,
    )
    assert not d.should_kill
