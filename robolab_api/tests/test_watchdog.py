"""Unit tests for Phase 3 watchdog kill policy (no live RunPod calls)."""

from __future__ import annotations

from robolab_api.watchdog import (
    classify_remote_pod_status,
    decide_kill,
    decide_pod_absent,
    decide_stale_heartbeat,
)


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


def test_single_stale_heartbeat_does_not_kill() -> None:
    d = decide_kill(
        run_id="r1",
        run_known=True,
        heartbeat_age_sec=11 * 60,
        runtime_min=5.0,
        max_runtime_min=120.0,
        accrued_usd=0.01,
        budget_usd=5.0,
        consecutive_stale_misses=1,
        miss_threshold=3,
    )
    assert not d.should_kill
    assert not d.mark_failed
    assert d.warn_only
    assert "grace" in d.reason


def test_stale_after_threshold_running_extends_grace() -> None:
    d = decide_stale_heartbeat(
        heartbeat_age_sec=11 * 60,
        consecutive_stale_misses=3,
        pod_remote_status="RUNNING",
        miss_threshold=3,
    )
    assert not d.should_kill
    assert not d.mark_failed
    assert d.warn_only
    assert "RUNNING" in d.reason


def test_stale_after_threshold_missing_marks_failed_not_kill() -> None:
    d = decide_stale_heartbeat(
        heartbeat_age_sec=11 * 60,
        consecutive_stale_misses=3,
        pod_remote_status="MISSING",
        miss_threshold=3,
    )
    assert not d.should_kill
    assert d.mark_failed
    assert "FAILED" in d.reason or "without kill" in d.reason


def test_stale_after_threshold_exited_marks_failed() -> None:
    d = decide_kill(
        run_id="r1",
        run_known=True,
        heartbeat_age_sec=15 * 60,
        runtime_min=5.0,
        max_runtime_min=120.0,
        accrued_usd=0.01,
        budget_usd=5.0,
        consecutive_stale_misses=4,
        pod_remote_status="EXITED",
        miss_threshold=3,
    )
    assert not d.should_kill
    assert d.mark_failed


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


def test_pod_absent_grace_before_threshold() -> None:
    d = decide_pod_absent(
        consecutive_absent=1,
        pod_remote_status=None,
        miss_threshold=3,
    )
    assert d.warn_only
    assert not d.mark_failed


def test_pod_absent_running_not_vanished() -> None:
    d = decide_pod_absent(
        consecutive_absent=3,
        pod_remote_status="RUNNING",
        miss_threshold=3,
    )
    assert d.warn_only
    assert not d.mark_failed
    assert "RUNNING" in d.reason


def test_pod_absent_confirmed_gone_marks_failed() -> None:
    d = decide_pod_absent(
        consecutive_absent=3,
        pod_remote_status="TERMINATED",
        miss_threshold=3,
    )
    assert d.mark_failed
    assert not d.should_kill


def test_classify_remote_pod_status() -> None:
    assert classify_remote_pod_status(None) == "MISSING"
    assert classify_remote_pod_status({"desiredStatus": "RUNNING"}) == "RUNNING"
    assert classify_remote_pod_status({"desiredStatus": "EXITED"}) == "EXITED"
    assert classify_remote_pod_status({"runtime": {"uptimeInSeconds": 10}}) == "RUNNING"
