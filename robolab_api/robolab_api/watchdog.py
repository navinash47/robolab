"""Backend cost / orphan / stale-heartbeat watchdog.

Every ~60s: terminate pods that violate safety rules and label runs
KILLED_BY_WATCHDOG. Orphan sweep on startup.

Stale heartbeats / list_pods gaps require N consecutive observations plus a
RunPod get_pod confirm before any kill or "pod vanished" failure — avoids
false orphans from a single flaky API/tunnel tick.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from sqlmodel import Session, select

from robolab.core.run import TERMINAL_RUN_STATUSES, RunStatus
from robolab_api.db import CostLedger, Pod, Run, engine

logger = logging.getLogger("robolab.watchdog")

STALE_HEARTBEAT_SEC = 10 * 60
# PROVISIONING + null pod_id (create never finished / API crash mid-launch).
STUCK_PROVISIONING_SEC = int(float(os.environ.get("STUCK_PROVISIONING_SEC", str(10 * 60))))
WATCHDOG_INTERVAL_SEC = 60
# Missed heartbeats / list_pods absences before acting (env override).
CONSECUTIVE_MISS_THRESHOLD = max(
    1, int(float(os.environ.get("WATCHDOG_CONSECUTIVE_MISSES", "3")))
)

# In-process counters — reset on API restart (extra grace; safer than false kill).
_stale_misses: dict[str, int] = {}
_absent_misses: dict[str, int] = {}

RemoteAction = Literal["none", "warn", "kill", "mark_failed"]


@dataclass
class KillDecision:
    should_kill: bool
    reason: str = ""
    # When True, mark run FAILED (logistics) without calling terminate_pod.
    mark_failed: bool = False
    # Soft miss: log only; do not kill or fail.
    warn_only: bool = False

    @property
    def action(self) -> RemoteAction:
        if self.should_kill:
            return "kill"
        if self.mark_failed:
            return "mark_failed"
        if self.warn_only:
            return "warn"
        return "none"


def consecutive_miss_threshold() -> int:
    return max(1, int(float(os.environ.get("WATCHDOG_CONSECUTIVE_MISSES", "3"))))


def accrued_from_rate(
    hourly_rate: float,
    started_at: datetime,
    *,
    now: datetime | None = None,
) -> float:
    now = now or datetime.now(timezone.utc)
    start = started_at
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    hours = max(0.0, (now - start).total_seconds() / 3600.0)
    return round(float(hourly_rate) * hours, 6)


def classify_remote_pod_status(pod: dict[str, Any] | None) -> str:
    """Normalize RunPod get_pod / list_pods shape → RUNNING|EXITED|TERMINATED|MISSING|OTHER."""
    if not pod:
        return "MISSING"
    desired = str(pod.get("desiredStatus") or pod.get("desired_status") or "").upper()
    # Some payloads nest status under runtime / machine.
    if not desired:
        desired = str(pod.get("status") or "").upper()
    if desired in {"RUNNING", "EXITED", "TERMINATED", "DEAD", "STOPPED"}:
        if desired == "DEAD":
            return "TERMINATED"
        if desired == "STOPPED":
            return "EXITED"
        return desired
    # Runtime present without desiredStatus usually means the machine is up.
    if pod.get("runtime") and desired in {"", "UNKNOWN"}:
        return "RUNNING"
    if desired:
        return "OTHER"
    return "MISSING"



def decide_stuck_provisioning(
    *,
    status: str,
    pod_id: str | None,
    age_sec: float | None,
    stuck_sec: float | None = None,
) -> KillDecision:
    """Fail runs left PROVISIONING with no pod_id past the stuck threshold."""
    limit = float(STUCK_PROVISIONING_SEC if stuck_sec is None else stuck_sec)
    if status != RunStatus.PROVISIONING.value:
        return KillDecision(False)
    if pod_id:
        return KillDecision(False)
    if age_sec is None or age_sec < limit:
        return KillDecision(False)
    return KillDecision(
        False,
        reason=(
            f"Stuck PROVISIONING {age_sec/60:.1f}min with null pod_id "
            f"(launch never allocated a RunPod). Cleared for retry."
        ),
        mark_failed=True,
    )


def decide_stale_heartbeat(
    *,
    heartbeat_age_sec: float | None,
    consecutive_stale_misses: int,
    pod_remote_status: str | None,
    stale_sec: float = STALE_HEARTBEAT_SEC,
    miss_threshold: int | None = None,
) -> KillDecision:
    """Policy for stale trainer heartbeats (unit-testable).

    A single stale observation never kills. After N consecutive misses, confirm
    via RunPod status: already gone → mark_failed; still RUNNING → warn/grace.
    """
    threshold = miss_threshold if miss_threshold is not None else consecutive_miss_threshold()
    if heartbeat_age_sec is None or heartbeat_age_sec <= stale_sec:
        return KillDecision(False)

    if consecutive_stale_misses < threshold:
        return KillDecision(
            False,
            f"stale heartbeat grace ({consecutive_stale_misses}/{threshold}, "
            f"age={heartbeat_age_sec:.0f}s)",
            warn_only=True,
        )

    status = (pod_remote_status or "MISSING").upper()
    if status in {"MISSING", "EXITED", "TERMINATED"}:
        return KillDecision(
            False,
            f"pod already {status.lower()} after {consecutive_stale_misses} stale "
            f"heartbeats (age={heartbeat_age_sec:.0f}s) — marking FAILED without kill",
            mark_failed=True,
        )
    if status == "RUNNING":
        return KillDecision(
            False,
            f"stale heartbeat ({heartbeat_age_sec:.0f}s) but RunPod says RUNNING — "
            "extending grace (no terminate)",
            warn_only=True,
        )
    # Unusual remote status after N misses: still prefer mark_failed over blind kill
    # when we cannot confirm a live billable machine.
    if status == "OTHER":
        return KillDecision(
            False,
            f"stale heartbeat + unclear pod status {status} — extending grace",
            warn_only=True,
        )
    return KillDecision(False, warn_only=True)


def decide_pod_absent(
    *,
    consecutive_absent: int,
    pod_remote_status: str | None,
    miss_threshold: int | None = None,
) -> KillDecision:
    """Policy when a DB pod is missing from list_pods (unit-testable)."""
    threshold = miss_threshold if miss_threshold is not None else consecutive_miss_threshold()
    if consecutive_absent < threshold:
        return KillDecision(
            False,
            f"list_pods absence grace ({consecutive_absent}/{threshold})",
            warn_only=True,
        )
    status = (pod_remote_status or "MISSING").upper()
    if status == "RUNNING":
        return KillDecision(
            False,
            "pod missing from list_pods but get_pod says RUNNING — not marking vanished",
            warn_only=True,
        )
    if status in {"MISSING", "EXITED", "TERMINATED"}:
        return KillDecision(
            False,
            f"pod confirmed {status.lower()} after {consecutive_absent} list_pods misses",
            mark_failed=True,
        )
    return KillDecision(
        False,
        f"pod absent + unclear status {status} — extending grace",
        warn_only=True,
    )


def decide_kill(
    *,
    run_id: str | None,
    run_known: bool,
    heartbeat_age_sec: float | None,
    runtime_min: float,
    max_runtime_min: float,
    accrued_usd: float,
    budget_usd: float,
    consecutive_stale_misses: int = 0,
    pod_remote_status: str | None = None,
    miss_threshold: int | None = None,
) -> KillDecision:
    """Pure kill policy (unit-testable).

    Budget / runtime / unknown RUN_ID still kill immediately.
    Stale heartbeats defer to decide_stale_heartbeat (N misses + RunPod confirm).
    """
    if not run_id:
        return KillDecision(True, "pod missing RUN_ID env")
    if not run_known:
        return KillDecision(True, f"unknown RUN_ID {run_id}")
    if runtime_min > max_runtime_min:
        return KillDecision(
            True,
            f"runtime {runtime_min:.1f}min > MAX_RUNTIME_MIN {max_runtime_min}",
        )
    if budget_usd > 0 and accrued_usd > budget_usd:
        return KillDecision(
            True,
            f"accrued ${accrued_usd:.4f} > budget_usd ${budget_usd:.4f}",
        )
    return decide_stale_heartbeat(
        heartbeat_age_sec=heartbeat_age_sec,
        consecutive_stale_misses=consecutive_stale_misses,
        pod_remote_status=pod_remote_status,
        miss_threshold=miss_threshold,
    )


def decide_render_pod_kill(
    *,
    runtime_min: float,
    max_runtime_min: float,
) -> KillDecision:
    """Kill policy for video-render pods attached to COMPLETE runs.

    Never flips training run status — only terminates the render pod (caller
    sets video_status=FAILED). No stale-heartbeat kill (render has no HB).
    """
    if runtime_min > max_runtime_min:
        return KillDecision(
            True,
            f"render runtime {runtime_min:.1f}min > max {max_runtime_min}",
        )
    return KillDecision(False)


def _env_from_pod(pod: dict[str, Any]) -> dict[str, str]:
    raw = pod.get("env") or {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    # Some GraphQL shapes return [{key,value}, ...]
    if isinstance(raw, list):
        out: dict[str, str] = {}
        for item in raw:
            if isinstance(item, dict) and "key" in item:
                out[str(item["key"])] = str(item.get("value", ""))
        return out
    return {}


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def finalize_cost(
    session: Session,
    run: Run,
    pod_row: Pod | None,
    *,
    reason: str,
) -> float:
    """Write CostLedger delta and update run.cost_usd / pod.accrued_usd."""
    now = datetime.now(timezone.utc)
    rate = float(run.hourly_rate or (pod_row.hourly_rate if pod_row else 0.0) or 0.0)
    started = _aware(pod_row.started_at if pod_row else run.created_at) or now
    accrued = accrued_from_rate(rate, started, now=now)
    prev = float(run.cost_usd or 0.0)
    # Ledger stores the final amount for this run as a single settle event if new,
    # or the delta if we already logged something.
    already = 0.0
    for row in session.exec(select(CostLedger).where(CostLedger.run_id == run.id)).all():
        already += float(row.amount_usd or 0.0)
    delta = round(max(0.0, accrued - already), 6)
    if delta > 0:
        session.add(
            CostLedger(
                run_id=run.id,
                pod_id=run.pod_id or (pod_row.id if pod_row else None),
                amount_usd=delta,
                reason=reason,
            )
        )
    run.cost_usd = accrued
    if pod_row:
        pod_row.accrued_usd = accrued
        if pod_row.terminated_at is None and reason.startswith("kill"):
            pod_row.terminated_at = now
            pod_row.status = "TERMINATED"
        elif reason.startswith(("complete", "fail", "abort")):
            pod_row.terminated_at = now
            pod_row.status = "TERMINATED"
    session.add(run)
    if pod_row:
        session.add(pod_row)
    return accrued


def _confirm_pod_status(pod_id: str) -> str:
    """Best-effort get_pod classification; MISSING on any failure."""
    try:
        from robolab.compute.runpod import get_pod

        remote = get_pod(pod_id)
        return classify_remote_pod_status(remote)
    except Exception as exc:
        logger.warning("get_pod(%s) confirm failed: %s", pod_id, exc)
        return "MISSING"


def _kill_run(
    session: Session,
    run: Run | None,
    pod_id: str,
    reason: str,
) -> None:
    from robolab.compute.runpod import terminate_pod

    logger.warning("watchdog terminating pod=%s reason=%s", pod_id, reason)
    try:
        terminate_pod(pod_id)
    except Exception as exc:
        logger.error("terminate_pod(%s) failed: %s", pod_id, exc)

    now = datetime.now(timezone.utc)
    pod_row = session.get(Pod, pod_id)
    if pod_row:
        pod_row.status = "TERMINATED"
        pod_row.terminated_at = now
        session.add(pod_row)

    if run is not None:
        run.status = RunStatus.KILLED_BY_WATCHDOG.value
        run.error = f"KILLED_BY_WATCHDOG: {reason}"
        run.updated_at = now
        finalize_cost(session, run, pod_row, reason=f"kill:{reason}")
        session.add(run)
        try:
            from robolab_api.failures import record_logistics_from_run

            record_logistics_from_run(session, run)
        except Exception:
            logger.exception("failed to record logistics failure for %s", run.id)
    session.commit()
    _stale_misses.pop(pod_id, None)
    _absent_misses.pop(pod_id, None)


def _kill_render_pod(
    session: Session,
    run: Run,
    pod_id: str,
    reason: str,
) -> None:
    """Terminate a video-render pod without flipping COMPLETE → KILLED."""
    from robolab.compute.runpod import terminate_pod

    logger.warning("watchdog terminating render pod=%s reason=%s", pod_id, reason)
    try:
        terminate_pod(pod_id)
    except Exception as exc:
        logger.error("terminate_pod(%s) failed: %s", pod_id, exc)

    now = datetime.now(timezone.utc)
    pod_row = session.get(Pod, pod_id)
    if pod_row:
        pod_row.status = "TERMINATED"
        pod_row.terminated_at = now
        session.add(pod_row)
    if run.video_status == "RENDERING":
        run.video_status = "FAILED"
        run.video_error = f"render pod killed: {reason}"
        run.updated_at = now
        session.add(run)
    session.commit()
    _stale_misses.pop(pod_id, None)
    _absent_misses.pop(pod_id, None)


def _mark_run_failed_pod_gone(
    session: Session,
    run: Run,
    pod_row: Pod,
    reason: str,
) -> None:
    """Pod already dead on RunPod — settle cost + FAILED logistics, no terminate."""
    now = datetime.now(timezone.utc)
    logger.warning(
        "watchdog marking FAILED (no terminate) pod=%s run=%s reason=%s",
        pod_row.id,
        run.id,
        reason,
    )
    pod_row.status = "TERMINATED"
    pod_row.terminated_at = now
    session.add(pod_row)
    if run.status in {
        RunStatus.PROVISIONING.value,
        RunStatus.RUNNING.value,
        RunStatus.QUEUED.value,
    }:
        run.status = RunStatus.FAILED.value
        run.error = reason
        run.updated_at = now
        finalize_cost(session, run, pod_row, reason="fail:pod_gone")
        session.add(run)
        try:
            from robolab_api.failures import record_logistics_from_run

            record_logistics_from_run(session, run)
        except Exception:
            logger.exception("failed to record logistics failure for %s", run.id)
    session.commit()
    _stale_misses.pop(pod_row.id, None)
    _absent_misses.pop(pod_row.id, None)



def sweep_stuck_provisioning(session: Session, *, now: datetime | None = None) -> list[str]:
    """Fail PROVISIONING runs with null pod_id past STUCK_PROVISIONING_SEC."""
    actions: list[str] = []
    now = now or datetime.now(timezone.utc)
    for run in session.exec(select(Run)).all():
        created = _aware(run.created_at) or _aware(run.updated_at)
        age = (now - created).total_seconds() if created else None
        decision = decide_stuck_provisioning(
            status=run.status,
            pod_id=run.pod_id,
            age_sec=age,
        )
        if not decision.mark_failed:
            continue
        run.status = RunStatus.FAILED.value
        run.error = decision.reason
        run.updated_at = now
        session.add(run)
        try:
            from robolab_api.failures import record_logistics_from_run

            record_logistics_from_run(session, run)
        except Exception:
            logger.exception("logistics record failed for stuck %s", run.id)
        session.commit()
        actions.append(f"{run.id}:stuck_provisioning_no_pod")
        logger.warning("watchdog %s", decision.reason)
    return actions


def sweep_once() -> list[str]:
    """One watchdog pass. Returns list of kill / fail / warn reasons applied."""
    actions: list[str] = []
    max_runtime = float(os.environ.get("MAX_RUNTIME_MIN", "120"))
    miss_threshold = consecutive_miss_threshold()
    api_key = (os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not api_key:
        with Session(engine) as session:
            actions.extend(sweep_stuck_provisioning(session))
        return actions

    try:
        from robolab.compute.runpod import list_pods
    except Exception as exc:
        logger.error("import list_pods failed: %s", exc)
        return actions

    try:
        live = list_pods()
    except Exception as exc:
        logger.error("list_pods failed: %s", exc)
        with Session(engine) as session:
            actions.extend(sweep_stuck_provisioning(session))
        return actions

    with Session(engine) as session:
        known_runs = {r.id: r for r in session.exec(select(Run)).all()}
        now = datetime.now(timezone.utc)
        actions.extend(sweep_stuck_provisioning(session, now=now))

        # Track which DB pods are still live
        live_ids = {str(p.get("id")) for p in live if p.get("id")}

        for remote in live:
            pod_id = str(remote.get("id") or "")
            if not pod_id:
                continue
            env = _env_from_pod(remote)
            # Prefer DB pod→run mapping. GraphQL list_pods often returns empty env
            # right after create; killing on missing RUN_ID was a false positive that
            # terminated the Secure smoke pod ~1min after launch.
            pod_row = session.get(Pod, pod_id)
            run_id = (env.get("RUN_ID") or "").strip() or None
            if not run_id and pod_row is not None:
                run_id = (pod_row.run_id or "").strip() or None
            run = known_runs.get(run_id) if run_id else None

            # Seen in list_pods → clear absence counter.
            _absent_misses.pop(pod_id, None)

            # Accrue cost for known runpod runs
            if run and run.compute == "runpod":
                env_job = (env.get("ROBOLAB_JOB") or "").strip().lower()
                is_render_pod = (
                    env_job == "render"
                    or (pod_row is not None and str(remote.get("name") or "").startswith("robolab-render-"))
                    or (
                        run.status == RunStatus.COMPLETE.value
                        and run.video_status == "RENDERING"
                        and pod_row is not None
                        and pod_row.id != run.pod_id
                    )
                )
                # Finished training runs must never have cost rewritten or status
                # flipped by a short-lived render pod.
                if run.status in TERMINAL_RUN_STATUSES:
                    if is_render_pod and run.video_status == "RENDERING":
                        started = (
                            _aware(pod_row.started_at if pod_row else None) or now
                        )
                        runtime_min = (now - started).total_seconds() / 60.0
                        render_max = float(
                            os.environ.get("ROBOLAB_RENDER_MAX_RUNTIME_MIN", "25")
                        )
                        decision = decide_render_pod_kill(
                            runtime_min=runtime_min,
                            max_runtime_min=render_max,
                        )
                        if pod_row and pod_row.status == "PROVISIONING":
                            desired = (remote.get("desiredStatus") or "").upper()
                            if desired == "RUNNING" or remote.get("runtime"):
                                pod_row.status = "RUNNING"
                                session.add(pod_row)
                        if decision.should_kill:
                            _kill_render_pod(session, run, pod_id, decision.reason)
                            actions.append(f"{pod_id}:render_kill:{decision.reason}")
                    continue

                rate = float(
                    run.hourly_rate
                    or remote.get("costPerHr")
                    or (pod_row.hourly_rate if pod_row else 0)
                    or 0
                )
                started = _aware(pod_row.started_at if pod_row else run.created_at) or now
                accrued = accrued_from_rate(rate, started, now=now)
                run.cost_usd = accrued
                run.hourly_rate = rate
                if pod_row:
                    pod_row.accrued_usd = accrued
                    pod_row.hourly_rate = rate
                    # Do NOT promote PROVISIONING→RUNNING from desiredStatus alone.
                    # That started the 10min stale-heartbeat clock before the entrypoint
                    # finished git clone / uv sync, and killed healthy smoke pods.
                    if pod_row.status == "PROVISIONING":
                        desired = (remote.get("desiredStatus") or "").upper()
                        if desired == "RUNNING" or remote.get("runtime"):
                            pod_row.status = "RUNNING"
                    session.add(pod_row)
                session.add(run)

                updated = _aware(run.updated_at)
                hb_age = (now - updated).total_seconds() if updated else None
                # Stale heartbeats only apply after the trainer has actually checked in
                # (status RUNNING via /heartbeat). While PROVISIONING, allow long image
                # pull + uv sync; still enforce MAX_RUNTIME_MIN / budget.
                if run.status == RunStatus.PROVISIONING.value:
                    hb_age_for_kill = None
                    _stale_misses.pop(pod_id, None)
                else:
                    hb_age_for_kill = hb_age
                    if hb_age is not None and hb_age > STALE_HEARTBEAT_SEC:
                        _stale_misses[pod_id] = _stale_misses.get(pod_id, 0) + 1
                    else:
                        _stale_misses.pop(pod_id, None)

                consecutive = _stale_misses.get(pod_id, 0)
                remote_status: str | None = None
                # Only hit get_pod when we are at/over the miss threshold for stale HB.
                if (
                    hb_age_for_kill is not None
                    and hb_age_for_kill > STALE_HEARTBEAT_SEC
                    and consecutive >= miss_threshold
                ):
                    remote_status = classify_remote_pod_status(remote)
                    # Prefer fresh get_pod if list row looks odd.
                    if remote_status != "RUNNING":
                        remote_status = _confirm_pod_status(pod_id)

                runtime_min = (now - started).total_seconds() / 60.0
                decision = decide_kill(
                    run_id=run_id,
                    run_known=run is not None,
                    heartbeat_age_sec=hb_age_for_kill,
                    runtime_min=runtime_min,
                    max_runtime_min=max_runtime,
                    accrued_usd=accrued,
                    budget_usd=float(run.budget_usd or 0.0),
                    consecutive_stale_misses=consecutive,
                    pod_remote_status=remote_status,
                    miss_threshold=miss_threshold,
                )
                # Healthy RUNNING after threshold → reset miss streak (extend grace).
                if (
                    decision.warn_only
                    and remote_status == "RUNNING"
                    and consecutive >= miss_threshold
                ):
                    _stale_misses.pop(pod_id, None)
                    logger.warning(
                        "watchdog grace: pod=%s run=%s %s",
                        pod_id,
                        run.id,
                        decision.reason,
                    )
                    actions.append(f"{pod_id}:warn:{decision.reason}")
            elif pod_row is not None and run_id:
                # DB knows the pod but run row missing / non-runpod — do not kill.
                decision = KillDecision(False)
            elif pod_row is not None and not run_id:
                # Our pod row exists; wait for env / run linkage — never orphan-kill.
                decision = KillDecision(False)
            else:
                # Live pod not in our Pod table. Only orphan-kill RoboLab-named pods
                # that advertise a RUN_ID we don't know. Never kill strangers or
                # env-less pods (GraphQL often omits env; that was a false positive).
                name = str(remote.get("name") or "")
                if run_id and not run and name.startswith("robolab-"):
                    decision = decide_kill(
                        run_id=run_id,
                        run_known=False,
                        heartbeat_age_sec=None,
                        runtime_min=0.0,
                        max_runtime_min=max_runtime,
                        accrued_usd=0.0,
                        budget_usd=0.0,
                    )
                else:
                    decision = KillDecision(False)

            if decision.should_kill:
                _kill_run(session, run, pod_id, decision.reason)
                actions.append(f"{pod_id}:{decision.reason}")
            elif decision.mark_failed and run is not None and pod_row is not None:
                _mark_run_failed_pod_gone(session, run, pod_row, decision.reason)
                actions.append(f"{pod_id}:mark_failed:{decision.reason}")
            elif decision.warn_only and decision.reason:
                # RUNNING-at-threshold path already appended; avoid duplicates.
                tag = f"{pod_id}:warn:{decision.reason}"
                if tag not in actions:
                    logger.info("watchdog warn pod=%s %s", pod_id, decision.reason)
                    actions.append(tag)

        # DB pods marked live but gone from list_pods — require N consecutive + confirm.
        for pod_row in session.exec(select(Pod).where(Pod.terminated_at.is_(None))).all():
            if pod_row.id in live_ids:
                continue
            _absent_misses[pod_row.id] = _absent_misses.get(pod_row.id, 0) + 1
            consecutive = _absent_misses[pod_row.id]
            remote_status: str | None = None
            if consecutive >= miss_threshold:
                remote_status = _confirm_pod_status(pod_row.id)
            decision = decide_pod_absent(
                consecutive_absent=consecutive,
                pod_remote_status=remote_status,
                miss_threshold=miss_threshold,
            )
            if decision.warn_only:
                logger.warning(
                    "watchdog: pod=%s absent from list_pods (%s)",
                    pod_row.id,
                    decision.reason,
                )
                if remote_status == "RUNNING":
                    # Flaky list — reset absence streak.
                    _absent_misses.pop(pod_row.id, None)
                actions.append(f"{pod_row.id}:warn:{decision.reason}")
                continue
            if not decision.mark_failed:
                continue

            run = session.get(Run, pod_row.run_id)
            if run and run.status in {
                RunStatus.PROVISIONING.value,
                RunStatus.RUNNING.value,
                RunStatus.QUEUED.value,
            }:
                reason = (
                    run.error
                    or "Pod disappeared from RunPod before the trainer finished "
                    "(entrypoint crash, self-terminate, or machine reclaim). "
                    f"Confirmed after {consecutive} consecutive list_pods misses "
                    f"(status={remote_status})."
                )
                _mark_run_failed_pod_gone(session, run, pod_row, reason)
                actions.append(f"{pod_row.id}:pod_vanished")
            else:
                # Terminal training status — close pod row; fail video if still RENDERING.
                if (
                    run is not None
                    and run.status == RunStatus.COMPLETE.value
                    and run.video_status == "RENDERING"
                    and pod_row.id != run.pod_id
                ):
                    run.video_status = "FAILED"
                    run.video_error = (
                        "Render pod disappeared before video-complete "
                        f"(status={remote_status})."
                    )
                    run.updated_at = now
                    session.add(run)
                    actions.append(f"{pod_row.id}:render_pod_vanished")
                pod_row.status = "TERMINATED"
                pod_row.terminated_at = now
                session.add(pod_row)
                session.commit()
                _absent_misses.pop(pod_row.id, None)
        session.commit()
    return actions


async def watchdog_loop(stop: asyncio.Event) -> None:
    logger.info(
        "watchdog started (interval=%ss, consecutive_misses=%s)",
        WATCHDOG_INTERVAL_SEC,
        consecutive_miss_threshold(),
    )
    # Startup orphan sweep
    try:
        actions = await asyncio.to_thread(sweep_once)
        if actions:
            logger.warning("startup sweep actions: %s", actions)
    except Exception:
        logger.exception("startup sweep failed")

    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=WATCHDOG_INTERVAL_SEC)
        except TimeoutError:
            pass
        if stop.is_set():
            break
        try:
            actions = await asyncio.to_thread(sweep_once)
            if actions:
                logger.warning("watchdog actions: %s", actions)
        except Exception:
            logger.exception("watchdog sweep failed")
    logger.info("watchdog stopped")
