"""Backend cost / orphan / stale-heartbeat watchdog.

Every ~60s: terminate pods that violate safety rules and label runs
KILLED_BY_WATCHDOG. Orphan sweep on startup.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from robolab.core.run import RunStatus
from robolab_api.db import CostLedger, Pod, Run, engine

logger = logging.getLogger("robolab.watchdog")

STALE_HEARTBEAT_SEC = 10 * 60
WATCHDOG_INTERVAL_SEC = 60


@dataclass
class KillDecision:
    should_kill: bool
    reason: str = ""


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


def decide_kill(
    *,
    run_id: str | None,
    run_known: bool,
    heartbeat_age_sec: float | None,
    runtime_min: float,
    max_runtime_min: float,
    accrued_usd: float,
    budget_usd: float,
) -> KillDecision:
    """Pure kill policy (unit-testable)."""
    if not run_id:
        return KillDecision(True, "pod missing RUN_ID env")
    if not run_known:
        return KillDecision(True, f"unknown RUN_ID {run_id}")
    if heartbeat_age_sec is not None and heartbeat_age_sec > STALE_HEARTBEAT_SEC:
        return KillDecision(
            True,
            f"stale heartbeat ({heartbeat_age_sec:.0f}s > {STALE_HEARTBEAT_SEC}s)",
        )
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
        elif reason.startswith("complete") or reason.startswith("fail"):
            pod_row.terminated_at = now
            pod_row.status = "TERMINATED"
    session.add(run)
    if pod_row:
        session.add(pod_row)
    return accrued


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
    session.commit()


def sweep_once() -> list[str]:
    """One watchdog pass. Returns list of kill reasons applied."""
    actions: list[str] = []
    max_runtime = float(os.environ.get("MAX_RUNTIME_MIN", "120"))
    api_key = (os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not api_key:
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
        return actions

    with Session(engine) as session:
        known_runs = {r.id: r for r in session.exec(select(Run)).all()}
        now = datetime.now(timezone.utc)

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

            # Accrue cost for known runpod runs
            if run and run.compute == "runpod":
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
                else:
                    hb_age_for_kill = hb_age

                runtime_min = (now - started).total_seconds() / 60.0
                decision = decide_kill(
                    run_id=run_id,
                    run_known=run is not None,
                    heartbeat_age_sec=hb_age_for_kill,
                    runtime_min=runtime_min,
                    max_runtime_min=max_runtime,
                    accrued_usd=accrued,
                    budget_usd=float(run.budget_usd or 0.0),
                )
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

        # DB pods marked live but gone from API
        for pod_row in session.exec(select(Pod).where(Pod.terminated_at.is_(None))).all():
            if pod_row.id not in live_ids:
                pod_row.status = "TERMINATED"
                pod_row.terminated_at = now
                session.add(pod_row)
                run = session.get(Run, pod_row.run_id)
                if run and run.status in {
                    RunStatus.PROVISIONING.value,
                    RunStatus.RUNNING.value,
                    RunStatus.QUEUED.value,
                }:
                    # Pod vanished without complete/fail callback — surface clearly.
                    if run.status != RunStatus.KILLED_BY_WATCHDOG.value:
                        run.status = RunStatus.FAILED.value
                        run.error = (
                            run.error
                            or "Pod disappeared from RunPod before the trainer finished "
                            "(watchdog kill, entrypoint crash, or machine reclaim). "
                            "Check worker logs / image / git clone."
                        )
                        run.updated_at = now
                        finalize_cost(session, run, pod_row, reason="fail:pod_vanished")
                        session.add(run)
                        actions.append(f"{pod_row.id}:pod_vanished")
                session.commit()
        session.commit()
    return actions


async def watchdog_loop(stop: asyncio.Event) -> None:
    logger.info("watchdog started (interval=%ss)", WATCHDOG_INTERVAL_SEC)
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
