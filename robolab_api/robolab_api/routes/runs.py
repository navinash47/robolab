"""Run routes: create, list, heartbeat, complete, SSE."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.sse import EventSourceResponse
from pydantic import BaseModel
from sqlmodel import select

from robolab.compute.local import launch_local
from robolab.core.run import RunConfig, RunStatus
from robolab_api.db import Run, SessionDep

router = APIRouter(prefix="/api/runs", tags=["runs"])


class HeartbeatBody(BaseModel):
    step: int
    total: int
    mean_return: float | None = None
    eta: float | None = None
    wandb_url: str | None = None
    status: str | None = None


class CompleteBody(BaseModel):
    status: str = "COMPLETE"
    wandb_url: str | None = None
    mean_return: float | None = None
    step: int | None = None
    checkpoint: str | None = None


class FailBody(BaseModel):
    error: str
    traceback: str | None = None


class ProgressEvent(BaseModel):
    id: str
    status: str
    step: int
    total: int
    mean_return: float | None = None
    wandb_url: str | None = None
    progress: float = 0.0
    error: str | None = None


def _run_to_dict(run: Run) -> dict[str, Any]:
    total = run.total_steps or 1
    progress = min(1.0, float(run.step) / float(total)) if total else 0.0
    return {
        "id": run.id,
        "name": run.name,
        "status": run.status,
        "sim": run.sim,
        "task": run.task,
        "robot": run.robot,
        "arch": run.arch,
        "compute": run.compute,
        "step": run.step,
        "total_steps": run.total_steps,
        "mean_return": run.mean_return,
        "wandb_url": run.wandb_url,
        "error": run.error,
        "progress": progress,
        "pid": run.pid,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
        "config": json.loads(run.config_json) if run.config_json else {},
    }


@router.get("")
def list_runs(session: SessionDep) -> dict:
    rows = session.exec(select(Run).order_by(Run.created_at.desc())).all()
    return {"runs": [_run_to_dict(r) for r in rows], "count": len(rows)}


@router.get("/{run_id}")
def get_run(run_id: str, session: SessionDep) -> dict:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    return _run_to_dict(run)


@router.post("")
def create_run(body: RunConfig, session: SessionDep) -> dict:
    if body.compute != "local":
        raise HTTPException(
            400,
            "Phase 1 only supports compute='local'. RunPod lands in Phase 3.",
        )
    run_id = uuid.uuid4().hex[:12]
    name = body.name or f"{body.task}-{body.arch}-{body.compute}"
    total = int(body.trainer.timesteps)
    row = Run(
        id=run_id,
        name=name,
        status=RunStatus.QUEUED.value,
        sim=body.sim,
        task=body.task,
        robot=body.robot,
        arch=body.arch,
        compute=body.compute,
        config_json=body.model_dump_json(),
        step=0,
        total_steps=total,
    )
    session.add(row)
    session.commit()
    session.refresh(row)

    try:
        proc = launch_local(body, run_id=run_id, backend_url="http://127.0.0.1:8000")
        row.status = RunStatus.RUNNING.value
        row.pid = proc.pid
        row.updated_at = datetime.now(timezone.utc)
        session.add(row)
        session.commit()
        session.refresh(row)
    except Exception as exc:
        row.status = RunStatus.FAILED.value
        row.error = str(exc)
        row.updated_at = datetime.now(timezone.utc)
        session.add(row)
        session.commit()
        raise HTTPException(500, f"Failed to launch local trainer: {exc}") from exc

    return _run_to_dict(row)


@router.post("/{run_id}/heartbeat")
def heartbeat(run_id: str, body: HeartbeatBody, session: SessionDep) -> dict:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    run.step = body.step
    run.total_steps = body.total or run.total_steps
    if body.mean_return is not None:
        run.mean_return = body.mean_return
    if body.wandb_url:
        run.wandb_url = body.wandb_url
    if body.status:
        run.status = body.status
    elif run.status == RunStatus.QUEUED.value:
        run.status = RunStatus.RUNNING.value
    run.updated_at = datetime.now(timezone.utc)
    session.add(run)
    session.commit()
    return {"ok": True}


@router.post("/{run_id}/complete")
def complete(run_id: str, body: CompleteBody, session: SessionDep) -> dict:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    run.status = RunStatus.COMPLETE.value
    if body.wandb_url:
        run.wandb_url = body.wandb_url
    if body.mean_return is not None:
        run.mean_return = body.mean_return
    if body.step is not None:
        run.step = body.step
        run.total_steps = max(run.total_steps, body.step)
    run.updated_at = datetime.now(timezone.utc)
    session.add(run)
    session.commit()
    return {"ok": True}


@router.post("/{run_id}/fail")
def fail(run_id: str, body: FailBody, session: SessionDep) -> dict:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    run.status = RunStatus.FAILED.value
    run.error = body.error
    run.updated_at = datetime.now(timezone.utc)
    session.add(run)
    session.commit()
    return {"ok": True}


@router.get("/{run_id}/events", response_class=EventSourceResponse)
async def run_events(run_id: str) -> AsyncIterable[ProgressEvent]:
    """SSE progress stream for the dashboard."""
    from sqlmodel import Session

    from robolab_api.db import engine

    last_payload: str | None = None
    idle_rounds = 0
    while True:
        with Session(engine) as session:
            run = session.get(Run, run_id)
            if not run:
                yield ProgressEvent(
                    id=run_id,
                    status="FAILED",
                    step=0,
                    total=0,
                    error="not found",
                )
                return
            total = run.total_steps or 1
            progress = min(1.0, float(run.step) / float(total)) if total else 0.0
            event = ProgressEvent(
                id=run.id,
                status=run.status,
                step=run.step,
                total=run.total_steps,
                mean_return=run.mean_return,
                wandb_url=run.wandb_url,
                progress=progress,
                error=run.error,
            )
        payload = event.model_dump_json()
        if payload != last_payload:
            yield event
            last_payload = payload
            idle_rounds = 0
        else:
            idle_rounds += 1
            if idle_rounds % 5 == 0:
                yield event

        if event.status in {
            RunStatus.COMPLETE.value,
            RunStatus.FAILED.value,
            RunStatus.KILLED_BY_WATCHDOG.value,
        }:
            return
        await asyncio.sleep(1.0)
