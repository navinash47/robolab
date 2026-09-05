"""Run routes: create, list, heartbeat, complete, SSE."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.sse import EventSourceResponse
from pydantic import BaseModel
from sqlmodel import select

from robolab.compute.local import launch_local, launch_render, repo_root
from robolab.compute.runpod import RunPodConfigError, launch_runpod
from robolab.core.run import RunConfig, RunStatus
from robolab_api.budget import refuse_if_over_cap
from robolab_api.db import Pod, Run, SessionDep

logger = logging.getLogger("robolab.runs")

router = APIRouter(prefix="/api/runs", tags=["runs"])


class HeartbeatBody(BaseModel):
    step: int
    total: int
    mean_return: float | None = None
    eta: float | None = None
    wandb_url: str | None = None
    status: str | None = None
    param_count: int | None = None


class CompleteBody(BaseModel):
    status: str = "COMPLETE"
    wandb_url: str | None = None
    mean_return: float | None = None
    step: int | None = None
    checkpoint: str | None = None
    checkpoint_artifact: str | None = None
    param_count: int | None = None


class FailBody(BaseModel):
    error: str
    traceback: str | None = None


class VideoCompleteBody(BaseModel):
    status: str = "READY"
    video_path: str | None = None
    video_url: str | None = None
    wandb_url: str | None = None
    mujoco_gl: str | None = None
    checkpoint: str | None = None


class VideoFailBody(BaseModel):
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
    pod_id: str | None = None
    hourly_rate: float | None = None
    cost_usd: float | None = None
    gpu_type: str | None = None
    video_status: str | None = None
    video_url: str | None = None
    video_error: str | None = None


def _checkpoints_for(run: Run) -> list[dict[str, Any]]:
    """Visible checkpoints for the run page: local zip and/or W&B artifact ref."""
    items: list[dict[str, Any]] = []
    local = repo_root() / "checkpoints" / run.id / "policy.zip"
    if local.is_file():
        items.append(
            {
                "kind": "local",
                "name": "policy.zip",
                "path": str(local),
                "exists": True,
            }
        )
    if run.checkpoint_artifact:
        art = run.checkpoint_artifact
        items.append(
            {
                "kind": "wandb_artifact",
                "name": art,
                "path": None,
                "exists": True,
            }
        )
    return items


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
        "param_count": run.param_count,
        "wandb_url": run.wandb_url,
        "error": run.error,
        "progress": progress,
        "pid": run.pid,
        "pod_id": run.pod_id,
        "gpu_type": run.gpu_type,
        "hourly_rate": run.hourly_rate,
        "cost_usd": run.cost_usd,
        "budget_usd": run.budget_usd,
        "video_status": run.video_status,
        "video_path": run.video_path,
        "video_url": run.video_url,
        "video_error": run.video_error,
        "checkpoint_artifact": run.checkpoint_artifact,
        "checkpoints": _checkpoints_for(run),
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
        "config": json.loads(run.config_json) if run.config_json else {},
    }


def _settle_runpod_cost(session, run: Run, reason: str) -> None:
    if run.compute != "runpod":
        return
    from robolab_api.watchdog import finalize_cost

    pod_row = session.get(Pod, run.pod_id) if run.pod_id else None
    finalize_cost(session, run, pod_row, reason=reason)


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
    if body.compute not in {"local", "runpod"}:
        raise HTTPException(400, f"Unsupported compute={body.compute!r}")

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
        budget_usd=float(body.budget_usd or 0.0),
        gpu_type=body.gpu_type,
    )
    session.add(row)
    session.commit()
    session.refresh(row)

    if body.compute == "local":
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

    # --- runpod ---
    try:
        refuse_if_over_cap(session)
    except RuntimeError as exc:
        row.status = RunStatus.FAILED.value
        row.error = str(exc)
        row.updated_at = datetime.now(timezone.utc)
        session.add(row)
        session.commit()
        raise HTTPException(400, str(exc)) from exc

    # Prefer CUDA on the pod even if the client left device=cpu
    if body.trainer.device == "cpu":
        body = body.model_copy(
            update={"trainer": body.trainer.model_copy(update={"device": "cuda"})}
        )
        row.config_json = body.model_dump_json()

    row.status = RunStatus.PROVISIONING.value
    row.updated_at = datetime.now(timezone.utc)
    session.add(row)
    session.commit()

    public = (os.environ.get("BACKEND_PUBLIC_URL") or "").strip()
    try:
        result = launch_runpod(body, run_id, backend_public_url=public or None)
    except RunPodConfigError as exc:
        row.status = RunStatus.FAILED.value
        row.error = str(exc)
        row.updated_at = datetime.now(timezone.utc)
        session.add(row)
        session.commit()
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("runpod launch failed")
        row.status = RunStatus.FAILED.value
        row.error = str(exc)
        row.updated_at = datetime.now(timezone.utc)
        session.add(row)
        session.commit()
        raise HTTPException(500, f"Failed to launch RunPod: {exc}") from exc

    pod_id = str(result["pod_id"])
    hourly = float(result["hourly_rate"])
    gpu_type = str(result["gpu_type"])
    now = datetime.now(timezone.utc)

    row.pod_id = pod_id
    row.hourly_rate = hourly
    row.gpu_type = gpu_type
    row.cost_usd = 0.0
    row.status = RunStatus.PROVISIONING.value
    row.updated_at = now
    session.add(row)
    session.add(
        Pod(
            id=pod_id,
            run_id=run_id,
            gpu_type=gpu_type,
            hourly_rate=hourly,
            status="PROVISIONING",
            accrued_usd=0.0,
            started_at=now,
        )
    )
    session.commit()
    session.refresh(row)
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
    if body.param_count is not None:
        run.param_count = body.param_count
    if body.status:
        run.status = body.status
    elif run.status in {RunStatus.QUEUED.value, RunStatus.PROVISIONING.value}:
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
    if body.param_count is not None:
        run.param_count = body.param_count
    if body.checkpoint_artifact:
        run.checkpoint_artifact = body.checkpoint_artifact
    elif body.checkpoint:
        # Persist local path when trainer did not (or could not) name a W&B artifact.
        run.checkpoint_artifact = body.checkpoint
    if body.step is not None:
        run.step = body.step
        run.total_steps = max(run.total_steps, body.step)
    # Heal common Phase 1–3 gap: COMPLETE + on-disk zip but null DB field.
    if not run.checkpoint_artifact:
        local = repo_root() / "checkpoints" / run_id / "policy.zip"
        if local.is_file():
            run.checkpoint_artifact = f"policy-{run_id}"
    run.updated_at = datetime.now(timezone.utc)
    _settle_runpod_cost(session, run, reason="complete")
    session.add(run)
    session.commit()
    return {"ok": True}


@router.post("/{run_id}/fail")
def fail(run_id: str, body: FailBody, session: SessionDep) -> dict:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    # Keep a more specific error if one is already recorded (entrypoint fallback
    # used to overwrite trainer/W&B detail with a generic exit message).
    incoming = (body.error or "").strip()
    existing = (run.error or "").strip()
    if not existing or (incoming and len(incoming) >= len(existing)):
        run.error = incoming or existing or "failed"
    run.status = RunStatus.FAILED.value
    run.updated_at = datetime.now(timezone.utc)
    _settle_runpod_cost(session, run, reason="fail")
    session.add(run)
    session.commit()
    return {"ok": True}


@router.post("/{run_id}/render")
def start_render(run_id: str, session: SessionDep) -> dict:
    """Queue headless playback recording for a COMPLETE run (Phase 4)."""
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    if run.status != RunStatus.COMPLETE.value:
        raise HTTPException(
            400,
            f"Render requires COMPLETE status (got {run.status})",
        )
    if not run.wandb_url:
        raise HTTPException(400, "Run has no wandb_url — cannot attach playback video")
    if run.video_status == "RENDERING":
        return _run_to_dict(run)

    config_path = repo_root() / "runs" / run_id / "config.yaml"
    if not config_path.is_file():
        raise HTTPException(400, f"Missing run config at {config_path}")

    ckpt = repo_root() / "checkpoints" / run_id / "policy.zip"
    ckpt_arg = ckpt if ckpt.is_file() else None
    if ckpt_arg is None and not run.checkpoint_artifact:
        # Still try — render_video may find an artifact on W&B from a later upload
        pass

    try:
        proc = launch_render(
            run_id=run_id,
            config_path=config_path,
            wandb_url=run.wandb_url,
            checkpoint=ckpt_arg,
            backend_url="http://127.0.0.1:8000",
        )
    except Exception as exc:
        run.video_status = "FAILED"
        run.video_error = str(exc)
        run.updated_at = datetime.now(timezone.utc)
        session.add(run)
        session.commit()
        raise HTTPException(500, f"Failed to launch render: {exc}") from exc

    run.video_status = "RENDERING"
    run.video_error = None
    run.pid = proc.pid  # reuse pid field for render process while COMPLETE
    run.updated_at = datetime.now(timezone.utc)
    session.add(run)
    session.commit()
    session.refresh(run)
    return _run_to_dict(run)


@router.post("/{run_id}/video-complete")
def video_complete(run_id: str, body: VideoCompleteBody, session: SessionDep) -> dict:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    run.video_status = "READY"
    if body.video_path:
        run.video_path = body.video_path
    run.video_url = body.video_url or f"/api/runs/{run_id}/video"
    run.video_error = None
    if body.wandb_url and not run.wandb_url:
        run.wandb_url = body.wandb_url
    if body.checkpoint and not run.checkpoint_artifact:
        # Render used a local zip; record artifact name convention for the UI.
        run.checkpoint_artifact = f"policy-{run_id}"
    run.updated_at = datetime.now(timezone.utc)
    session.add(run)
    session.commit()
    return {"ok": True}


@router.post("/{run_id}/video-fail")
def video_fail(run_id: str, body: VideoFailBody, session: SessionDep) -> dict:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    run.video_status = "FAILED"
    run.video_error = body.error
    run.updated_at = datetime.now(timezone.utc)
    session.add(run)
    session.commit()
    return {"ok": True}


@router.get("/{run_id}/video")
def get_video(run_id: str, session: SessionDep):
    from fastapi.responses import FileResponse

    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    if run.video_status != "READY" or not run.video_path:
        raise HTTPException(404, "Video not ready")
    path = Path(run.video_path)
    if not path.is_file():
        raise HTTPException(404, f"Video file missing: {path}")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{run_id}-playback.mp4",
    )


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
                pod_id=run.pod_id,
                hourly_rate=run.hourly_rate,
                cost_usd=run.cost_usd,
                gpu_type=run.gpu_type,
                video_status=run.video_status,
                video_url=run.video_url,
                video_error=run.video_error,
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
