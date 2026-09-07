"""Run routes: create, list, heartbeat, complete, SSE."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import uuid
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.sse import EventSourceResponse
from pydantic import BaseModel
from sqlmodel import select

from robolab.compute.local import launch_local, launch_render, repo_root
from robolab.compute.runpod import (
    RunPodConfigError,
    launch_render_runpod,
    launch_runpod,
    needs_remote_render,
)
from robolab.core.run import TERMINAL_RUN_STATUSES, RunConfig, RunStatus
from robolab_api.budget import refuse_if_over_cap
from robolab_api.db import Pod, Run, SessionDep
from robolab_api.failures import record_logistics_from_run

logger = logging.getLogger("robolab.runs")

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _log_logistics_fail(session: SessionDep, run: Run) -> None:
    """Persist a logistics FailureRecord for a run that could not proceed."""
    try:
        record_logistics_from_run(session, run)
    except Exception:
        logger.exception("failed to record logistics failure for %s", run.id)


class HeartbeatBody(BaseModel):
    step: int
    total: int
    mean_return: float | None = None
    eta: float | None = None
    wandb_url: str | None = None
    status: str | None = None
    param_count: int | None = None
    obs_dim: int | None = None
    act_dim: int | None = None
    control_hz: float | None = None
    physics_substeps: int | None = None


class CompleteBody(BaseModel):
    status: str = "COMPLETE"
    wandb_url: str | None = None
    mean_return: float | None = None
    step: int | None = None
    checkpoint: str | None = None
    checkpoint_artifact: str | None = None
    param_count: int | None = None
    obs_dim: int | None = None
    act_dim: int | None = None
    control_hz: float | None = None
    physics_substeps: int | None = None


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


def _space_dims_for(run: Run) -> tuple[int | None, int | None]:
    """Prefer DB columns; fall back to TaskSpec / config so old MuJoCo rows match PyBullet."""
    obs = run.obs_dim
    act = run.act_dim
    if obs is not None and act is not None:
        return obs, act
    try:
        cfg = json.loads(run.config_json) if run.config_json else {}
    except Exception:
        cfg = {}
    task_name = run.task or cfg.get("task") or "wall_follow"
    try:
        import robolab.tasks  # noqa: F401
        from robolab.core.task import get_task

        spec = get_task(task_name)
        obs = obs if obs is not None else int(spec.observation.shape[0])
        act = act if act is not None else int(spec.action.shape[0])
    except Exception:
        if task_name == "wall_follow":
            obs = obs if obs is not None else 5
            act = act if act is not None else 2
    return obs, act


def _domain_timing_for(run: Run) -> tuple[float | None, int | None]:
    hz = run.control_hz
    sub = run.physics_substeps
    if hz is not None and sub is not None:
        return hz, sub
    try:
        cfg = json.loads(run.config_json) if run.config_json else {}
        domain = cfg.get("domain") or {}
        if hz is None and domain.get("control_hz") is not None:
            hz = float(domain["control_hz"])
        if sub is None and domain.get("physics_substeps") is not None:
            sub = int(domain["physics_substeps"])
    except Exception:
        pass
    return hz, sub


def _run_to_dict(run: Run) -> dict[str, Any]:
    total = run.total_steps or 1
    progress = min(1.0, float(run.step) / float(total)) if total else 0.0
    obs_dim, act_dim = _space_dims_for(run)
    control_hz, physics_substeps = _domain_timing_for(run)
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
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "control_hz": control_hz,
        "physics_substeps": physics_substeps,
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

    from robolab_api.routes.architectures import resolve_arch_for_run

    base_arch, arch_cfg, saved_name = resolve_arch_for_run(
        session, body.arch, dict(body.arch_cfg or {})
    )
    body = body.model_copy(update={"arch": base_arch, "arch_cfg": arch_cfg})

    # Tabular arch always uses Q-learning; neural FA Q only for mlp/kan/kaf/gpkan/fan.
    algo = str(body.trainer.algo or "ppo").lower()
    if base_arch == "avinash_wall":
        if algo != "q_learning":
            body = body.model_copy(
                update={"trainer": body.trainer.model_copy(update={"algo": "q_learning"})}
            )
    elif algo == "q_learning":
        from robolab.train.q_fa import QL_ARCHS

        if base_arch not in QL_ARCHS:
            raise HTTPException(
                400,
                f"trainer.algo=q_learning requires arch in {sorted(QL_ARCHS)} "
                f"or avinash_wall; got {base_arch!r}",
            )

    run_id = uuid.uuid4().hex[:12]
    arch_label = saved_name or body.arch
    name = body.name or f"{body.task}-{arch_label}-{body.compute}"
    if saved_name and not body.name:
        body = body.model_copy(update={"name": name})
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
        control_hz=float(body.domain.control_hz),
        physics_substeps=int(body.domain.physics_substeps),
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
            _log_logistics_fail(session, row)
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
        _log_logistics_fail(session, row)
        session.commit()
        raise HTTPException(400, str(exc)) from exc

    # Prefer CUDA on the pod even if the client left device=cpu
    if body.trainer.device == "cpu":
        body = body.model_copy(
            update={"trainer": body.trainer.model_copy(update={"device": "cuda"})}
        )
        row.config_json = body.model_dump_json()

    # Stay QUEUED until create_pod returns a pod_id. Committing PROVISIONING
    # before launch made the dashboard look "stuck" forever when create hung
    # or the API died mid-flight (null pod_id, no RunPod object).
    public = (os.environ.get("BACKEND_PUBLIC_URL") or "").strip()
    launch_timeout = float(os.environ.get("RUNPOD_LAUNCH_TIMEOUT_SEC", "180"))
    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(
                launch_runpod, body, run_id, backend_public_url=public or None
            )
            try:
                result = fut.result(timeout=launch_timeout)
            except FuturesTimeout as exc:
                raise TimeoutError(
                    f"RunPod launch timed out after {launch_timeout:.0f}s "
                    f"(capacity loop / API hang). Retry with Best available or "
                    f"another GPU; check Failure Resolution."
                ) from exc
    except RunPodConfigError as exc:
        row.status = RunStatus.FAILED.value
        row.error = str(exc)
        row.updated_at = datetime.now(timezone.utc)
        session.add(row)
        _log_logistics_fail(session, row)
        session.commit()
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("runpod launch failed")
        row.status = RunStatus.FAILED.value
        row.error = str(exc)
        row.updated_at = datetime.now(timezone.utc)
        session.add(row)
        _log_logistics_fail(session, row)
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
    # Never revive a terminal run (especially ABORTED) from a late worker heartbeat.
    if run.status in TERMINAL_RUN_STATUSES:
        return {"ok": True, "ignored": True, "status": run.status}
    run.step = body.step
    run.total_steps = body.total or run.total_steps
    if body.mean_return is not None:
        run.mean_return = body.mean_return
    if body.wandb_url:
        run.wandb_url = body.wandb_url
    if body.param_count is not None:
        run.param_count = body.param_count
    if body.obs_dim is not None:
        run.obs_dim = body.obs_dim
    if body.act_dim is not None:
        run.act_dim = body.act_dim
    if body.control_hz is not None:
        run.control_hz = body.control_hz
    if body.physics_substeps is not None:
        run.physics_substeps = body.physics_substeps
    if body.status:
        run.status = body.status
    elif run.status in {RunStatus.QUEUED.value, RunStatus.PROVISIONING.value}:
        run.status = RunStatus.RUNNING.value
    run.updated_at = datetime.now(timezone.utc)
    session.add(run)
    session.commit()
    return {"ok": True}


@router.post("/{run_id}/abort")
def abort_run(run_id: str, session: SessionDep) -> dict:
    """Abruptly stop a run: terminate RunPod pod / local pid, status=ABORTED."""
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    if run.status == RunStatus.ABORTED.value:
        return {"ok": True, "status": RunStatus.ABORTED.value, "already": True}
    if run.status in TERMINAL_RUN_STATUSES:
        raise HTTPException(
            400,
            f"Cannot abort terminal status {run.status}",
        )
    if run.status not in {
        RunStatus.QUEUED.value,
        RunStatus.PROVISIONING.value,
        RunStatus.RUNNING.value,
    }:
        raise HTTPException(400, f"Cannot abort status {run.status}")

    # Stop local trainer subprocess if any.
    if run.pid:
        try:
            os.kill(int(run.pid), signal.SIGTERM)
        except OSError as exc:
            logger.info("abort: local pid %s already gone (%s)", run.pid, exc)
        run.pid = None

    # Terminate RunPod pod if mapped (best-effort; may already be gone).
    if run.pod_id and run.compute == "runpod":
        try:
            from robolab.compute.runpod import terminate_pod

            terminate_pod(run.pod_id)
        except Exception as exc:
            logger.warning(
                "abort: terminate_pod(%s) for run %s failed: %s",
                run.pod_id,
                run.id,
                exc,
            )

    now = datetime.now(timezone.utc)
    run.status = RunStatus.ABORTED.value
    run.error = "aborted by user"
    run.updated_at = now
    _settle_runpod_cost(session, run, reason="abort:user")
    session.add(run)
    session.commit()
    session.refresh(run)
    return {"ok": True, "status": RunStatus.ABORTED.value, "run": _run_to_dict(run)}


@router.post("/{run_id}/complete")
def complete(run_id: str, body: CompleteBody, session: SessionDep) -> dict:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    if run.status == RunStatus.ABORTED.value:
        return {"ok": True, "ignored": True, "status": run.status}
    run.status = RunStatus.COMPLETE.value
    if body.wandb_url:
        run.wandb_url = body.wandb_url
    if body.mean_return is not None:
        run.mean_return = body.mean_return
    if body.param_count is not None:
        run.param_count = body.param_count
    if body.obs_dim is not None:
        run.obs_dim = body.obs_dim
    if body.act_dim is not None:
        run.act_dim = body.act_dim
    if body.control_hz is not None:
        run.control_hz = body.control_hz
    if body.physics_substeps is not None:
        run.physics_substeps = body.physics_substeps
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
    incoming = (body.error or "").strip()
    # COMPLETE + render worker posting /fail (old bootstrap) → video-fail only.
    if run.status == RunStatus.COMPLETE.value:
        run.video_status = "FAILED"
        run.video_error = incoming or run.video_error or "render worker failed"
        run.updated_at = datetime.now(timezone.utc)
        session.add(run)
        session.commit()
        return {"ok": True, "routed": "video-fail"}
    # Keep ABORTED sticky — late worker /fail must not overwrite user abort.
    if run.status == RunStatus.ABORTED.value:
        return {"ok": True, "ignored": True, "status": run.status}
    # Keep a more specific error if one is already recorded (entrypoint fallback
    # used to overwrite trainer/W&B detail with a generic exit message).
    existing = (run.error or "").strip()
    if not existing or (incoming and len(incoming) >= len(existing)):
        run.error = incoming or existing or "failed"
    run.status = RunStatus.FAILED.value
    run.updated_at = datetime.now(timezone.utc)
    _settle_runpod_cost(session, run, reason="fail")
    session.add(run)
    _log_logistics_fail(session, run)
    session.commit()
    return {"ok": True}


@router.post("/{run_id}/render")
def start_render(run_id: str, session: SessionDep) -> dict:
    """Queue headless playback recording for a COMPLETE run (Phase 4).

    mujoco/pybullet → local subprocess. genesis/isaac → short RunPod worker
    with the matching image (Mac cannot import genesis-world / Isaac).
    """
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

    cfg: RunConfig | None = None
    try:
        if run.config_json:
            cfg = RunConfig.model_validate_json(run.config_json)
    except Exception:
        cfg = None
    if cfg is None:
        try:
            import yaml

            raw = yaml.safe_load(config_path.read_text())
            cfg = RunConfig.model_validate(raw)
        except Exception as exc:
            raise HTTPException(400, f"Cannot parse run config: {exc}") from exc

    sim_name = (run.sim or cfg.sim or "").strip().lower()
    remote = needs_remote_render(sim_name)
    try:
        if remote:
            try:
                refuse_if_over_cap(session)
            except RuntimeError as exc:
                raise HTTPException(400, str(exc)) from exc
            # Ensure cfg.sim matches DB (remote image selection keys off cfg.sim).
            if (cfg.sim or "").strip().lower() != sim_name:
                cfg = cfg.model_copy(update={"sim": sim_name})
            public = (os.environ.get("BACKEND_PUBLIC_URL") or "").strip()
            result = launch_render_runpod(
                cfg,
                run_id=run_id,
                wandb_url=run.wandb_url,
                backend_public_url=public or None,
            )
            pod_id = str(result["pod_id"])
            session.add(
                Pod(
                    id=pod_id,
                    run_id=run.id,
                    gpu_type=str(result.get("gpu_type") or ""),
                    hourly_rate=float(result.get("hourly_rate") or 0.0),
                    status="PROVISIONING",
                )
            )
            # Keep training pod_id on the run row for history; render pod is in Pod table.
            run.video_status = "RENDERING"
            run.video_error = None
            run.updated_at = datetime.now(timezone.utc)
            session.add(run)
            session.commit()
            session.refresh(run)
            logger.info(
                "remote render pod=%s image=%s run=%s",
                pod_id,
                result.get("image"),
                run_id,
            )
            return _run_to_dict(run)

        proc = launch_render(
            run_id=run_id,
            config_path=config_path,
            wandb_url=run.wandb_url,
            checkpoint=ckpt_arg,
            backend_url="http://127.0.0.1:8000",
            sim=sim_name,
        )
    except RunPodConfigError as exc:
        run.video_status = "FAILED"
        run.video_error = str(exc)
        run.updated_at = datetime.now(timezone.utc)
        session.add(run)
        session.commit()
        raise HTTPException(400, str(exc)) from exc
    except HTTPException:
        raise
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


@router.post("/{run_id}/video-upload")
async def video_upload(
    run_id: str,
    session: SessionDep,
    file: UploadFile = File(...),
) -> dict:
    """Accept MP4 bytes from a RunPod render worker; store under videos/{run_id}/."""
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    if run.status != RunStatus.COMPLETE.value:
        raise HTTPException(400, f"Video upload requires COMPLETE run (got {run.status})")

    dest_dir = repo_root() / "videos" / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "playback.mp4"
    payload = await file.read()
    if not payload:
        raise HTTPException(400, "Empty video upload")
    dest.write_bytes(payload)

    run.video_path = str(dest)
    run.video_url = f"/api/runs/{run_id}/video"
    # Stay RENDERING until video-complete (wandb upload may still be in flight).
    if run.video_status != "READY":
        run.video_status = "RENDERING"
    run.video_error = None
    run.updated_at = datetime.now(timezone.utc)
    session.add(run)
    session.commit()
    return {"ok": True, "video_path": str(dest), "bytes": len(payload)}


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


@router.api_route("/{run_id}/video", methods=["GET", "HEAD"])
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
    # inline (not attachment) so the dashboard <video> can play via Vite proxy
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{run_id}-playback.mp4",
        content_disposition_type="inline",
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

        if event.status in TERMINAL_RUN_STATUSES:
            return
        await asyncio.sleep(1.0)
