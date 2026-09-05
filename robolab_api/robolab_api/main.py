"""RoboLab FastAPI entrypoint."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import select

from robolab_api.budget import get_budget
from robolab_api.db import Run, SessionDep, create_db_and_tables
from robolab_api.routes import compare_router, costs_router, failures_router, runs_router
from robolab_api.routes.failures import ensure_seeded
from robolab_api.wandb_status import get_wandb_status
from robolab_api.watchdog import watchdog_loop

logger = logging.getLogger("robolab.api")


def _run_checkpoints(run: Run) -> list[dict]:
    from pathlib import Path

    from robolab.compute.local import repo_root

    items: list[dict] = []
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
        items.append(
            {
                "kind": "wandb_artifact",
                "name": run.checkpoint_artifact,
                "path": None,
                "exists": True,
            }
        )
    return items


@asynccontextmanager
async def lifespan(_app: FastAPI):
    create_db_and_tables()
    try:
        ensure_seeded()
    except Exception:
        logger.exception("failure seed failed")
    stop = asyncio.Event()
    task = asyncio.create_task(watchdog_loop(stop), name="robolab-watchdog")
    logger.info("watchdog task started")
    try:
        yield
    finally:
        stop.set()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()


app = FastAPI(title="RoboLab API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs_router)
app.include_router(compare_router)
app.include_router(costs_router)
app.include_router(failures_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/archs")
def list_archs_endpoint() -> dict:
    from robolab.core.arch import list_archs

    return {"archs": list_archs()}


@app.get("/api/sims")
def list_sims_endpoint() -> dict:
    import robolab.sims.mujoco  # noqa: F401
    import robolab.sims.pybullet  # noqa: F401
    from robolab.core.sim import list_sims

    return {"sims": list_sims()}


@app.get("/api/experiments")
def list_experiments(session: SessionDep) -> dict:
    """Phase 0 compatibility: now backed by Run rows."""
    from robolab_api.routes.runs import _run_to_dict

    rows = session.exec(select(Run).order_by(Run.created_at.desc())).all()
    experiments = []
    for r in rows:
        d = _run_to_dict(r)
        # Experiments table uses a slightly flatter shape historically.
        experiments.append(
            {
                "id": d["id"],
                "name": d["name"],
                "sim": d["sim"],
                "arch": d["arch"],
                "status": d["status"],
                "progress": d["progress"],
                "mean_return": d["mean_return"],
                "param_count": d["param_count"],
                "obs_dim": d["obs_dim"],
                "act_dim": d["act_dim"],
                "control_hz": d["control_hz"],
                "physics_substeps": d["physics_substeps"],
                "wandb_url": d["wandb_url"],
                "task": d["task"],
                "compute": d["compute"],
                "step": d["step"],
                "total_steps": d["total_steps"],
                "pod_id": d["pod_id"],
                "gpu_type": d["gpu_type"],
                "hourly_rate": d["hourly_rate"],
                "cost_usd": d["cost_usd"],
                "budget_usd": d["budget_usd"],
                "video_status": d["video_status"],
                "video_url": d["video_url"],
                "video_error": d["video_error"],
                "checkpoint_artifact": d["checkpoint_artifact"],
                "checkpoints": d["checkpoints"],
                "created_at": d["created_at"],
            }
        )
    return {"experiments": experiments, "count": len(experiments)}


@app.get("/api/budget")
def budget() -> dict[str, float]:
    return get_budget()


@app.get("/api/wandb/status")
def wandb_status() -> dict:
    """Presence / shape / auth probe for WANDB_API_KEY (never returns the key)."""
    return get_wandb_status(probe=True)


def run() -> None:
    import uvicorn

    uvicorn.run(
        "robolab_api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
