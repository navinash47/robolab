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
from robolab_api.routes import compare_router, runs_router
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/archs")
def list_archs_endpoint() -> dict:
    from robolab.core.arch import list_archs

    return {"archs": list_archs()}


@app.get("/api/experiments")
def list_experiments(session: SessionDep) -> dict:
    """Phase 0 compatibility: now backed by Run rows."""
    rows = session.exec(select(Run).order_by(Run.created_at.desc())).all()
    return {
        "experiments": [
            {
                "id": r.id,
                "name": r.name,
                "sim": r.sim,
                "arch": r.arch,
                "status": r.status,
                "progress": min(1.0, r.step / r.total_steps) if r.total_steps else 0.0,
                "mean_return": r.mean_return,
                "param_count": r.param_count,
                "wandb_url": r.wandb_url,
                "task": r.task,
                "compute": r.compute,
                "step": r.step,
                "total_steps": r.total_steps,
                "pod_id": r.pod_id,
                "gpu_type": r.gpu_type,
                "hourly_rate": r.hourly_rate,
                "cost_usd": r.cost_usd,
                "budget_usd": r.budget_usd,
                "video_status": r.video_status,
                "video_url": r.video_url,
                "video_error": r.video_error,
                "checkpoint_artifact": r.checkpoint_artifact,
                "checkpoints": _run_checkpoints(r),
            }
            for r in rows
        ],
        "count": len(rows),
    }


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
