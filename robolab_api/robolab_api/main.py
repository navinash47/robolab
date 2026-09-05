"""RoboLab FastAPI entrypoint (Phase 0)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import select

from robolab_api.budget import get_budget
from robolab_api.db import Experiment, SessionDep, create_db_and_tables


@asynccontextmanager
async def lifespan(_app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(title="RoboLab API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/experiments")
def list_experiments(session: SessionDep) -> dict:
    rows = session.exec(select(Experiment)).all()
    return {
        "experiments": [{"id": r.id, "name": r.name} for r in rows],
        "count": len(rows),
    }


@app.get("/api/budget")
def budget() -> dict[str, float]:
    return get_budget()


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
