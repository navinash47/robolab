"""SQLModel models and SQLite engine."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends
from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel, create_engine

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _REPO_ROOT / "robolab.db"
sqlite_url = f"sqlite:///{_DB_PATH}"
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


class Experiment(SQLModel, table=True):
    """Legacy Phase 0 table kept for migration compatibility."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str


class Run(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    status: str = "QUEUED"
    sim: str = "mujoco"
    task: str = "wall_follow"
    robot: str = "diffdrive_lidar"
    arch: str = "mlp"
    compute: str = "local"
    config_json: str = "{}"
    step: int = 0
    total_steps: int = 0
    mean_return: Optional[float] = None
    param_count: Optional[int] = None
    wandb_url: Optional[str] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    pod_id: Optional[str] = None
    gpu_type: Optional[str] = None
    hourly_rate: Optional[float] = None
    cost_usd: Optional[float] = None
    budget_usd: float = 0.0
    video_status: Optional[str] = None  # None | RENDERING | READY | FAILED
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    video_error: Optional[str] = None
    checkpoint_artifact: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Pod(SQLModel, table=True):
    """Live / historical RunPod mapping for a run."""

    id: str = Field(primary_key=True)  # RunPod pod id
    run_id: str = Field(index=True)
    gpu_type: str = ""
    hourly_rate: float = 0.0
    status: str = "PROVISIONING"
    accrued_usd: float = 0.0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    terminated_at: Optional[datetime] = None


class CostLedger(SQLModel, table=True):
    """Immutable cost events (month spend = sum of amount_usd)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    pod_id: Optional[str] = None
    amount_usd: float
    reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


_RUN_EXTRA_COLS: dict[str, str] = {
    "param_count": "INTEGER",
    "pod_id": "TEXT",
    "gpu_type": "TEXT",
    "hourly_rate": "REAL",
    "cost_usd": "REAL",
    "budget_usd": "REAL DEFAULT 0",
    "video_status": "TEXT",
    "video_path": "TEXT",
    "video_url": "TEXT",
    "video_error": "TEXT",
    "checkpoint_artifact": "TEXT",
}


def _migrate_columns() -> None:
    """SQLite create_all does not ADD columns — patch Run extras if missing."""
    with engine.connect() as conn:
        rows = conn.execute(text("PRAGMA table_info(run)")).fetchall()
        cols = {r[1] for r in rows}
        if not rows:
            return
        for name, decl in _RUN_EXTRA_COLS.items():
            if name not in cols:
                conn.execute(text(f"ALTER TABLE run ADD COLUMN {name} {decl}"))
        conn.commit()


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)
    _migrate_columns()


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
