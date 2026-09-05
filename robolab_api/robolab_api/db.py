"""SQLModel models and SQLite engine."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends
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
    wandb_url: Optional[str] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
