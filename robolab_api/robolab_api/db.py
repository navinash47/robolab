"""SQLModel models and SQLite engine for Phase 0."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlmodel import Field, Session, SQLModel, create_engine

# Persist DB at repo root (parent of robolab_api package project).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _REPO_ROOT / "robolab.db"
sqlite_url = f"sqlite:///{_DB_PATH}"
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


class Experiment(SQLModel, table=True):
    """Experiment row. Phase 0 keeps this empty (no seed data)."""

    id: int | None = Field(default=None, primary_key=True)
    name: str


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
