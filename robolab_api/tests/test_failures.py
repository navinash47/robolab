"""Failure Resolution helpers."""

from __future__ import annotations

from sqlmodel import Session, SQLModel, create_engine, select

from robolab_api.db import FailureRecord, Run
from robolab_api.failures import (
    CATEGORY_EXPERIMENT,
    CATEGORY_LOGISTICS,
    list_failures,
    record_failure,
    record_logistics_from_run,
    seed_known_failures,
    suggest_fix_for_error,
)


def test_suggest_fix_capacity():
    fix = suggest_fix_for_error("There are no longer any instances available")
    assert fix is not None
    assert "SECURE" in fix


def test_record_and_list(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        run = Run(
            id="abc123",
            name="test-run",
            status="FAILED",
            error="RUNPOD_API_KEY is missing",
        )
        session.add(run)
        session.commit()
        record_logistics_from_run(session, run)
        session.commit()
        rows = list_failures(session, category=CATEGORY_LOGISTICS)
        assert len(rows) == 1
        assert rows[0].run_id == "abc123"
        assert rows[0].source == "auto"
        # Dedupe auto
        record_logistics_from_run(session, run)
        session.commit()
        assert len(list_failures(session, category=CATEGORY_LOGISTICS)) == 1

        record_failure(
            session,
            category=CATEGORY_EXPERIMENT,
            title="bad wall follow",
            reason="robot spins; reward inverted",
            run_id="abc123",
            source="manual",
            dedupe_auto=False,
        )
        session.commit()
        assert len(list_failures(session, category=CATEGORY_EXPERIMENT)) == 1


def test_seed_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't2.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        n1 = seed_known_failures(session)
        assert n1 >= 1
        n2 = seed_known_failures(session)
        assert n2 == 0
        seeds = session.exec(
            select(FailureRecord).where(FailureRecord.source == "seed")
        ).all()
        assert any("502" in (r.title or "") for r in seeds)
        assert any(r.run_id == "15131a02b3f3" for r in seeds)
