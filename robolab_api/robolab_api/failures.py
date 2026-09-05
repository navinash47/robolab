"""Failure Resolution — persist logistics vs experiment failures for debugging."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from robolab_api.db import FailureRecord, Run, engine

logger = logging.getLogger("robolab.failures")

CATEGORY_LOGISTICS = "logistics"
CATEGORY_EXPERIMENT = "experiment"
VALID_CATEGORIES = frozenset({CATEGORY_LOGISTICS, CATEGORY_EXPERIMENT})
VALID_SOURCES = frozenset({"auto", "manual", "seed"})


def suggest_fix_for_error(error: str | None) -> str | None:
    """Cheap keyword → suggested fix map (no ML)."""
    e = (error or "").lower()
    if not e:
        return None
    if "runpod_api_key" in e or "api key is missing" in e:
        return "Set RUNPOD_API_KEY in .env and restart `make dev`."
    if "no longer any instances" in e or "no instances available" in e or "capacity" in e:
        return "Set RUNPOD_CLOUD_TYPE=SECURE (Community+EU-RO-1 volume often refuses); retry."
    if "budget" in e and ("exhaust" in e or "cap" in e or "refuse" in e):
        return "Raise BUDGET_USD_CAP or wait for next month; check /spend."
    if "backend_public_url" in e or "localhost" in e and "backend" in e:
        return "Expose :8000 via cloudflared/ngrok; set BACKEND_PUBLIC_URL (no localhost)."
    if "wandb" in e and ("401" in e or "unauthorized" in e or "auth" in e):
        return "Replace WANDB_API_KEY at https://wandb.ai/authorize; restart `make dev`."
    if "killed_by_watchdog" in e and "missing run_id" in e:
        return "Watchdog must map pod_id→run via DB when GraphQL env is empty (already fixed)."
    if "false positive" in e:
        return "Ensure watchdog uses Pod.run_id when list_pods env is empty."
    if "egl" in e or "eglquerystring" in e:
        return "Worker image needs EGL/MuJoCo GL deps; rebuild/push worker image."
    if "nvidia driver" in e or "cuda" in e and ("old" in e or "driver" in e):
        return "Pick a Secure GPU template whose driver matches the torch CUDA build."
    if "git" in e and ("dirty" in e or "push" in e or "remote" in e):
        return "Commit + push HEAD; set ROBOLAB_GIT_URL; launch only from a clean tree."
    if "502" in e or "connection refused" in e or "api down" in e:
        return "Keep `make dev` + tunnel alive; refresh BACKEND_PUBLIC_URL if tunnel rotated."
    if "pod disappeared" in e:
        return "Check entrypoint/git clone/uv sync logs; avoid false watchdog kills during provision."
    return None


def title_for_run_failure(run: Run | None, error: str | None) -> str:
    name = (run.name if run else None) or (run.id if run else "unknown")
    err = (error or "").strip()
    if not err:
        return f"{name}: FAILED"
    # Keep titles short for the table.
    short = err.split("\n", 1)[0]
    if len(short) > 120:
        short = short[:117] + "…"
    return f"{name}: {short}"


def _existing_auto_for_run(session: Session, run_id: str) -> FailureRecord | None:
    return session.exec(
        select(FailureRecord).where(
            FailureRecord.run_id == run_id,
            FailureRecord.source == "auto",
            FailureRecord.category == CATEGORY_LOGISTICS,
        )
    ).first()


def record_failure(
    session: Session,
    *,
    category: str,
    title: str,
    reason: str,
    run_id: str | None = None,
    suggested_fix: str | None = None,
    source: str = "auto",
    created_at: datetime | None = None,
    dedupe_auto: bool = True,
) -> FailureRecord | None:
    """Insert a failure row. Auto logistics for the same run_id updates in place."""
    if category not in VALID_CATEGORIES:
        raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}")
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}")

    reason = (reason or "").strip()
    title = (title or "").strip() or "Untitled failure"
    if suggested_fix is None:
        suggested_fix = suggest_fix_for_error(reason)

    if dedupe_auto and source == "auto" and run_id and category == CATEGORY_LOGISTICS:
        existing = _existing_auto_for_run(session, run_id)
        if existing is not None:
            # Prefer longer / newer detail.
            if reason and len(reason) >= len(existing.reason or ""):
                existing.reason = reason
                existing.title = title
                existing.suggested_fix = suggested_fix
                existing.created_at = created_at or datetime.now(timezone.utc)
                session.add(existing)
            return existing

    row = FailureRecord(
        created_at=created_at or datetime.now(timezone.utc),
        category=category,
        run_id=run_id,
        title=title,
        reason=reason,
        suggested_fix=suggested_fix,
        source=source,
    )
    session.add(row)
    return row


def record_logistics_from_run(
    session: Session,
    run: Run,
    *,
    error: str | None = None,
    source: str = "auto",
) -> FailureRecord | None:
    """Auto-log a logistics failure from a FAILED / killed run."""
    err = (error if error is not None else run.error) or "failed"
    return record_failure(
        session,
        category=CATEGORY_LOGISTICS,
        title=title_for_run_failure(run, err),
        reason=err,
        run_id=run.id,
        source=source,
        created_at=run.updated_at or run.created_at,
        dedupe_auto=True,
    )


def failure_to_dict(row: FailureRecord) -> dict[str, Any]:
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "category": row.category,
        "run_id": row.run_id,
        "title": row.title,
        "reason": row.reason,
        "suggested_fix": row.suggested_fix,
        "source": row.source,
    }


def list_failures(
    session: Session,
    *,
    category: str | None = None,
) -> list[FailureRecord]:
    stmt = select(FailureRecord).order_by(FailureRecord.created_at.desc())
    if category:
        if category not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}")
        stmt = stmt.where(FailureRecord.category == category)
    return list(session.exec(stmt).all())


# Known historical incidents (seeded once; accurate reasons from Phase 3 docs / DB).
_SEED_SPECS: list[dict[str, Any]] = [
    {
        "key": "phase3-refuse-check",
        "run_id": "ceb358ae34ca",
        "category": CATEGORY_LOGISTICS,
        "title": "phase3-refuse-check: RUNPOD_API_KEY missing (launch refuse)",
        "reason": (
            "Intentional refuse while API process lacked RUNPOD_API_KEY "
            "(HTTP 400 + FAILED at 0 steps). Not a capacity bug. "
            "Sibling row d6f4a87ca2fe same cause."
        ),
        "suggested_fix": "Set RUNPOD_API_KEY in .env and restart `make dev`.",
    },
    {
        "key": "community-capacity",
        "run_id": "15131a02b3f3",
        "category": CATEGORY_LOGISTICS,
        "title": "wall_follow-mlp-runpod: Community capacity refuse (EU-RO-1)",
        "reason": (
            "Community cloud create_pod returned no instances available for "
            "4090/3090/A4000 with EU-RO-1 volume. Catalog can show High while "
            "Community+volume still refuses. Secure create succeeds."
        ),
        "suggested_fix": "Set RUNPOD_CLOUD_TYPE=SECURE and retry.",
    },
    {
        "key": "watchdog-false-positive",
        "run_id": "b569bf7b275e",
        "category": CATEGORY_LOGISTICS,
        "title": "phase3-secure-smoke-2k: watchdog false kill (missing RUN_ID)",
        "reason": (
            "KILLED_BY_WATCHDOG false positive: GraphQL list_pods env empty → "
            "missing RUN_ID; fixed in watchdog (map pod_id→run via DB)."
        ),
        "suggested_fix": "Watchdog must map pod_id→run via DB when GraphQL env is empty.",
    },
    {
        "key": "secure-smoke-egl",
        "run_id": "9edaa7cac8ca",
        "category": CATEGORY_LOGISTICS,
        "title": "phase3-secure-smoke-2k-g: MuJoCo EGL import failure on worker",
        "reason": (
            "trainer rc=1: OpenGL EGL eglQueryString AttributeError during mujoco import "
            "on the pod (worker image / GL stack)."
        ),
        "suggested_fix": "Rebuild worker image with working EGL/MuJoCo GL deps.",
    },
    {
        "key": "secure-smoke-cuda-driver",
        "run_id": "b642eb2a6e76",
        "category": CATEGORY_LOGISTICS,
        "title": "phase3-secure-smoke-2k-h: CUDA driver too old (trainer rc=139)",
        "reason": (
            "trainer segfault/rc=139 after W&B init; torch warned NVIDIA driver too old "
            "for the CUDA build on that Secure template."
        ),
        "suggested_fix": "Use a Secure GPU template whose driver matches torch CUDA.",
    },
    {
        "key": "api-502-down",
        "run_id": None,
        "category": CATEGORY_LOGISTICS,
        "title": "API 502 / backend unreachable from pods or UI",
        "reason": (
            "Historical ops failure: localhost API or cloudflared tunnel down → "
            "heartbeats/callbacks 502, UI fetch fails, pods cannot reach BACKEND_URL. "
            "No single run_id — infra must stay up during RunPod launches."
        ),
        "suggested_fix": (
            "Keep `make dev` + cloudflared alive; refresh BACKEND_PUBLIC_URL if the "
            "tunnel URL rotated; never point pods at localhost."
        ),
    },
]


def _seed_exists(session: Session, *, title: str, run_id: str | None) -> bool:
    rows = session.exec(
        select(FailureRecord).where(
            FailureRecord.source == "seed",
            FailureRecord.title == title,
        )
    ).all()
    for r in rows:
        if (r.run_id or None) == (run_id or None):
            return True
    return False


def seed_known_failures(session: Session | None = None) -> int:
    """Idempotent seed of known historical logistics failures. Returns rows added."""
    own = session is None
    sess = session or Session(engine)
    added = 0
    try:
        for spec in _SEED_SPECS:
            title = str(spec["title"])
            run_id = spec.get("run_id")
            if _seed_exists(sess, title=title, run_id=run_id):
                continue
            # Prefer DB timestamps when the run exists.
            created = datetime.now(timezone.utc)
            if run_id:
                run = sess.get(Run, run_id)
                if run is not None:
                    created = run.updated_at or run.created_at or created
            record_failure(
                sess,
                category=str(spec["category"]),
                title=title,
                reason=str(spec["reason"]),
                run_id=run_id,
                suggested_fix=spec.get("suggested_fix"),
                source="seed",
                created_at=created,
                dedupe_auto=False,
            )
            added += 1

        # Backfill any other FAILED / KILLED runs not yet covered by auto/seed.
        for run in sess.exec(select(Run)).all():
            if run.status not in {"FAILED", "KILLED_BY_WATCHDOG"}:
                continue
            if not (run.error or "").strip():
                continue
            existing = sess.exec(
                select(FailureRecord).where(FailureRecord.run_id == run.id)
            ).first()
            if existing is not None:
                continue
            record_logistics_from_run(sess, run, source="seed")
            added += 1

        sess.commit()
        if added:
            logger.info("seeded %s failure record(s)", added)
        return added
    finally:
        if own:
            sess.close()
