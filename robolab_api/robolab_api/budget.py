"""Budget helpers — month spend from CostLedger vs BUDGET_USD_CAP."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlmodel import Session, select

from robolab_api.db import CostLedger, engine


def month_spend_usd(session: Session | None = None) -> float:
    """Sum CostLedger rows for the current UTC calendar month."""
    now = datetime.now(timezone.utc)
    start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)

    def _sum(s: Session) -> float:
        rows = s.exec(select(CostLedger)).all()
        total = 0.0
        for row in rows:
            ts = row.created_at
            if ts is None:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= start:
                total += float(row.amount_usd or 0.0)
        return round(total, 6)

    if session is not None:
        return _sum(session)
    with Session(engine) as s:
        return _sum(s)


def get_budget(session: Session | None = None) -> dict[str, float]:
    """Return budget cap, month spend, and remaining USD."""
    cap = float(os.environ.get("BUDGET_USD_CAP", "100"))
    month_spend = month_spend_usd(session)
    return {
        "budget_usd_cap": cap,
        "month_spend_usd": month_spend,
        "remaining_usd": round(cap - month_spend, 6),
    }


def refuse_if_over_cap(session: Session | None = None) -> None:
    """Raise RuntimeError if month ledger already at/over BUDGET_USD_CAP."""
    b = get_budget(session)
    if b["month_spend_usd"] >= b["budget_usd_cap"]:
        raise RuntimeError(
            f"Monthly budget exhausted: spent ${b['month_spend_usd']:.4f} "
            f">= BUDGET_USD_CAP ${b['budget_usd_cap']:.2f}. "
            "Refuse new RunPod launches until next month or raise the cap."
        )
