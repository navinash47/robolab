"""Spend / cost breakdown — JSON API and a simple dedicated HTML page."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from html import escape

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from sqlmodel import select

from robolab_api.budget import get_budget
from robolab_api.db import CostLedger, Pod, Run, SessionDep

router = APIRouter(tags=["costs"])


def _as_utc(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def _hours_between(start: datetime | None, end: datetime | None) -> float | None:
    s = _as_utc(start)
    e = _as_utc(end)
    if s is None:
        return None
    if e is None:
        e = datetime.now(timezone.utc)
    return round(max(0.0, (e - s).total_seconds() / 3600.0), 6)


def build_costs_payload(session: SessionDep) -> dict:
    """Line items from CostLedger joined to Run / Pod, plus budget totals."""
    cloud = (os.environ.get("RUNPOD_CLOUD_TYPE") or "SECURE").strip().upper()
    budget = get_budget(session)

    ledger_rows = session.exec(select(CostLedger).order_by(CostLedger.created_at)).all()
    run_ids = {row.run_id for row in ledger_rows}
    pod_ids = {row.pod_id for row in ledger_rows if row.pod_id}

    runs: dict[str, Run] = {}
    if run_ids:
        for r in session.exec(select(Run)).all():
            if r.id in run_ids:
                runs[r.id] = r
    pods: dict[str, Pod] = {}
    if pod_ids:
        for p in session.exec(select(Pod)).all():
            if p.id in pod_ids:
                pods[p.id] = p

    items: list[dict] = []
    total = 0.0
    for row in ledger_rows:
        run = runs.get(row.run_id)
        pod = pods.get(row.pod_id) if row.pod_id else None
        amount = float(row.amount_usd or 0.0)
        total += amount
        hours = None
        hourly = None
        gpu = None
        if pod is not None:
            hours = _hours_between(pod.started_at, pod.terminated_at)
            hourly = float(pod.hourly_rate) if pod.hourly_rate is not None else None
            gpu = pod.gpu_type or None
        if gpu is None and run is not None:
            gpu = run.gpu_type
        if hourly is None and run is not None and run.hourly_rate is not None:
            hourly = float(run.hourly_rate)

        items.append(
            {
                "ledger_id": row.id,
                "run_id": row.run_id,
                "name": run.name if run else row.run_id,
                "pod_id": row.pod_id,
                "gpu": gpu or "",
                "cloud": cloud,
                "hourly_rate_usd": hourly,
                "hours": hours,
                "cost_usd": round(amount, 6),
                "status": run.status if run else "",
                "reason": row.reason or "",
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )

    return {
        "items": items,
        "count": len(items),
        "total_usd": round(total, 6),
        "month_spend_usd": budget["month_spend_usd"],
        "budget_usd_cap": budget["budget_usd_cap"],
        "remaining_usd": budget["remaining_usd"],
        "cloud_default": cloud,
    }


@router.get("/api/costs")
def get_costs(session: SessionDep) -> dict:
    """JSON spend breakdown from CostLedger + Pod/Run."""
    return build_costs_payload(session)


@router.get("/api/billing")
def get_billing(session: SessionDep) -> dict:
    """Alias of /api/costs."""
    return build_costs_payload(session)


def _fmt_usd(n: float | None) -> str:
    if n is None:
        return "—"
    return f"${n:,.6f}"


def _fmt_hours(h: float | None) -> str:
    if h is None:
        return "—"
    return f"{h:.4f}"


@router.get("/spend", response_class=HTMLResponse)
def spend_page(session: SessionDep) -> HTMLResponse:
    """Simple dedicated spend page — table of charges + totals."""
    data = build_costs_payload(session)
    rows_html: list[str] = []
    for it in data["items"]:
        rows_html.append(
            "<tr>"
            f"<td><code>{escape(str(it['run_id']))}</code></td>"
            f"<td>{escape(str(it['name']))}</td>"
            f"<td>{escape(str(it['gpu']))}</td>"
            f"<td>{escape(str(it['cloud']))}</td>"
            f"<td class='num'>{_fmt_usd(it['hourly_rate_usd'])}/hr</td>"
            f"<td class='num'>{_fmt_hours(it['hours'])}</td>"
            f"<td class='num'><strong>{_fmt_usd(it['cost_usd'])}</strong></td>"
            f"<td>{escape(str(it['status']))}</td>"
            f"<td class='muted'>{escape(str(it['reason']))}</td>"
            "</tr>"
        )
    if not rows_html:
        rows_html.append(
            '<tr><td colspan="9" class="muted">No CostLedger charges yet.</td></tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RoboLab Spend</title>
  <style>
    :root {{
      --bg: #e8eef2;
      --surface: #fff;
      --border: #c5ced6;
      --text: #15202b;
      --muted: #5c6773;
      --accent: #0b6e4f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(ellipse 70% 45% at 0% -10%, #c5ddd4 0%, transparent 55%),
        radial-gradient(ellipse 50% 35% at 100% 0%, #d5dde6 0%, transparent 50%),
        var(--bg);
      padding: 2rem 1.5rem 3rem;
    }}
    main {{ max-width: 1100px; margin: 0 auto; }}
    h1 {{ margin: 0 0 0.25rem; font-size: 1.75rem; }}
    .sub {{ color: var(--muted); margin: 0 0 1.5rem; font-size: 0.95rem; }}
    .summary {{
      display: flex; gap: 1.5rem; flex-wrap: wrap;
      margin-bottom: 1.25rem; font-size: 0.95rem;
    }}
    .summary strong {{ color: var(--accent); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--border);
      font-size: 0.875rem;
    }}
    th, td {{
      text-align: left;
      padding: 0.55rem 0.65rem;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      background: #f3f6f8;
      font-weight: 600;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }}
    tr:last-child td {{ border-bottom: none; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    code {{ font-size: 0.8em; }}
    .muted {{ color: var(--muted); }}
    .total-row {{
      margin-top: 1rem;
      font-size: 1.05rem;
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;
    }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <main>
    <h1>RoboLab Spend</h1>
    <p class="sub">CostLedger charges for the current month · cloud default {escape(data['cloud_default'])}</p>
    <div class="summary">
      <div>Month spend: <strong>{_fmt_usd(data['month_spend_usd'])}</strong></div>
      <div>Budget cap: {_fmt_usd(data['budget_usd_cap'])}</div>
      <div>Remaining: <strong>{_fmt_usd(data['remaining_usd'])}</strong></div>
      <div>Line items: {data['count']}</div>
    </div>
    <table>
      <thead>
        <tr>
          <th>Run ID</th>
          <th>Name</th>
          <th>GPU</th>
          <th>Cloud</th>
          <th>$/hr</th>
          <th>Hours</th>
          <th>Cost</th>
          <th>Status</th>
          <th>Reason</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    <div class="total-row">
      <span>Ledger total</span>
      <strong>{_fmt_usd(data['total_usd'])}</strong>
    </div>
    <p class="sub" style="margin-top:1.5rem">
      JSON: <a href="/api/costs"><code>/api/costs</code></a>
    </p>
  </main>
</body>
</html>
"""
    return HTMLResponse(content=html)
