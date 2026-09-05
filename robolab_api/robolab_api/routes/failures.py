"""Failure Resolution — JSON API and dedicated HTML page (like /spend)."""

from __future__ import annotations

from html import escape
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlmodel import Session

from robolab_api.db import Run, SessionDep, engine
from robolab_api.failures import (
    CATEGORY_EXPERIMENT,
    CATEGORY_LOGISTICS,
    VALID_CATEGORIES,
    failure_to_dict,
    list_failures,
    record_failure,
    seed_known_failures,
)

router = APIRouter(tags=["failures"])


class FailureCreateBody(BaseModel):
    category: Literal["logistics", "experiment"] = CATEGORY_EXPERIMENT
    title: Optional[str] = None
    reason: str = Field(..., min_length=1)
    run_id: Optional[str] = None
    suggested_fix: Optional[str] = None


@router.get("/api/failures")
def get_failures(
    session: SessionDep,
    category: Optional[str] = Query(default=None),
) -> dict:
    """List failure records, optionally filtered by category."""
    seed_known_failures(session)
    try:
        rows = list_failures(session, category=category)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "failures": [failure_to_dict(r) for r in rows],
        "count": len(rows),
        "categories": {
            "logistics": "Logistics failures",
            "experiment": "Experiment failures",
        },
    }


@router.post("/api/failures")
def create_failure(body: FailureCreateBody, session: SessionDep) -> dict:
    """Manually flag a failure (typically experiment / run-quality)."""
    if body.category not in VALID_CATEGORIES:
        raise HTTPException(400, f"category must be one of {sorted(VALID_CATEGORIES)}")

    run: Run | None = None
    if body.run_id:
        run = session.get(Run, body.run_id)
        if not run:
            raise HTTPException(404, f"Run {body.run_id} not found")

    title = (body.title or "").strip()
    if not title:
        if run is not None:
            title = f"{run.name}: experiment failure"
        else:
            title = "Manual experiment failure"

    row = record_failure(
        session,
        category=body.category,
        title=title,
        reason=body.reason.strip(),
        run_id=body.run_id,
        suggested_fix=body.suggested_fix,
        source="manual",
        dedupe_auto=False,
    )
    session.commit()
    if row is not None:
        session.refresh(row)
    return {"ok": True, "failure": failure_to_dict(row) if row else None}


@router.get("/failures", response_class=HTMLResponse)
def failures_page(session: SessionDep) -> HTMLResponse:
    """Dedicated Failure Resolution page with logistics / experiment tabs."""
    seed_known_failures(session)
    rows = list_failures(session)
    logistics = [r for r in rows if r.category == CATEGORY_LOGISTICS]
    experiment = [r for r in rows if r.category == CATEGORY_EXPERIMENT]

    def _rows_html(items: list) -> str:
        if not items:
            return (
                '<tr><td colspan="6" class="muted">No failures in this category yet.</td></tr>'
            )
        parts: list[str] = []
        for it in items:
            ts = it.created_at.isoformat(sep=" ", timespec="seconds") if it.created_at else "—"
            run_cell = (
                f"<code>{escape(str(it.run_id))}</code>" if it.run_id else "—"
            )
            fix = escape(it.suggested_fix or "—")
            parts.append(
                "<tr>"
                f"<td class='muted'>{escape(ts)}</td>"
                f"<td>{run_cell}</td>"
                f"<td><strong>{escape(it.title)}</strong></td>"
                f"<td class='reason'>{escape(it.reason or '')}</td>"
                f"<td class='muted'>{fix}</td>"
                f"<td><span class='pill'>{escape(it.source)}</span></td>"
                "</tr>"
            )
        return "".join(parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Failure Resolution — RoboLab</title>
  <style>
    :root {{
      --bg: #e8eef2;
      --surface: #fff;
      --border: #c5ced6;
      --text: #15202b;
      --muted: #5c6773;
      --accent: #0b6e4f;
      --warn: #8a4b08;
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
    main {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ margin: 0 0 0.25rem; font-size: 1.75rem; }}
    .sub {{ color: var(--muted); margin: 0 0 1.25rem; font-size: 0.95rem; }}
    .tabs {{
      display: flex; gap: 0.5rem; flex-wrap: wrap;
      margin-bottom: 1rem;
    }}
    .tabs button {{
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      padding: 0.45rem 0.9rem;
      font: inherit;
      font-size: 0.9rem;
      cursor: pointer;
      border-radius: 4px;
    }}
    .tabs button.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      font-weight: 600;
    }}
    .panel {{ display: none; }}
    .panel.active {{ display: block; }}
    .blurb {{
      margin: 0 0 1rem;
      font-size: 0.9rem;
      color: var(--muted);
      max-width: 70ch;
    }}
    .summary {{
      display: flex; gap: 1.5rem; flex-wrap: wrap;
      margin-bottom: 1rem; font-size: 0.95rem;
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
    code {{ font-size: 0.8em; }}
    .muted {{ color: var(--muted); }}
    .reason {{ white-space: pre-wrap; max-width: 36rem; }}
    .pill {{
      display: inline-block;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      padding: 0.15rem 0.4rem;
      border: 1px solid var(--border);
      border-radius: 3px;
      color: var(--muted);
    }}
    a {{ color: var(--accent); }}
    .manual {{
      margin-top: 1.75rem;
      padding: 1rem;
      background: var(--surface);
      border: 1px solid var(--border);
    }}
    .manual h2 {{ margin: 0 0 0.5rem; font-size: 1.05rem; }}
    .manual label {{ display: block; font-size: 0.8rem; color: var(--muted); margin-top: 0.6rem; }}
    .manual input, .manual textarea, .manual select {{
      width: 100%;
      margin-top: 0.25rem;
      padding: 0.4rem 0.5rem;
      font: inherit;
      border: 1px solid var(--border);
    }}
    .manual textarea {{ min-height: 4.5rem; }}
    .manual button {{
      margin-top: 0.75rem;
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 0.45rem 0.9rem;
      font: inherit;
      cursor: pointer;
    }}
    .manual .msg {{ margin-top: 0.5rem; font-size: 0.85rem; color: var(--warn); }}
  </style>
</head>
<body>
  <main>
    <h1>Failure Resolution</h1>
    <p class="sub">
      Logistics (infra/ops) vs experiment (science/engineering) failures —
      so future agents and humans can rectify them.
    </p>
    <div class="summary">
      <div>Logistics: <strong>{len(logistics)}</strong></div>
      <div>Experiment: <strong>{len(experiment)}</strong></div>
      <div>Total: <strong>{len(rows)}</strong></div>
    </div>
    <div class="tabs" role="tablist">
      <button type="button" class="active" data-tab="logistics" role="tab">
        Logistics failures ({len(logistics)})
      </button>
      <button type="button" data-tab="experiment" role="tab">
        Experiment failures ({len(experiment)})
      </button>
    </div>

    <section id="panel-logistics" class="panel active" role="tabpanel">
      <p class="blurb">
        Infra/ops that prevent or abort a run wrongly: RunPod capacity/auth/image/git/tunnel,
        API 502, missing keys, budget refuse, watchdog false kills, Docker/worker bootstrap.
        Often FAILED at 0 steps or launch refused.
      </p>
      <table>
        <thead>
          <tr>
            <th>When</th>
            <th>Run ID</th>
            <th>Title</th>
            <th>Reason</th>
            <th>Suggested fix</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {_rows_html(logistics)}
        </tbody>
      </table>
    </section>

    <section id="panel-experiment" class="panel" role="tabpanel">
      <p class="blurb">
        Run may be COMPLETE or FAILED, but the science/engineering is wrong:
        bad reward, wrong config, obs/action bugs, policy does not wall-follow,
        NaNs, absurd returns, video nonsense. Distinct from &ldquo;couldn&rsquo;t start the pod&rdquo;.
      </p>
      <table>
        <thead>
          <tr>
            <th>When</th>
            <th>Run ID</th>
            <th>Title</th>
            <th>Reason</th>
            <th>Suggested fix</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {_rows_html(experiment)}
        </tbody>
      </table>
    </section>

    <div class="manual">
      <h2>Flag experiment failure</h2>
      <p class="sub" style="margin-bottom:0">
        Manually record a run-quality issue (COMPLETE runs included). Also available from the dashboard.
      </p>
      <form id="flag-form">
        <label>Run ID (optional)
          <input name="run_id" placeholder="e.g. 873109d8da56" />
        </label>
        <label>Title (optional)
          <input name="title" placeholder="e.g. wall_follow video: robot spins in place" />
        </label>
        <label>Reason
          <textarea name="reason" required placeholder="What went wrong scientifically / in config?"></textarea>
        </label>
        <label>Suggested fix (optional)
          <input name="suggested_fix" placeholder="e.g. fix lidar obs indexing in wall_follow" />
        </label>
        <button type="submit">Save experiment failure</button>
        <p class="msg" id="flag-msg" hidden></p>
      </form>
    </div>

    <p class="sub" style="margin-top:1.5rem">
      JSON: <a href="/api/failures"><code>/api/failures</code></a>
      · filter <code>?category=logistics</code> or <code>experiment</code>
      · POST <code>/api/failures</code> for manual flags
    </p>
  </main>
  <script>
    const tabs = document.querySelectorAll(".tabs button");
    const panels = {{
      logistics: document.getElementById("panel-logistics"),
      experiment: document.getElementById("panel-experiment"),
    }};
    tabs.forEach((btn) => {{
      btn.addEventListener("click", () => {{
        const key = btn.dataset.tab;
        tabs.forEach((b) => b.classList.toggle("active", b === btn));
        Object.entries(panels).forEach(([k, el]) => {{
          el.classList.toggle("active", k === key);
        }});
      }});
    }});

    const form = document.getElementById("flag-form");
    const msg = document.getElementById("flag-msg");
    form.addEventListener("submit", async (ev) => {{
      ev.preventDefault();
      msg.hidden = true;
      const fd = new FormData(form);
      const body = {{
        category: "experiment",
        run_id: (fd.get("run_id") || "").toString().trim() || null,
        title: (fd.get("title") || "").toString().trim() || null,
        reason: (fd.get("reason") || "").toString().trim(),
        suggested_fix: (fd.get("suggested_fix") || "").toString().trim() || null,
      }};
      try {{
        const res = await fetch("/api/failures", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(body),
        }});
        const data = await res.json().catch(() => ({{}}));
        if (!res.ok) {{
          throw new Error(data.detail || ("HTTP " + res.status));
        }}
        window.location.reload();
      }} catch (err) {{
        msg.textContent = err instanceof Error ? err.message : String(err);
        msg.hidden = false;
      }}
    }});
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


def ensure_seeded() -> None:
    """Call from lifespan so the page is useful on first open."""
    with Session(engine) as session:
        seed_known_failures(session)
