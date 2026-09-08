"""Cursor / agent token tracker — JSON API + dedicated HTML page (mirrors /spend)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from html import escape
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlmodel import select

from robolab_api.db import SessionDep, TokenLedger

router = APIRouter(tags=["tokens"])


def _usd_per_1m_input() -> float:
    try:
        return float(os.environ.get("TOKEN_USD_PER_1M_INPUT", "3.0"))
    except ValueError:
        return 3.0


def _usd_per_1m_output() -> float:
    try:
        return float(os.environ.get("TOKEN_USD_PER_1M_OUTPUT", "15.0"))
    except ValueError:
        return 15.0


def estimate_usd(input_tokens: int, output_tokens: int) -> float:
    """Rough USD estimate from env rates ($ / 1M tokens)."""
    inp = max(0, int(input_tokens or 0))
    out = max(0, int(output_tokens or 0))
    return round(
        (inp / 1_000_000.0) * _usd_per_1m_input()
        + (out / 1_000_000.0) * _usd_per_1m_output(),
        6,
    )


def _probe_headroom_proxy() -> dict[str, Any]:
    """Best-effort check of local headroom proxy (default :8787)."""
    import urllib.error
    import urllib.request

    url = (os.environ.get("HEADROOM_PROXY_URL") or "http://127.0.0.1:8787").rstrip("/")
    try:
        req = urllib.request.Request(f"{url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            body = resp.read(200).decode("utf-8", errors="replace")
            return {"url": url, "status": "ok", "http_code": resp.status, "body": body[:120]}
    except urllib.error.HTTPError as e:
        # Some proxies 404 /health but are still up — treat non-connection as reachable.
        return {"url": url, "status": "reachable", "http_code": e.code, "body": ""}
    except Exception as e:  # noqa: BLE001 — surface any connect failure cleanly
        return {"url": url, "status": "unreachable", "error": str(e)[:200]}


def build_tokens_payload(session: SessionDep) -> dict:
    rows = session.exec(select(TokenLedger).order_by(TokenLedger.created_at.desc())).all()
    items: list[dict] = []
    total_in = total_out = total_saved = 0
    total_usd = 0.0
    by_source: dict[str, dict[str, float | int]] = {}
    by_model: dict[str, dict[str, float | int]] = {}

    for row in rows:
        tin = int(row.input_tokens or 0)
        tout = int(row.output_tokens or 0)
        saved = int(row.tokens_saved or 0)
        usd = float(row.estimated_usd or 0.0)
        total_in += tin
        total_out += tout
        total_saved += saved
        total_usd += usd
        src = row.source or "manual"
        model = row.model or "(unknown)"
        for bucket, key in ((by_source, src), (by_model, model)):
            slot = bucket.setdefault(
                key,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "tokens_saved": 0,
                    "estimated_usd": 0.0,
                    "count": 0,
                },
            )
            slot["input_tokens"] = int(slot["input_tokens"]) + tin
            slot["output_tokens"] = int(slot["output_tokens"]) + tout
            slot["tokens_saved"] = int(slot["tokens_saved"]) + saved
            slot["estimated_usd"] = round(float(slot["estimated_usd"]) + usd, 6)
            slot["count"] = int(slot["count"]) + 1

        items.append(
            {
                "id": row.id,
                "source": src,
                "model": model,
                "session_label": row.session_label or "",
                "input_tokens": tin,
                "output_tokens": tout,
                "total_tokens": tin + tout,
                "tokens_saved": saved,
                "estimated_usd": round(usd, 6),
                "note": row.note or "",
                "meta": _safe_json(row.meta_json),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )

    return {
        "items": items,
        "count": len(items),
        "totals": {
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "tokens_saved": total_saved,
            "estimated_usd": round(total_usd, 6),
        },
        "by_source": by_source,
        "by_model": by_model,
        "rates": {
            "input_usd_per_1m": _usd_per_1m_input(),
            "output_usd_per_1m": _usd_per_1m_output(),
        },
        "headroom_proxy": _probe_headroom_proxy(),
        "notes": [
            "Not Cursor billing truth — local ledger for discipline (ops cost §5).",
            "Headroom MCP compress/retrieve/stats is available in Cursor; proxy optional on :8787.",
            "Bob8259 Cursor-Token-Saver proxy is archived/closed — monitor pattern only, not wired.",
        ],
    }


def _safe_json(raw: str | None) -> Any:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw[:200]}


class TokenEventCreate(BaseModel):
    source: str = Field(default="manual", max_length=64)
    model: str = Field(default="", max_length=128)
    session_label: str = Field(default="", max_length=256)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    tokens_saved: int = Field(default=0, ge=0)
    estimated_usd: Optional[float] = Field(default=None, ge=0)
    note: str = Field(default="", max_length=2000)
    meta: dict[str, Any] = Field(default_factory=dict)


class HeadroomSnapshot(BaseModel):
    """Ingest a headroom_stats-like payload into TokenLedger."""

    session_label: str = Field(default="headroom-session", max_length=256)
    model: str = Field(default="headroom", max_length=128)
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    total_tokens_saved: int = Field(default=0, ge=0)
    estimated_cost_saved_usd: float = Field(default=0.0, ge=0)
    compressions: int = Field(default=0, ge=0)
    note: str = Field(default="", max_length=2000)
    meta: dict[str, Any] = Field(default_factory=dict)


@router.get("/api/tokens")
def get_tokens(session: SessionDep) -> dict:
    return build_tokens_payload(session)


@router.post("/api/tokens")
def create_token_event(body: TokenEventCreate, session: SessionDep) -> dict:
    usd = body.estimated_usd
    if usd is None:
        usd = estimate_usd(body.input_tokens, body.output_tokens)
    row = TokenLedger(
        source=(body.source or "manual").strip()[:64] or "manual",
        model=(body.model or "").strip()[:128],
        session_label=(body.session_label or "").strip()[:256],
        input_tokens=int(body.input_tokens),
        output_tokens=int(body.output_tokens),
        tokens_saved=int(body.tokens_saved),
        estimated_usd=float(usd),
        note=(body.note or "")[:2000],
        meta_json=json.dumps(body.meta or {}, default=str)[:8000],
        created_at=datetime.now(timezone.utc),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return {"ok": True, "id": row.id, "estimated_usd": row.estimated_usd}


@router.post("/api/tokens/headroom")
def ingest_headroom(body: HeadroomSnapshot, session: SessionDep) -> dict:
    """Record one headroom session snapshot (tokens saved + optional I/O totals)."""
    meta = dict(body.meta or {})
    meta["compressions"] = body.compressions
    meta["estimated_cost_saved_usd"] = body.estimated_cost_saved_usd
    row = TokenLedger(
        source="headroom",
        model=(body.model or "headroom")[:128],
        session_label=(body.session_label or "headroom-session")[:256],
        input_tokens=int(body.total_input_tokens),
        output_tokens=int(body.total_output_tokens),
        tokens_saved=int(body.total_tokens_saved),
        estimated_usd=estimate_usd(body.total_input_tokens, body.total_output_tokens),
        note=(body.note or f"headroom snapshot · saved≈${body.estimated_cost_saved_usd:.4f}")[
            :2000
        ],
        meta_json=json.dumps(meta, default=str)[:8000],
        created_at=datetime.now(timezone.utc),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return {"ok": True, "id": row.id, "estimated_usd": row.estimated_usd}


@router.delete("/api/tokens/{event_id}")
def delete_token_event(event_id: int, session: SessionDep) -> dict:
    row = session.get(TokenLedger, event_id)
    if row is None:
        raise HTTPException(status_code=404, detail="token event not found")
    session.delete(row)
    session.commit()
    return {"ok": True, "id": event_id}


def _fmt_int(n: int | None) -> str:
    return f"{int(n or 0):,}"


def _fmt_usd(n: float | None) -> str:
    if n is None:
        return "—"
    return f"${n:,.6f}"


@router.get("/tokens", response_class=HTMLResponse)
def tokens_page(session: SessionDep) -> HTMLResponse:
    """Dedicated token tracker page — pattern inspired by spend + Token-Saver monitor."""
    data = build_tokens_payload(session)
    rows_html: list[str] = []
    for it in data["items"]:
        rows_html.append(
            "<tr class='usage-row'"
            f" data-input='{it['input_tokens']}' data-output='{it['output_tokens']}'"
            f" data-cost='{it['estimated_usd']}'>"
            f"<td>{escape(str(it['created_at'] or ''))}</td>"
            f"<td>{escape(str(it['source']))}</td>"
            f"<td>{escape(str(it['model']))}</td>"
            f"<td>{escape(str(it['session_label']))}</td>"
            f"<td class='num'>{_fmt_int(it['input_tokens'])}</td>"
            f"<td class='num'>{_fmt_int(it['output_tokens'])}</td>"
            f"<td class='num'>{_fmt_int(it['tokens_saved'])}</td>"
            f"<td class='num'><strong>{_fmt_usd(it['estimated_usd'])}</strong></td>"
            f"<td class='muted'>{escape(str(it['note']))}</td>"
            "</tr>"
        )
    if not rows_html:
        rows_html.append(
            '<tr><td colspan="9" class="muted">No token events yet. '
            "Log via <code>POST /api/tokens</code> or "
            "<code>python scripts/log_token_usage.py</code>.</td></tr>"
        )

    by_model_rows = []
    for model, slot in sorted(data["by_model"].items(), key=lambda kv: -float(kv[1]["estimated_usd"])):
        by_model_rows.append(
            "<tr>"
            f"<td>{escape(model)}</td>"
            f"<td class='num'>{_fmt_int(int(slot['count']))}</td>"
            f"<td class='num'>{_fmt_int(int(slot['input_tokens']))}</td>"
            f"<td class='num'>{_fmt_int(int(slot['output_tokens']))}</td>"
            f"<td class='num'>{_fmt_usd(float(slot['estimated_usd']))}</td>"
            "</tr>"
        )
    if not by_model_rows:
        by_model_rows.append('<tr><td colspan="5" class="muted">—</td></tr>')

    hr = data["headroom_proxy"]
    hr_line = f"{escape(str(hr.get('status')))} · {escape(str(hr.get('url', '')))}"
    if hr.get("error"):
        hr_line += f" · {escape(str(hr['error']))}"

    rates = data["rates"]
    totals = data["totals"]
    notes = "".join(f"<li>{escape(n)}</li>" for n in data["notes"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RoboLab Tokens</title>
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
    h2 {{ margin: 1.75rem 0 0.75rem; font-size: 1.1rem; }}
    .sub {{ color: var(--muted); margin: 0 0 1.5rem; font-size: 0.95rem; }}
    .summary {{
      display: flex; gap: 1.5rem; flex-wrap: wrap;
      margin-bottom: 1.25rem; font-size: 0.95rem;
    }}
    .summary strong {{ color: var(--accent); }}
    .controls {{
      display: flex; gap: 1rem; flex-wrap: wrap; align-items: end;
      margin-bottom: 1rem; padding: 0.85rem 1rem;
      background: var(--surface); border: 1px solid var(--border);
    }}
    .controls label {{ font-size: 0.8rem; color: var(--muted); display: flex; flex-direction: column; gap: 0.25rem; }}
    .controls input {{ padding: 0.35rem 0.5rem; border: 1px solid var(--border); font: inherit; width: 7rem; }}
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
    tr.usage-row.selected {{ background: #d7ebe3; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    code {{ font-size: 0.8em; }}
    .muted {{ color: var(--muted); }}
    a {{ color: var(--accent); }}
    ul.notes {{ color: var(--muted); font-size: 0.9rem; }}
  </style>
</head>
<body>
  <main>
    <h1>RoboLab Tokens</h1>
    <p class="sub">Local Cursor / agent token ledger · companion to <a href="/spend">Spend</a> (GPU)</p>
    <div class="summary">
      <div>Events: <strong>{data['count']}</strong></div>
      <div>Input: <strong>{_fmt_int(totals['input_tokens'])}</strong></div>
      <div>Output: <strong>{_fmt_int(totals['output_tokens'])}</strong></div>
      <div>Saved (headroom): <strong>{_fmt_int(totals['tokens_saved'])}</strong></div>
      <div>Est. cost: <strong id="totalCost">{_fmt_usd(totals['estimated_usd'])}</strong></div>
      <div>Headroom proxy: {hr_line}</div>
    </div>
    <div class="controls">
      <label>Input $/1M
        <input type="number" id="inputPrice" value="{rates['input_usd_per_1m']}" step="0.01" />
      </label>
      <label>Output $/1M
        <input type="number" id="outputPrice" value="{rates['output_usd_per_1m']}" step="0.01" />
      </label>
      <div class="muted" style="align-self:center">
        Selected reprice: $<span id="selectedCost">0.0000</span>
        · in <span id="selectedInput">0</span>
        · out <span id="selectedOutput">0</span>
      </div>
    </div>
    <table>
      <thead>
        <tr>
          <th>When (UTC)</th>
          <th>Source</th>
          <th>Model</th>
          <th>Session</th>
          <th>Input</th>
          <th>Output</th>
          <th>Saved</th>
          <th>Est. $</th>
          <th>Note</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    <h2>By model</h2>
    <table>
      <thead>
        <tr><th>Model</th><th>Events</th><th>Input</th><th>Output</th><th>Est. $</th></tr>
      </thead>
      <tbody>
        {''.join(by_model_rows)}
      </tbody>
    </table>
    <h2>Notes</h2>
    <ul class="notes">{notes}</ul>
    <p class="muted"><a href="/">API</a> · <a href="/spend">Spend</a> · <a href="/failures">Failures</a></p>
  </main>
  <script>
    function reprice() {{
      const ip = parseFloat(document.getElementById('inputPrice').value) || 0;
      const op = parseFloat(document.getElementById('outputPrice').value) || 0;
      let tin = 0, tout = 0, cost = 0;
      document.querySelectorAll('.usage-row.selected').forEach(row => {{
        const i = parseInt(row.dataset.input || '0', 10);
        const o = parseInt(row.dataset.output || '0', 10);
        tin += i; tout += o;
        cost += (i / 1e6) * ip + (o / 1e6) * op;
      }});
      document.getElementById('selectedInput').textContent = tin.toLocaleString();
      document.getElementById('selectedOutput').textContent = tout.toLocaleString();
      document.getElementById('selectedCost').textContent = cost.toFixed(4);
    }}
    document.querySelectorAll('.usage-row').forEach(row => {{
      row.addEventListener('click', () => {{
        row.classList.toggle('selected');
        reprice();
      }});
    }});
    document.getElementById('inputPrice').addEventListener('input', reprice);
    document.getElementById('outputPrice').addEventListener('input', reprice);
  </script>
</body>
</html>
"""
    return HTMLResponse(html)
