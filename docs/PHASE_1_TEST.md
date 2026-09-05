# Phase 1 human test

**Blunt status (2026-09-04):** Phase 1 **application code is in place** (core interfaces, MuJoCo adapter, `diffdrive_lidar`, `wall_follow`, `mlp`, SB3 custom policy, local runner, `POST /runs`, heartbeat + SSE, dashboard New Run).  
**Auth key recovered from screenshot into `.env` (86-char `wandb_v1_`); GraphQL viewer probe returns 200.** Do **not** mark the Phase 1 human gate passed until a full local train shows COMPLETE + W&B curve. Phase 2 has **not** started.

## Root cause of W&B 401 (diagnosed → fixed for auth)

| Check | Result |
|---|---|
| `.env` has `WANDB_API_KEY` / `WANDB_PROJECT` | Yes (`.gitignore` covers `.env`); `WANDB_PROJECT=robolab` |
| Key whitespace / quotes | Clean (no quotes, no leading/trailing WS) |
| Key shape | **OK now:** `wandb_v1_` + **77** chars (**total 86**), transcribed from the new-key modal |
| `GET https://api.wandb.ai/viewer` Bearer | **HTTP 404** (endpoint gone; do not use) |
| GraphQL `viewer { id }` with Bearer | **HTTP 200** (auth OK) |
| `wandb` SDK | 0.29.0 (supports long keys) |
| Env → trainer subprocess | **OK** — child inherits API env via `os.environ.copy()` / explicit forward |

**Verdict:** prior 401 was a truncated/wrong key. Screenshot key is complete and authenticates. Human gate still needs a successful train+W&B path.

---

## 0. Fix W&B auth (required before the gate)

### Exact steps

1. Open https://wandb.ai/authorize (log in if needed).
2. Copy a **new** personal API key (expect ~86 chars starting with `wandb_v1_`, or a legacy 40-char hex key).
3. Edit `/Users/avinashnandyala/Projects/robolab/.env` (never commit this file). Replace the whole line — do not paste over a truncated value mid-line:

```bash
WANDB_API_KEY=paste_the_new_key_here
WANDB_PROJECT=robolab
BUDGET_USD_CAP=100
# Optional only if your default entity is wrong for org-scoped keys:
# WANDB_ENTITY=your_team_or_username
```

4. Restart the stack so the API subprocess inherits the new env:

```bash
# stop make dev (Ctrl-C), then:
cd /Users/avinashnandyala/Projects/robolab
make dev
```

5. Verify auth (must print `graphql_status 200` with a viewer id — not 401):

```bash
cd /Users/avinashnandyala/Projects/robolab
uv run --env-file .env --package robolab python - <<'PY'
import os, json, urllib.request
key = os.environ.get("WANDB_API_KEY","").strip()
assert key, "WANDB_API_KEY empty"
print("key_len", len(key), "prefix", key[:8])
req = urllib.request.Request(
    "https://api.wandb.ai/graphql",
    data=b'{"query":"query Viewer { viewer { id username } }"}',
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=20) as r:
    body = json.loads(r.read().decode())
    print("graphql_status", r.status, "viewer", body.get("data", {}).get("viewer"))
PY
```

6. Dashboard header should show **W&B: key valid**. Or: `curl -s http://127.0.0.1:8000/api/wandb/status`.

---

## 1. Start stack

```bash
cd /Users/avinashnandyala/Projects/robolab
make dev
```

- Web: http://localhost:5173  
- API: http://127.0.0.1:8000  

Expect header: **Budget: $100.00 remaining** (or your `BUDGET_USD_CAP`) and **W&B: key valid**.

---

## 2. Human gate clicks

1. Open http://localhost:5173
2. Click **New Run**
3. Leave defaults: **local → mlp → wall_follow → 50k**
4. Click **Start**

### What you must see (PASS)

| Check | Expected |
|---|---|
| Row appears | Name like `wall_follow-mlp-local`, sim `mujoco`, arch `mlp` |
| Status chip | `RUNNING` (then later `COMPLETE`) |
| Progress bar | Advances in real time (SSE + heartbeat; also polls every 5s) |
| Mean return | Number updates (not stuck on `—` forever after a few thousand steps) |
| W&B link | Blue **W&B** link appears; opens a run on wandb.ai |
| W&B chart | Run page shows a `rollout/ep_rew_mean` (or similar) return curve |
| Final status | Chip flips to **`COMPLETE`** |

Optional smoke (faster): choose **5k (smoke)** in the timesteps dropdown first. Same visuals; just fewer steps.

---

## 3. Fail modes (be blunt)

| Symptom | Likely cause |
|---|---|
| Header **W&B: key invalid (truncated?)** | Key length ≠ 86 (`wandb_v1_`) or ≠ 40 (legacy) — redo section 0 |
| Header **W&B: key rejected (401)** | Revoked/wrong key — redo section 0 |
| Status → `FAILED` before launch, error mentions truncated/corrupt key | Same as above; local runner now fails fast |
| Status → `FAILED`, error mentions W&B / 401 / unauthorized | Invalid `WANDB_API_KEY` — redo section 0 |
| Status stuck `RUNNING`, progress 0, no W&B link | Trainer crashed early; inspect `runs/<id>/trainer.log` |
| `POST /api/runs` 400 about runpod | Phase 1 only supports `local` |
| Progress never moves but process alive | Heartbeat/SSE broken; check API logs and `EventSource` in browser Network tab |
| Dashboard empty / budget `…` | `make dev` not running or API not on :8000 |
| NaN / unstable sim warnings | Should not happen with kinematic drive; file a bug if it returns |

---

## 4. CLI cross-check (optional)

```bash
curl -s http://127.0.0.1:8000/api/wandb/status | python3 -m json.tool
curl -s http://127.0.0.1:8000/api/runs | python3 -m json.tool | head
# After starting a run from UI, stream SSE:
# curl -N http://127.0.0.1:8000/api/runs/<id>/events
```

Config template: `configs/experiments/wall_follow_mlp_local.yaml`

---

## 5. Checklist (human)

- [ ] Valid `WANDB_API_KEY` (viewer endpoint returns 200; header shows **W&B: key valid**)
- [ ] `make dev` running; budget header visible
- [ ] New Run → local → mlp → wall_follow → 50k → Start
- [ ] Progress bar advances live
- [ ] Mean return updates live
- [ ] W&B link opens a run with a return curve
- [ ] Status becomes `COMPLETE`

**Gate result:** ☐ PASS / ☐ FAIL  

If FAIL because of W&B 401 / truncated key only: code path is built; **auth is the blocker** — not Phase 2.

**Phase 2 not started. Phase 1 human gate not passed.**
