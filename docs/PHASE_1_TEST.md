# Phase 1 human test

**Blunt status (2026-09-04):** Phase 1 **application code is in place** (core interfaces, MuJoCo adapter, `diffdrive_lidar`, `wall_follow`, `mlp`, SB3 custom policy, local runner, `POST /runs`, heartbeat + SSE, dashboard New Run).  
**Not human-gate-ready yet for COMPLETE + W&B curve:** the `WANDB_API_KEY` currently in `.env` is **rejected by W&B with HTTP 401**. Until you paste a valid key, runs fail at `wandb.init` and never reach `COMPLETE` with a live return curve.

---

## 0. Fix W&B auth (required before the gate)

Verified during Phase 1: `POST https://api.wandb.ai/graphql` → **401 Unauthorized** with the key currently in `.env`.

### Exact steps

1. Open https://wandb.ai/authorize (log in if needed).
2. Copy a **new** personal API key.
3. Edit `/Users/avinashnandyala/Projects/robolab/.env` (never commit this file):

```bash
WANDB_API_KEY=paste_the_new_key_here
WANDB_PROJECT=robolab
BUDGET_USD_CAP=100
```

4. Restart the stack so the API subprocess inherits the new env:

```bash
# stop make dev (Ctrl-C), then:
cd /Users/avinashnandyala/Projects/robolab
make dev
```

5. Verify auth (must print `viewer_status 200`, not 401):

```bash
cd /Users/avinashnandyala/Projects/robolab
uv run --env-file .env python - <<'PY'
import os, urllib.request
key = os.environ.get("WANDB_API_KEY","").strip()
assert key, "WANDB_API_KEY empty"
req = urllib.request.Request(
    "https://api.wandb.ai/viewer",
    headers={"Authorization": f"Bearer {key}"},
)
with urllib.request.urlopen(req, timeout=20) as r:
    print("viewer_status", r.status)
PY
```

---

## 1. Start stack

```bash
cd /Users/avinashnandyala/Projects/robolab
make dev
```

- Web: http://localhost:5173  
- API: http://127.0.0.1:8000  

Expect header: **Budget: $100.00 remaining** (or your `BUDGET_USD_CAP`).

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
| Status → `FAILED`, error mentions W&B / 401 / unauthorized | Invalid or empty `WANDB_API_KEY` — redo section 0 |
| Status stuck `RUNNING`, progress 0, no W&B link | Trainer crashed early; inspect `runs/<id>/trainer.log` |
| `POST /api/runs` 400 about runpod | Phase 1 only supports `local` |
| Progress never moves but process alive | Heartbeat/SSE broken; check API logs and `EventSource` in browser Network tab |
| Dashboard empty / budget `…` | `make dev` not running or API not on :8000 |
| NaN / unstable sim warnings | Should not happen with kinematic drive; file a bug if it returns |

---

## 4. CLI cross-check (optional)

```bash
curl -s http://127.0.0.1:8000/api/runs | python3 -m json.tool | head
# After starting a run from UI, stream SSE:
# curl -N http://127.0.0.1:8000/api/runs/<id>/events
```

Config template: `configs/experiments/wall_follow_mlp_local.yaml`

---

## 5. Checklist (human)

- [ ] Valid `WANDB_API_KEY` (viewer endpoint returns 200)
- [ ] `make dev` running; budget header visible
- [ ] New Run → local → mlp → wall_follow → 50k → Start
- [ ] Progress bar advances live
- [ ] Mean return updates live
- [ ] W&B link opens a run with a return curve
- [ ] Status becomes `COMPLETE`

**Gate result:** ☐ PASS / ☐ FAIL  

If FAIL because of W&B 401 only: code path is built; auth is the blocker — not Phase 2.

**Phase 2 not started.**
