# Phase 3 test — human gate checklist

**Blunt status:** Phase 3 **code + docs shipped**. `RUNPOD_API_KEY` is present in local `.env`; git remote exists and is pushed (`https://github.com/navinash47/robolab.git`). Paid smoke still **blocked** on: network volume, pushed worker image, `BACKEND_PUBLIC_URL`, and `GITHUB_TOKEN` (private repo). **Not gate-passed.**

Full zero-spend setup steps: [`docs/PHASE_3_SETUP.md`](PHASE_3_SETUP.md).

## Human blockers (do these before any paid launch)

MCP OAuth in Cursor (`user-runpod`) is **not** the RoboLab app key. The FastAPI process only reads `.env`.

### 1. API key → `.env`

1. Open https://www.runpod.io/console/user/settings → create an **API key**
2. Put it in project `.env` (never commit):

```bash
RUNPOD_API_KEY=rpa_…
```

3. Restart `make dev` so uvicorn reloads env

### 2. Network volume → `.env`

Create **when you are ready to pay storage** (agent will not auto-create unless you ask):

- **Size tip:** **50 GB** recommended
- **Region tip:** US DC that has your GPU type (e.g. RTX 4090)
- **Console:** https://www.runpod.io/console/user/storage
- **Or MCP** (only if you request it): `create-network-volume` with `name`, `size` (50), `dataCenterId`

Then:

```bash
RUNPOD_NETWORK_VOLUME_ID=…   # id from console / MCP response
```

### 3. Worker image (build + push)

Docker must be installed locally (Desktop or Colima). Agent host had no `docker` binary.

```bash
# Tag must be pullable by RunPod (Docker Hub / GHCR / etc.)
make worker-image IMAGE=YOURUSER/robolab-worker:phase3
docker login
docker push YOURUSER/robolab-worker:phase3
```

```bash
ROBOLAB_WORKER_IMAGE=YOURUSER/robolab-worker:phase3
```

### 4. Public backend URL

Pods cannot reach `localhost:8000`. Free option (`cloudflared` is on PATH here):

```bash
make dev   # API on :8000
cloudflared tunnel --url http://127.0.0.1:8000
```

```bash
BACKEND_PUBLIC_URL=https://YOUR-SUBDOMAIN.trycloudflare.com
```

(ngrok also fine: `ngrok http 8000`. Kingdom dashboard tunnel scripts are **not** for RoboLab.)

### 5. Git clone URL + clean push

```bash
ROBOLAB_GIT_URL=https://github.com/navinash47/robolab.git
GITHUB_TOKEN=…   # required while repo is private (repo scope)
```

Launch **refuses** if: dirty working tree, no git remote, or HEAD SHA not on a remote.

### 6. Restart

After editing `.env`: stop `make dev`, start again.

## Prereqs checklist

- [x] `RUNPOD_API_KEY` set (len > 0) — local `.env` (do not commit)
- [ ] `RUNPOD_NETWORK_VOLUME_ID` set (50GB US recommended; you create)
- [ ] `ROBOLAB_WORKER_IMAGE` pushed + set
- [x] git remote + pushed (`origin` → navinash47/robolab) — still set `ROBOLAB_GIT_URL` + `GITHUB_TOKEN` in `.env`
- [ ] `BACKEND_PUBLIC_URL` is HTTPS public (not localhost)
- [ ] `WANDB_API_KEY` / `WANDB_PROJECT` set (already used in Phases 1–2)
- [ ] `BUDGET_USD_CAP` sensible (default 100)
- [ ] `make dev` running → http://localhost:5173 + API :8000
- [ ] Watchdog running (API lifespan starts it; no separate process)

## Offline / free checks (no GPU spend)

```bash
# Watchdog policy unit tests
uv run --package robolab-api python -m pytest robolab_api/tests/test_watchdog.py -q

# Refuse launch without keys (expect HTTP 400 mentioning RUNPOD_API_KEY)
curl -s -X POST http://127.0.0.1:8000/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"compute":"runpod","task":"wall_follow","arch":"mlp","trainer":{"timesteps":2048,"device":"cuda"}}'
```

UI: **New Run → compute runpod** should show GPU + Budget USD fields. Starting without env → error banner (400 text), no pod created.

## Exact clicks — happy path (costs money)

Use **2k (quick)** or **5k (smoke)** first. Keep `BUDGET_USD_CAP` low if nervous.

1. Open http://localhost:5173
2. **New Run** → Compute: **runpod**
3. Arch: **mlp**, Task: **wall_follow**, Timesteps: **2k (quick)**
4. GPU: RTX 4090 (or fallback), Budget USD: **0** (no per-run cap; month cap still applies)
5. **Start**
6. **Expect:**
   - Status `PROVISIONING` with **pod id** under the chip
   - Cost column shows **$/hr** and accrued $
   - RunPod console shows the pod
   - Status → `RUNNING`, progress advances, W&B link
   - Status → `COMPLETE`
   - Pod **disappears** from RunPod console (entrypoint EXIT DELETE)
   - Dashboard shows final accrued cost; budget header month spend increases

## Exact clicks — watchdog kill (`budget_usd: 0.05`)

1. **New Run → runpod**, timesteps **50k** (long enough to accrue), Budget USD: **0.05**
2. **Start**
3. **Expect:** within ~1–2 watchdog ticks after accrual exceeds $0.05 → status `KILLED_BY_WATCHDOG`, pod gone, ledger updated

## Fail modes

| Symptom | Likely cause |
|---|---|
| 400 `RUNPOD_API_KEY is missing` | Empty `.env` key or forgot restart `make dev` |
| 400 `RUNPOD_NETWORK_VOLUME_ID` | No volume / wrong id |
| 400 `BACKEND_PUBLIC_URL` / localhost | Tunnel missing; pods need public URL |
| 400 dirty tree / not on remote | Commit + push before launch |
| 400 monthly budget exhausted | `CostLedger` sum ≥ `BUDGET_USD_CAP` |
| PROVISIONING forever | Image pull fail / DC capacity / volume DC mismatch |
| Pod stays after COMPLETE | Entrypoint missing `RUNPOD_API_KEY` / `RUNPOD_POD_ID`; watchdog should still kill on stale heartbeat |
| MCP list-pods works but app 400 | Expected — MCP OAuth ≠ `RUNPOD_API_KEY` in `.env` |

## Local-only limitation

If this clone has **no git remote**, RunPod launch will always refuse (pod must `git clone` `ROBOLAB_GIT_URL` at `GIT_SHA`). Add a remote, push, set `ROBOLAB_GIT_URL`.

## Gate pass criteria (you confirm)

- [ ] New Run → **runpod** fields visible (GPU, budget)
- [ ] Launch shows `PROVISIONING → RUNNING` with pod id, $/hr, accrued $
- [ ] Pod visible in RunPod console during run
- [ ] On COMPLETE, pod **self-terminates** (gone from console)
- [ ] Final cost on dashboard / budget spend updated
- [ ] Separate run with `budget_usd: 0.05` → `KILLED_BY_WATCHDOG`

Until those boxes are checked with a real key + volume: **not gate-passed.** Do **not** start Phase 4 until this gate is human-confirmed.
