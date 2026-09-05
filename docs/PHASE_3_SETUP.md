# Phase 3 setup — zero-spend progress

**Blunt status (2026-09-04 evening):** Worker image **built locally** and `ROBOLAB_WORKER_IMAGE` is set in `.env`. Tunnel + EU volume + tokens already done. **Blocked only on Docker Hub login + push** — agent has no Hub credentials; you must run `docker login` in your terminal, then `docker push navinash47/robolab-worker:phase3`. Do **not** launch paid pods until push succeeds.

Paid gate needs the artifacts below in `.env` plus a clean pushed SHA. This doc gets you there **without** launching pods until you choose to.

**Repo remote (already set):** `https://github.com/navinash47/robolab.git`  
→ `ROBOLAB_GIT_URL`. Repo is **private** → `GITHUB_TOKEN` (classic PAT / `gh auth token` with `repo` scope) so pods can `git clone`.

## Checklist (cost) — live status

| Step | Costs money? | Status |
|---|---|---|
| `RUNPOD_API_KEY` in `.env` | No (key alone) | **Done** |
| `RUNPOD_NETWORK_VOLUME_ID=1hyuaan8i2` | **Yes** (~$2.80/mo) | **Done** — volume `robolab-workspace`, **EU-RO-1**, 40 GB |
| `RUNPOD_DATA_CENTER_ID=EU-RO-1` | No | **Done** — pods must launch in same DC as volume |
| Push code to `origin` | No | Remote exists; keep tree clean + push before each launch |
| Build worker image (`linux/amd64`) | No | **Done** — local `navinash47/robolab-worker:phase3` (~7.46GB, sha `5b8a9448ea16`) |
| Push worker image | No (registry free tier OK) | **Human blocker:** not logged in to Docker Hub (no `~/.docker` auths / no `DOCKERHUB_TOKEN`). Run `docker login` then `docker push navinash47/robolab-worker:phase3` |
| `BACKEND_PUBLIC_URL` tunnel | No | cloudflared quick tunnel → set in `.env` |
| `GITHUB_TOKEN` | No | **Done** (via `gh auth token`) |
| Unused US volume `ljecesg9a8` | was storage $/mo | **Deleted** via MCP (user chose EU volume) |
| Launch RunPod GPU pod | **Yes** | Only after image is **pushed** + tunnel alive |

## 1. `.env` keys (never commit `.env`)

Copy from `.env.example`. Minimum for launch:

```bash
RUNPOD_API_KEY=…                    # console → User Settings → API Keys
RUNPOD_NETWORK_VOLUME_ID=1hyuaan8i2  # robolab-workspace @ EU-RO-1
RUNPOD_DATA_CENTER_ID=EU-RO-1       # MUST match volume DC (create_pod uses this)
ROBOLAB_WORKER_IMAGE=navinash47/robolab-worker:phase3
ROBOLAB_GIT_URL=https://github.com/navinash47/robolab.git
GITHUB_TOKEN=…                      # required while repo is private
BACKEND_PUBLIC_URL=https://….trycloudflare.com   # no trailing slash; not localhost
WANDB_API_KEY=…
WANDB_PROJECT=robolab
BUDGET_USD_CAP=100                  # keep low while smoking
```

Restart `make dev` after every `.env` edit.

### Critical: volume DC = pod DC

Network volumes are **data-center scoped**. The active volume is in **EU-RO-1**.  
`create_training_pod` passes `data_center_id` from `RUNPOD_DATA_CENTER_ID` (default `EU-RO-1`).  
If you pick a GPU that has no stock in EU-RO-1, create fails — change GPU type or create a volume in another DC (and update both env vars).

## 2. Worker image (build + push)

Colima + Docker CLI are installed. Build is **amd64** (RunPod GPUs):

```bash
colima start   # if needed
cd /Users/avinashnandyala/Projects/robolab
make worker-image IMAGE=navinash47/robolab-worker:phase3
# Human gate:
docker login
docker push navinash47/robolab-worker:phase3
```

Then:

```bash
ROBOLAB_WORKER_IMAGE=navinash47/robolab-worker:phase3
```

Image must be **pullable by RunPod** (public Docker Hub, or registry auth on the account). Local-only build is **not** enough.

**GHCR alternative:**

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u navinash47 --password-stdin
make worker-image IMAGE=ghcr.io/navinash47/robolab-worker:phase3
docker push ghcr.io/navinash47/robolab-worker:phase3
```

## 3. `BACKEND_PUBLIC_URL` — expose `:8000` (free)

Pods heartbeat to the FastAPI API. `localhost:8000` / `127.0.0.1` are refused.

```bash
make dev
cloudflared tunnel --url http://127.0.0.1:8000
```

Copy the printed `https://….trycloudflare.com` URL into `.env` (no trailing slash). Restart API. Keep the tunnel alive for the whole RunPod run.

## 4. Git hygiene (launch refuse conditions)

```bash
git status          # must be clean
git push origin HEAD
```

## 5. Network volume (active)

| Field | Value |
|---|---|
| Name | `robolab-workspace` |
| ID | `1hyuaan8i2` |
| Size | 40 GB |
| DC | **EU-RO-1** |
| Mount | `/workspace` |

Earlier US-IL-1 volume `ljecesg9a8` was deleted after you chose the EU volume (avoid double storage billing).

## 6. Free checks before any paid pod

See `docs/PHASE_3_TEST.md` § Offline / free checks.

## Blunt: can we run the paid gate yet?

**Not until** `navinash47/robolab-worker:phase3` is **pushed** to a registry RunPod can pull, and `BACKEND_PUBLIC_URL` tunnel is alive + `make dev` restarted with current `.env`. Volume + API key + `GITHUB_TOKEN` + DC preference: **done**.
