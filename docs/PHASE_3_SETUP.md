# Phase 3 setup — zero-spend progress

Paid gate needs four artifacts in `.env` plus a clean pushed SHA. This doc gets you there **without** launching pods or creating volumes until you choose to.

**Repo remote (already set):** `https://github.com/navinash47/robolab.git`  
→ put that in `.env` as `ROBOLAB_GIT_URL`. Repo is **private** → also set `GITHUB_TOKEN` (classic PAT with `repo` scope) so pods can `git clone`.

## Checklist (cost)

| Step | Costs money? | Status help |
|---|---|---|
| `RUNPOD_API_KEY` in `.env` | No (key alone) | Required before any API create |
| Push code to `origin` | No | Remote exists; keep tree clean + push before each launch |
| Build/push worker image | No (registry free tier OK) | Needs Docker Desktop / Colima locally |
| `BACKEND_PUBLIC_URL` tunnel | No (cloudflared free quick tunnel) | See below |
| Network volume **50 GB** (US DC) | **Yes** (storage $/mo) | **You** create when ready — do not auto-create |
| Launch RunPod GPU pod | **Yes** | Only after all env vars set |

## 1. `.env` keys (never commit `.env`)

Copy from `.env.example`. Minimum for launch:

```bash
RUNPOD_API_KEY=…                    # console → User Settings → API Keys
RUNPOD_NETWORK_VOLUME_ID=…          # after you create the volume (step 5)
ROBOLAB_WORKER_IMAGE=YOURUSER/robolab-worker:phase3
ROBOLAB_GIT_URL=https://github.com/navinash47/robolab.git
GITHUB_TOKEN=…                      # required while repo is private
BACKEND_PUBLIC_URL=https://….trycloudflare.com   # no trailing slash; not localhost
WANDB_API_KEY=…
WANDB_PROJECT=robolab
BUDGET_USD_CAP=100                  # keep low while smoking
```

Restart `make dev` after every `.env` edit.

## 2. Worker image (build + push)

Docker was **not** available in the agent environment (`docker` missing). On your Mac:

```bash
# Install Docker Desktop or: brew install colima docker && colima start
cd /Users/avinashnandyala/Projects/robolab

# Build (tag must match what you push + set in .env)
make worker-image IMAGE=YOURDOCKERHUB/robolab-worker:phase3

# Log in + push (Docker Hub example)
docker login
docker push YOURDOCKERHUB/robolab-worker:phase3
```

**GHCR alternative:**

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u YOUR_GH_USER --password-stdin
make worker-image IMAGE=ghcr.io/navinash47/robolab-worker:phase3
docker push ghcr.io/navinash47/robolab-worker:phase3
# make package public, or use a pull credential RunPod can use
```

Then:

```bash
ROBOLAB_WORKER_IMAGE=YOURDOCKERHUB/robolab-worker:phase3
```

Image must be **pullable by RunPod** (public, or registry auth configured on the account).

## 3. `BACKEND_PUBLIC_URL` — expose `:8000` (free)

Pods heartbeart to the FastAPI API. `localhost:8000` / `127.0.0.1` are refused.

**Recommended (already on this machine: `cloudflared`):**

```bash
# Terminal A — API (and web if you want UI)
make dev

# Terminal B — public HTTPS → local :8000
cloudflared tunnel --url http://127.0.0.1:8000
```

Copy the printed `https://….trycloudflare.com` URL into `.env`:

```bash
BACKEND_PUBLIC_URL=https://YOUR-SUBDOMAIN.trycloudflare.com
```

Restart the API process so it picks up the env. Keep the tunnel process alive for the whole RunPod run.

**ngrok** (if you install it): `ngrok http 8000` → same pattern.

**Kingdom tunnels skill** (`avinashs-kingdom` scripts) is for Kingdom dashboard ports, **not** RoboLab `:8000`. Use raw `cloudflared` here.

## 4. Git hygiene (launch refuse conditions)

Launch **refuses** if dirty tree, no remote, or HEAD SHA not on remote:

```bash
git status          # must be clean
git push origin HEAD
# ROBOLAB_GIT_URL=https://github.com/navinash47/robolab.git
```

## 5. Network volume (you; costs storage) — when ready

**Not created by the agent unless you explicitly ask.** Size tip: **50 GB** recommended (checkpoints + uv cache + clones). Put it in a **US** data center that has your GPU type (e.g. RTX 4090).

- Console: https://www.runpod.io/console/user/storage → create → copy id
- Or Cursor MCP `user-runpod` → `create-network-volume` (only when you request it)

```bash
RUNPOD_NETWORK_VOLUME_ID=…   # mounts at /workspace
```

## 6. Free checks before any paid pod

See `docs/PHASE_3_TEST.md` § Offline / free checks (watchdog unit tests + launch 400 without full env).

## Blunt: can we run the paid gate yet?

**No.** Key may be present, but volume id, pushed worker image, `BACKEND_PUBLIC_URL`, and (for private clone) `GITHUB_TOKEN` are still required. Offline refuse-path / UI field checks: **yes**.
