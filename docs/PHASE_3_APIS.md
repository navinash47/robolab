# Phase 3 APIs (researched before code)

Exact RunPod / Docker / env signatures for Phase 3. **Doc wins** if it contradicts the build prompt — differences flagged below.

Sources (checked 2026-09-04):

- Live OpenAPI: [`https://api.runpod.io/v2/openapi.json`](https://api.runpod.io/v2/openapi.json) — **preferred control-plane shape**
- Cursor MCP `user-runpod` (OAuth) — curated projection of REST v2; **not** RoboLab app auth
- [Agent setup (official)](https://docs.runpod.io/agent-setup.md) — skills + hosted MCP for the coding agent
- [runpod-python `ctl_commands.py`](https://github.com/runpod/runpod-python/blob/main/runpod/api/ctl_commands.py) (`runpod` PyPI ≈ 1.12.x) — GraphQL wrappers still used for create/list/price in-app
- [Manage Pods (REST)](https://docs.runpod.io/pods/manage-pods) — still documents `rest.runpod.io/v1` in places
- [Network volumes](https://docs.runpod.io/storage/network-volumes)

## Conflict: `agent-setup.md` vs Phase 3 RoboLab app

Official [agent-setup.md](https://docs.runpod.io/agent-setup.md) is **agent onboarding only** (install Runpod skills + wire `https://mcp.getrunpod.io/`). It explicitly says:

- Do **not** install `runpodctl` / Flash / API keys during setup — skills install those later on demand.
- MCP auth is **OAuth** (“Sign in with Runpod”); no API key created or stored by that flow.
- Prefer MCP tools for control-plane CRUD once connected; MCP drives **REST v2** (`api.runpod.io`), not hand-rolled `rest.runpod.io/v1`.

| Concern | Prefer | Why |
|---|---|---|
| Cursor/agent infra tools | Hosted MCP + skills per agent-setup | Official agent path |
| RoboLab backend `create_pod` / watchdog / entrypoint self-terminate | `RUNPOD_API_KEY` in project `.env` + SDK create + **REST v2 DELETE** | App runs headlessly outside the IDE; OAuth MCP cannot replace process env |
| Entrypoint self-kill | `DELETE https://api.runpod.io/v2/pods/$ID` (v1 fallback) | Docs/OpenAPI: terminate is irreversible delete; v2 is the live contract |

**Phase 3 still requires `RUNPOD_API_KEY` (and related) in `.env` for the FastAPI runner** — that is separate from agent MCP OAuth and is **not** set up by agent-setup. Do not commit those secrets.

## Auth

```python
import os
import runpod

runpod.api_key = os.environ["RUNPOD_API_KEY"]
# SDK also reads RUNPOD_API_KEY from the environment if set.
```

REST (self-terminate / DELETE / any v2 call):

```http
Authorization: Bearer <RUNPOD_API_KEY>
```

| API | Base URL | Used for |
|---|---|---|
| **REST v2 (preferred)** | `https://api.runpod.io/v2` | Terminate DELETE; OpenAPI truth; MCP |
| REST v1 (legacy docs) | `https://rest.runpod.io/v1` | Entrypoint fallback only |
| GraphQL via `runpod-python` | SDK internals | App `create_pod` / `get_pods` / `get_gpu` today |

## REST v2 create pod (preferred shape — OpenAPI `CreatePodRequest`)

MCP `create-pod` and older prompt/docs use camelCase (`imageName`, `gpuTypeIds`, `networkVolumeId`). **Live v2** uses nested `gpu` / `mounts` / `image` / `cloud`:

```bash
curl -sS -X POST "https://api.runpod.io/v2/pods" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "robolab-<run_id>",
    "image": "'"$ROBOLAB_WORKER_IMAGE"'",
    "cloud": "COMMUNITY",
    "gpu": { "id": "NVIDIA GeForce RTX 4090", "count": 1 },
    "disk": 30,
    "mounts": {
      "network": [
        { "volumeId": "'"$RUNPOD_NETWORK_VOLUME_ID"'", "path": "/workspace" }
      ]
    },
    "env": {
      "RUN_ID": "<run_id>",
      "GIT_SHA": "<sha>",
      "CONFIG_B64": "<…>",
      "WANDB_API_KEY": "…",
      "WANDB_PROJECT": "robolab",
      "RUNPOD_API_KEY": "…",
      "BACKEND_URL": "https://…",
      "MAX_RUNTIME_MIN": "120",
      "ROBOLAB_GIT_URL": "https://github.com/…/robolab.git"
    }
  }'
```

Network volume create (v2 / MCP):

```bash
# OpenAPI: POST /v2/network-volumes  { name, size, dataCenter, type? }
# MCP: create-network-volume(name, size, dataCenterId, volumeType?)
```

### Flags vs build prompt / MCP camelCase

| Prompt / MCP assumption | Actual REST v2 OpenAPI | What we do |
|---|---|---|
| `POST rest.runpod.io/v1/pods` + `imageName` / `gpuTypeIds` / `networkVolumeId` | `POST api.runpod.io/v2/pods` + `image` / `gpu.id` / `mounts.network[].volumeId` | Document v2 as truth; app still creates via **SDK GraphQL** (stable today) |
| `pod_id = runpod.create_pod(...)` as if string | SDK returns **dict** with `id` | Use `pod["id"]` |
| Mount at `/workspace` | v2 `path` has **no default** — must set explicitly; SDK default `/runpod-volume` | Pass `/workspace` |
| `cloudType: COMMUNITY` | v2 field is `cloud` (default `SECURE`) | Pass COMMUNITY via SDK `cloud_type` / env `RUNPOD_CLOUD_TYPE` |
| Entrypoint `DELETE rest.runpod.io/v1/pods/$ID` | Prefer `DELETE api.runpod.io/v2/pods/$ID` | Entrypoint tries v2 first, then v1 |
| MCP OAuth enough for app | MCP ≠ process env | Refuse launch without `.env` `RUNPOD_API_KEY` |

## `runpod.create_pod` (Python SDK — current app launch path)

Exact signature from current `runpod-python` source (GraphQL under the hood — **not** the v2 JSON body above):

```python
def create_pod(
    name: str,
    image_name: Optional[str] = "",
    gpu_type_id: Optional[str] = None,
    cloud_type: str = "ALL",                    # "ALL" | "COMMUNITY" | "SECURE"
    support_public_ip: bool = True,
    start_ssh: bool = True,
    data_center_id: Optional[str] = None,
    country_code: Optional[str] = None,
    gpu_count: int = 1,
    volume_in_gb: int = 0,
    container_disk_in_gb: Optional[int] = None, # default 10 if no template
    min_vcpu_count: int = 1,
    min_memory_in_gb: int = 1,
    docker_args: str = "",
    ports: Optional[str] = None,                # e.g. "8888/http,22/tcp"
    volume_mount_path: str = "/runpod-volume",  # ← pass "/workspace"
    env: Optional[dict] = None,                 # {KEY: "value", ...}
    template_id: Optional[str] = None,
    network_volume_id: Optional[str] = None,
    allowed_cuda_versions: Optional[list] = None,
    min_download=None,
    min_upload=None,
    instance_id: Optional[str] = None,          # CPU pods
) -> dict:
    """Returns GraphQL pod object (includes at least `id`), NOT a bare string."""
```

RoboLab Phase 3 call shape:

```python
pod = runpod.create_pod(
    name=f"robolab-{run_id}",
    image_name=os.environ["ROBOLAB_WORKER_IMAGE"],
    gpu_type_id=gpu_type_id,           # default "NVIDIA GeForce RTX 4090"
    cloud_type="COMMUNITY",            # override via RUNPOD_CLOUD_TYPE
    gpu_count=1,
    volume_in_gb=0,                    # network volume replaces local volume
    container_disk_in_gb=30,
    volume_mount_path="/workspace",    # explicit (not SDK default)
    network_volume_id=os.environ["RUNPOD_NETWORK_VOLUME_ID"],
    env={
        "RUN_ID": run_id,
        "GIT_SHA": git_sha,
        "CONFIG_B64": config_b64,
        "WANDB_API_KEY": ...,
        "WANDB_PROJECT": ...,
        "RUNPOD_API_KEY": ...,         # needed for EXIT self-terminate
        "BACKEND_URL": backend_public_url,
        "MAX_RUNTIME_MIN": str(max_runtime_min),
        "ROBOLAB_GIT_URL": git_url,
        # optional: GITHUB_TOKEN, WANDB_ENTITY
    },
)
pod_id = pod["id"]
```

Fallback GPU ids if preferred unavailable / create fails: `NVIDIA GeForce RTX 3090`, `NVIDIA RTX A4000` (exact `id` strings from `get_gpus()`).

## List / get / terminate / prices

```python
runpod.get_gpus() -> list[dict]   # id, displayName, memoryInGb
runpod.get_gpu(gpu_id: str, gpu_quantity: int = 1) -> dict
# includes communityPrice, securePrice, lowestPrice.uninterruptablePrice, …

runpod.get_pods() -> list[dict]   # id, costPerHr, desiredStatus, env, uptimeSeconds, …
runpod.get_pod(pod_id: str) -> dict

runpod.stop_pod(pod_id: str) -> dict
runpod.terminate_pod(pod_id: str) -> None   # GraphQL; app prefers REST v2 DELETE first
```

Hourly rate for ledger: prefer `pod["costPerHr"]` after create/get; else `get_gpu(...).get("communityPrice")` (COMMUNITY) or `securePrice` (SECURE).

## REST terminate (entrypoint EXIT trap + watchdog)

**Preferred (OpenAPI `DELETE /v2/pods/{id}`):**

```bash
curl -sS -X DELETE \
  "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

Legacy fallback (Manage Pods docs / build prompt):

```bash
curl -sS -X DELETE \
  "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

`RUNPOD_POD_ID` is injected by the RunPod runtime into the container environment.

Backend watchdog: `terminate_pod()` tries REST v2 DELETE, then SDK GraphQL.

## Worker image + entrypoint

- Base: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Build: `make worker-image IMAGE=user/robolab-worker:phase3` then `docker push …`
- Image must be **pushed** to a registry RunPod can pull (`ROBOLAB_WORKER_IMAGE`)
- `entrypoint.sh`: clone `ROBOLAB_GIT_URL` at `GIT_SHA` → `uv sync` → train → **always** REST DELETE on EXIT

## Env vars (Phase 3)

| Var | Required for runpod launch | Meaning |
|---|---|---|
| `RUNPOD_API_KEY` | yes | Account API key (≠ MCP OAuth) |
| `RUNPOD_NETWORK_VOLUME_ID` | yes | Persistent `/workspace` volume id |
| `ROBOLAB_WORKER_IMAGE` | yes | Image RunPod pulls (e.g. `user/robolab-worker:phase3`) |
| `ROBOLAB_GIT_URL` | yes | Clone URL for `GIT_SHA` checkout on the pod |
| `BACKEND_PUBLIC_URL` | yes | Public base URL pods use for heartbeats (not `localhost`) |
| `BUDGET_USD_CAP` | yes (default 100) | Refuse launches when month ledger ≥ cap |
| `MAX_RUNTIME_MIN` | no (default 120) | Watchdog + pod env hard wall-clock |
| `RUNPOD_CLOUD_TYPE` | no (default **SECURE**) | `COMMUNITY` \| `SECURE` \| `ALL` — EU-RO-1 + network volume often refuses Community; app falls back Community↔Secure |
| `WANDB_API_KEY` / `WANDB_PROJECT` | yes | Passed into pod env |
| `GITHUB_TOKEN` | if private repo | Optional clone auth |
| `BACKEND_URL` | on pod | Set from `BACKEND_PUBLIC_URL` at launch |

## Git gate before launch

Resolve a SHA the pod can `git clone` from `ROBOLAB_GIT_URL`. **Dirty working tree does not refuse.**

| Condition | Behavior |
|---|---|
| `GIT_SHA` env set | Pin that SHA (must exist on a remote) |
| Clean tree + HEAD on remote | Use HEAD (reproducible) |
| Dirty tree | Warn + Failure Resolution logistics note; use pushed HEAD or `origin/main` |
| Unpushed HEAD (clean or dirty) | Fall back to `origin/main` (or other remote tip) with warning |
| No git remote / no resolvable remote tip | Refuse |

Pods never see uncommitted local edits — commit+push when you need the pod to run those changes.

## Watchdog rules (backend)

Every 60s (and orphan sweep on startup):

1. Unknown `RUN_ID` on a live pod → terminate + `KILLED_BY_WATCHDOG`
2. Heartbeat `updated_at` stale **> 10 minutes** for **N consecutive** sweeps (`WATCHDOG_CONSECUTIVE_MISSES`, default **3**) → confirm via RunPod `get_pod`: already EXITED/TERMINATED/MISSING → mark `FAILED` (logistics, **no** terminate); still `RUNNING` → warn + extend grace (**no** kill). Budget / `MAX_RUNTIME_MIN` still kill immediately.
3. Runtime **> `MAX_RUNTIME_MIN`** → terminate + kill label
4. Accrued cost **> run `budget_usd`** (when `budget_usd > 0`) → terminate + kill label
5. Month `CostLedger` sum ≥ `BUDGET_USD_CAP` → **refuse new launches**
6. DB pod missing from `list_pods` → same N-miss + `get_pod` confirm before `pod_vanished` FAILED (avoids false orphans from a flaky list tick)
7. Launch preflight: `GET $BACKEND_PUBLIC_URL/health` must be 200 (refuse + logistics); if `/tmp/robolab-cloudflared.log` shows a different tunnel and health fails, error includes that hint

## Status machine (Phase 3 addition)

```text
QUEUED → PROVISIONING → RUNNING → COMPLETE | FAILED | KILLED_BY_WATCHDOG
```

`PROVISIONING` is added to `RunStatus` for the human gate (pod id visible while machine starts).
