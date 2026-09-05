# Phase 3 APIs (researched before code)

Exact RunPod / Docker / env signatures for Phase 3. **Doc wins** if it contradicts the build prompt — differences flagged below.

Sources (checked 2026-09-04):

- [runpod-python `ctl_commands.py`](https://github.com/runpod/runpod-python/blob/main/runpod/api/ctl_commands.py) (`runpod` PyPI ≈ 1.12.x)
- [Manage Pods (REST)](https://docs.runpod.io/pods/manage-pods)
- [Create Pod REST `POST /pods`](https://docs.runpod.io/api-reference/pods/POST/pods)
- [Network volumes](https://docs.runpod.io/storage/network-volumes)
- GPU price fields via SDK GraphQL `get_gpu` / `get_gpus`

## Auth

```python
import os
import runpod

runpod.api_key = os.environ["RUNPOD_API_KEY"]
# SDK also reads RUNPOD_API_KEY from the environment if set.
```

REST (self-terminate / DELETE):

```http
Authorization: Bearer <RUNPOD_API_KEY>
```

Base URL: `https://rest.runpod.io/v1`

## `runpod.create_pod` (Python SDK — primary launch path)

Exact signature from current `runpod-python` source:

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
    volume_mount_path: str = "/runpod-volume",  # ← see flag below
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

### Flags vs build prompt

| Prompt assumption | Actual docs / SDK | What we do |
|---|---|---|
| `pod_id = runpod.create_pod(...)` as if string | Returns **dict** with `id` | Use `pod["id"]` |
| Mount at `/workspace` | SDK default `volume_mount_path="/runpod-volume"` | Pass `volume_mount_path="/workspace"` explicitly |
| `cloud_type="COMMUNITY"` | SDK default `"ALL"`; REST CreatePod default `SECURE` | Pass `"COMMUNITY"` (or `RUNPOD_CLOUD_TYPE`) |
| REST body camelCase (`gpuTypeIds`, …) | SDK snake_case wrappers over GraphQL | Prefer **SDK** for create/list/get/price; REST DELETE for self-terminate (matches prompt) |
| Network volume auto | If `network_volume_id` set and `data_center_id` is None, SDK looks up volume’s data center from `get_user()` | Rely on SDK behavior |

## List / get / terminate / prices

```python
runpod.get_gpus() -> list[dict]   # id, displayName, memoryInGb
runpod.get_gpu(gpu_id: str, gpu_quantity: int = 1) -> dict
# includes communityPrice, securePrice, lowestPrice.uninterruptablePrice, …

runpod.get_pods() -> list[dict]   # id, costPerHr, desiredStatus, env, uptimeSeconds, …
runpod.get_pod(pod_id: str) -> dict

runpod.stop_pod(pod_id: str) -> dict
runpod.terminate_pod(pod_id: str) -> None   # GraphQL terminate; no useful return
```

Hourly rate for ledger: prefer `pod["costPerHr"]` after create/get; else `get_gpu(...).get("communityPrice")` (COMMUNITY) or `securePrice` (SECURE).

## REST terminate (entrypoint EXIT trap)

Matches [Manage Pods → Terminate](https://docs.runpod.io/pods/manage-pods):

```bash
curl -sS -X DELETE \
  "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

`RUNPOD_POD_ID` is injected by the RunPod runtime into the container environment.

Equivalent from backend watchdog: `runpod.terminate_pod(pod_id)`.

## REST create (reference only — not primary)

```bash
curl --request POST \
  --url https://rest.runpod.io/v1/pods \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  --header "Content-Type: application/json" \
  --data '{
    "name": "my-pod",
    "imageName": "runpod/pytorch:…",
    "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
    "gpuCount": 1,
    "containerDiskInGb": 50,
    "networkVolumeId": "…",
    "volumeMountPath": "/workspace",
    "cloudType": "COMMUNITY",
    "env": { "RUN_ID": "…" }
  }'
```

## Worker image + entrypoint

- Base: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (matches existing stub)
- Image must be **pushed** to a registry RunPod can pull (`ROBOLAB_WORKER_IMAGE`)
- `entrypoint.sh`: clone `ROBOLAB_GIT_URL` at `GIT_SHA` → `uv sync` → train → **always** REST DELETE on EXIT

## Env vars (Phase 3)

| Var | Required for runpod launch | Meaning |
|---|---|---|
| `RUNPOD_API_KEY` | yes | Account API key |
| `RUNPOD_NETWORK_VOLUME_ID` | yes | Persistent `/workspace` volume id |
| `ROBOLAB_WORKER_IMAGE` | yes | Image RunPod pulls (e.g. `user/robolab-worker:phase3`) |
| `ROBOLAB_GIT_URL` | yes | Clone URL for `GIT_SHA` checkout on the pod |
| `BACKEND_PUBLIC_URL` | yes | Public base URL pods use for heartbeats (not `localhost`) |
| `BUDGET_USD_CAP` | yes (default 100) | Refuse launches when month ledger ≥ cap |
| `MAX_RUNTIME_MIN` | no (default 120) | Watchdog + pod env hard wall-clock |
| `RUNPOD_CLOUD_TYPE` | no (default COMMUNITY) | `COMMUNITY` \| `SECURE` \| `ALL` |
| `WANDB_API_KEY` / `WANDB_PROJECT` | yes | Passed into pod env |
| `GITHUB_TOKEN` | if private repo | Optional clone auth |
| `BACKEND_URL` | on pod | Set from `BACKEND_PUBLIC_URL` at launch |

## Git gate before launch

Prompt: resolve current git SHA; **must be clean and pushed**; refuse otherwise.

- Dirty tree → refuse
- No `git remote` → refuse (local-only repo cannot be cloned on the pod) — document in `PHASE_3_TEST.md`
- SHA not on any remote → refuse (not pushed)

Override is **not** implemented (cost safety > convenience).

## Watchdog rules (backend)

Every 60s (and orphan sweep on startup):

1. Unknown `RUN_ID` on a live pod → terminate + `KILLED_BY_WATCHDOG`
2. Heartbeat `updated_at` stale **> 10 minutes** → terminate + kill label
3. Runtime **> `MAX_RUNTIME_MIN`** → terminate + kill label
4. Accrued cost **> run `budget_usd`** (when `budget_usd > 0`) → terminate + kill label
5. Month `CostLedger` sum ≥ `BUDGET_USD_CAP` → **refuse new launches**

## Status machine (Phase 3 addition)

```text
QUEUED → PROVISIONING → RUNNING → COMPLETE | FAILED | KILLED_BY_WATCHDOG
```

`PROVISIONING` is added to `RunStatus` for the human gate (pod id visible while machine starts).
