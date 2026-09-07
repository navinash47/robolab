"""RunPod compute runner: create / poll / terminate / hourly rate.

Signatures follow current runpod-python + REST docs (see docs/PHASE_3_APIS.md).
"""

from __future__ import annotations

import base64
import logging
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from robolab.compute.local import repo_root, write_run_config
from robolab.core.run import RunConfig

logger = logging.getLogger("robolab.runpod")

_TUNNEL_URL_RE = re.compile(
    r"https://[a-z0-9-]+\.(?:trycloudflare\.com|lhr\.life|loca\.lt|ngrok-free\.app|ngrok\.io)",
    re.IGNORECASE,
)
_CF_LOG_CANDIDATES = (
    Path("/tmp/robolab-cloudflared.log"),
    Path("/tmp/cloudflared.log"),
)

# Prefer mid-tier GPUs that are usually stocked in EU-RO-1 with network volumes.
# Keep this list cost-safe (no H100/A100/H200 auto-fallback).
# Order used for UI "Best available" (gpu_type=best) and null/empty gpu_type.
DEFAULT_GPU_FALLBACKS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 3070",
    "NVIDIA RTX A4000",
    "NVIDIA RTX A4500",
    "NVIDIA RTX 4000 Ada Generation",
]

# Sent from New Run UI when user picks "Best available".
BEST_AVAILABLE_GPU = "best"
_BEST_AVAILABLE_ALIASES = frozenset(
    {BEST_AVAILABLE_GPU, "auto", "any", "best_available"}
)


def _is_best_available(gpu_type: str | None) -> bool:
    """True when create should walk DEFAULT_GPU_FALLBACKS instead of one GPU."""
    if gpu_type is None:
        return True
    key = gpu_type.strip().lower()
    return not key or key in _BEST_AVAILABLE_ALIASES


def _cloud_type_candidates(preferred: str) -> list[str]:
    """Order cloud types to try. EU-RO-1 + network volume often has Secure stock only."""
    preferred = preferred.strip().upper()
    if preferred == "ALL":
        return ["ALL"]
    if preferred == "COMMUNITY":
        # Community frequently refuses with volume attached; fall back to Secure.
        return ["COMMUNITY", "SECURE"]
    if preferred == "SECURE":
        return ["SECURE", "COMMUNITY"]
    return ["SECURE", "COMMUNITY"]


class RunPodConfigError(RuntimeError):
    """Missing keys / git gate / budget — safe to surface as HTTP 400."""


def _require_env(name: str) -> str:
    val = (os.environ.get(name) or "").strip()
    if not val:
        raise RunPodConfigError(
            f"{name} is missing. Set it in .env and restart `make dev`. "
            "See docs/PHASE_3_TEST.md for setup steps."
        )
    return val


def require_runpod_launch_env() -> dict[str, str]:
    """Fail fast before any paid API call."""
    return {
        "RUNPOD_API_KEY": _require_env("RUNPOD_API_KEY"),
        "RUNPOD_NETWORK_VOLUME_ID": _require_env("RUNPOD_NETWORK_VOLUME_ID"),
        "ROBOLAB_WORKER_IMAGE": _require_env("ROBOLAB_WORKER_IMAGE"),
        "ROBOLAB_GIT_URL": _require_env("ROBOLAB_GIT_URL"),
        "BACKEND_PUBLIC_URL": _require_env("BACKEND_PUBLIC_URL"),
    }


def _latest_tunnel_url_from_logs() -> str | None:
    """Best-effort: pick the newest trycloudflare/lhr URL from keep_dev_alive logs."""
    found: list[tuple[float, str]] = []
    for path in _CF_LOG_CANDIDATES:
        try:
            if not path.is_file():
                continue
            text = path.read_text(errors="ignore")
            matches = _TUNNEL_URL_RE.findall(text)
            if not matches:
                continue
            found.append((path.stat().st_mtime, matches[-1].rstrip("/")))
        except OSError:
            continue
    if not found:
        return None
    found.sort(key=lambda x: x[0], reverse=True)
    return found[0][1]


def preflight_backend_public_url(
    public_url: str | None = None,
    *,
    timeout_sec: float = 8.0,
) -> str:
    """Verify BACKEND_PUBLIC_URL /health is reachable before creating a paid pod.

    Also warns when /tmp/robolab-cloudflared.log shows a newer tunnel URL than .env
    (stale trycloudflare after cloudflared restart — see tmp/keep_dev_alive.sh).
    Returns the normalized base URL on success; raises RunPodConfigError otherwise.
    """
    public = (public_url or os.environ.get("BACKEND_PUBLIC_URL") or "").strip().rstrip("/")
    if not public:
        raise RunPodConfigError(
            "BACKEND_PUBLIC_URL is missing. Start cloudflared "
            "(`cloudflared tunnel --url http://127.0.0.1:8000` or tmp/keep_dev_alive.sh), "
            "set BACKEND_PUBLIC_URL in .env, restart `make dev`."
        )
    lower = public.lower()
    if "localhost" in lower or "127.0.0.1" in lower:
        raise RunPodConfigError(
            "BACKEND_PUBLIC_URL must be reachable from RunPod (not localhost). "
            "Use ngrok or Cloudflare Tunnel and set BACKEND_PUBLIC_URL in .env."
        )
    if not lower.startswith("https://") and not lower.startswith("http://"):
        raise RunPodConfigError(
            f"BACKEND_PUBLIC_URL must be an http(s) URL, got {public!r}."
        )

    live_tunnel = _latest_tunnel_url_from_logs()
    if live_tunnel and live_tunnel.rstrip("/") != public:
        stale_hint = (
            f" Hint: cloudflared log shows a different tunnel ({live_tunnel}) — "
            "update BACKEND_PUBLIC_URL in .env and restart the API "
            "(or re-run tmp/keep_dev_alive.sh) if heartbeats fail."
        )
    else:
        stale_hint = ""

    health = f"{public}/health"
    req = urllib.request.Request(health, method="GET", headers={"User-Agent": "robolab-preflight"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            body = resp.read(256)
    except urllib.error.HTTPError as exc:
        raise RunPodConfigError(
            f"BACKEND_PUBLIC_URL preflight failed: GET {health} → HTTP {exc.code}. "
            "Pods cannot heartbeat if the public tunnel is down."
            f"{stale_hint}"
        ) from exc
    except Exception as exc:
        raise RunPodConfigError(
            f"BACKEND_PUBLIC_URL preflight failed: GET {health} unreachable ({exc}). "
            "Keep `make dev` + cloudflared alive; refresh BACKEND_PUBLIC_URL if the "
            f"tunnel rotated.{stale_hint}"
        ) from exc

    if int(code) != 200:
        raise RunPodConfigError(
            f"BACKEND_PUBLIC_URL preflight failed: GET {health} → HTTP {code} "
            f"(body={body[:80]!r}). Expected 200.{stale_hint}"
        )

    if live_tunnel and live_tunnel.rstrip("/") != public:
        logger.warning(
            "BACKEND_PUBLIC_URL=%s differs from cloudflared log %s (health OK — launching)",
            public,
            live_tunnel,
        )
    return public


# Sims that cannot reliably render on the Mac/API host (optional heavy deps / GPU).
REMOTE_RENDER_SIMS = frozenset({"genesis", "isaaclab", "isaac_sim"})


def needs_remote_render(sim: str | None) -> bool:
    """True when playback must run on a RunPod worker (not the local API host)."""
    return (sim or "").strip().lower() in REMOTE_RENDER_SIMS


def worker_image_for_sim(sim: str, default_image: str | None = None) -> str:
    """Pick worker image by sim. Genesis/Isaac prefer dedicated tags when set."""
    default = (default_image or os.environ.get("ROBOLAB_WORKER_IMAGE") or "").strip()
    if not default:
        raise RunPodConfigError("ROBOLAB_WORKER_IMAGE is missing.")
    sim_key = (sim or "").strip().lower()
    if sim_key == "genesis":
        override = (os.environ.get("ROBOLAB_WORKER_IMAGE_GENESIS") or "").strip()
        if override:
            return override
        if default.endswith(":phase3"):
            return default[: -len("phase3")] + "genesis"
        return default
    if sim_key in {"isaaclab", "isaac_sim"}:
        override = (os.environ.get("ROBOLAB_WORKER_IMAGE_ISAAC") or "").strip()
        if override:
            return override
        if default.endswith(":phase3"):
            return default[: -len("phase3")] + "isaac"
        return default
    return default


def _configure_sdk(api_key: str | None = None) -> Any:
    import runpod

    key = (api_key or os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not key:
        raise RunPodConfigError(
            "RUNPOD_API_KEY is missing. Get a key at "
            "https://www.runpod.io/console/user/settings and set it in .env."
        )
    runpod.api_key = key
    return runpod


def _git(args: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _sha_on_any_remote(sha: str, remotes: list[str], root: Path) -> bool:
    for remote in remotes:
        ls = _git(["ls-remote", remote], root)
        if ls.returncode == 0 and sha in ls.stdout:
            return True
    return False


def _remote_fallback_sha(root: Path, remotes: list[str]) -> str | None:
    """Prefer origin/main (or master), then upstream tip, then any local remote-tracking tip on a remote."""
    candidates: list[str] = []
    if "origin" in remotes:
        candidates.extend(["origin/main", "origin/master"])
    for remote in remotes:
        if remote != "origin":
            candidates.extend([f"{remote}/main", f"{remote}/master"])
    upstream = _git(["rev-parse", "--abbrev-ref", "@{u}"], root)
    if upstream.returncode == 0 and upstream.stdout.strip():
        candidates.append(upstream.stdout.strip())

    seen: set[str] = set()
    for ref in candidates:
        if ref in seen:
            continue
        seen.add(ref)
        tip = _git(["rev-parse", ref], root)
        if tip.returncode != 0:
            continue
        sha = tip.stdout.strip()
        if len(sha) >= 7 and _sha_on_any_remote(sha, remotes, root):
            return sha
    return None


def _note_dirty_git_fallback(sha: str, reason: str) -> None:
    """Warn + best-effort Failure Resolution logistics note (API may be absent)."""
    msg = (
        f"Working tree dirty or HEAD not on remote ({reason}). "
        f"Launching with remote SHA {sha[:12]} instead of local HEAD. "
        "Uncommitted local changes will not be on the pod."
    )
    logger.warning(msg)
    try:
        from sqlmodel import Session

        from robolab_api.db import engine
        from robolab_api.failures import CATEGORY_LOGISTICS, record_failure

        with Session(engine) as session:
            record_failure(
                session,
                category=CATEGORY_LOGISTICS,
                title="git gate: dirty tree → remote SHA fallback",
                reason=msg,
                run_id=None,
                suggested_fix=(
                    "Optional: commit + push so launches pin your exact HEAD. "
                    "Dirty trees no longer block; pods clone the remote fallback SHA."
                ),
                source="auto",
                dedupe_auto=False,
            )
            session.commit()
    except Exception:
        # Compute layer / unit tests may run without the API package or DB.
        pass


def resolve_git_sha(root: Path | None = None) -> str:
    """Return a git SHA the pod can clone from ROBOLAB_GIT_URL.

    Resolution order:
    1. ``GIT_SHA`` env override (explicit pin; must exist on a remote)
    2. Clean tree + HEAD on a remote → HEAD (reproducible)
    3. Dirty tree and/or unpushed HEAD → ``origin/main`` (or other remote tip),
       with a warning — **never refuse solely because the tree is dirty**
    4. No remote / no resolvable remote tip → refuse
    """
    root = root or repo_root()

    remotes_proc = _git(["remote"], root)
    if remotes_proc.returncode != 0 or not remotes_proc.stdout.strip():
        raise RunPodConfigError(
            "No git remote configured. RunPod workers `git clone` ROBOLAB_GIT_URL "
            "at GIT_SHA — add a remote, push, set ROBOLAB_GIT_URL in .env. "
            "See docs/PHASE_3_TEST.md (local-only limitation)."
        )
    remotes = remotes_proc.stdout.split()

    override = (os.environ.get("GIT_SHA") or "").strip()
    if override:
        if len(override) < 7:
            raise RunPodConfigError("GIT_SHA env override is too short to be a commit SHA.")
        if not _sha_on_any_remote(override, remotes, root):
            raise RunPodConfigError(
                f"GIT_SHA override {override[:12]} is not on any remote. "
                "Push that commit (or unset GIT_SHA) before RunPod launch."
            )
        return override

    porcelain = _git(["status", "--porcelain"], root)
    if porcelain.returncode != 0:
        raise RunPodConfigError(f"git status failed: {porcelain.stderr.strip()}")
    dirty = bool(porcelain.stdout.strip())

    sha_proc = _git(["rev-parse", "HEAD"], root)
    if sha_proc.returncode != 0:
        raise RunPodConfigError(f"git rev-parse failed: {sha_proc.stderr.strip()}")
    head = sha_proc.stdout.strip()
    if len(head) < 7:
        raise RunPodConfigError("Could not resolve git HEAD SHA.")

    head_on_remote = _sha_on_any_remote(head, remotes, root)

    # Clean + pushed HEAD → pin exact local checkout for reproducibility.
    if not dirty and head_on_remote:
        return head

    # Dirty tree: never refuse. Prefer pushed HEAD; else origin/main (or remote tip).
    if dirty and head_on_remote:
        _note_dirty_git_fallback(head, "dirty working tree")
        return head

    fallback = _remote_fallback_sha(root, remotes)
    if fallback:
        reason = "dirty working tree" if dirty else f"HEAD {head[:12]} not on remote"
        _note_dirty_git_fallback(fallback, reason)
        return fallback

    raise RunPodConfigError(
        f"No resolvable remote git SHA (HEAD {head[:12]} not on remote; "
        "origin/main unavailable). Push main (or set GIT_SHA) before RunPod launch "
        "(pod must clone from ROBOLAB_GIT_URL)."
    )


def list_pods(api_key: str | None = None) -> list[dict[str, Any]]:
    runpod = _configure_sdk(api_key)
    pods = runpod.get_pods()
    return list(pods or [])


def get_pod(pod_id: str, api_key: str | None = None) -> dict[str, Any] | None:
    runpod = _configure_sdk(api_key)
    try:
        return runpod.get_pod(pod_id)
    except Exception:
        return None


def terminate_pod(pod_id: str, api_key: str | None = None) -> None:
    """Terminate via REST v2 DELETE; fall back to runpod-python GraphQL."""
    key = (api_key or os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not key:
        raise RunPodConfigError("RUNPOD_API_KEY is missing for terminate_pod.")

    req = urllib.request.Request(
        f"https://api.runpod.io/v2/pods/{pod_id}",
        method="DELETE",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
        return
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 410}:
            return
        # Fall through to SDK
    except Exception:
        pass

    runpod = _configure_sdk(key)
    runpod.terminate_pod(pod_id)


def query_hourly_rate(
    gpu_type_id: str,
    *,
    cloud_type: str = "COMMUNITY",
    api_key: str | None = None,
) -> float:
    """Best-effort $/hr from get_gpu pricing fields."""
    runpod = _configure_sdk(api_key)
    try:
        info = runpod.get_gpu(gpu_type_id)
    except Exception as exc:
        raise RunPodConfigError(f"Failed to query GPU price for {gpu_type_id}: {exc}") from exc

    cloud = cloud_type.upper()
    if cloud == "SECURE":
        price = info.get("securePrice")
    else:
        price = info.get("communityPrice") or info.get("securePrice")
    lowest = info.get("lowestPrice") or {}
    if price is None:
        price = lowest.get("uninterruptablePrice")
    if price is None:
        raise RunPodConfigError(
            f"No hourly price returned for GPU {gpu_type_id}. Check RunPod catalog."
        )
    return float(price)


def estimate_cost_usd(hourly_rate: float, max_runtime_min: int | None = None) -> float:
    minutes = max_runtime_min
    if minutes is None:
        minutes = int(float(os.environ.get("MAX_RUNTIME_MIN", "120")))
    return round(hourly_rate * (max(1, minutes) / 60.0), 4)


def _gpu_candidates(preferred: str | None) -> list[str]:
    """Resolve GPU type ids to try for create_pod.

    - best / auto / any / empty / None → DEFAULT_GPU_FALLBACKS (EU-RO-1 cost-safe order)
    - specific catalog name → that GPU only (capacity miss fails; no silent swap)
    """
    if _is_best_available(preferred):
        return list(DEFAULT_GPU_FALLBACKS)
    assert preferred is not None
    return [preferred.strip()]


def create_training_pod(
    *,
    run_id: str,
    cfg: RunConfig,
    git_sha: str,
    env_bundle: dict[str, str] | None = None,
    job: str = "train",
    wandb_url: str | None = None,
) -> dict[str, Any]:
    """Create an on-demand pod. Returns {pod_id, gpu_type, hourly_rate, raw}.

    job='train' runs the trainer; job='render' runs playback video on the worker
    (required for genesis/isaac — Mac API host lacks those optional packages).
    """
    env_bundle = env_bundle or require_runpod_launch_env()
    runpod = _configure_sdk(env_bundle["RUNPOD_API_KEY"])

    public = preflight_backend_public_url(env_bundle["BACKEND_PUBLIC_URL"])
    env_bundle = {**env_bundle, "BACKEND_PUBLIC_URL": public}

    # Secure is the reliable default for EU-RO-1 + network volume; Community often
    # returns "no instances available" even when the capacity catalog shows High stock.
    cloud_pref = (os.environ.get("RUNPOD_CLOUD_TYPE") or "SECURE").strip().upper()
    if cloud_pref not in {"ALL", "COMMUNITY", "SECURE"}:
        cloud_pref = "SECURE"
    job_key = (job or "train").strip().lower() or "train"
    if job_key == "render":
        max_runtime = str(
            int(float(os.environ.get("ROBOLAB_RENDER_MAX_RUNTIME_MIN", "25")))
        )
    else:
        max_runtime = str(int(float(os.environ.get("MAX_RUNTIME_MIN", "120"))))

    config_json = cfg.model_dump_json()
    config_b64 = base64.b64encode(config_json.encode("utf-8")).decode("ascii")

    pod_env: dict[str, str] = {
        "RUN_ID": run_id,
        "GIT_SHA": git_sha,
        "CONFIG_B64": config_b64,
        "WANDB_API_KEY": (os.environ.get("WANDB_API_KEY") or "").strip(),
        "WANDB_PROJECT": (os.environ.get("WANDB_PROJECT") or "robolab").strip() or "robolab",
        "RUNPOD_API_KEY": env_bundle["RUNPOD_API_KEY"],
        "BACKEND_URL": public,
        "MAX_RUNTIME_MIN": max_runtime,
        "ROBOLAB_GIT_URL": env_bundle["ROBOLAB_GIT_URL"],
        "ROBOLAB_JOB": job_key,
    }
    entity = (os.environ.get("WANDB_ENTITY") or "").strip()
    if entity:
        pod_env["WANDB_ENTITY"] = entity
    gh = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if gh:
        pod_env["GITHUB_TOKEN"] = gh
    # Allow override; default OSMesa is safest across Secure GPU hosts.
    pod_env["MUJOCO_GL"] = (os.environ.get("MUJOCO_GL") or "glfw").strip() or "glfw"
    if (cfg.sim or "").strip() == "genesis":
        pod_env["ROBOLAB_INSTALL_GENESIS"] = "1"
        pod_env["ROBOLAB_GENESIS_GPU"] = (
            (os.environ.get("ROBOLAB_GENESIS_GPU") or "1").strip() or "1"
        )
    sim_key = (cfg.sim or "").strip().lower()
    if sim_key in {"isaaclab", "isaac_sim"}:
        # Belt-and-suspenders: image should already set this; force thin for COMPLETE path.
        pod_env["ROBOLAB_ISAAC_MODE"] = (
            (os.environ.get("ROBOLAB_ISAAC_MODE") or "thin").strip() or "thin"
        )
    if job_key == "render":
        wb = (wandb_url or "").strip()
        if not wb:
            raise RunPodConfigError(
                "wandb_url is required to launch a remote video render pod."
            )
        pod_env["WANDB_URL"] = wb

    if not pod_env["WANDB_API_KEY"]:
        raise RunPodConfigError(
            "WANDB_API_KEY is missing. Set it in .env before launching RunPod runs."
        )

    # Network volumes are DC-scoped; pods must launch in the same data center.
    # Default EU-RO-1 matches the user's robolab-workspace volume (1hyuaan8i2).
    data_center_id = (os.environ.get("RUNPOD_DATA_CENTER_ID") or "EU-RO-1").strip() or None

    image_name = worker_image_for_sim(sim_key, env_bundle["ROBOLAB_WORKER_IMAGE"])

    # UI "Best available" sends gpu_type=best; null/empty also walks the fallback list.
    # Render jobs: prefer best/cheap GPUs even if training used a specific type.
    preferred = cfg.gpu_type if job_key != "render" else (
        (os.environ.get("ROBOLAB_RENDER_GPU_TYPE") or "best").strip() or "best"
    )
    candidates = _gpu_candidates(preferred)
    last_err: Exception | None = None
    attempts: list[str] = []
    pod_name = f"robolab-render-{run_id}" if job_key == "render" else f"robolab-{run_id}"
    for gpu_type_id in candidates:
        for cloud_type in _cloud_type_candidates(cloud_pref):
            try:
                rate_cloud = "SECURE" if cloud_type == "ALL" else cloud_type
                hourly = query_hourly_rate(gpu_type_id, cloud_type=rate_cloud)
                create_kwargs: dict[str, Any] = {
                    "name": pod_name,
                    "image_name": image_name,
                    "gpu_type_id": gpu_type_id,
                    "cloud_type": cloud_type,
                    "gpu_count": 1,
                    "volume_in_gb": 0,
                    "container_disk_in_gb": 50,
                    "volume_mount_path": "/workspace",
                    "network_volume_id": env_bundle["RUNPOD_NETWORK_VOLUME_ID"],
                    "env": pod_env,
                }
                if data_center_id:
                    create_kwargs["data_center_id"] = data_center_id
                raw = runpod.create_pod(**create_kwargs)
                pod_id = raw.get("id") if isinstance(raw, dict) else None
                if not pod_id:
                    raise RunPodConfigError(f"create_pod returned no id: {raw!r}")
                # Prefer live costPerHr when present
                live = None
                try:
                    live = runpod.get_pod(pod_id)
                except Exception:
                    live = None
                if live and live.get("costPerHr") is not None:
                    hourly = float(live["costPerHr"])
                est_min = int(float(max_runtime))
                return {
                    "pod_id": str(pod_id),
                    "gpu_type": gpu_type_id,
                    "hourly_rate": float(hourly),
                    "raw": raw,
                    "estimated_cost_usd": estimate_cost_usd(
                        float(hourly), max_runtime_min=est_min
                    ),
                    "cloud_type": cloud_type,
                    "data_center_id": data_center_id,
                    "job": job_key,
                    "image": image_name,
                }
            except RunPodConfigError:
                raise
            except Exception as exc:
                last_err = exc
                attempts.append(f"{gpu_type_id}/{cloud_type}: {exc}")
                continue
    dc = data_center_id or "any-DC"
    vol = env_bundle["RUNPOD_NETWORK_VOLUME_ID"]
    detail = attempts[-1] if attempts else str(last_err)
    hint = (
        "Try GPU type 'Best available', RUNPOD_CLOUD_TYPE=SECURE (or ALL), "
        "a different GPU, or retry later."
        if not _is_best_available(preferred)
        else "Try RUNPOD_CLOUD_TYPE=SECURE (or ALL), or retry later."
    )
    raise RunPodConfigError(
        f"No RunPod capacity in {dc} (volume {vol}) for "
        f"GPUs {candidates} / clouds {_cloud_type_candidates(cloud_pref)}. "
        f"Last error: {detail}. {hint}"
    )


def poll_until_running(
    pod_id: str,
    *,
    timeout_sec: float = 600.0,
    interval_sec: float = 5.0,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Poll get_pod until desiredStatus/runtime looks running or timeout."""
    deadline = time.time() + timeout_sec
    last: dict[str, Any] | None = None
    while time.time() < deadline:
        last = get_pod(pod_id, api_key=api_key) or {}
        status = (last.get("desiredStatus") or "").upper()
        if status in {"RUNNING", "EXITED", "TERMINATED"}:
            return last
        if last.get("runtime"):
            return last
        time.sleep(interval_sec)
    return last or {}


def launch_runpod(
    cfg: RunConfig,
    run_id: str,
    *,
    backend_public_url: str | None = None,
) -> dict[str, Any]:
    """High-level: git gate + env gate + create_pod. Does not write DB."""
    if cfg.compute != "runpod":
        raise ValueError("launch_runpod requires compute='runpod'")

    # Ensure config on disk for local debugging (pod uses CONFIG_B64)
    write_run_config(cfg, run_id)

    env_bundle = require_runpod_launch_env()
    if backend_public_url:
        env_bundle = {**env_bundle, "BACKEND_PUBLIC_URL": backend_public_url.rstrip("/")}

    git_sha = resolve_git_sha()
    return create_training_pod(run_id=run_id, cfg=cfg, git_sha=git_sha, env_bundle=env_bundle)


def launch_render_runpod(
    cfg: RunConfig,
    run_id: str,
    *,
    wandb_url: str,
    backend_public_url: str | None = None,
) -> dict[str, Any]:
    """Spin a short-lived :genesis/:isaac worker to record playback video."""
    write_run_config(cfg, run_id)
    env_bundle = require_runpod_launch_env()
    if backend_public_url:
        env_bundle = {**env_bundle, "BACKEND_PUBLIC_URL": backend_public_url.rstrip("/")}
    git_sha = resolve_git_sha()
    return create_training_pod(
        run_id=run_id,
        cfg=cfg,
        git_sha=git_sha,
        env_bundle=env_bundle,
        job="render",
        wandb_url=wandb_url,
    )
