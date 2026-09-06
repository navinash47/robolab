"""RunPod compute runner: create / poll / terminate / hourly rate.

Signatures follow current runpod-python + REST docs (see docs/PHASE_3_APIS.md).
"""

from __future__ import annotations

import base64
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from robolab.compute.local import repo_root, write_run_config
from robolab.core.run import RunConfig

# Prefer mid-tier GPUs that are usually stocked in EU-RO-1 with network volumes.
# Keep this list cost-safe (no H100/A100/H200 auto-fallback).
DEFAULT_GPU_FALLBACKS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 3070",
    "NVIDIA RTX A4000",
    "NVIDIA RTX A4500",
    "NVIDIA RTX 4000 Ada Generation",
]


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


def resolve_git_sha(root: Path | None = None) -> str:
    """Return HEAD SHA only if working tree is clean and SHA is on a remote.

    No remote / not pushed → refuse (pod must `git clone` this SHA).
    """
    root = root or repo_root()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if porcelain.returncode != 0:
        raise RunPodConfigError(f"git status failed: {porcelain.stderr.strip()}")
    if porcelain.stdout.strip():
        raise RunPodConfigError(
            "Working tree is dirty. Commit or stash before launching a RunPod run "
            "(pod clones a fixed GIT_SHA)."
        )

    remotes = subprocess.run(
        ["git", "remote"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if remotes.returncode != 0 or not remotes.stdout.strip():
        raise RunPodConfigError(
            "No git remote configured. RunPod workers `git clone` ROBOLAB_GIT_URL "
            "at GIT_SHA — add a remote, push, set ROBOLAB_GIT_URL in .env. "
            "See docs/PHASE_3_TEST.md (local-only limitation)."
        )

    sha_proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if sha_proc.returncode != 0:
        raise RunPodConfigError(f"git rev-parse failed: {sha_proc.stderr.strip()}")
    sha = sha_proc.stdout.strip()
    if len(sha) < 7:
        raise RunPodConfigError("Could not resolve git HEAD SHA.")

    # Any remote containing this commit counts as "pushed".
    found = False
    for remote in remotes.stdout.split():
        ls = subprocess.run(
            ["git", "ls-remote", remote],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if ls.returncode == 0 and sha in ls.stdout:
            found = True
            break
    if not found:
        raise RunPodConfigError(
            f"HEAD {sha[:12]} is not on any remote. Push before RunPod launch "
            f"(pod must clone this SHA from ROBOLAB_GIT_URL)."
        )
    return sha


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
    import urllib.error
    import urllib.request

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
    ordered: list[str] = []
    if preferred:
        ordered.append(preferred)
    for g in DEFAULT_GPU_FALLBACKS:
        if g not in ordered:
            ordered.append(g)
    return ordered


def create_training_pod(
    *,
    run_id: str,
    cfg: RunConfig,
    git_sha: str,
    env_bundle: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create an on-demand pod. Returns {pod_id, gpu_type, hourly_rate, raw}."""
    env_bundle = env_bundle or require_runpod_launch_env()
    runpod = _configure_sdk(env_bundle["RUNPOD_API_KEY"])

    public = env_bundle["BACKEND_PUBLIC_URL"].rstrip("/")
    if "localhost" in public or "127.0.0.1" in public:
        raise RunPodConfigError(
            "BACKEND_PUBLIC_URL must be reachable from RunPod (not localhost). "
            "Use ngrok or Cloudflare Tunnel and set BACKEND_PUBLIC_URL in .env."
        )

    # Secure is the reliable default for EU-RO-1 + network volume; Community often
    # returns "no instances available" even when the capacity catalog shows High stock.
    cloud_pref = (os.environ.get("RUNPOD_CLOUD_TYPE") or "SECURE").strip().upper()
    if cloud_pref not in {"ALL", "COMMUNITY", "SECURE"}:
        cloud_pref = "SECURE"
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

    if not pod_env["WANDB_API_KEY"]:
        raise RunPodConfigError(
            "WANDB_API_KEY is missing. Set it in .env before launching RunPod runs."
        )

    # Network volumes are DC-scoped; pods must launch in the same data center.
    # Default EU-RO-1 matches the user's robolab-workspace volume (1hyuaan8i2).
    data_center_id = (os.environ.get("RUNPOD_DATA_CENTER_ID") or "EU-RO-1").strip() or None

    image_name = worker_image_for_sim(sim_key, env_bundle["ROBOLAB_WORKER_IMAGE"])

    preferred = cfg.gpu_type or DEFAULT_GPU_FALLBACKS[0]
    last_err: Exception | None = None
    attempts: list[str] = []
    for gpu_type_id in _gpu_candidates(preferred):
        for cloud_type in _cloud_type_candidates(cloud_pref):
            try:
                rate_cloud = "SECURE" if cloud_type == "ALL" else cloud_type
                hourly = query_hourly_rate(gpu_type_id, cloud_type=rate_cloud)
                create_kwargs: dict[str, Any] = {
                    "name": f"robolab-{run_id}",
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
                return {
                    "pod_id": str(pod_id),
                    "gpu_type": gpu_type_id,
                    "hourly_rate": float(hourly),
                    "raw": raw,
                    "estimated_cost_usd": estimate_cost_usd(float(hourly)),
                    "cloud_type": cloud_type,
                    "data_center_id": data_center_id,
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
    raise RunPodConfigError(
        f"No RunPod capacity in {dc} (volume {vol}) for "
        f"GPUs {_gpu_candidates(preferred)} / clouds {_cloud_type_candidates(cloud_pref)}. "
        f"Last error: {detail}. "
        f"Try RUNPOD_CLOUD_TYPE=SECURE (or ALL), a different GPU, or retry later."
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
