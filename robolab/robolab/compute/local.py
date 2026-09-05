"""Local compute runner: spawn trainer subprocess (CPU)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from robolab.core.run import RunConfig

_LEGACY_KEY = re.compile(r"^[a-f0-9]{40}$")
_V1_KEY = re.compile(r"^wandb_v1_[A-Za-z0-9_]{77}$")


def repo_root() -> Path:
    # robolab/robolab/compute/local.py → parents[3] = workspace root
    return Path(__file__).resolve().parents[3]


def write_run_config(cfg: RunConfig, run_id: str) -> Path:
    out_dir = repo_root() / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.yaml"
    import yaml

    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    return path


def _trainer_env(backend_url: str) -> dict[str, str]:
    """Copy parent env and ensure W&B + backend vars reach the trainer child.

    `make dev` loads `.env` via `uv run --env-file .env` into the API process;
    the child only sees what we pass here (defaults to a full environ copy).
    """
    env = os.environ.copy()
    env["BACKEND_URL"] = backend_url
    env["CUDA_VISIBLE_DEVICES"] = ""

    api_key = (env.get("WANDB_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "WANDB_API_KEY is missing in the API process environment. "
            "Set it in .env and restart `make dev` (uv --env-file)."
        )
    if not (_LEGACY_KEY.fullmatch(api_key) or _V1_KEY.fullmatch(api_key)):
        if api_key.startswith("wandb_v1_"):
            body = len(api_key) - len("wandb_v1_")
            raise RuntimeError(
                f"WANDB_API_KEY looks truncated/corrupt (wandb_v1_ body len={body}, "
                f"expected 77; total len={len(api_key)}, expected 86). "
                "Paste a fresh key from https://wandb.ai/authorize into .env and restart."
            )
        raise RuntimeError(
            f"WANDB_API_KEY has unexpected length {len(api_key)} "
            "(want legacy 40-char hex or wandb_v1_ 86-char). "
            "Paste a fresh key from https://wandb.ai/authorize into .env and restart."
        )
    env["WANDB_API_KEY"] = api_key

    project = (env.get("WANDB_PROJECT") or "robolab").strip() or "robolab"
    env["WANDB_PROJECT"] = project

    entity = (env.get("WANDB_ENTITY") or "").strip()
    if entity:
        env["WANDB_ENTITY"] = entity
    else:
        env.pop("WANDB_ENTITY", None)

    mode = (env.get("WANDB_MODE") or "").strip().lower()
    if mode == "offline":
        raise RuntimeError(
            "WANDB_MODE=offline is set; Phase 1 needs a live W&B return curve. "
            "Unset WANDB_MODE in .env and restart `make dev`."
        )
    return env


def launch_local(
    cfg: RunConfig,
    run_id: str,
    backend_url: str = "http://127.0.0.1:8000",
) -> subprocess.Popen:
    """Start `python -m robolab.train.trainer` as a subprocess. CPU only."""
    if cfg.compute != "local":
        raise ValueError("launch_local requires compute='local'")

    config_path = write_run_config(cfg, run_id)
    env = _trainer_env(backend_url)

    cmd = [
        sys.executable,
        "-m",
        "robolab.train.trainer",
        "--run-id",
        run_id,
        "--config",
        str(config_path),
        "--backend-url",
        backend_url,
    ]
    log_path = repo_root() / "runs" / run_id / "trainer.log"
    log_f = open(log_path, "w", encoding="utf-8")  # kept open for subprocess lifetime
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root()),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._robolab_log = log_f  # type: ignore[attr-defined]
    return proc


def _mujoco_gl_for_local() -> str:
    """Match docs/PHASE_4_APIS.md: Darwin=cgl, Linux headless=osmesa."""
    import platform

    system = platform.system()
    if system == "Darwin":
        return "cgl"
    if system == "Linux":
        return "osmesa"
    return ""


def launch_render(
    *,
    run_id: str,
    config_path: Path,
    wandb_url: str,
    checkpoint: Path | None,
    backend_url: str = "http://127.0.0.1:8000",
) -> subprocess.Popen:
    """Spawn playback renderer subprocess with MUJOCO_GL set before import."""
    env = _trainer_env(backend_url)
    gl = _mujoco_gl_for_local()
    if gl:
        env["MUJOCO_GL"] = gl

    meta = {
        "wandb_url": wandb_url,
        "checkpoint": str(checkpoint) if checkpoint else None,
    }
    meta_path = repo_root() / "runs" / run_id / "render_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta))

    cmd = [
        sys.executable,
        "-m",
        "robolab.train.render_video",
        "--run-id",
        run_id,
        "--config",
        str(config_path),
        "--wandb-url",
        wandb_url,
        "--backend-url",
        backend_url,
        "--meta",
        str(meta_path),
    ]
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])

    log_path = repo_root() / "runs" / run_id / "render.log"
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root()),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._robolab_log = log_f  # type: ignore[attr-defined]
    return proc
