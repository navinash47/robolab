"""Local compute runner: spawn trainer subprocess (CPU)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from robolab.core.run import RunConfig


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


def launch_local(
    cfg: RunConfig,
    run_id: str,
    backend_url: str = "http://127.0.0.1:8000",
) -> subprocess.Popen:
    """Start `python -m robolab.train.trainer` as a subprocess. CPU only."""
    if cfg.compute != "local":
        raise ValueError("launch_local requires compute='local'")

    config_path = write_run_config(cfg, run_id)
    env = os.environ.copy()
    env["BACKEND_URL"] = backend_url
    env.setdefault("WANDB_PROJECT", "robolab")
    env["CUDA_VISIBLE_DEVICES"] = ""

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
