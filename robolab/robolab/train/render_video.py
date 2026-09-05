"""Headless policy rollout → MP4 via Gymnasium RecordVideo → wandb.Video.

Hard rules (see docs/PHASE_4_APIS.md):
- Fresh env per recording (never reuse a stepped MuJoCo env).
- MUJOCO_GL must be set in the process env *before* MuJoCo GL init
  (this module is launched as a subprocess from the API).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from urllib.parse import urlparse

import yaml

# Ensure registries are populated before SB3 load / env build
import robolab.archs  # noqa: F401
import robolab.sims.mujoco  # noqa: F401
import robolab.tasks  # noqa: F401
from robolab.core.run import RunConfig
from robolab.core.sim import get_sim
from robolab.core.task import get_task
from robolab.sims.mujoco.adapter import urdf_path
from robolab.train.callbacks import _post_json


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def configure_mujoco_gl() -> str:
    """Pick a valid MUJOCO_GL for this OS; set env if unset. Returns chosen value."""
    existing = (os.environ.get("MUJOCO_GL") or "").strip().lower()
    system = platform.system()
    if existing:
        return existing
    if system == "Linux":
        # Prefer EGL on GPU pods; OSMesa for CPU/headless Linux.
        choice = "egl" if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1") else "osmesa"
        # RunPod worker Dockerfile already sets egl; respect if we got here unset.
        if Path("/.dockerenv").exists() or os.environ.get("RUNPOD_POD_ID"):
            choice = "egl"
        os.environ["MUJOCO_GL"] = choice
        return choice
    if system == "Darwin":
        os.environ["MUJOCO_GL"] = "cgl"
        return "cgl"
    # Windows / other — leave default (glfw/wgl)
    return existing or "default"


def parse_wandb_path(wandb_url: str) -> tuple[str, str, str]:
    path = urlparse(wandb_url).path.strip("/")
    parts = path.split("/")
    if len(parts) < 4 or parts[2] != "runs":
        raise ValueError(f"Unrecognized wandb_url path: {wandb_url}")
    return parts[0], parts[1], parts[3]


def load_config(path: Path) -> RunConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return RunConfig.model_validate(raw)


def artifact_name_for(run_id: str) -> str:
    return f"policy-{run_id}"


def ensure_checkpoint_artifact(
    *,
    run_id: str,
    wandb_url: str,
    local_ckpt: Path | None,
) -> Path:
    """Return a local policy.zip, preferring W&B artifact download.

    If the artifact is missing but a local checkpoint exists, upload it to the
    resumed W&B run, then download (canonical Phase 4 path is artifact-backed).
    """
    import wandb

    entity, project, wb_id = parse_wandb_path(wandb_url)
    api = wandb.Api(timeout=120)
    art_path = f"{entity}/{project}/{artifact_name_for(run_id)}:latest"

    try:
        artifact = api.artifact(art_path)
    except Exception:
        artifact = None
        # Fall back: look at run's logged artifacts
        try:
            wrun = api.run(f"{entity}/{project}/{wb_id}")
            for art in wrun.logged_artifacts():
                if art.type == "model" and (
                    art.name.startswith(artifact_name_for(run_id))
                    or "policy" in art.name
                ):
                    artifact = art
                    break
        except Exception:
            artifact = None

    if artifact is None:
        if local_ckpt is None or not local_ckpt.is_file():
            raise FileNotFoundError(
                f"No W&B artifact {art_path} and no local checkpoint at "
                f"{local_ckpt}. Re-train or place policy.zip under checkpoints/{run_id}/."
            )
        # Upload local checkpoint onto the original run, then re-fetch.
        with wandb.init(
            project=project,
            entity=entity,
            id=wb_id,
            resume="allow",
            job_type="checkpoint-backfill",
            settings=wandb.Settings(init_timeout=120),
        ) as run:
            art = wandb.Artifact(name=artifact_name_for(run_id), type="model")
            art.add_file(str(local_ckpt), name="policy.zip")
            logged = run.log_artifact(art)
            try:
                logged.wait(timeout=300)
            except Exception:
                pass
        # Public API can lag a beat after upload
        artifact = None
        last_err: Exception | None = None
        for _ in range(8):
            try:
                artifact = api.artifact(art_path)
                break
            except Exception as exc:
                last_err = exc
                import time

                time.sleep(2.0)
        if artifact is None:
            raise RuntimeError(
                f"Uploaded artifact but could not fetch {art_path}: {last_err}"
            )

    download_root = repo_root() / "artifacts" / run_id
    download_root.mkdir(parents=True, exist_ok=True)
    root = Path(artifact.download(root=str(download_root)))
    ckpt = root / "policy.zip"
    if not ckpt.is_file():
        # Some uploads nest differently — search
        matches = list(root.rglob("policy.zip"))
        if not matches:
            raise FileNotFoundError(f"policy.zip missing inside artifact download {root}")
        ckpt = matches[0]
    return ckpt


def record_playback_mp4(
    *,
    cfg: RunConfig,
    ckpt_path: Path,
    video_dir: Path,
    seed: int = 0,
) -> Path:
    """Create a fresh render env, wrap RecordVideo, roll out one episode."""
    from gymnasium.wrappers import RecordVideo
    from stable_baselines3 import PPO

    video_dir.mkdir(parents=True, exist_ok=True)
    # Clear prior clips so we know which file is new
    for old in video_dir.glob("*.mp4"):
        old.unlink(missing_ok=True)

    sim = get_sim(cfg.sim)
    task = get_task(cfg.task)
    robot = sim.load_robot(urdf_path(cfg.robot), robot=cfg.robot)

    # FRESH env — never reuse after close / prior recording
    env = sim.make_env(task=task, robot=robot, domain=cfg.domain, render=True)
    env = RecordVideo(
        env,
        video_folder=str(video_dir),
        name_prefix="playback",
        episode_trigger=lambda _ep: True,
        disable_logger=True,
    )

    model = PPO.load(str(ckpt_path), device="cpu")
    try:
        obs, _info = env.reset(seed=seed)
        terminated = truncated = False
        steps = 0
        max_steps = int(task.max_steps) + 10
        while not (terminated or truncated) and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
    finally:
        env.close()

    mp4s = sorted(video_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4s:
        raise RuntimeError(f"RecordVideo produced no MP4 in {video_dir}")
    return mp4s[-1]


def upload_wandb_video(*, wandb_url: str, mp4_path: Path, caption: str) -> str | None:
    import wandb

    entity, project, wb_id = parse_wandb_path(wandb_url)
    with wandb.init(
        project=project,
        entity=entity,
        id=wb_id,
        resume="allow",
        job_type="playback",
        settings=wandb.Settings(init_timeout=120),
    ) as run:
        run.log(
            {
                "playback": wandb.Video(
                    str(mp4_path),
                    format="mp4",
                    caption=caption,
                )
            }
        )
        return run.url


def render_run(
    *,
    run_id: str,
    config_path: Path,
    backend_url: str,
    wandb_url: str,
    local_ckpt: Path | None = None,
) -> dict:
    gl = configure_mujoco_gl()
    backend = backend_url.rstrip("/")
    cfg = load_config(config_path)
    if cfg.seeds:
        cfg.trainer.seed = int(cfg.seeds[0])

    try:
        ckpt = ensure_checkpoint_artifact(
            run_id=run_id,
            wandb_url=wandb_url,
            local_ckpt=local_ckpt,
        )
        video_dir = repo_root() / "videos" / run_id
        mp4 = record_playback_mp4(
            cfg=cfg,
            ckpt_path=ckpt,
            video_dir=video_dir,
            seed=int(cfg.trainer.seed),
        )
        caption = f"{cfg.task} / {cfg.arch} / {cfg.sim} playback"
        wb_url = upload_wandb_video(wandb_url=wandb_url, mp4_path=mp4, caption=caption)
        result = {
            "status": "READY",
            "video_path": str(mp4),
            "video_url": f"/api/runs/{run_id}/video",
            "wandb_url": wb_url or wandb_url,
            "mujoco_gl": gl,
            "checkpoint": str(ckpt),
        }
        _post_json(f"{backend}/api/runs/{run_id}/video-complete", result)
        return result
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        _post_json(
            f"{backend}/api/runs/{run_id}/video-fail",
            {"error": err, "traceback": traceback.format_exc()},
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RoboLab playback video renderer")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--wandb-url", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional local policy.zip if artifact not yet logged",
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"),
    )
    # Optional JSON sidecar from API with the same fields (avoids huge CLI)
    parser.add_argument("--meta", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.meta and args.meta.is_file():
        meta = json.loads(args.meta.read_text())
        wandb_url = meta.get("wandb_url") or args.wandb_url
        ckpt = Path(meta["checkpoint"]) if meta.get("checkpoint") else args.checkpoint
    else:
        wandb_url = args.wandb_url
        ckpt = args.checkpoint

    render_run(
        run_id=args.run_id,
        config_path=args.config,
        backend_url=args.backend_url,
        wandb_url=wandb_url,
        local_ckpt=ckpt,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
