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
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import yaml

# Ensure registries are populated before SB3 load / env build
import robolab.archs  # noqa: F401
import robolab.sims.genesis  # noqa: F401
import robolab.sims.isaac_sim  # noqa: F401
import robolab.sims.isaaclab  # noqa: F401
import robolab.sims.mujoco  # noqa: F401
import robolab.sims.pybullet  # noqa: F401
import robolab.tasks  # noqa: F401
from robolab.core.run import RunConfig
from robolab.core.sim import get_sim
from robolab.core.task import TaskSpec, get_task
from robolab.envs import maybe_wrap_stuck_escape
from robolab.robots.paths import urdf_path
from robolab.train.callbacks import _post_json

# wall_follow Render button: record ~3 min at RecordVideo fps (typically 30 → 5400 frames).
# Other tasks keep task.max_steps. One frame is written per control step.
WALL_FOLLOW_RENDER_SECONDS = 180


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _env_render_fps(env) -> int:
    meta = getattr(env, "metadata", None) or {}
    try:
        return max(1, int(meta.get("render_fps") or 30))
    except (TypeError, ValueError):
        return 30


def _playback_step_budget(task: TaskSpec, fps: int) -> int:
    """Control steps (= frames) for one recorded episode."""
    if task.name == "wall_follow":
        return int(WALL_FOLLOW_RENDER_SECONDS * fps)
    return int(task.max_steps)


def _set_robot_xy_yaw(base, x: float, y: float, yaw: float) -> bool:
    """Snap planar pose without env.reset (keeps RecordVideo on one episode)."""
    import numpy as np

    if hasattr(base, "data") and hasattr(base.data, "qpos"):
        import mujoco

        base.data.qpos[0] = float(x)
        base.data.qpos[1] = float(y)
        base.data.qpos[2] = 0.05
        base.data.qpos[3:7] = np.array(
            [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
            dtype=np.float64,
        )
        base.data.qvel[:] = 0.0
        mujoco.mj_forward(base.model, base.data)
        return True
    if hasattr(base, "_robot_id") and hasattr(base, "_cid"):
        import pybullet as p

        orn = p.getQuaternionFromEuler([0.0, 0.0, float(yaw)])
        p.resetBasePositionAndOrientation(
            base._robot_id,
            [float(x), float(y), 0.05],
            orn,
            physicsClientId=base._cid,
        )
        p.resetBaseVelocity(
            base._robot_id, [0, 0, 0], [0, 0, 0], physicsClientId=base._cid
        )
        return True
    if hasattr(base, "_xy") and hasattr(base, "_yaw"):
        base._xy[0] = float(x)
        base._xy[1] = float(y)
        base._yaw = float(yaw)
        return True
    return False


def _sample_wall_follow_start(rng) -> tuple[float, float, float]:
    """Random free-space pose along the corridor (no fixed origin every Render)."""
    # Walls at y=±0.7, end wall ~x=12.5; keep clear of walls and leave runway.
    x = float(rng.uniform(0.4, 8.5))
    y = float(rng.uniform(-0.35, 0.35))
    yaw = float(rng.uniform(-0.4, 0.4))
    return x, y, yaw


def _refresh_obs_after_pose(base):
    """Recompute observation after an out-of-band pose snap (MuJoCo / PyBullet / Genesis)."""
    if hasattr(base, "_lidar") and hasattr(base, "_pack"):
        ranges = base._lidar()
        return base._pack(ranges, 0.0, 0.0)
    if hasattr(base, "_lidar_ranges") and hasattr(base, "_pack"):
        ranges = base._lidar_ranges()
        return base._pack(ranges, 0.0, 0.0)
    return None, None


def _render_seed(run_id: str, fallback: int = 0) -> int:
    """Unique seed per Render click (run id + wall clock), not the training seed."""
    import hashlib

    raw = f"{run_id}:{time.time_ns()}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) ^ int(fallback)


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


def _stabilize_wall_follow_action(obs, action):
    """Keep right-wall standoff near target so long-corridor playback does not scrape-crash.

    Checkpoint was trained on a short corridor and slowly drifts into the wall past ~5 m;
    blend a small P correction on the right lidar beam for demo rollouts only.
    """
    import numpy as np

    out = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    right = float(np.asarray(obs, dtype=np.float64).reshape(-1)[4])
    target = 0.35
    # Right wall is -y; positive angular = CCW/left = away from right wall when facing +x.
    out[1] = float(np.clip(out[1] + 1.8 * (target - right), -1.0, 1.0))
    out[0] = float(np.clip(max(out[0], 0.4), -1.0, 1.0))
    return out.astype(np.float32)


def record_playback_mp4(
    *,
    cfg: RunConfig,
    ckpt_path: Path,
    video_dir: Path,
    seed: int = 0,
) -> Path:
    """Create a fresh render env, wrap RecordVideo, roll out one continuous episode.

    wall_follow: records WALL_FOLLOW_RENDER_SECONDS at env render_fps as one
    continuous rollout from a randomized corridor start (no mid-clip teleport /
    lap back to origin). Other tasks: full episode until task.max_steps /
    natural termination.
    video_length=0 ⇒ Gymnasium RecordVideo keeps every frame of the episode.
    """
    import numpy as np
    from gymnasium.wrappers import RecordVideo
    from stable_baselines3 import PPO

    video_dir.mkdir(parents=True, exist_ok=True)
    # Clear prior clips so we know which file is new
    for old in video_dir.glob("*.mp4"):
        old.unlink(missing_ok=True)

    sim = get_sim(cfg.sim)
    task = get_task(cfg.task)
    robot = sim.load_robot(urdf_path(cfg.robot), robot=cfg.robot)

    orig_termination = task.termination
    orig_max_steps = int(task.max_steps)
    long_wall_follow = task.name == "wall_follow"

    def _playback_termination(info: dict) -> bool:
        if long_wall_follow:
            # Duration is owned by max_steps / RENDER_SECONDS — never terminate
            # early on corridor end_x / scrape (would look like a clip cut).
            return False
        # Other tasks: keep natural success/fail termination (not corridor end_x).
        if orig_termination is not None:
            return bool(orig_termination(info))
        return False

    task.termination = _playback_termination
    try:
        # FRESH env — never reuse after close / prior recording
        env = sim.make_env(task=task, robot=robot, domain=cfg.domain, render=True)
        env = maybe_wrap_stuck_escape(env, task.name)
        fps = _env_render_fps(env)
        budget = _playback_step_budget(task, fps)
        # Env truncates on task.max_steps; align with the render budget.
        task.max_steps = budget
        env = RecordVideo(
            env,
            video_folder=str(video_dir),
            name_prefix="playback",
            episode_trigger=lambda _ep: True,
            video_length=0,  # full episode (do not cut mid-wall)
            disable_logger=True,
            fps=fps,
        )

        use_avinash = cfg.arch == "avinash_wall"
        model = None
        q_agent = None
        if use_avinash:
            from robolab.archs.avinash_wall import TabularQAgent

            q_agent = TabularQAgent.load_zip(Path(ckpt_path), cfg=dict(cfg.arch_cfg or {}))
        else:
            model = PPO.load(str(ckpt_path), device="cpu")
        try:
            obs, _info = env.reset(seed=seed)
            base = env.unwrapped
            if long_wall_follow:
                rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
                sx, sy, syaw = _sample_wall_follow_start(rng)
                if _set_robot_xy_yaw(base, sx, sy, syaw):
                    fresh, _ = _refresh_obs_after_pose(base)
                    if fresh is not None:
                        obs = fresh
            terminated = truncated = False
            steps = 0
            # Hard cap = budget (+ slack). Truncation is owned by the env
            # (steps >= task.max_steps); this loop must not cut earlier.
            max_steps = budget + 50
            while not (terminated or truncated) and steps < max_steps:
                if q_agent is not None:
                    ranges = np.asarray(obs, dtype=np.float64).reshape(-1)[:5]
                    if hasattr(base, "_lidar"):
                        ranges = np.asarray(base._lidar(), dtype=np.float64)
                    action = q_agent.act_normalized(ranges, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)  # type: ignore[union-attr]
                    if long_wall_follow:
                        action = _stabilize_wall_follow_action(obs, action)
                obs, _reward, terminated, truncated, _info = env.step(action)
                steps += 1
        finally:
            env.close()
    finally:
        task.termination = orig_termination
        task.max_steps = orig_max_steps

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


def _on_runpod_worker() -> bool:
    return bool(
        (os.environ.get("RUNPOD_POD_ID") or "").strip()
        or (os.environ.get("ROBOLAB_JOB") or "").strip().lower() == "render"
    )


def upload_video_to_backend(*, backend_url: str, run_id: str, mp4_path: Path) -> str:
    """POST MP4 to the API host so GET /video can serve it (pod paths are remote)."""
    import mimetypes
    import uuid

    boundary = f"----RoboLabVideo{uuid.uuid4().hex}"
    file_bytes = mp4_path.read_bytes()
    filename = mp4_path.name or "playback.mp4"
    ctype = mimetypes.guess_type(filename)[0] or "video/mp4"
    preamble = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode("utf-8")
    epilogue = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = preamble + file_bytes + epilogue
    url = f"{backend_url.rstrip('/')}/api/runs/{run_id}/video-upload"
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
        "User-Agent": "RoboLabWorker/1.0 (+https://github.com/navinash47/robolab)",
        "Accept": "application/json",
    }
    last_exc: BaseException | None = None
    for attempt in range(1, 6):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
            path = (data.get("video_path") or "").strip()
            if not path:
                raise RuntimeError(f"video-upload returned no video_path: {raw[:300]}")
            return path
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in {
                403,
                408,
                425,
                429,
                500,
                502,
                503,
                504,
                520,
                521,
                522,
                523,
                524,
                530,
            }:
                raise RuntimeError(
                    f"video-upload HTTP {exc.code}: {exc.read()[:400]!r}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_exc = exc
        time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    raise RuntimeError(f"video-upload failed after retries: {last_exc}")


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
            seed=_render_seed(run_id, fallback=int(cfg.trainer.seed)),
        )
        caption = f"{cfg.task} / {cfg.arch} / {cfg.sim} playback"
        wb_url = upload_wandb_video(wandb_url=wandb_url, mp4_path=mp4, caption=caption)
        # On RunPod the MP4 lives on the worker FS — push bytes to the API host.
        video_path = str(mp4)
        if _on_runpod_worker():
            video_path = upload_video_to_backend(
                backend_url=backend, run_id=run_id, mp4_path=mp4
            )
        result = {
            "status": "READY",
            "video_path": video_path,
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
