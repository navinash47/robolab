# Phase 4 APIs (researched before code)

Exact Gymnasium / W&B / MuJoCo signatures for playback + video. **Doc wins** if it contradicts the build prompt — differences flagged below.

Sources (checked 2026-09-04):

- [Gymnasium RecordVideo](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo) (v1.3.0 installed)
- [Gymnasium recording agents guide](https://gymnasium.farama.org/introduction/record_agent/)
- [wandb.Video](https://docs.wandb.ai/models/ref/python/data-types/video)
- [Log media](https://docs.wandb.ai/models/track/log/media)
- [Download & use artifacts](https://docs.wandb.ai/models/artifacts/download-and-use-an-artifact)
- [Artifact.download](https://docs.wandb.ai/models/ref/python/experiments/artifact)
- MuJoCo 3.12 `mujoco/rendering/classic/gl_context.py` (installed) + [Programming / OpenGL](https://mujoco.readthedocs.io/en/stable/programming/)

## Conflict: `MUJOCO_GL=osmesa` “locally”

| Prompt assumption | Actual MuJoCo 3.12 | What we do |
|---|---|---|
| Local always `MUJOCO_GL=osmesa` | `osmesa` / `egl` are **Linux-only** valid values | Linux local without GPU → `osmesa`; RunPod → `egl` (already in worker Dockerfile) |
| macOS can use osmesa | Darwin valid backends: default / `glfw` / `cgl` | On Darwin leave unset (CGL) or set `cgl` — **never** `osmesa`/`egl` |
| Set after import is fine | Backend chosen at **first** GL import; must be set before MuJoCo creates a GL context | Render in a **subprocess** with env set before Python starts |

```python
# From mujoco 3.12 gl_context.py (installed):
# Linux: glfw, glx, egl, osmesa
# Darwin: glfw, cgl
# Windows: wgl
import os, platform
system = platform.system()
if system == "Linux":
    os.environ.setdefault("MUJOCO_GL", "egl" if has_gpu else "osmesa")
elif system == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")  # or leave unset
```

## Gymnasium `RecordVideo`

**Docs:** [Misc wrappers — RecordVideo](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo)

Installed signature (`gymnasium==1.3.0`):

```python
from gymnasium.wrappers import RecordVideo

env = RecordVideo(
    env,                          # must support render_mode="rgb_array"
    video_folder: str,            # output directory
    episode_trigger=None,         # Callable[[int], bool]; default cubic schedule if both triggers None
    step_trigger=None,
    video_length: int = 0,        # 0 = full episode; >0 = fixed frame count
    name_prefix: str = "rl-video",
    fps: int | None = None,       # else env.metadata["render_fps"] or 30
    disable_logger: bool = True,  # silences MoviePy logger
    gc_trigger=lambda episode: True,
)
```

Eval-style one-shot recording (every episode):

```python
env = make_env(..., render=True)  # DiffDriveLidarEnv render_mode="rgb_array"
# HARD RULE (prompt + known MuJoCo issue): fresh env per video — never reuse.
env = RecordVideo(
    env,
    video_folder=str(out_dir),
    name_prefix="playback",
    episode_trigger=lambda ep: True,
    disable_logger=True,
)
obs, info = env.reset(seed=seed)
terminated = truncated = False
while not (terminated or truncated):
    action = policy.predict(obs, deterministic=True)[0]
    obs, reward, terminated, truncated, info = env.step(action)
env.close()  # flushes MP4 via MoviePy ImageSequenceClip.write_videofile
```

**Dependency (not optional):** RecordVideo imports MoviePy at record time:

```text
MoviePy is not installed, run `pip install "gymnasium[other]"`
```

Phase 4 adds `moviepy` to the `robolab` package (ask rule: required by documented Gymnasium API — not optional fluff).

**Flag vs older tips:** Some guides mention imageio; Gymnasium 1.3 uses **MoviePy** only for `RecordVideo`.

## W&B Artifact (checkpoint)

**Docs:** [Download and use](https://docs.wandb.ai/models/artifacts/download-and-use-an-artifact)

Log at end of training (inside the active run):

```python
import wandb
from pathlib import Path

ckpt = Path("checkpoints/<run_id>/policy.zip")
art = wandb.Artifact(name=f"policy-{run_id}", type="model")
art.add_file(str(ckpt), name="policy.zip")
wandb_run.log_artifact(art)
# optional alias
wandb_run.link_artifact(...)  # not required for Phase 4
```

Download without starting a new training run (Public API):

```python
import wandb

api = wandb.Api()
# path forms: entity/project/artifact:alias  OR  entity/project/artifact:v0
artifact = api.artifact(f"{entity}/{project}/policy-{run_id}:latest")
root = artifact.download(root="/tmp/robolab-ckpt/<run_id>")  # returns FilePathStr
ckpt_path = Path(root) / "policy.zip"
```

Also available from a run’s logged artifacts:

```python
wrun = api.run(f"{entity}/{project}/{wandb_run_id}")
for art in wrun.logged_artifacts():
    if art.type == "model" and art.name.startswith("policy-"):
        ...
```

**Phase 1–3 gap:** Existing COMPLETE runs saved `checkpoints/<id>/policy.zip` locally but did **not** log artifacts. Render path: if artifact missing, upload local zip onto a resumed W&B run as `policy-<id>`, then download (canonical path is still artifact-backed).

## `wandb.Video`

**Docs:** [Video class](https://docs.wandb.ai/models/ref/python/data-types/video), [Log media](https://docs.wandb.ai/models/track/log/media)

```python
wandb.Video(
    data_or_path: str | Path | np.ndarray | BytesIO,
    caption: str | None = None,
    fps: int | None = None,       # ignored when path is a file string
    format: Literal["gif", "mp4", "webm", "ogg"] | None = None,
)

# Prefer file path (avoids loading huge arrays):
run.log({"playback": wandb.Video(str(mp4_path), format="mp4", caption="wall_follow playback")})
```

Resume finished training run to attach media:

```python
with wandb.init(
    project=project,
    entity=entity,
    id=wandb_run_id,
    resume="allow",
    job_type="playback",
) as run:
    run.log({"playback": wandb.Video(str(mp4_path), format="mp4")})
```

Browser players need a browser-friendly codec (H.264). MoviePy/ffmpeg default is usually fine; if the dashboard `<video>` is blank but the file plays locally, re-encode with `h264`.

## SB3 load for rollout

```python
from stable_baselines3 import PPO

model = PPO.load(str(ckpt_path), device="cpu")
action, _ = model.predict(obs, deterministic=True)
```

Policy custom objects (`RoboLabActorCriticPolicy` + arch registry) must be importable before `PPO.load` (same as train).

## Dashboard / API shape (RoboLab)

| Endpoint | Role |
|---|---|
| `POST /api/runs/{id}/render` | Queue render: **local** subprocess for mujoco/pybullet; **RunPod** (`ROBOLAB_JOB=render`, `:genesis` / `:isaac` image) for genesis/isaac — Mac cannot import those packages |
| `POST /api/runs/{id}/video-upload` | Multipart MP4 from a RunPod render worker → `videos/{id}/playback.mp4` |
| `GET /api/runs/{id}` | Includes `video_status`, `video_url`, `video_error` |
| `GET /api/runs/{id}/video` | `FileResponse` MP4 when `READY` (dashboard player) |

UI: on COMPLETE run, **Render video** → poll until READY → HTML5 `<video controls src="/api/runs/{id}/video">`.

**Genesis / Isaac:** do not install `genesis-world` on the Mac for playback. Render launches the same RunPod worker path as training (`ROBOLAB_WORKER_IMAGE_GENESIS` / `:isaac`); the worker posts the MP4 back via `video-upload` then `video-complete`.

## Hard rules baked into Phase 4

1. **Fresh env per recording** — create env → RecordVideo → one episode → `close()`; never wrap an already-stepped MuJoCo env.
2. **Subprocess / remote render** — set `MUJOCO_GL` in child env before import; genesis/isaac render on RunPod only.
3. **Never commit** `.env`, keys, `checkpoints/`, `*.mp4`, `videos/`, `artifacts/`.
4. **Render failures must not demote COMPLETE** — workers use `/video-fail` (API also routes `/fail` → video-fail when status is COMPLETE).