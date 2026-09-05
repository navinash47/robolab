# Phase 1 APIs (researched before code)

Exact APIs and signatures for Phase 1. **Doc wins** if it contradicts the build prompt — flagged below.

## Gymnasium `Env`

**Docs:** [Env API](https://gymnasium.farama.org/api/env/), [Creating custom envs](https://gymnasium.farama.org/introduction/create_custom_env/)

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self):
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = ...  # np.ndarray
        info: dict = {}
        return obs, info

    def step(self, action):
        # returns: obs, reward, terminated, truncated, info
        # (done was removed; use terminated + truncated)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        # rgb_array → np.ndarray (H, W, 3)
        ...

    def close(self) -> None:
        ...
```

Spaces used in Phase 1: `spaces.Box` for continuous lidar + (v, ω) actions.

## MuJoCo (pip `mujoco`)

**Docs:** [Python bindings](https://mujoco.readthedocs.io/en/stable/python.html), [URDF support](https://mujoco.readthedocs.io/en/stable/modeling.html#urdf), [Overview / model instances](https://mujoco.readthedocs.io/en/stable/overview.html)

```python
import mujoco
import numpy as np

# Load MJCF or URDF from path (compiler handles .urdf)
model = mujoco.MjModel.from_xml_path("/path/to/robot.urdf")
# Optional assets VFS: MjModel.from_xml_path(path, assets_dict)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
mujoco.mj_step(model, data)           # one physics step
# mj_step(model, data, nstep=N)       # N substeps (Python binding)

# Named access
jid = model.joint("left_wheel").id
# Actuators via model.actuator("name") when present
```

**Headless GL (Phase 4 video; note for Phase 1):** set `MUJOCO_GL=osmesa` (local) or `egl` (GPU pod) before import. Phase 1 gate does **not** require rendering.

**Flag vs prompt:** Prompt says “MuJoCo compiles URDF natively.” Docs confirm URDF is a supported model description; prefer simple geoms (no mesh packages) to avoid mesh-path / closed-loop issues. Document constraints in `robolab/robots/README.md`.

## Stable-Baselines3

**Docs:** [Custom policy](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html), [Callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html), [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [Logger keys](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)

### Custom `ActorCriticPolicy` (registry-backed arch)

```python
from typing import Callable
from gymnasium import spaces
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class ArchMlpExtractor(nn.Module):
    def __init__(self, feature_dim: int, last_layer_dim_pi: int = 64, last_layer_dim_vf: int = 64):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        # ... nets from Architecture.forward features ...

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor: ...
    def forward_critic(self, features: th.Tensor) -> th.Tensor: ...

class RoboLabActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule: Callable[[float], float], *args, **kwargs):
        # arch_name / arch_cfg pulled via policy_kwargs then popped before super
        kwargs["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ArchMlpExtractor(self.features_dim, ...)

model = PPO(RoboLabActorCriticPolicy, env, policy_kwargs={"arch_name": "mlp", "arch_cfg": {...}}, verbose=1)
model.learn(total_timesteps=50_000, callback=callbacks)
model.save(path)
```

### Callbacks + mean return

```python
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

class HeartbeatCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # self.num_timesteps, self.model, self.locals
        # ep_rew_mean available when Monitor wrapper is used:
        # self.logger.name_to_value.get("rollout/ep_rew_mean")
        return True  # False stops training
```

`rollout/ep_rew_mean`: mean episodic training reward (Monitor required; auto-added by `make_vec_env`).

```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

env = make_vec_env(env_fn, n_envs=1, vec_env_cls=DummyVecEnv)  # wraps Monitor
```

PPO (Phase 1 defaults): `PPO(policy, env, learning_rate=..., n_steps=..., batch_size=..., verbose=1, device="cpu")`.

## Weights & Biases

**Docs:** [`wandb.init`](https://docs.wandb.ai/models/ref/python/functions/init), [API key](https://docs.wandb.ai/support/find_api), [log](https://docs.wandb.ai/models/track/log)

```python
import wandb
import os

# Auth: WANDB_API_KEY in environment (or wandb.login(key=...))
run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "robolab"),
    name="wall_follow-mlp-...",
    config={...},          # hyperparameters
    job_type="train",
    tags=["phase1", "mlp", "wall_follow", "mujoco"],
)
run.log({"rollout/ep_rew_mean": mean_return, "train/step": step}, step=step)
url = run.url  # https://wandb.ai/<entity>/<project>/runs/<id>
run.finish()
```

Env vars: `WANDB_API_KEY` (required), `WANDB_PROJECT` (default `robolab`).

**Flag:** Prefer `run.log` / `run.finish` on the returned `Run` object (current docs) over bare `wandb.log` when a local `run` handle exists.

## FastAPI SSE + runs API

**Docs:** [Server-Sent Events](https://fastapi.tiangolo.com/tutorial/server-sent-events/) (**added FastAPI 0.135.0**), [StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)

```python
from collections.abc import AsyncIterable
from fastapi import FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel

class ProgressEvent(BaseModel):
    step: int
    total: int
    mean_return: float | None
    status: str
    wandb_url: str | None = None

@app.get("/api/runs/{run_id}/events", response_class=EventSourceResponse)
async def run_events(run_id: str) -> AsyncIterable[ProgressEvent]:
    while True:
        yield ProgressEvent(...)
        # or: yield ServerSentEvent(data=..., event="progress", id=...)
```

**Flag vs older patterns:** Prefer native `fastapi.sse.EventSourceResponse` (no `sse-starlette` dep) given FastAPI ≥0.135. Pin `fastapi>=0.135.0`. Fallback if needed: `StreamingResponse` with `media_type="text/event-stream"` and manual `data: {...}\n\n` framing.

Browser client: `new EventSource("/api/runs/{id}/events")` ([MDN EventSource](https://developer.mozilla.org/en-US/docs/Web/API/EventSource)).

### Runs endpoints (Phase 1)

| Method | Path | Body / notes |
|---|---|---|
| `POST` | `/api/runs` | `RunConfig` JSON → create row, spawn local subprocess |
| `GET` | `/api/runs` | list runs (also powers experiments table) |
| `GET` | `/api/runs/{id}` | single run record |
| `POST` | `/api/runs/{id}/heartbeat` | `{step, total, mean_return, eta?, wandb_url?}` |
| `POST` | `/api/runs/{id}/complete` | success payload; status → `COMPLETE` |
| `POST` | `/api/runs/{id}/fail` | error message; status → `FAILED` |
| `GET` | `/api/runs/{id}/events` | SSE progress stream |

## Pydantic + PyYAML config

**Docs:** [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/), [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)

```python
from pydantic import BaseModel, Field
from typing import Literal
import yaml

class RunConfig(BaseModel):
    sim: str
    task: str
    robot: str
    arch: str
    arch_cfg: dict = Field(default_factory=dict)
    # ...

raw = yaml.safe_load(Path("configs/experiments/foo.yaml").read_text())
cfg = RunConfig.model_validate(raw)
```

## Local runner / subprocess

**Docs:** [subprocess](https://docs.python.org/3/library/subprocess.html)

```python
import subprocess, sys
proc = subprocess.Popen(
    [sys.executable, "-m", "robolab.train.trainer", "--run-id", run_id, "--config", config_path],
    cwd=repo_root,
    env={**os.environ, "WANDB_API_KEY": ..., "BACKEND_URL": "http://127.0.0.1:8000"},
)
```

CPU only for Phase 1 (`device="cpu"` in PPO).

## HTTP client (heartbeat from trainer)

**Docs:** [urllib / httpx](https://www.python-httpx.org/) — prefer stdlib `urllib.request` to avoid an extra dep, or `httpx` if already pulled by FastAPI stack.

```python
import json, urllib.request
req = urllib.request.Request(
    f"{backend}/api/runs/{run_id}/heartbeat",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
urllib.request.urlopen(req, timeout=5)
```

## Dependencies added this phase (Section 3 stack)

| Package | Purpose |
|---|---|
| `mujoco` | Sim |
| `gymnasium` | Env API |
| `stable-baselines3` | PPO + custom policy |
| `torch` | Arch nets / SB3 backend |
| `wandb` | Metrics + run URL |
| `pydantic` | RunConfig / TaskSpec |
| `pyyaml` | Experiment YAML |
| `httpx` | Optional; prefer stdlib for heartbeat |

Ask before anything else.

## Sources

- [Gymnasium Env](https://gymnasium.farama.org/api/env/)
- [MuJoCo Python](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo URDF](https://mujoco.readthedocs.io/en/stable/modeling.html#urdf)
- [SB3 custom policy](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
- [SB3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)
- [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [wandb.init](https://docs.wandb.ai/models/ref/python/functions/init)
- [FastAPI SSE](https://fastapi.tiangolo.com/tutorial/server-sent-events/)
- [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/)
