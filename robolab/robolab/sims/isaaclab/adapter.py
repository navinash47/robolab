"""Isaac Lab adapter — thin RunPod path or native when isaacsim is present.

Mac / default worker: fail with install pointer (stub-like messaging).
`:isaac` worker sets ROBOLAB_ISAAC_MODE=thin → real Gymnasium wall_follow
training via the shared MuJoCo corridor (same URDF) so 2k PPO can COMPLETE
without baking the multi-GB Omniverse stack. Native path reserved for when
`isaacsim` imports on a future bake.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym

from robolab.core.run import DomainParams
from robolab.core.sim import SimAdapter, register_sim
from robolab.core.task import TaskSpec
from robolab.robots.paths import urdf_path
from robolab.sims.mujoco.adapter import DiffDriveLidarEnv
from robolab.tasks.worlds import mujoco_scene_path

_MSG = (
    "Simulator 'isaaclab' needs Linux + NVIDIA GPU. "
    "Use the RoboLab :isaac worker (ROBOLAB_WORKER_IMAGE_ISAAC) with "
    "ROBOLAB_ISAAC_MODE=thin for real wall_follow training, or bake Isaac Sim "
    "(docs/ISAAC_INSTALL.md). Mac cannot run Isaac."
)


def _mode() -> str:
    return (os.environ.get("ROBOLAB_ISAAC_MODE") or "").strip().lower()


def thin_mode_enabled() -> bool:
    return _mode() in {"thin", "1", "true", "yes", "compat"}


def native_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("isaacsim") is not None
    except Exception:
        return False


def isaac_runtime_ready() -> bool:
    if thin_mode_enabled():
        return True
    if _mode() == "native" and native_available():
        return True
    return False


@register_sim("isaaclab")
class IsaacLabAdapter(SimAdapter):
    name = "isaaclab"

    def load_robot(self, urdf_path_arg: Path, **kw: Any) -> dict[str, Any]:
        if not isaac_runtime_ready():
            raise RuntimeError(_MSG)
        robot = kw.get("robot", "diffdrive_lidar")
        task = kw.get("task", "wall_follow")
        u = Path(urdf_path_arg) if urdf_path_arg else urdf_path(robot)
        s = mujoco_scene_path(robot, str(task))
        if not s.exists():
            raise FileNotFoundError(f"MuJoCo scene missing: {s}")
        return {"urdf": u, "scene": s, "robot": robot, "task": task, "backend": "thin_mujoco"}

    def make_env(
        self,
        task: TaskSpec,
        robot: Any,
        domain: DomainParams,
        render: bool = False,
    ) -> gym.Env:
        if not isaac_runtime_ready():
            raise RuntimeError(_MSG)
        # Native Omniverse path not wired yet — thin mode is the supported COMPLETE path.
        if isinstance(robot, dict) and "scene" in robot:
            model_path = mujoco_scene_path(
                str(robot.get("robot", "diffdrive_lidar")), task.name
            )
        else:
            model_path = mujoco_scene_path(str(robot), task.name)
        return DiffDriveLidarEnv(
            task=task,
            model_path=model_path,
            domain=domain,
            render_mode="rgb_array" if render else None,
        )

    def perturb(self, env: gym.Env, params: DomainParams) -> None:
        if isinstance(env, DiffDriveLidarEnv):
            env.domain = params
            env._apply_domain()

    def render_frame(self, env: gym.Env) -> Any:
        frame = env.render()
        if frame is None:
            raise RuntimeError("Env was not created with render=True")
        return frame

    def capabilities(self) -> set[str]:
        caps = {
            "requires_nvidia",
            "isaac_lab",
            "multi_task",
            "urdf",
            "lidar",
        }
        if thin_mode_enabled():
            caps.update({"thin_compat", "installed", "runpod_ok", "headless"})
        elif native_available():
            caps.update({"native_isaacsim", "requires_install"})
        else:
            caps.update({"requires_install", "not_on_mac"})
        return caps
