"""Isaac Lab stub — capability-flagged until a real NVIDIA adapter exists."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym

from robolab.core.run import DomainParams
from robolab.core.sim import SimAdapter, register_sim
from robolab.core.task import TaskSpec

_MSG = (
    "Simulator 'isaaclab' is a capability-flagged stub. "
    "Isaac Lab requires Linux + NVIDIA GPU (Ubuntu 22.04/24.04, ≥16GB VRAM) — "
    "NOT macOS. Install via Isaac Sim pip/binary then Isaac Lab source "
    "(https://isaac-sim.github.io/IsaacLab/). "
    "RoboLab does not bake Isaac into the default worker (multi-GB). "
    "Run wall_follow on RunPod Secure (EU-RO-1) with mujoco/pybullet/genesis; "
    "approve a separate :isaac worker image before Isaac smoke. "
    "See docs/ISAAC_INSTALL.md."
)


@register_sim("isaaclab")
class IsaacLabAdapter(SimAdapter):
    name = "isaaclab"

    def load_robot(self, urdf_path: Path, **kw: Any) -> Any:
        raise RuntimeError(_MSG)

    def make_env(
        self,
        task: TaskSpec,
        robot: Any,
        domain: DomainParams,
        render: bool = False,
    ) -> gym.Env:
        raise RuntimeError(_MSG)

    def perturb(self, env: gym.Env, params: DomainParams) -> None:
        raise RuntimeError(_MSG)

    def render_frame(self, env: gym.Env) -> Any:
        raise RuntimeError(_MSG)

    def capabilities(self) -> set[str]:
        return {
            "stub",
            "requires_nvidia",
            "requires_install",
            "not_implemented",
            "isaac_lab",
        }
