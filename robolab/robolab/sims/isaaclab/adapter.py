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
    "Isaac Lab requires a heavy NVIDIA / Omniverse install "
    "(https://isaac-sim.github.io/IsaacLab/). "
    "RoboLab does not bundle Isaac — use mujoco or pybullet for training now. "
    "See docs/SIMULATORS.md and docs/PHASE_5B_APIS.md."
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
