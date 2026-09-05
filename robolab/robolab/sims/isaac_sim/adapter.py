"""Isaac Sim stub — runtime separate from Isaac Lab (NVIDIA naming)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym

from robolab.core.run import DomainParams
from robolab.core.sim import SimAdapter, register_sim
from robolab.core.task import TaskSpec

_MSG = (
    "Simulator 'isaac_sim' is a capability-flagged stub for the Isaac Sim / "
    "Omniverse runtime (distinct from 'isaaclab', the RL framework). "
    "Isaac Sim is Linux/Windows + NVIDIA only — NOT macOS. "
    "RoboLab has no in-process Isaac Sim adapter yet and does not bake the "
    "multi-GB Omniverse stack into the default RunPod worker. "
    "Use mujoco/pybullet/genesis on RunPod Secure; see docs/ISAAC_INSTALL.md."
)


@register_sim("isaac_sim")
class IsaacSimAdapter(SimAdapter):
    name = "isaac_sim"

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
            "isaac_sim_runtime",
        }
