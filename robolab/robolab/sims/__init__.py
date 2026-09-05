"""Sim adapters package — import subpackages to populate the registry."""

from robolab.sims.mujoco.adapter import MujocoAdapter
from robolab.sims.pybullet.adapter import PybulletAdapter

__all__ = ["MujocoAdapter", "PybulletAdapter"]
