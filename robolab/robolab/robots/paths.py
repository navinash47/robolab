"""Shared paths to canonical robot URDFs (sim-agnostic)."""

from __future__ import annotations

from pathlib import Path

ROBOTS_ROOT = Path(__file__).resolve().parent


def robot_dir(robot: str) -> Path:
    return ROBOTS_ROOT / robot


def urdf_path(robot: str) -> Path:
    return robot_dir(robot) / "robot.urdf"


def scene_path(robot: str) -> Path:
    """MuJoCo MJCF scene mirroring the URDF + arena (not used by PyBullet)."""
    return robot_dir(robot) / "scene.xml"
