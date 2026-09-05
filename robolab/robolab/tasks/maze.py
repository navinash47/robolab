"""Obstacle corridor maze — navigate to exit without colliding."""

from __future__ import annotations

from typing import Any

import numpy as np

from robolab.core.task import ActSpec, ObsSpec, TaskSpec, register_task

COLLISION_DIST = 0.12
EXIT_RADIUS = 0.45


def _reward(info: dict[str, Any]) -> float:
    dist = float(info.get("dist_to_goal") or 10.0)
    yaw_err = abs(float(info.get("yaw_err") or 0.0))
    forward = float(info.get("forward_speed", 0.0))
    min_r = float(info.get("min_range", 5.0))
    progress = np.exp(-((dist / 3.0) ** 2))
    heading = 0.4 * np.exp(-((yaw_err / 1.2) ** 2))
    clear = 0.3 * np.clip((min_r - 0.2) / 1.0, 0.0, 1.0)
    speed = 0.3 * np.clip(forward / 0.5, -0.5, 1.0)
    crash = -5.0 if min_r < COLLISION_DIST else 0.0
    near = -0.5 if min_r < 0.22 else 0.0
    bonus = 2.5 if dist < EXIT_RADIUS else 0.0
    return float(1.4 * progress + heading + clear + speed + crash + near + bonus)


def _termination(info: dict[str, Any]) -> bool:
    if float(info.get("min_range", 5.0)) < COLLISION_DIST:
        return True
    return bool(info.get("success"))


def _success(info: dict[str, Any]) -> bool:
    dist = info.get("dist_to_goal")
    return dist is not None and float(dist) < EXIT_RADIUS


@register_task("maze")
def make_maze() -> TaskSpec:
    return TaskSpec(
        name="maze",
        observation=ObsSpec(
            keys=["lidar", "exit_rel"],
            shape=(8,),
            low=-10.0,
            high=10.0,
            notes="5 lidar rays + exit bearing (robot-frame dx, dy, yaw_err)",
        ),
        action=ActSpec(
            shape=(2,),
            low=-1.0,
            high=1.0,
            notes="[linear_vel, angular_vel] normalized",
        ),
        max_steps=2500,
        reward=_reward,
        termination=_termination,
        success=_success,
        meta={
            "exit_xy": [7.2, 1.4],
            "exit_radius": EXIT_RADIUS,
            "robot": "diffdrive_lidar",
        },
    )
