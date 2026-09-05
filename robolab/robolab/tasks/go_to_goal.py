"""Point navigation: reach a fixed goal in an open arena."""

from __future__ import annotations

from typing import Any

import numpy as np

from robolab.core.task import ActSpec, ObsSpec, TaskSpec, register_task

COLLISION_DIST = 0.12
GOAL_RADIUS = 0.25


def _reward(info: dict[str, Any]) -> float:
    dist = float(info.get("dist_to_goal") or 10.0)
    yaw_err = abs(float(info.get("yaw_err") or 0.0))
    forward = float(info.get("forward_speed", 0.0))
    min_r = float(info.get("min_range", 5.0))
    progress = np.exp(-((dist / 2.0) ** 2))
    heading = np.exp(-((yaw_err / 1.0) ** 2))
    crash = -5.0 if min_r < COLLISION_DIST else 0.0
    speed = 0.2 * np.clip(forward / 0.5, -0.5, 1.0)
    bonus = 2.0 if dist < GOAL_RADIUS else 0.0
    return float(1.5 * progress + 0.5 * heading + speed + crash + bonus)


def _termination(info: dict[str, Any]) -> bool:
    if float(info.get("min_range", 5.0)) < COLLISION_DIST:
        return True
    dist = info.get("dist_to_goal")
    return dist is not None and float(dist) < GOAL_RADIUS and bool(info.get("success"))


def _success(info: dict[str, Any]) -> bool:
    dist = info.get("dist_to_goal")
    return dist is not None and float(dist) < GOAL_RADIUS


@register_task("go_to_goal")
def make_go_to_goal() -> TaskSpec:
    return TaskSpec(
        name="go_to_goal",
        observation=ObsSpec(
            keys=["lidar", "goal_rel"],
            shape=(8,),
            low=-10.0,
            high=10.0,
            notes="5 lidar rays + goal (robot-frame dx, dy, yaw_err)",
        ),
        action=ActSpec(
            shape=(2,),
            low=-1.0,
            high=1.0,
            notes="[linear_vel, angular_vel] normalized",
        ),
        max_steps=1500,
        reward=_reward,
        termination=_termination,
        success=_success,
        meta={
            "goal_xy": [5.0, 1.5],
            "goal_radius": GOAL_RADIUS,
            "robot": "diffdrive_lidar",
        },
    )
