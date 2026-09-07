"""Parallel-park style docking into a side bay."""

from __future__ import annotations

from typing import Any

import numpy as np

from robolab.core.task import ActSpec, ObsSpec, TaskSpec, register_task
from robolab.tasks.worlds import is_wall_crash

COLLISION_DIST = 0.12
POS_TOL = 0.22
YAW_TOL = 0.25


def _reward(info: dict[str, Any]) -> float:
    dist = float(info.get("dist_to_goal") or 10.0)
    yaw_err = abs(float(info.get("yaw_err") or 0.0))
    speed = abs(float(info.get("forward_speed", 0.0)))
    pos_term = np.exp(-((dist / 1.2) ** 2))
    yaw_term = np.exp(-((yaw_err / 0.6) ** 2))
    # Prefer slowing near the bay
    slow = 0.3 if dist < 0.6 and speed < 0.15 else 0.0
    crash = -5.0 if is_wall_crash(info, COLLISION_DIST) else 0.0
    bonus = 3.0 if dist < POS_TOL and yaw_err < YAW_TOL else 0.0
    return float(1.2 * pos_term + 1.0 * yaw_term + slow + crash + bonus)


def _termination(info: dict[str, Any]) -> bool:
    if is_wall_crash(info, COLLISION_DIST):
        return True
    return bool(info.get("success"))


def _success(info: dict[str, Any]) -> bool:
    dist = info.get("dist_to_goal")
    yaw_err = info.get("yaw_err")
    if dist is None or yaw_err is None:
        return False
    return float(dist) < POS_TOL and abs(float(yaw_err)) < YAW_TOL


@register_task("docking")
def make_docking() -> TaskSpec:
    return TaskSpec(
        name="docking",
        observation=ObsSpec(
            keys=["lidar", "dock_rel"],
            shape=(8,),
            low=-10.0,
            high=10.0,
            notes="5 lidar rays + dock (robot-frame dx, dy, yaw_err)",
        ),
        action=ActSpec(
            shape=(2,),
            low=-1.0,
            high=1.0,
            notes="[linear_vel, angular_vel] normalized",
        ),
        max_steps=2000,
        reward=_reward,
        termination=_termination,
        success=_success,
        meta={
            "dock_xy": [2.75, 0.55],
            "dock_yaw": 1.5708,  # +y, into bay
            "pos_tol": POS_TOL,
            "yaw_tol": YAW_TOL,
            "robot": "diffdrive_lidar",
        },
    )
