"""Figure-8 (lemniscate) smooth path tracking."""

from __future__ import annotations

from typing import Any

import numpy as np

from robolab.core.task import ActSpec, ObsSpec, TaskSpec, register_task
from robolab.tasks.worlds import is_wall_crash

COLLISION_DIST = 0.12
CROSSTRACK_OK = 0.28
YAW_OK = 0.45
# One full lemniscate parameter sweep is 2π; require slightly more than one loop.
LOOP_S = 2.0 * np.pi + 0.3


def _reward(info: dict[str, Any]) -> float:
    cross = abs(float(info.get("crosstrack") or 0.0))
    yaw_err = abs(float(info.get("yaw_err") or 0.0))
    forward = float(info.get("forward_speed", 0.0))
    track = np.exp(-((cross / 0.4) ** 2))
    heading = np.exp(-((yaw_err / 0.8) ** 2))
    speed = 0.5 * np.clip(forward / 0.45, 0.0, 1.0)
    crash = -5.0 if is_wall_crash(info, COLLISION_DIST) else 0.0
    return float(1.4 * track + 0.8 * heading + speed + crash)


def _termination(info: dict[str, Any]) -> bool:
    if is_wall_crash(info, COLLISION_DIST):
        return True
    return bool(info.get("success"))


def _success(info: dict[str, Any]) -> bool:
    path_s = float(info.get("path_s") or 0.0)
    cross = abs(float(info.get("crosstrack") or 99.0))
    yaw_err = abs(float(info.get("yaw_err") or 99.0))
    return path_s >= LOOP_S and cross < CROSSTRACK_OK and yaw_err < YAW_OK


@register_task("figure8_tracking")
def make_figure8_tracking() -> TaskSpec:
    return TaskSpec(
        name="figure8_tracking",
        observation=ObsSpec(
            keys=["lidar", "track_err"],
            shape=(8,),
            low=-10.0,
            high=10.0,
            notes="5 lidar rays + (along, crosstrack, yaw_err) vs lemniscate",
        ),
        action=ActSpec(
            shape=(2,),
            low=-1.0,
            high=1.0,
            notes="[linear_vel, angular_vel] normalized",
        ),
        max_steps=3000,
        reward=_reward,
        termination=_termination,
        success=_success,
        meta={
            "path_scale": 1.6,
            "path_center": [3.0, 0.0],
            "crosstrack_ok": CROSSTRACK_OK,
            "robot": "diffdrive_lidar",
        },
    )
