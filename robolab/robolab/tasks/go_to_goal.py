"""Point navigation: reach a fixed goal in an open arena."""

from __future__ import annotations

from typing import Any

import numpy as np

from robolab.core.task import ActSpec, ObsSpec, TaskSpec, register_task
from robolab.tasks.worlds import is_wall_crash

COLLISION_DIST = 0.12
GOAL_RADIUS = 0.25
# Match kinematic adapters (mujoco / pybullet / genesis).
_W_MAX = 1.2
_FORWARD_GATE = 0.05  # m/s — heading bonus requires forward motion
# Typical max step length ≈ v_max * dt = 0.5 / 50 = 0.01 m; use 0.02 for headroom.
_DELTA_SCALE = 0.02


def _reward(info: dict[str, Any]) -> float:
    """Dense shaping that does not reward spin-in-place.

    Absolute proximity + unconditional heading made yaw-rate farming cheaper than
    navigating: spinning periodically zeros ``yaw_err`` with no angular penalty,
    while far-from-goal ``exp(-(d/2)^2)`` is nearly flat so the only dense signal
    was heading. Prefer delta-distance progress, gate heading on forward motion,
    and penalize |yaw-rate| when not closing on the goal.
    """
    dist = float(info.get("dist_to_goal") or 10.0)
    prev = info.get("prev_dist_to_goal")
    prev_dist = float(prev) if prev is not None else dist
    delta = prev_dist - dist  # >0 when closer to goal

    yaw_err = abs(float(info.get("yaw_err") or 0.0))
    forward = float(info.get("forward_speed", 0.0))
    yaw_rate = abs(float(info.get("angular_vel", 0.0)))
    w_norm = float(np.clip(yaw_rate / _W_MAX, 0.0, 1.0))

    # Progress: reward closing the gap (not just being near).
    progress = 3.0 * float(np.clip(delta / _DELTA_SCALE, -1.0, 1.0))
    # Weak proximity so near-goal still has a gentle basin (cannot replace progress).
    proximity = 0.2 * float(np.exp(-((dist / 2.0) ** 2)))

    # Heading bonus only while moving forward — spinning to face the goal alone pays 0.
    if forward > _FORWARD_GATE:
        heading = 0.3 * float(np.exp(-((yaw_err / 1.0) ** 2)))
    else:
        heading = 0.0

    speed = 0.2 * float(np.clip(forward / 0.5, -0.5, 1.0))

    # Spin penalty: strong when not making progress, mild otherwise (turning while driving OK).
    if delta <= 1e-4:
        spin = -0.6 * w_norm
    else:
        spin = -0.05 * w_norm

    crash = -5.0 if is_wall_crash(info, COLLISION_DIST) else 0.0
    bonus = 2.0 if dist < GOAL_RADIUS else 0.0
    return float(progress + proximity + heading + speed + spin + crash + bonus)


def _termination(info: dict[str, Any]) -> bool:
    if is_wall_crash(info, COLLISION_DIST):
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
