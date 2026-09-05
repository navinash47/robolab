"""Wall-following task: keep a target standoff on the right wall while moving forward."""

from __future__ import annotations

from typing import Any

import numpy as np

from robolab.core.task import ActSpec, ObsSpec, TaskSpec, register_task

TARGET_DIST = 0.35
COLLISION_DIST = 0.12


def _reward(info: dict[str, Any]) -> float:
    ranges = np.asarray(info["ranges"], dtype=np.float64)
    # indices: 0 front, 1 +45, 2 -45, 3 +90 (left), 4 -90 (right)
    right = float(ranges[4])
    front = float(ranges[0])
    forward = float(info.get("forward_speed", 0.0))

    dist_err = abs(right - TARGET_DIST)
    wall_term = np.exp(-((dist_err / 0.25) ** 2))
    forward_term = np.clip(forward / 0.6, -0.5, 1.0)
    front_penalty = -2.0 if front < 0.25 else 0.0
    crash = -5.0 if front < COLLISION_DIST or right < COLLISION_DIST else 0.0
    return float(1.2 * wall_term + 0.8 * forward_term + front_penalty + crash)


def _termination(info: dict[str, Any]) -> bool:
    ranges = np.asarray(info["ranges"], dtype=np.float64)
    if float(ranges.min()) < COLLISION_DIST:
        return True
    pos = info.get("position")
    # Match scene.xml wall_end (~12.5); finish once past the far wall.
    if pos is not None and float(pos[0]) > 12.0:
        return True  # reached end of corridor (episode success terminal)
    return False


def _success(info: dict[str, Any]) -> bool:
    ranges = np.asarray(info["ranges"], dtype=np.float64)
    right = float(ranges[4])
    forward = float(info.get("forward_speed", 0.0))
    return abs(right - TARGET_DIST) < 0.12 and forward > 0.15


@register_task("wall_follow")
def make_wall_follow() -> TaskSpec:
    return TaskSpec(
        name="wall_follow",
        observation=ObsSpec(
            keys=["lidar"],
            shape=(5,),
            low=0.0,
            high=5.0,
            notes="5 planar rays: front, ±45°, ±90°",
        ),
        action=ActSpec(
            shape=(2,),
            low=-1.0,
            high=1.0,
            notes="[linear_vel, angular_vel] normalized",
        ),
        # ~12 m corridor at ~0.5 m/s and control_hz=50 needs ~1200+ steps;
        # keep headroom so playback reaches the far wall (not a short stub clip).
        max_steps=2000,
        reward=_reward,
        termination=_termination,
        success=_success,
        meta={
            "target_wall_distance": TARGET_DIST,
            "side": "right",
            "corridor_end_x": 12.0,
        },
    )
