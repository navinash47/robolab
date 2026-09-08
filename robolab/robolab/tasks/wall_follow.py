"""Wall-following task: keep a target standoff on the right wall while moving forward.

PPO uses continuous shaping aimed at the P2_D3 **medium** band (~0.8 m).
Q-learning trainers use PDF ``pdf_reward`` instead of this function.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robolab.core.task import ActSpec, ObsSpec, TaskSpec, register_task
from robolab.tasks.worlds import is_wall_crash

# P2_D3 medium band is 0.7–0.9 m — center that for continuous PPO shaping.
TARGET_DIST = 0.80
COLLISION_DIST = 0.18

# Stall/scrape shaping: tiny min_range + low |v| → bail earlier than crash.
STALL_MIN_RANGE = 0.25  # m
STALL_SPEED_EPS = 0.08  # m/s
STALL_PENALTY = -3.0


def _stall_scrape_penalty(info: dict[str, Any]) -> float:
    """Bigger cost when scraping a nearby surface while barely moving."""
    min_r = float(info.get("min_range", 5.0))
    speed = abs(float(info.get("forward_speed", 0.0)))
    if min_r < STALL_MIN_RANGE and speed < STALL_SPEED_EPS:
        return STALL_PENALTY
    return 0.0


def _reward(info: dict[str, Any]) -> float:
    ranges = np.asarray(info["ranges"], dtype=np.float64)
    # indices: 0 front, 1 +45, 2 -45, 3 +90 (left), 4 -90 (right)
    right = float(ranges[4])
    front = float(ranges[0])
    forward = float(info.get("forward_speed", 0.0))

    dist_err = abs(right - TARGET_DIST)
    wall_term = np.exp(-((dist_err / 0.25) ** 2))
    forward_term = np.clip(forward / 0.6, -0.5, 1.0)
    front_penalty = -2.0 if front < 0.35 else 0.0
    crash = -5.0 if is_wall_crash(info, COLLISION_DIST) else 0.0
    stall = _stall_scrape_penalty(info)
    return float(1.2 * wall_term + 0.8 * forward_term + front_penalty + crash + stall)


def _termination(info: dict[str, Any]) -> bool:
    if is_wall_crash(info, COLLISION_DIST):
        return True
    # Closed largemaze has no linear corridor end; episodes end on crash or max_steps.
    return False


def _success(info: dict[str, Any]) -> bool:
    ranges = np.asarray(info["ranges"], dtype=np.float64)
    right = float(ranges[4])
    forward = float(info.get("forward_speed", 0.0))
    return abs(right - TARGET_DIST) < 0.12 and forward > 0.15


def _make_wall_task(name: str, *, layout: str, max_steps: int = 3000) -> TaskSpec:
    return TaskSpec(
        name=name,
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
        max_steps=max_steps,
        reward=_reward,
        termination=_termination,
        success=_success,
        meta={
            "target_wall_distance": TARGET_DIST,
            "side": "right",
            "layout": layout,
            "wall_layout_scale": 4.0 if layout == "largemaze" else 1.0,
            "p2_d3_scenario": layout,
        },
    )


@register_task("wall_follow")
def make_wall_follow() -> TaskSpec:
    return _make_wall_task("wall_follow", layout="largemaze", max_steps=4000)


@register_task("wall_straight")
def make_wall_straight() -> TaskSpec:
    return _make_wall_task("wall_straight", layout="straight", max_steps=2000)


@register_task("wall_l_inside")
def make_wall_l_inside() -> TaskSpec:
    return _make_wall_task("wall_l_inside", layout="l_inside", max_steps=2500)


@register_task("wall_l_outside")
def make_wall_l_outside() -> TaskSpec:
    return _make_wall_task("wall_l_outside", layout="l_outside", max_steps=2500)


@register_task("wall_i_corner")
def make_wall_i_corner() -> TaskSpec:
    return _make_wall_task("wall_i_corner", layout="i_corner", max_steps=2500)


@register_task("wall_uturn")
def make_wall_uturn() -> TaskSpec:
    return _make_wall_task("wall_uturn", layout="uturn", max_steps=3000)
