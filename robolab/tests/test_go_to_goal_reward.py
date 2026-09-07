"""go_to_goal reward: spin-in-place must lose to progressing toward the goal."""

from __future__ import annotations

import math

from robolab.core.task import get_task
from robolab.tasks.go_to_goal import _reward


def _cum(
    *,
    n: int,
    dist0: float,
    delta_per_step: float,
    yaw_err: float,
    forward: float,
    angular_vel: float,
) -> float:
    """Accumulate reward for a synthetic trajectory (no sim)."""
    total = 0.0
    dist = dist0
    for _ in range(n):
        prev = dist
        dist = max(0.0, dist - delta_per_step)
        total += _reward(
            {
                "dist_to_goal": dist,
                "prev_dist_to_goal": prev,
                "yaw_err": yaw_err,
                "forward_speed": forward,
                "angular_vel": angular_vel,
                "min_range": 5.0,
                "wall_contact": False,
            }
        )
    return total


def test_spin_in_place_worse_than_approach():
    """Constant spin farms less return than closing distance over N steps."""
    n = 100
    dist0 = 5.0
    # Spin: full yaw-rate, zero forward, distance unchanged; yaw_err briefly 0
    # (best-case heading alignment while spinning — still should lose).
    spin = _cum(
        n=n,
        dist0=dist0,
        delta_per_step=0.0,
        yaw_err=0.0,
        forward=0.0,
        angular_vel=1.2,
    )
    # Approach: move toward goal with modest forward speed, no spin.
    approach = _cum(
        n=n,
        dist0=dist0,
        delta_per_step=0.01,  # ~v_max * dt at 50 Hz
        yaw_err=0.2,
        forward=0.4,
        angular_vel=0.0,
    )
    assert approach > spin, f"approach={approach:.3f} should beat spin={spin:.3f}"
    assert spin < 0.0, f"spin cumulative should be negative, got {spin:.3f}"
    assert approach > 0.0, f"approach cumulative should be positive, got {approach:.3f}"


def test_spin_worse_than_standing_still():
    """Constant spin is worse than standing still (no free heading farm)."""
    n = 50
    dist0 = 4.0
    spin = _cum(
        n=n,
        dist0=dist0,
        delta_per_step=0.0,
        yaw_err=0.0,
        forward=0.0,
        angular_vel=1.2,
    )
    still = _cum(
        n=n,
        dist0=dist0,
        delta_per_step=0.0,
        yaw_err=0.0,
        forward=0.0,
        angular_vel=0.0,
    )
    assert spin < still, f"spin={spin:.3f} should be < still={still:.3f}"


def test_heading_requires_forward_motion():
    """Facing the goal with zero forward speed yields no heading bonus."""
    base = {
        "dist_to_goal": 3.0,
        "prev_dist_to_goal": 3.0,
        "yaw_err": 0.0,
        "angular_vel": 0.0,
        "min_range": 5.0,
        "wall_contact": False,
    }
    r_still = _reward({**base, "forward_speed": 0.0})
    r_fwd = _reward({**base, "forward_speed": 0.3})
    assert r_fwd > r_still
    # Still facing goal ≈ proximity only (weak); must not match old 0.5 heading farm.
    assert r_still < 0.25


def test_go_to_goal_task_wires_reward():
    task = get_task("go_to_goal")
    assert task.reward is _reward
    assert math.isclose(float(task.meta["goal_radius"]), 0.25)
