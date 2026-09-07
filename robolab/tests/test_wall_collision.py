"""Wall solidity for kinematic go_to_goal (and shared layout boxes)."""

from __future__ import annotations

import numpy as np
import pytest

from robolab.core.run import DomainParams
from robolab.core.task import get_task
import robolab.tasks  # noqa: F401 — register tasks
from robolab.sims.mujoco.adapter import MujocoAdapter
from robolab.sims.pybullet.adapter import PybulletAdapter
from robolab.tasks.worlds import (
    OPEN_PERIMETER,
    ROBOT_COLLISION_RADIUS,
    is_wall_crash,
    layout_for_task,
    resolve_wall_collision,
)


def test_resolve_rejects_tunnel_through_east_wall():
    # East wall center x=8, half-extent 0.1 → faces at 7.9 / 8.1.
    inside, hit = resolve_wall_collision([7.95, 0.0], OPEN_PERIMETER)
    assert hit
    assert inside[0] <= 7.9 - ROBOT_COLLISION_RADIUS + 1e-6


def test_resolve_one_step_cannot_cross_wall():
    # From clear space inside, a jump past the thin east wall must stop at contact.
    out, hit = resolve_wall_collision(
        [9.5, 0.0], OPEN_PERIMETER, prev_xy=[7.5, 0.0]
    )
    assert hit
    assert out[0] <= 7.9 - ROBOT_COLLISION_RADIUS + 1e-6
    assert out[0] < 8.0


def test_is_wall_crash_from_contact_flag():
    assert is_wall_crash({"wall_contact": True, "min_range": 5.0})
    assert not is_wall_crash({"wall_contact": False, "min_range": 1.0})


def test_spawn_not_in_collision():
    layout = layout_for_task("go_to_goal")
    xy, hit = resolve_wall_collision(layout.spawn_xy, layout.boxes)
    assert not hit
    np.testing.assert_allclose(xy[:2], layout.spawn_xy, atol=1e-9)


def _drive_into_east_wall(env, steps: int = 800) -> tuple[float, bool]:
    """Drive +x at full throttle; return final x and whether wall_contact ever fired."""
    env.reset(seed=0)
    saw_contact = False
    last_x = 0.0
    for _ in range(steps):
        _obs, _r, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        saw_contact = saw_contact or bool(info.get("wall_contact"))
        last_x = float(info["position"][0])
        if term or trunc:
            break
    return last_x, saw_contact


@pytest.mark.parametrize("adapter_cls", [MujocoAdapter, PybulletAdapter])
def test_go_to_goal_cannot_drive_through_east_wall(adapter_cls):
    task = get_task("go_to_goal")
    adapter = adapter_cls()
    robot = adapter.load_robot(None, robot="diffdrive_lidar", task="go_to_goal")
    env = adapter.make_env(task, robot, DomainParams(), render=False)
    try:
        last_x, saw_contact = _drive_into_east_wall(env)
        # Wall face at 7.9; robot center must stay west of face − radius.
        assert last_x <= 7.9 - ROBOT_COLLISION_RADIUS + 0.05
        assert saw_contact
    finally:
        env.close()
