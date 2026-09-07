"""wall_follow stuck escape + stall/scrape penalty."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from robolab.envs.stuck_escape import (
    ESCAPE_LINEAR,
    ESCAPE_YAW,
    StuckEscapeWrapper,
)
from robolab.tasks.wall_follow import (
    STALL_MIN_RANGE,
    STALL_PENALTY,
    STALL_SPEED_EPS,
    _reward,
    _stall_scrape_penalty,
)


def _base_info(**overrides):
    info = {
        "ranges": np.array([1.0, 1.0, 1.0, 1.0, 0.35], dtype=np.float64),
        "forward_speed": 0.4,
        "min_range": 1.0,
        "wall_contact": False,
        "position": np.array([1.0, 0.0, 0.05], dtype=np.float64),
    }
    info.update(overrides)
    return info


def test_stall_scrape_penalty_fires_when_scraping_and_slow():
    pen = _stall_scrape_penalty(
        _base_info(min_range=STALL_MIN_RANGE - 0.01, forward_speed=STALL_SPEED_EPS / 2)
    )
    assert pen == STALL_PENALTY


def test_stall_scrape_penalty_skips_when_moving_or_clear():
    assert _stall_scrape_penalty(_base_info(min_range=0.05, forward_speed=0.3)) == 0.0
    assert _stall_scrape_penalty(_base_info(min_range=1.0, forward_speed=0.0)) == 0.0


def test_reward_includes_stall_penalty():
    clear = _reward(_base_info(min_range=1.0, forward_speed=0.4))
    stuck = _reward(
        _base_info(min_range=STALL_MIN_RANGE - 0.02, forward_speed=0.0)
    )
    assert stuck < clear
    assert stuck <= clear + STALL_PENALTY + 0.5  # other terms similar


class _StuckFakeEnv(gym.Env):
    """Fixed pose / scrape stall; records the last action seen."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(0.0, 5.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.last_action = None
        self._xy = np.array([0.0, 0.0], dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_action = None
        return self.observation_space.sample(), self._info()

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float64).copy()
        obs = np.zeros(5, dtype=np.float32)
        return obs, 0.0, False, False, self._info()

    def _info(self):
        return {
            "position": np.array([self._xy[0], self._xy[1], 0.05], dtype=np.float64),
            "min_range": 0.10,
            "forward_speed": 0.0,
            "ranges": np.array([0.15, 0.2, 0.2, 0.5, 0.12], dtype=np.float64),
            "wall_contact": False,
        }


def test_stuck_escape_activates_on_scrape_stall():
    env = StuckEscapeWrapper(
        _StuckFakeEnv(),
        stuck_steps=10_000,  # disable pose path
        scrape_steps=5,
        escape_steps=4,
        seed=0,
    )
    env.reset(seed=0)
    triggered = False
    for _ in range(5):
        _obs, _r, _t, _tr, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        if info.get("stuck_escape_triggered"):
            triggered = True
            break
    assert triggered, "scrape stall should trigger escape"

    env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env.unwrapped.last_action is not None
    assert env.unwrapped.last_action[0] == ESCAPE_LINEAR
    assert abs(env.unwrapped.last_action[1]) == ESCAPE_YAW


def test_stuck_escape_activates_on_pose_stall():
    class PoseStuckEnv(_StuckFakeEnv):
        def _info(self):
            # Far from walls so scrape path stays quiet; pose never moves.
            return {
                "position": np.array([0.0, 0.0, 0.05], dtype=np.float64),
                "min_range": 2.0,
                "forward_speed": 0.4,
                "ranges": np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64),
                "wall_contact": False,
            }

    env = StuckEscapeWrapper(
        PoseStuckEnv(),
        stuck_steps=4,
        scrape_steps=10_000,
        escape_steps=3,
        pose_eps=0.05,
        seed=1,
    )
    env.reset(seed=1)
    triggered = False
    for _ in range(4):
        _obs, _r, _t, _tr, info = env.step(np.array([0.5, 0.0], dtype=np.float32))
        if info.get("stuck_escape_triggered"):
            triggered = True
            break
    assert triggered, "pose stall should trigger escape"
    env.step(np.array([0.5, 0.0], dtype=np.float32))
    assert env.unwrapped.last_action[0] == ESCAPE_LINEAR
