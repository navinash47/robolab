"""Stuck detector + brief reverse/yaw escape kick (arch-agnostic env wrapper).

Triggers when pose barely moves for N steps, or when scraping a nearby surface
with near-zero forward speed. Does not touch wall collision resolution.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

# Tunable named constants (~50 Hz control → stuck_steps≈1 s).
STUCK_POSE_EPS = 0.05  # m; progress below this counts as stuck
STUCK_STEPS = 50  # consecutive low-progress steps (~1 s)
SCRAPE_MIN_RANGE = 0.18  # m; lidar scrape proximity
SCRAPE_SPEED_EPS = 0.08  # m/s; |forward_speed| considered stalled
SCRAPE_STEPS = 30  # consecutive scrape+stall steps
ESCAPE_STEPS = 18  # recover action duration
ESCAPE_LINEAR = -0.55  # normalized reverse
ESCAPE_YAW = 0.85  # normalized |yaw| during recover


class StuckEscapeWrapper(gym.Wrapper):
    """Override actions briefly when the robot is stuck in a corner/U-turn."""

    def __init__(
        self,
        env: gym.Env,
        *,
        pose_eps: float = STUCK_POSE_EPS,
        stuck_steps: int = STUCK_STEPS,
        scrape_min_range: float = SCRAPE_MIN_RANGE,
        scrape_speed_eps: float = SCRAPE_SPEED_EPS,
        scrape_steps: int = SCRAPE_STEPS,
        escape_steps: int = ESCAPE_STEPS,
        escape_linear: float = ESCAPE_LINEAR,
        escape_yaw: float = ESCAPE_YAW,
        seed: int | None = None,
    ):
        super().__init__(env)
        self.pose_eps = float(pose_eps)
        self.stuck_steps = int(stuck_steps)
        self.scrape_min_range = float(scrape_min_range)
        self.scrape_speed_eps = float(scrape_speed_eps)
        self.scrape_steps = int(scrape_steps)
        self.escape_horizon = int(escape_steps)
        self.escape_linear = float(escape_linear)
        self.escape_yaw = float(escape_yaw)
        self._rng = np.random.default_rng(seed)
        self._reset_detector()

    def _reset_detector(self) -> None:
        self._last_xy: np.ndarray | None = None
        self._pose_stall = 0
        self._scrape_stall = 0
        self._escape_left = 0
        self._escape_yaw_sign = 1.0
        self._escape_active = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._reset_detector()
        if seed is not None:
            self._rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_xy = self._xy_from_info(info)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1).copy()
        recovering = self._escape_left > 0
        if recovering:
            action[0] = self.escape_linear
            action[1] = self._escape_yaw_sign * self.escape_yaw
            self._escape_left -= 1
            self._escape_active = True
        else:
            self._escape_active = False

        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["stuck_escape_active"] = bool(recovering or self._escape_left > 0)
        info["stuck_escape_triggered"] = False

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        if recovering:
            # Do not accumulate stall while kicking free.
            self._last_xy = self._xy_from_info(info)
            self._pose_stall = 0
            self._scrape_stall = 0
            return obs, reward, terminated, truncated, info

        xy = self._xy_from_info(info)
        if xy is not None and self._last_xy is not None:
            delta = float(np.linalg.norm(xy - self._last_xy))
            if delta < self.pose_eps:
                self._pose_stall += 1
            else:
                self._pose_stall = 0
            self._last_xy = xy
        elif xy is not None:
            self._last_xy = xy

        min_r = float(info.get("min_range", 5.0))
        speed = abs(float(info.get("forward_speed", 0.0)))
        if min_r < self.scrape_min_range and speed < self.scrape_speed_eps:
            self._scrape_stall += 1
        else:
            self._scrape_stall = 0

        if self._pose_stall >= self.stuck_steps or self._scrape_stall >= self.scrape_steps:
            self._begin_escape(info)
            info["stuck_escape_triggered"] = True
            info["stuck_escape_active"] = True

        return obs, reward, terminated, truncated, info

    def _begin_escape(self, info: dict[str, Any]) -> None:
        self._escape_left = self.escape_horizon
        self._pose_stall = 0
        self._scrape_stall = 0
        # Prefer yaw toward the more open side when lidar is available.
        ranges = info.get("ranges")
        if ranges is not None:
            r = np.asarray(ranges, dtype=np.float64).reshape(-1)
            if r.size >= 5:
                left, right = float(r[3]), float(r[4])
                if left > right + 1e-6:
                    self._escape_yaw_sign = 1.0
                elif right > left + 1e-6:
                    self._escape_yaw_sign = -1.0
                else:
                    self._escape_yaw_sign = float(self._rng.choice([-1.0, 1.0]))
                return
        self._escape_yaw_sign = float(self._rng.choice([-1.0, 1.0]))

    @staticmethod
    def _xy_from_info(info: dict[str, Any] | None) -> np.ndarray | None:
        if not info:
            return None
        pos = info.get("position")
        if pos is None:
            return None
        arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            return None
        return arr[:2].copy()


def maybe_wrap_stuck_escape(env: gym.Env, task_name: str | None) -> gym.Env:
    """Apply StuckEscapeWrapper for wall_follow only (all arches share this path)."""
    if task_name == "wall_follow":
        return StuckEscapeWrapper(env)
    return env
