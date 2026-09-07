"""Avinash Wall Follow — tabular Q-learning from course project P2_D3.

Replicates the PDF wall-following design:
- 3 lidar sectors (front / right / left) × 3 distance bins → 27 states
- 3 discrete actions (turn left / forward / turn right) at constant linear speed
- Piecewise reward R(s, a) exactly as specified
- Q-learning (chosen over SARSA: faster final-phase convergence + smoother corners)

Not an SB3 neural backbone — trainer.py branches to the tabular runner.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from robolab.core.arch import Architecture, register_arch

# --- PDF distance bins (meters) ---
NEAR_MAX = 0.7
MEDIUM_MAX = 0.9

# --- PDF continuous speeds (m/s, rad/s) ---
LINEAR_VEL = 0.3
ANGULAR_VEL = 0.7

# RoboLab DiffDriveLidarEnv scales: v = a0 * v_max, w = a1 * w_max
_V_MAX = 0.5
_W_MAX = 1.2

# Discrete actions (semantic names match PDF reward text).
# RoboLab: +angular = CCW = left. PDF listed left=-0.7 in its own frame;
# we map by semantic left/right so "turning left when front near" matches.
ACT_TURN_LEFT = 0
ACT_FORWARD = 1
ACT_TURN_RIGHT = 2
N_ACTIONS = 3
N_STATES = 27  # 3 × 3 × 3

DIST_NEAR = 0
DIST_MEDIUM = 1
DIST_FAR = 2

AlgoName = Literal["q_learning", "sarsa"]


def default_cfg() -> dict[str, Any]:
    return {
        "algorithm": "q_learning",
        "alpha": 0.1,
        "gamma": 1.0,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        # Legacy PDF episode schedule knobs (kept for epsilon_for_episode / docs).
        "epsilon_decay": 0.05,
        "explore_episodes": 200,
        "episode_max_steps": 1200,
        "linear_vel": LINEAR_VEL,
        "angular_vel": ANGULAR_VEL,
        "near_max": NEAR_MAX,
        "medium_max": MEDIUM_MAX,
        "lr_default": 0.1,
    }


def discretize_distance(d: float, near_max: float = NEAR_MAX, medium_max: float = MEDIUM_MAX) -> int:
    """Near: [0, near_max], Medium: (near_max, medium_max], Far: > medium_max."""
    x = float(d)
    if x <= near_max:
        return DIST_NEAR
    if x <= medium_max:
        return DIST_MEDIUM
    return DIST_FAR


def sectors_from_lidar(ranges: np.ndarray) -> tuple[float, float, float]:
    """Map RoboLab 5-ray lidar to PDF sectors (front / right / left).

    Ray layout: 0 front, 1 +45°, 2 -45°, 3 +90° (left), 4 -90° (right).
    PDF sectors are centered on front / right / left — single matching rays.
    """
    r = np.asarray(ranges, dtype=np.float64).reshape(-1)
    if r.size < 5:
        raise ValueError(f"expected ≥5 lidar ranges, got {r.size}")
    front = float(r[0])
    right = float(r[4])
    left = float(r[3])
    return front, right, left


def state_index(
    ranges: np.ndarray,
    *,
    near_max: float = NEAR_MAX,
    medium_max: float = MEDIUM_MAX,
) -> int:
    """State = (front, right, left) each in {near, medium, far} → index in [0, 26]."""
    front, right, left = sectors_from_lidar(ranges)
    f = discretize_distance(front, near_max, medium_max)
    ri = discretize_distance(right, near_max, medium_max)
    le = discretize_distance(left, near_max, medium_max)
    return int(f * 9 + ri * 3 + le)


def decode_state(idx: int) -> tuple[int, int, int]:
    f = idx // 9
    rem = idx % 9
    ri = rem // 3
    le = rem % 3
    return f, ri, le


def pdf_reward(state_idx: int, action: int) -> float:
    """Exact piecewise R(s, a) from the PDF (mutually exclusive, safety first).

    +20 if optimal distance (right = medium)
    +15 if turning left when too close to front wall
    −8  if too close to front wall without turning left
    −5  if too far from wall (right = far)
    −1  if too close to wall (right = near)
    """
    front, right, _left = decode_state(int(state_idx))
    a = int(action)
    if front == DIST_NEAR:
        if a == ACT_TURN_LEFT:
            return 15.0
        return -8.0
    if right == DIST_MEDIUM:
        return 20.0
    if right == DIST_FAR:
        return -5.0
    # right == DIST_NEAR
    return -1.0


def discrete_to_continuous(
    action: int,
    *,
    linear_vel: float = LINEAR_VEL,
    angular_vel: float = ANGULAR_VEL,
    v_max: float = _V_MAX,
    w_max: float = _W_MAX,
) -> np.ndarray:
    """Map discrete PDF action → RoboLab normalized [lin, ang] in [-1, 1]."""
    a = int(action)
    v_n = float(linear_vel) / float(v_max)
    w_mag = float(angular_vel) / float(w_max)
    if a == ACT_TURN_LEFT:
        w_n = w_mag  # CCW / left in RoboLab
    elif a == ACT_TURN_RIGHT:
        w_n = -w_mag
    else:
        w_n = 0.0
    return np.asarray([v_n, w_n], dtype=np.float32)


def continuous_to_discrete(action: np.ndarray) -> int:
    """Nearest discrete action from a continuous [lin, ang] command."""
    w = float(np.asarray(action, dtype=np.float64).reshape(-1)[1])
    if w > 0.15:
        return ACT_TURN_LEFT
    if w < -0.15:
        return ACT_TURN_RIGHT
    return ACT_FORWARD


def epsilon_for_episode(
    episode: int,
    *,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay: float = 0.05,
    explore_episodes: int = 200,
) -> float:
    """PDF ε schedule: decay by 0.05/ep until floor, hold until explore_episodes, then 0.

    Kept for course-PDF reference / unit tests. Trainers use
    :func:`epsilon_for_progress` so custom ``total_timesteps`` scale correctly.
    """
    ep = int(episode)
    if ep >= int(explore_episodes):
        return 0.0
    return float(max(epsilon_end, epsilon_start - epsilon_decay * ep))


def epsilon_for_progress(
    step: int,
    total_timesteps: int,
    *,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
) -> float:
    """Linear ε over the full training budget: ``start → end`` as ``step/total → 1``.

    Default RoboLab schedule for any custom ``total_timesteps`` (e.g. 2M):
    ε = 1.0 at step 0, ε = ``epsilon_end`` (0.05) at ``total_timesteps``.
    """
    total = max(int(total_timesteps), 1)
    t = min(max(int(step), 0), total)
    frac = t / float(total)
    return float(epsilon_start + (epsilon_end - epsilon_start) * frac)


class TabularQAgent:
    """27×3 Q-table with ε-greedy + Q-learning or SARSA updates."""

    def __init__(self, cfg: dict[str, Any] | None = None, rng: np.random.Generator | None = None):
        self.cfg = {**default_cfg(), **(cfg or {})}
        self.rng = rng or np.random.default_rng(0)
        self.q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self.algorithm: AlgoName = (
            "sarsa" if str(self.cfg.get("algorithm", "q_learning")).lower() == "sarsa" else "q_learning"
        )

    @property
    def alpha(self) -> float:
        return float(self.cfg.get("alpha", 0.1))

    @property
    def gamma(self) -> float:
        return float(self.cfg.get("gamma", 1.0))

    def select_action(self, state: int, epsilon: float) -> int:
        if self.rng.random() < float(epsilon):
            return int(self.rng.integers(0, N_ACTIONS))
        return int(np.argmax(self.q[int(state)]))

    def update_q_learning(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        target = float(r)
        if not done:
            target += self.gamma * float(np.max(self.q[int(s_next)]))
        self.q[int(s), int(a)] += self.alpha * (target - self.q[int(s), int(a)])

    def update_sarsa(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool) -> None:
        target = float(r)
        if not done:
            target += self.gamma * float(self.q[int(s_next), int(a_next)])
        self.q[int(s), int(a)] += self.alpha * (target - self.q[int(s), int(a)])

    def act_normalized(self, ranges: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        s = state_index(
            ranges,
            near_max=float(self.cfg.get("near_max", NEAR_MAX)),
            medium_max=float(self.cfg.get("medium_max", MEDIUM_MAX)),
        )
        a = int(np.argmax(self.q[s])) if deterministic else self.select_action(s, 0.0)
        return discrete_to_continuous(
            a,
            linear_vel=float(self.cfg.get("linear_vel", LINEAR_VEL)),
            angular_vel=float(self.cfg.get("angular_vel", ANGULAR_VEL)),
        )

    def save_zip(self, path: Path) -> None:
        import io

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            bio = io.BytesIO()
            np.save(bio, self.q)
            zf.writestr("q_table.npy", bio.getvalue())
            meta = {
                "kind": "avinash_wall_q_table",
                "algorithm": self.algorithm,
                "cfg": self.cfg,
                "n_states": N_STATES,
                "n_actions": N_ACTIONS,
            }
            zf.writestr("meta.json", json.dumps(meta, indent=2))

    @classmethod
    def load_zip(cls, path: Path, cfg: dict[str, Any] | None = None) -> TabularQAgent:
        path = Path(path)
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if "q_table.npy" not in names:
                raise FileNotFoundError(f"q_table.npy missing in {path}")
            import io

            q = np.load(io.BytesIO(zf.read("q_table.npy")))
            meta: dict[str, Any] = {}
            if "meta.json" in names:
                meta = json.loads(zf.read("meta.json").decode("utf-8"))
        merged = {**default_cfg(), **(meta.get("cfg") or {}), **(cfg or {})}
        agent = cls(cfg=merged)
        if q.shape != (N_STATES, N_ACTIONS):
            raise ValueError(f"bad Q shape {q.shape}, expected {(N_STATES, N_ACTIONS)}")
        agent.q = np.asarray(q, dtype=np.float64)
        if meta.get("algorithm"):
            agent.algorithm = (
                "sarsa" if str(meta["algorithm"]).lower() == "sarsa" else "q_learning"
            )
        return agent

    def param_count(self) -> int:
        return int(self.q.size)


@register_arch("avinash_wall")
class AvinashWallArch(Architecture):
    """Registry entry for New Run / Architectures UI.

    Forward is unused — tabular training lives in ``train/q_tabular.py``.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict | None = None):
        merged = {**default_cfg(), **(cfg or {})}
        super().__init__(obs_dim, act_dim, merged)
        self._latent_pi = 1
        self._latent_vf = 1
        # Tiny placeholders so accidental SB3 construction does not crash on shape.
        self.pi_net = nn.Linear(obs_dim, 1)
        self.vf_net = nn.Linear(obs_dim, 1)

    @property
    def latent_dim_pi(self) -> int:
        return self._latent_pi

    @property
    def latent_dim_vf(self) -> int:
        return self._latent_vf

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "avinash_wall is tabular Q-learning — use the dedicated trainer path, not SB3 PPO."
        )

    def param_count(self) -> int:
        return N_STATES * N_ACTIONS
