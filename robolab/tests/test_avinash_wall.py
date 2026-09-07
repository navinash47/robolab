"""Unit tests for avinash_wall PDF state / reward / ε schedule (no inventing)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from robolab.archs.avinash_wall import (
    ACT_FORWARD,
    ACT_TURN_LEFT,
    ACT_TURN_RIGHT,
    DIST_FAR,
    DIST_MEDIUM,
    DIST_NEAR,
    N_ACTIONS,
    N_STATES,
    TabularQAgent,
    decode_state,
    discrete_to_continuous,
    discretize_distance,
    epsilon_for_episode,
    epsilon_for_progress,
    pdf_reward,
    state_index,
)
from robolab.core.arch import get_arch, list_archs


def test_registered():
    assert "avinash_wall" in list_archs()
    cls = get_arch("avinash_wall")
    assert cls.name == "avinash_wall"


def test_distance_bins_match_pdf():
    assert discretize_distance(0.0) == DIST_NEAR
    assert discretize_distance(0.7) == DIST_NEAR
    assert discretize_distance(0.7001) == DIST_MEDIUM
    assert discretize_distance(0.9) == DIST_MEDIUM
    assert discretize_distance(0.9001) == DIST_FAR


def test_state_index_uses_front_right_left_rays():
    # ranges: front, +45, -45, left(+90), right(-90)
    ranges = np.array([0.5, 9.0, 9.0, 1.5, 0.8], dtype=np.float64)
    idx = state_index(ranges)
    f, r, le = decode_state(idx)
    assert f == DIST_NEAR
    assert r == DIST_MEDIUM
    assert le == DIST_FAR


def test_pdf_reward_cases():
    # front near + turn left → +15
    s_front_near = DIST_NEAR * 9 + DIST_MEDIUM * 3 + DIST_FAR
    assert pdf_reward(s_front_near, ACT_TURN_LEFT) == 15.0
    assert pdf_reward(s_front_near, ACT_FORWARD) == -8.0
    assert pdf_reward(s_front_near, ACT_TURN_RIGHT) == -8.0

    # front not near, right medium → +20
    s_ok = DIST_FAR * 9 + DIST_MEDIUM * 3 + DIST_FAR
    assert pdf_reward(s_ok, ACT_FORWARD) == 20.0

    # right far → -5
    s_far = DIST_FAR * 9 + DIST_FAR * 3 + DIST_FAR
    assert pdf_reward(s_far, ACT_FORWARD) == -5.0

    # right near → -1
    s_near = DIST_FAR * 9 + DIST_NEAR * 3 + DIST_FAR
    assert pdf_reward(s_near, ACT_FORWARD) == -1.0


def test_epsilon_schedule_pdf():
    """Course PDF episode schedule (reference only; trainers use progress)."""
    assert epsilon_for_episode(0) == pytest.approx(1.0)
    assert epsilon_for_episode(1) == pytest.approx(0.95)
    assert epsilon_for_episode(18) == pytest.approx(0.1)
    assert epsilon_for_episode(100) == pytest.approx(0.1)
    assert epsilon_for_episode(199) == pytest.approx(0.1)
    assert epsilon_for_episode(200) == pytest.approx(0.0)


def test_epsilon_for_progress_scales_to_total_timesteps():
    total = 1_000_000
    assert epsilon_for_progress(0, total) == pytest.approx(1.0)
    mid = epsilon_for_progress(total // 2, total)
    assert mid == pytest.approx(0.525)  # midpoint of 1.0 → 0.05
    assert epsilon_for_progress(total, total) == pytest.approx(0.05)
    # Custom end still linear over full budget (e.g. 2M run).
    assert epsilon_for_progress(1_000_000, 2_000_000, epsilon_end=0.05) == pytest.approx(0.525)
    assert epsilon_for_progress(2_000_000, 2_000_000, epsilon_end=0.05) == pytest.approx(0.05)


def test_actions_constant_linear_and_signed_yaw():
    left = discrete_to_continuous(ACT_TURN_LEFT)
    fwd = discrete_to_continuous(ACT_FORWARD)
    right = discrete_to_continuous(ACT_TURN_RIGHT)
    assert left[0] == pytest.approx(0.3 / 0.5)
    assert fwd[0] == pytest.approx(0.3 / 0.5)
    assert right[0] == pytest.approx(0.3 / 0.5)
    assert left[1] > 0  # RoboLab left = +yaw
    assert right[1] < 0
    assert fwd[1] == pytest.approx(0.0)
    assert abs(left[1]) == pytest.approx(0.7 / 1.2)


def test_q_learning_update_and_checkpoint_roundtrip():
    agent = TabularQAgent(cfg={"algorithm": "q_learning", "alpha": 0.1, "gamma": 1.0})
    assert agent.q.shape == (N_STATES, N_ACTIONS)
    s, a, r, s2 = 0, ACT_FORWARD, 20.0, 1
    before = agent.q[s, a]
    agent.update_q_learning(s, a, r, s2, done=False)
    assert agent.q[s, a] != before

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "policy.zip"
        agent.save_zip(path)
        loaded = TabularQAgent.load_zip(path)
        assert np.allclose(loaded.q, agent.q)
        assert loaded.algorithm == "q_learning"
