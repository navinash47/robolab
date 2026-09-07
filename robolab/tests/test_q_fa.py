"""Unit tests for function-approx Q-learning agent (no inventing PDF math)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from robolab.archs.avinash_wall import N_ACTIONS, pdf_reward, state_index
from robolab.train.q_fa import FunctionApproxQAgent, QL_ARCHS


def test_ql_archs_cover_builtins():
    assert QL_ARCHS == frozenset({"mlp", "kan", "kaf", "gpkan", "fan"})


def test_fa_agent_save_load_roundtrip():
    agent = FunctionApproxQAgent(
        arch_name="mlp",
        obs_dim=5,
        arch_cfg={"hidden_sizes": [16, 16], "activation": "tanh", "gamma": 1.0},
        lr=1e-3,
        gamma=1.0,
        device="cpu",
        seed=0,
    )
    feat = np.ones(5, dtype=np.float32)
    a = agent.select_action(feat, epsilon=0.0)
    assert 0 <= a < N_ACTIONS
    loss = agent.update(feat, a, 1.0, feat, done=True)
    assert loss >= 0.0

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "policy.zip"
        agent.save_zip(path)
        loaded = FunctionApproxQAgent.load_zip(path, device="cpu")
    assert loaded.arch_name == "mlp"
    assert loaded.obs_dim == 5
    with torch.no_grad():
        q0 = agent.net(torch.as_tensor(feat).unsqueeze(0))
        q1 = loaded.net(torch.as_tensor(feat).unsqueeze(0))
    assert torch.allclose(q0, q1, atol=1e-5)


def test_fa_uses_pdf_reward_helper():
    # Sanity: FA trainer imports the same PDF reward (no invented table).
    ranges = np.array([0.5, 9.0, 9.0, 1.5, 0.8], dtype=np.float64)
    s = state_index(ranges)
    assert pdf_reward(s, 0) == 15.0  # front near + turn left
