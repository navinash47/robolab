"""Unit tests for remote genesis/isaac video render routing."""

from __future__ import annotations

import os

from robolab.compute.runpod import needs_remote_render, worker_image_for_sim
from robolab_api.watchdog import decide_render_pod_kill


def test_needs_remote_render_sims() -> None:
    assert needs_remote_render("genesis")
    assert needs_remote_render("isaaclab")
    assert needs_remote_render("isaac_sim")
    assert not needs_remote_render("mujoco")
    assert not needs_remote_render("pybullet")
    assert not needs_remote_render(None)


def test_worker_image_rewrites_phase3_to_genesis(monkeypatch) -> None:
    monkeypatch.delenv("ROBOLAB_WORKER_IMAGE_GENESIS", raising=False)
    img = worker_image_for_sim(
        "genesis", "avinashnandyala2/robolab-worker:phase3"
    )
    assert img.endswith(":genesis")


def test_worker_image_genesis_override(monkeypatch) -> None:
    monkeypatch.setenv(
        "ROBOLAB_WORKER_IMAGE_GENESIS", "avinashnandyala2/robolab-worker:genesis"
    )
    img = worker_image_for_sim("genesis", "avinashnandyala2/robolab-worker:phase3")
    assert img == "avinashnandyala2/robolab-worker:genesis"


def test_render_pod_kill_only_on_runtime() -> None:
    ok = decide_render_pod_kill(runtime_min=5.0, max_runtime_min=25.0)
    assert not ok.should_kill
    bad = decide_render_pod_kill(runtime_min=30.0, max_runtime_min=25.0)
    assert bad.should_kill
    assert "render runtime" in bad.reason
