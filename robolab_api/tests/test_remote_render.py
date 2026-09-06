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


def test_local_launch_render_refuses_genesis(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    import pytest

    from robolab.compute.local import launch_render

    cfg = tmp_path / "config.yaml"
    cfg.write_text("sim: genesis\ntask: go_to_goal\nrobot: diffdrive_lidar\narch: mlp\n")
    with pytest.raises(RuntimeError, match="Local video render refuses"):
        launch_render(
            run_id="deadbeef",
            config_path=cfg,
            wandb_url="https://wandb.ai/e/p/runs/x",
            checkpoint=None,
            sim="genesis",
        )
    # Path-only (no sim=) still refuses after reading config
    with pytest.raises(RuntimeError, match="Local video render refuses"):
        launch_render(
            run_id="deadbeef",
            config_path=cfg,
            wandb_url="https://wandb.ai/e/p/runs/x",
            checkpoint=None,
        )
