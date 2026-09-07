"""Tabular Q-learning / SARSA trainer for ``avinash_wall`` (PDF wall-follow)."""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import robolab.archs  # noqa: F401
import robolab.sims.genesis  # noqa: F401
import robolab.sims.isaac_sim  # noqa: F401
import robolab.sims.isaaclab  # noqa: F401
import robolab.sims.mujoco  # noqa: F401
import robolab.sims.pybullet  # noqa: F401
import robolab.tasks  # noqa: F401
from robolab.archs.avinash_wall import (
    TabularQAgent,
    discrete_to_continuous,
    epsilon_for_episode,
    pdf_reward,
    state_index,
)
from robolab.core.run import RunConfig
from robolab.core.sim import get_sim
from robolab.core.task import get_task
from robolab.robots.paths import urdf_path
from robolab.train.callbacks import _post_json


def _ranges_from_obs_info(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    if "ranges" in info:
        return np.asarray(info["ranges"], dtype=np.float64)
    return np.asarray(obs, dtype=np.float64).reshape(-1)[:5]


def _build_env(cfg: RunConfig):
    sim = get_sim(cfg.sim)
    task = get_task(cfg.task)
    # PDF episode length (1200) — do not use stuck-escape (not in the project).
    ep_max = int(cfg.arch_cfg.get("episode_max_steps", 1200))
    task.max_steps = ep_max
    robot = sim.load_robot(urdf_path(cfg.robot), robot=cfg.robot)
    env = sim.make_env(task=task, robot=robot, domain=cfg.domain, render=False)
    return env, task


def train_avinash_wall(cfg: RunConfig, run_id: str, backend_url: str) -> dict:
    import wandb

    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not api_key:
        err = (
            "WANDB_API_KEY is missing. Get a key at https://wandb.ai/authorize "
            "and set it in .env, then restart make dev."
        )
        _post_json(
            f"{backend_url.rstrip('/')}/api/runs/{run_id}/fail",
            {"error": err},
        )
        raise RuntimeError(err)

    project = (os.environ.get("WANDB_PROJECT") or "robolab").strip() or "robolab"
    entity = (os.environ.get("WANDB_ENTITY") or "").strip() or None
    run_name = cfg.name or f"{cfg.task}-{cfg.arch}-{cfg.sim}-{run_id[:8]}"
    backend = backend_url.rstrip("/")
    wandb_run = None
    env = None

    arch_cfg = dict(cfg.arch_cfg or {})
    # Prefer trainer.lr as α when UI sends it; PDF default α=0.1.
    if "alpha" not in arch_cfg and cfg.trainer.lr is not None:
        arch_cfg["alpha"] = float(cfg.trainer.lr)
    if "gamma" not in arch_cfg:
        arch_cfg["gamma"] = float(cfg.trainer.gamma)

    total_steps = int(cfg.trainer.timesteps)
    seed = int(cfg.trainer.seed)

    try:
        try:
            init_kwargs: dict = {
                "project": project,
                "name": run_name,
                "config": {**cfg.model_dump(), "trainer_kind": "avinash_wall_tabular"},
                "job_type": "train",
                "tags": ["avinash_wall", "q_tabular", cfg.arch, cfg.task, cfg.sim, cfg.compute],
                "settings": wandb.Settings(init_timeout=120),
            }
            if entity:
                init_kwargs["entity"] = entity
            wandb_run = wandb.init(**init_kwargs)
        except Exception as exc:
            err = (
                f"W&B init failed ({type(exc).__name__}: {exc}). "
                "API returned unauthorized/unreachable. "
                "Create a fresh key at https://wandb.ai/authorize, put "
                "WANDB_API_KEY=... and WANDB_PROJECT=robolab in .env, "
                "restart `make dev`, and retry."
            )
            _post_json(
                f"{backend}/api/runs/{run_id}/fail",
                {"error": err, "traceback": traceback.format_exc()},
            )
            raise RuntimeError(err) from exc

        wandb_url = wandb_run.url
        _post_json(
            f"{backend}/api/runs/{run_id}/heartbeat",
            {
                "step": 0,
                "total": total_steps,
                "mean_return": None,
                "wandb_url": wandb_url,
                "status": "RUNNING",
            },
        )

        env, task = _build_env(cfg)
        obs_dim = int(env.observation_space.shape[0])  # type: ignore[index]
        act_dim = int(env.action_space.shape[0])  # type: ignore[index]
        control_hz = float(cfg.domain.control_hz)
        physics_substeps = int(cfg.domain.physics_substeps)
        physics_dt = (1.0 / max(1e-6, control_hz)) / max(1, physics_substeps)

        agent = TabularQAgent(cfg=arch_cfg, rng=np.random.default_rng(seed))
        param_count = agent.param_count()

        space_meta = {
            "param_count": param_count,
            "arch": cfg.arch,
            "algorithm": agent.algorithm,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "control_hz": control_hz,
            "physics_substeps": physics_substeps,
            "control_dt": 1.0 / max(1e-6, control_hz),
            "physics_dt": physics_dt,
            "q_table_shape": list(agent.q.shape),
        }
        if wandb_run is not None:
            wandb_run.config.update(space_meta, allow_val_change=True)
            for k, v in space_meta.items():
                wandb_run.summary[k] = v

        _post_json(
            f"{backend}/api/runs/{run_id}/heartbeat",
            {
                "step": 0,
                "total": total_steps,
                "mean_return": None,
                "wandb_url": wandb_url,
                "status": "RUNNING",
                "param_count": param_count,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "control_hz": control_hz,
                "physics_substeps": physics_substeps,
            },
        )

        near_max = float(arch_cfg.get("near_max", 0.7))
        medium_max = float(arch_cfg.get("medium_max", 0.9))
        lin = float(arch_cfg.get("linear_vel", 0.3))
        ang = float(arch_cfg.get("angular_vel", 0.7))

        global_step = 0
        episode = 0
        recent_returns: list[float] = []
        last_hb = 0.0
        mean_ret: float | None = None

        while global_step < total_steps:
            eps = epsilon_for_episode(
                episode,
                epsilon_start=float(arch_cfg.get("epsilon_start", 1.0)),
                epsilon_end=float(arch_cfg.get("epsilon_end", 0.1)),
                epsilon_decay=float(arch_cfg.get("epsilon_decay", 0.05)),
                explore_episodes=int(arch_cfg.get("explore_episodes", 200)),
            )
            obs, info = env.reset(seed=seed + episode)
            ranges = _ranges_from_obs_info(obs, info)
            s = state_index(ranges, near_max=near_max, medium_max=medium_max)
            a = agent.select_action(s, eps)
            ep_ret = 0.0
            done = False

            while not done and global_step < total_steps:
                action = discrete_to_continuous(a, linear_vel=lin, angular_vel=ang)
                obs, _r_env, terminated, truncated, info = env.step(action)
                # PDF reward R(s, a) from pre-transition state + discrete action.
                r = pdf_reward(s, a)
                ranges_next = _ranges_from_obs_info(obs, info)
                s_next = state_index(ranges_next, near_max=near_max, medium_max=medium_max)
                done = bool(terminated or truncated)

                if agent.algorithm == "sarsa":
                    a_next = agent.select_action(s_next, 0.0 if done else eps)
                    agent.update_sarsa(s, a, r, s_next, a_next, done)
                    a = a_next
                else:
                    agent.update_q_learning(s, a, r, s_next, done)
                    if not done:
                        a = agent.select_action(s_next, eps)

                s = s_next
                ep_ret += r
                global_step += 1

                now = time.time()
                if global_step % 256 == 0 or (now - last_hb) >= 2.0:
                    last_hb = now
                    mean_ret = (
                        float(sum(recent_returns[-20:]) / len(recent_returns[-20:]))
                        if recent_returns
                        else None
                    )
                    _post_json(
                        f"{backend}/api/runs/{run_id}/heartbeat",
                        {
                            "step": global_step,
                            "total": total_steps,
                            "mean_return": mean_ret,
                            "wandb_url": wandb_url,
                            "status": "RUNNING",
                            "param_count": param_count,
                        },
                    )
                    if wandb_run is not None and mean_ret is not None:
                        wandb_run.log(
                            {
                                "rollout/ep_rew_mean": mean_ret,
                                "train/epsilon": eps,
                                "train/episode": episode,
                            },
                            step=global_step,
                        )

            recent_returns.append(ep_ret)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "rollout/ep_rew": ep_ret,
                        "train/epsilon": eps,
                        "train/episode": episode,
                    },
                    step=global_step,
                )
            episode += 1

        mean_ret = (
            float(sum(recent_returns[-20:]) / len(recent_returns[-20:]))
            if recent_returns
            else None
        )
        if mean_ret is not None and wandb_run is not None:
            wandb_run.log({"rollout/ep_rew_mean": mean_ret}, step=global_step)

        ckpt_dir = Path(os.environ.get("ROBOLAB_CKPT_DIR", "checkpoints")) / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "policy.zip"
        agent.save_zip(ckpt_path)

        artifact_name = f"policy-{run_id}"
        artifact_ref = artifact_name
        if wandb_run is not None:
            art = wandb.Artifact(name=artifact_name, type="model")
            art.add_file(str(ckpt_path), name="policy.zip")
            logged = wandb_run.log_artifact(art)
            try:
                logged.wait(timeout=120)
                ver = getattr(logged, "version", None)
                if ver:
                    artifact_ref = f"{artifact_name}:{ver}"
            except Exception:
                pass

        result = {
            "status": "COMPLETE",
            "wandb_url": wandb_url,
            "mean_return": mean_ret,
            "step": global_step,
            "checkpoint": str(ckpt_path.resolve()),
            "checkpoint_artifact": artifact_ref,
            "param_count": param_count,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "control_hz": control_hz,
            "physics_substeps": physics_substeps,
        }
        _post_json(f"{backend}/api/runs/{run_id}/complete", result)
        return result
    except Exception as exc:
        if "W&B init failed" not in str(exc):
            err = f"{type(exc).__name__}: {exc}"
            _post_json(
                f"{backend}/api/runs/{run_id}/fail",
                {"error": err, "traceback": traceback.format_exc()},
            )
        raise
    finally:
        if env is not None:
            env.close()
        if wandb_run is not None:
            wandb_run.finish()


# Silence unused-import lint for helpers re-exported for tests / render.
__all__ = [
    "train_avinash_wall",
    "discrete_to_continuous",
]
