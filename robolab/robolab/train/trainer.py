"""Build env → policy → learn → checkpoint → heartbeat/complete."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure registries are populated
import robolab.archs  # noqa: F401
import robolab.sims.genesis  # noqa: F401
import robolab.sims.isaac_sim  # noqa: F401
import robolab.sims.isaaclab  # noqa: F401
import robolab.sims.mujoco  # noqa: F401
import robolab.sims.pybullet  # noqa: F401
import robolab.tasks  # noqa: F401
from robolab.core.run import RunConfig
from robolab.core.sim import get_sim
from robolab.core.task import get_task
from robolab.envs import maybe_wrap_stuck_escape
from robolab.robots.paths import urdf_path
from robolab.train.callbacks import WandbAndHeartbeatCallback, _post_json
from robolab.train.sb3_policy import RoboLabActorCriticPolicy


def load_config(path: Path) -> RunConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return RunConfig.model_validate(raw)


def build_env(cfg: RunConfig):
    sim = get_sim(cfg.sim)
    task = get_task(cfg.task)
    robot = sim.load_robot(urdf_path(cfg.robot), robot=cfg.robot)

    def _thunk():
        env = sim.make_env(task=task, robot=robot, domain=cfg.domain, render=False)
        return maybe_wrap_stuck_escape(env, cfg.task)

    return make_vec_env(
        _thunk,
        n_envs=cfg.trainer.n_envs,
        seed=cfg.trainer.seed,
        vec_env_cls=DummyVecEnv,
    )


def train(cfg: RunConfig, run_id: str, backend_url: str) -> dict:
    # PDF wall-follow Q-learning: tabular (avinash_wall) or function-approx (kan/kaf/…).
    algo = str(cfg.trainer.algo or "ppo").lower()
    if cfg.arch == "avinash_wall" or algo == "q_learning":
        if cfg.arch == "avinash_wall":
            from robolab.train.q_tabular import train_avinash_wall

            return train_avinash_wall(cfg, run_id, backend_url)
        from robolab.train.q_fa import train_q_fa

        return train_q_fa(cfg, run_id, backend_url)

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

    try:
        try:
            init_kwargs: dict = {
                "project": project,
                "name": run_name,
                "config": cfg.model_dump(),
                "job_type": "train",
                "tags": [
                    "arch-builder" if cfg.arch in {"kaf", "gpkan", "fan"} else ("phase2" if cfg.arch == "kan" else "phase1"),
                    cfg.arch,
                    cfg.task,
                    cfg.sim,
                    cfg.compute,
                ],
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
            _post_json(f"{backend}/api/runs/{run_id}/fail", {"error": err, "traceback": traceback.format_exc()})
            raise RuntimeError(err) from exc

        wandb_url = wandb_run.url

        # Announce URL early so dashboard can show the link
        _post_json(
            f"{backend}/api/runs/{run_id}/heartbeat",
            {
                "step": 0,
                "total": cfg.trainer.timesteps,
                "mean_return": None,
                "wandb_url": wandb_url,
                "status": "RUNNING",
            },
        )

        device = cfg.trainer.device
        if device == "cuda":
            import subprocess
            import sys

            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import torch; assert torch.cuda.is_available(); "
                    "torch.zeros(1, device='cuda'); print('cuda_ok')",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if probe.returncode != 0 or "cuda_ok" not in (probe.stdout or ""):
                detail = ((probe.stderr or "") + (probe.stdout or ""))[-400:]
                print(
                    f"CUDA subprocess probe failed (rc={probe.returncode}): {detail}\n"
                    "Falling back to device=cpu and hiding GPUs.",
                    flush=True,
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                device = "cpu"
                cfg = cfg.model_copy(
                    update={"trainer": cfg.trainer.model_copy(update={"device": "cpu"})}
                )

        env = build_env(cfg)
        # Obs / action dims from the live env (must match across mujoco|pybullet).
        obs_dim = int(env.observation_space.shape[0])  # type: ignore[index]
        act_dim = int(env.action_space.shape[0])  # type: ignore[index]
        control_hz = float(cfg.domain.control_hz)
        physics_substeps = int(cfg.domain.physics_substeps)
        physics_dt = (1.0 / max(1e-6, control_hz)) / max(1, physics_substeps)

        policy_kwargs = {
            "arch_name": cfg.arch,
            "arch_cfg": dict(cfg.arch_cfg),
        }
        model = PPO(
            RoboLabActorCriticPolicy,
            env,
            learning_rate=cfg.trainer.lr,
            n_steps=cfg.trainer.n_steps,
            batch_size=cfg.trainer.batch_size,
            gamma=cfg.trainer.gamma,
            verbose=1,
            device=device,
            seed=cfg.trainer.seed,
            policy_kwargs=policy_kwargs,
        )

        param_count = int(model.policy.mlp_extractor.arch.param_count())
        space_meta = {
            "param_count": param_count,
            "arch": cfg.arch,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "control_hz": control_hz,
            "physics_substeps": physics_substeps,
            "control_dt": 1.0 / max(1e-6, control_hz),
            "physics_dt": physics_dt,
        }
        if wandb_run is not None:
            wandb_run.config.update(space_meta, allow_val_change=True)
            for k, v in space_meta.items():
                wandb_run.summary[k] = v

        _post_json(
            f"{backend}/api/runs/{run_id}/heartbeat",
            {
                "step": 0,
                "total": cfg.trainer.timesteps,
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

        cb = WandbAndHeartbeatCallback(
            run_id=run_id,
            total_timesteps=cfg.trainer.timesteps,
            backend_url=backend,
            wandb_run=wandb_run,
            heartbeat_every_steps=max(256, cfg.trainer.n_steps // 4),
            param_count=param_count,
        )

        model.learn(total_timesteps=cfg.trainer.timesteps, callback=CallbackList([cb]))
        mean_ret = cb._mean_return()
        if mean_ret is not None and wandb_run is not None:
            # SB3 often overshoots configured timesteps (n_steps alignment); never
            # log a lower step than wandb already saw or W&B drops the point.
            step = int(getattr(model, "num_timesteps", 0) or cfg.trainer.timesteps)
            wandb_run.log({"rollout/ep_rew_mean": mean_ret}, step=step)

        ckpt_dir = Path(os.environ.get("ROBOLAB_CKPT_DIR", "checkpoints")) / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "policy.zip"
        model.save(str(ckpt_path))

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
            "step": cfg.trainer.timesteps,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RoboLab trainer")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"),
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    # Single-seed local run for Phase 1: use first seed
    if cfg.seeds:
        cfg.trainer.seed = int(cfg.seeds[0])
    train(cfg, run_id=args.run_id, backend_url=args.backend_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
