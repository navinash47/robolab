"""Function-approx Q-learning for wall_follow using builtin arch towers.

Same PDF discrete actions / reward as ``avinash_wall`` tabular; ε decays
linearly over ``total_timesteps`` (see ``epsilon_for_progress``).
Q(s,·) is a neural net whose feature tower is ``mlp|kan|kaf|gpkan|fan``.
"""

from __future__ import annotations

import io
import json
import os
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import robolab.archs  # noqa: F401
import robolab.sims.genesis  # noqa: F401
import robolab.sims.isaac_sim  # noqa: F401
import robolab.sims.isaaclab  # noqa: F401
import robolab.sims.mujoco  # noqa: F401
import robolab.sims.pybullet  # noqa: F401
import robolab.tasks  # noqa: F401
from robolab.archs.avinash_wall import (
    N_ACTIONS,
    N_STATES,
    default_cfg as wall_default_cfg,
    discrete_to_continuous,
    epsilon_for_progress,
    pdf_reward,
    state_index,
)
from robolab.core.arch import get_arch
from robolab.core.run import RunConfig
from robolab.core.sim import get_sim
from robolab.core.task import get_task
from robolab.robots.paths import urdf_path
from robolab.train.callbacks import _post_json

QL_ARCHS = frozenset({"mlp", "kan", "kaf", "gpkan", "fan"})


def _ranges_from_obs_info(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    if "ranges" in info:
        return np.asarray(info["ranges"], dtype=np.float64)
    return np.asarray(obs, dtype=np.float64).reshape(-1)[:5]


def _build_env(cfg: RunConfig):
    sim = get_sim(cfg.sim)
    task = get_task(cfg.task)
    ep_max = int(cfg.arch_cfg.get("episode_max_steps", 1200))
    task.max_steps = ep_max
    robot = sim.load_robot(urdf_path(cfg.robot), robot=cfg.robot)
    env = sim.make_env(task=task, robot=robot, domain=cfg.domain, render=False)
    return env, task


def _obs_features(
    obs: np.ndarray,
    ranges: np.ndarray,
    *,
    mode: str,
    near_max: float,
    medium_max: float,
) -> np.ndarray:
    """Feature vector for the Q-network.

    ``lidar`` — raw env observation (continuous).
    ``onehot27`` — one-hot of the PDF 27-state index (tabular-compatible input).
    """
    if mode == "onehot27":
        s = state_index(ranges, near_max=near_max, medium_max=medium_max)
        feat = np.zeros(N_STATES, dtype=np.float32)
        feat[int(s)] = 1.0
        return feat
    return np.asarray(obs, dtype=np.float32).reshape(-1)


class DiscreteQNet(nn.Module):
    """Arch feature tower + linear head → n_actions Q-values."""

    def __init__(self, arch_name: str, obs_dim: int, arch_cfg: dict[str, Any]):
        super().__init__()
        if arch_name not in QL_ARCHS:
            raise ValueError(f"q_learning FA requires arch in {sorted(QL_ARCHS)}, got {arch_name!r}")
        # act_dim unused by feature towers; pass n_actions for registry consistency.
        self.backbone = get_arch(arch_name)(obs_dim=obs_dim, act_dim=N_ACTIONS, cfg=dict(arch_cfg))
        self.q_head = nn.Linear(int(self.backbone.latent_dim_pi), N_ACTIONS)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pi_feat, _vf = self.backbone(obs)
        return self.q_head(pi_feat)


class FunctionApproxQAgent:
    """Online semi-gradient Q-learning with ε-greedy over discrete PDF actions."""

    def __init__(
        self,
        arch_name: str,
        obs_dim: int,
        arch_cfg: dict[str, Any] | None = None,
        *,
        lr: float = 3e-4,
        gamma: float = 1.0,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.arch_name = arch_name
        self.obs_dim = int(obs_dim)
        self.cfg = {**wall_default_cfg(), **(arch_cfg or {})}
        # Drop tabular-only α; neural uses Adam lr.
        self.cfg.pop("alpha", None)
        self.gamma = float(self.cfg.get("gamma", gamma))
        self.lr = float(lr)
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(int(seed))

        net_cfg = {
            k: v
            for k, v in self.cfg.items()
            if k
            not in {
                "algorithm",
                "alpha",
                "gamma",
                "epsilon_start",
                "epsilon_end",
                "epsilon_decay",
                "explore_episodes",
                "episode_max_steps",
                "linear_vel",
                "angular_vel",
                "near_max",
                "medium_max",
                "lr_default",
                "obs_mode",
            }
        }
        self.net = DiscreteQNet(arch_name, self.obs_dim, net_cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.obs_mode = str(self.cfg.get("obs_mode", "lidar")).lower()
        if self.obs_mode not in {"lidar", "onehot27"}:
            self.obs_mode = "lidar"

    def param_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def _to_t(self, feat: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, feat: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < float(epsilon):
            return int(self.rng.integers(0, N_ACTIONS))
        with torch.no_grad():
            q = self.net(self._to_t(feat))
            return int(torch.argmax(q, dim=-1).item())

    def update(
        self,
        feat: np.ndarray,
        action: int,
        reward: float,
        feat_next: np.ndarray,
        done: bool,
    ) -> float:
        self.net.train()
        q_all = self.net(self._to_t(feat))
        q_sa = q_all[0, int(action)]
        with torch.no_grad():
            target = float(reward)
            if not done:
                q_next = self.net(self._to_t(feat_next))
                target += self.gamma * float(torch.max(q_next).item())
            target_t = torch.tensor(target, dtype=torch.float32, device=self.device)
        loss = nn.functional.mse_loss(q_sa, target_t)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def act_normalized(self, obs: np.ndarray, ranges: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        near_max = float(self.cfg.get("near_max", 0.7))
        medium_max = float(self.cfg.get("medium_max", 0.9))
        feat = _obs_features(
            obs, ranges, mode=self.obs_mode, near_max=near_max, medium_max=medium_max
        )
        if feat.shape[0] != self.obs_dim:
            # Fallback: rebuild from ranges if obs layout drifted
            feat = _obs_features(
                ranges, ranges, mode=self.obs_mode, near_max=near_max, medium_max=medium_max
            )
        a = self.select_action(feat, 0.0 if deterministic else 0.0)
        return discrete_to_continuous(
            a,
            linear_vel=float(self.cfg.get("linear_vel", 0.3)),
            angular_vel=float(self.cfg.get("angular_vel", 0.7)),
        )

    def save_zip(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "arch_name": self.arch_name,
                "obs_dim": self.obs_dim,
                "cfg": self.cfg,
                "obs_mode": self.obs_mode,
                "lr": self.lr,
                "gamma": self.gamma,
            },
            buf,
        )
        meta = {
            "kind": "q_fa",
            "arch": self.arch_name,
            "obs_dim": self.obs_dim,
            "n_actions": N_ACTIONS,
            "obs_mode": self.obs_mode,
            "cfg": self.cfg,
            "lr": self.lr,
            "gamma": self.gamma,
        }
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("q_fa.pt", buf.getvalue())
            zf.writestr("meta.json", json.dumps(meta, indent=2))

    @classmethod
    def load_zip(cls, path: Path, device: str = "cpu") -> FunctionApproxQAgent:
        path = Path(path)
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if "q_fa.pt" not in names:
                raise FileNotFoundError(f"q_fa.pt missing in {path}")
            blob = torch.load(io.BytesIO(zf.read("q_fa.pt")), map_location="cpu", weights_only=False)
            meta: dict[str, Any] = {}
            if "meta.json" in names:
                meta = json.loads(zf.read("meta.json").decode("utf-8"))
        arch_name = str(blob.get("arch_name") or meta.get("arch") or "mlp")
        obs_dim = int(blob.get("obs_dim") or meta.get("obs_dim") or 5)
        cfg = dict(blob.get("cfg") or meta.get("cfg") or {})
        agent = cls(
            arch_name=arch_name,
            obs_dim=obs_dim,
            arch_cfg=cfg,
            lr=float(blob.get("lr") or meta.get("lr") or 3e-4),
            gamma=float(blob.get("gamma") or meta.get("gamma") or cfg.get("gamma", 1.0)),
            device=device,
            seed=0,
        )
        agent.net.load_state_dict(blob["state_dict"])
        agent.net.eval()
        return agent


def train_q_fa(cfg: RunConfig, run_id: str, backend_url: str) -> dict:
    import wandb

    if cfg.arch not in QL_ARCHS:
        raise ValueError(
            f"trainer.algo=q_learning requires arch in {sorted(QL_ARCHS)}; "
            f"got {cfg.arch!r} (use avinash_wall for tabular)."
        )

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
    run_name = cfg.name or f"{cfg.task}-{cfg.arch}-ql-{cfg.sim}-{run_id[:8]}"
    backend = backend_url.rstrip("/")
    wandb_run = None
    env = None

    arch_cfg = {**wall_default_cfg(), **dict(cfg.arch_cfg or {})}
    # Neural FA uses Adam lr from trainer; keep PDF γ / ε unless overridden.
    lr = float(cfg.trainer.lr)
    if "gamma" not in (cfg.arch_cfg or {}):
        arch_cfg["gamma"] = float(cfg.trainer.gamma)

    total_steps = int(cfg.trainer.timesteps)
    seed = int(cfg.trainer.seed)
    device = str(cfg.trainer.device)

    try:
        try:
            init_kwargs: dict = {
                "project": project,
                "name": run_name,
                "config": {**cfg.model_dump(), "trainer_kind": "q_fa"},
                "job_type": "train",
                "tags": ["q_fa", "q_learning", cfg.arch, cfg.task, cfg.sim, cfg.compute],
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
        obs0, info0 = env.reset(seed=seed)
        ranges0 = _ranges_from_obs_info(obs0, info0)
        near_max = float(arch_cfg.get("near_max", 0.7))
        medium_max = float(arch_cfg.get("medium_max", 0.9))
        obs_mode = str(arch_cfg.get("obs_mode", "lidar")).lower()
        feat0 = _obs_features(
            obs0, ranges0, mode=obs_mode, near_max=near_max, medium_max=medium_max
        )
        obs_dim = int(feat0.shape[0])
        act_dim = int(env.action_space.shape[0])  # type: ignore[index]
        control_hz = float(cfg.domain.control_hz)
        physics_substeps = int(cfg.domain.physics_substeps)
        physics_dt = (1.0 / max(1e-6, control_hz)) / max(1, physics_substeps)

        agent = FunctionApproxQAgent(
            arch_name=cfg.arch,
            obs_dim=obs_dim,
            arch_cfg=arch_cfg,
            lr=lr,
            gamma=float(arch_cfg.get("gamma", 1.0)),
            device=device,
            seed=seed,
        )
        param_count = agent.param_count()

        space_meta = {
            "param_count": param_count,
            "arch": cfg.arch,
            "algorithm": "q_learning_fa",
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "obs_mode": agent.obs_mode,
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

        lin = float(arch_cfg.get("linear_vel", 0.3))
        ang = float(arch_cfg.get("angular_vel", 0.7))
        eps_start = float(arch_cfg.get("epsilon_start", 1.0))
        eps_end = float(arch_cfg.get("epsilon_end", 0.05))

        global_step = 0
        episode = 0
        recent_returns: list[float] = []
        last_hb = 0.0
        mean_ret: float | None = None

        # Continue from the initial reset used for obs_dim probing.
        obs, info = obs0, info0
        while global_step < total_steps:
            eps = epsilon_for_progress(
                global_step,
                total_steps,
                epsilon_start=eps_start,
                epsilon_end=eps_end,
            )
            if episode > 0:
                obs, info = env.reset(seed=seed + episode)
            ranges = _ranges_from_obs_info(obs, info)
            s = state_index(ranges, near_max=near_max, medium_max=medium_max)
            feat = _obs_features(
                obs, ranges, mode=agent.obs_mode, near_max=near_max, medium_max=medium_max
            )
            a = agent.select_action(feat, eps)
            ep_ret = 0.0
            done = False

            while not done and global_step < total_steps:
                eps = epsilon_for_progress(
                    global_step,
                    total_steps,
                    epsilon_start=eps_start,
                    epsilon_end=eps_end,
                )
                action = discrete_to_continuous(a, linear_vel=lin, angular_vel=ang)
                obs, _r_env, terminated, truncated, info = env.step(action)
                r = pdf_reward(s, a)
                ranges_next = _ranges_from_obs_info(obs, info)
                s_next = state_index(ranges_next, near_max=near_max, medium_max=medium_max)
                feat_next = _obs_features(
                    obs,
                    ranges_next,
                    mode=agent.obs_mode,
                    near_max=near_max,
                    medium_max=medium_max,
                )
                done = bool(terminated or truncated)
                agent.update(feat, a, r, feat_next, done)

                if not done:
                    eps_next = epsilon_for_progress(
                        global_step + 1,
                        total_steps,
                        epsilon_start=eps_start,
                        epsilon_end=eps_end,
                    )
                    a = agent.select_action(feat_next, eps_next)
                feat = feat_next
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
                    if wandb_run is not None:
                        payload: dict[str, Any] = {
                            "train/epsilon": eps,
                            "train/episode": episode,
                        }
                        if mean_ret is not None:
                            payload["rollout/ep_rew_mean"] = mean_ret
                        wandb_run.log(payload, step=global_step)

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


__all__ = [
    "FunctionApproxQAgent",
    "DiscreteQNet",
    "QL_ARCHS",
    "train_q_fa",
]
