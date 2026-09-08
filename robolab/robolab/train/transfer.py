"""Phase 6: zero-shot sim-to-sim transfer eval + robustness sweeps."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import robolab.archs  # noqa: F401
import robolab.sims.genesis  # noqa: F401
import robolab.sims.isaac_sim  # noqa: F401
import robolab.sims.isaaclab  # noqa: F401
import robolab.sims.mujoco  # noqa: F401
import robolab.sims.pybullet  # noqa: F401
import robolab.tasks  # noqa: F401
from robolab.core.run import DomainParams, RunConfig
from robolab.core.sim import get_sim
from robolab.core.task import get_task
from robolab.envs import maybe_wrap_stuck_escape
from robolab.robots.paths import urdf_path
from robolab.train.callbacks import _post_json

FRICTION_SWEEP = [0.5, 1.0, 1.5, 2.0]
MASS_SWEEP = [0.75, 1.0, 1.25]
NOISE_SWEEP = [0.0, 0.02, 0.05]
MAC_SKIP_SIMS = frozenset({"genesis", "isaaclab", "isaac_sim"})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def transfer_dir(run_id: str) -> Path:
    return repo_root() / "transfer" / run_id


def _mean_std_ci(returns: list[float]) -> dict[str, Any]:
    n = len(returns)
    if n == 0:
        return {
            "mean": None,
            "std": None,
            "ci_low": None,
            "ci_high": None,
            "n": 0,
            "returns": [],
        }
    arr = np.asarray(returns, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n >= 2 else 0.0
    if n >= 2 and std >= 0.0:
        half = 1.96 * std / math.sqrt(n)
        ci_low, ci_high = mean - half, mean + half
    else:
        ci_low = ci_high = None
    return {
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "returns": [float(x) for x in returns],
    }


def transfer_ratio(source_mean: float | None, target_mean: float | None) -> float | None:
    if source_mean is None or target_mean is None:
        return None
    if abs(source_mean) < 1e-6:
        return None
    return float(target_mean / source_mean)


def gap(source_mean: float | None, target_mean: float | None) -> float | None:
    if source_mean is None or target_mean is None:
        return None
    return float(source_mean - target_mean)


def _default_targets(cfg: RunConfig, targets: list[str] | None) -> list[str]:
    if targets:
        return [str(t).strip().lower() for t in targets if str(t).strip()]
    if cfg.transfer_to:
        return [str(t).strip().lower() for t in cfg.transfer_to if str(t).strip()]
    src = (cfg.sim or "mujoco").strip().lower()
    if src == "mujoco":
        return ["pybullet"]
    return ["mujoco"]


def _sim_available(sim_name: str) -> tuple[bool, str | None]:
    key = sim_name.strip().lower()
    if key in MAC_SKIP_SIMS and platform.system() == "Darwin":
        return False, f"{key} skipped on macOS (RunPod-only)"
    try:
        get_sim(key)
    except KeyError as exc:
        return False, str(exc)
    if key == "genesis":
        try:
            import genesis  # noqa: F401
        except Exception as exc:  # pragma: no cover
            return False, f"genesis not importable: {exc}"
    if key in {"isaaclab", "isaac_sim"}:
        # Stubs may register; refuse real Mac transfer work.
        if platform.system() == "Darwin":
            return False, f"{key} skipped on macOS"
    return True, None


def _load_policy(cfg: RunConfig, checkpoint: Path):
    use_avinash = cfg.arch == "avinash_wall"
    use_q_fa = (not use_avinash) and str(cfg.trainer.algo or "").lower() == "q_learning"
    if use_avinash:
        from robolab.archs.avinash_wall import TabularQAgent

        return ("avinash", TabularQAgent.load_zip(checkpoint, cfg=dict(cfg.arch_cfg or {})))
    if use_q_fa:
        from robolab.train.q_fa import FunctionApproxQAgent

        return ("q_fa", FunctionApproxQAgent.load_zip(checkpoint, device="cpu"))
    from stable_baselines3 import PPO

    return ("ppo", PPO.load(str(checkpoint), device="cpu"))


def _act(kind: str, agent: Any, obs: Any, env: Any) -> Any:
    base = getattr(env, "unwrapped", env)
    if kind == "avinash":
        ranges = np.asarray(obs, dtype=np.float64).reshape(-1)[:5]
        if hasattr(base, "_lidar"):
            ranges = np.asarray(base._lidar(), dtype=np.float64)
        return agent.act_normalized(ranges, deterministic=True)
    if kind == "q_fa":
        ranges = np.asarray(obs, dtype=np.float64).reshape(-1)[:5]
        if hasattr(base, "_lidar"):
            ranges = np.asarray(base._lidar(), dtype=np.float64)
        return agent.act_normalized(
            np.asarray(obs, dtype=np.float32), ranges, deterministic=True
        )
    action, _ = agent.predict(obs, deterministic=True)
    return action


def _make_env(cfg: RunConfig, sim_name: str, domain: DomainParams):
    sim = get_sim(sim_name)
    task = get_task(cfg.task)
    # Cap episode length for transfer smoke (override via env).
    max_steps = int(os.environ.get("ROBOLAB_TRANSFER_MAX_STEPS", "500"))
    if max_steps > 0:
        task.max_steps = min(int(task.max_steps), max_steps)
    robot = sim.load_robot(urdf_path(cfg.robot), robot=cfg.robot)
    env = sim.make_env(task=task, robot=robot, domain=domain, render=False)
    return maybe_wrap_stuck_escape(env, cfg.task)


def _rollout_return(
    *,
    kind: str,
    agent: Any,
    cfg: RunConfig,
    sim_name: str,
    domain: DomainParams,
    seed: int,
) -> float:
    env = _make_env(cfg, sim_name, domain)
    try:
        obs, _info = env.reset(seed=seed)
        terminated = truncated = False
        total = 0.0
        steps = 0
        limit = int(getattr(getattr(env, "unwrapped", env), "task", None).max_steps or 500) + 50
        while not (terminated or truncated) and steps < limit:
            action = _act(kind, agent, obs, env)
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            steps += 1
        return total
    finally:
        env.close()


def _eval_sim(
    *,
    kind: str,
    agent: Any,
    cfg: RunConfig,
    sim_name: str,
    domain: DomainParams,
    n_episodes: int,
    seed: int,
) -> dict[str, Any]:
    ok, reason = _sim_available(sim_name)
    if not ok:
        return {"sim": sim_name, "status": "skipped", "reason": reason}
    returns: list[float] = []
    try:
        for i in range(n_episodes):
            ret = _rollout_return(
                kind=kind,
                agent=agent,
                cfg=cfg,
                sim_name=sim_name,
                domain=domain,
                seed=seed + i,
            )
            returns.append(ret)
    except Exception as exc:
        return {
            "sim": sim_name,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    stats = _mean_std_ci(returns)
    return {"sim": sim_name, "status": "ok", **stats}


def _sweep_axis(
    *,
    kind: str,
    agent: Any,
    cfg: RunConfig,
    sim_name: str,
    base_domain: DomainParams,
    axis: str,
    values: list[float],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ok, reason = _sim_available(sim_name)
    if not ok:
        for v in values:
            rows.append(
                {
                    "sim": sim_name,
                    "axis": axis,
                    "value": v,
                    "mean_return": None,
                    "status": "skipped",
                    "reason": reason,
                }
            )
        return rows
    for i, v in enumerate(values):
        domain = base_domain.model_copy(update={axis: v})
        try:
            ret = _rollout_return(
                kind=kind,
                agent=agent,
                cfg=cfg,
                sim_name=sim_name,
                domain=domain,
                seed=seed + i,
            )
            rows.append(
                {
                    "sim": sim_name,
                    "axis": axis,
                    "value": v,
                    "mean_return": ret,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "sim": sim_name,
                    "axis": axis,
                    "value": v,
                    "mean_return": None,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def run_transfer_eval(
    *,
    cfg: RunConfig,
    checkpoint: Path,
    run_id: str,
    targets: list[str] | None = None,
    n_episodes: int = 5,
    seed: int | None = None,
    out_dir: Path | None = None,
    wandb_run: Any = None,
) -> dict[str, Any]:
    """Zero-shot transfer + perturbation curves → transfer/{run_id}/report.json."""
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    target_sims = _default_targets(cfg, targets)
    # Never eval source as a "target" duplicate unless explicitly listed alone.
    target_sims = [t for t in target_sims if t]
    ep_seed = int(cfg.trainer.seed if seed is None else seed)
    n_ep = max(1, int(n_episodes))
    kind, agent = _load_policy(cfg, ckpt)
    base_domain = cfg.domain.model_copy()

    source = _eval_sim(
        kind=kind,
        agent=agent,
        cfg=cfg,
        sim_name=cfg.sim,
        domain=base_domain,
        n_episodes=n_ep,
        seed=ep_seed,
    )
    source_mean = source.get("mean")

    target_rows: list[dict[str, Any]] = []
    for t in target_sims:
        row = _eval_sim(
            kind=kind,
            agent=agent,
            cfg=cfg,
            sim_name=t,
            domain=base_domain,
            n_episodes=n_ep,
            seed=ep_seed + 1000,
        )
        t_mean = row.get("mean")
        row["transfer_ratio"] = transfer_ratio(source_mean, t_mean if isinstance(t_mean, float) else None)
        row["gap"] = gap(source_mean, t_mean if isinstance(t_mean, float) else None)
        target_rows.append(row)

    sims_for_robust = [cfg.sim] + [
        r["sim"] for r in target_rows if r.get("status") == "ok"
    ]
    # Dedupe preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for s in sims_for_robust:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    robustness: dict[str, list[dict[str, Any]]] = {
        "friction": [],
        "mass_scale": [],
        "sensor_noise_std": [],
    }
    for sim_name in ordered:
        robustness["friction"].extend(
            _sweep_axis(
                kind=kind,
                agent=agent,
                cfg=cfg,
                sim_name=sim_name,
                base_domain=base_domain,
                axis="friction",
                values=FRICTION_SWEEP,
                seed=ep_seed + 2000,
            )
        )
        robustness["mass_scale"].extend(
            _sweep_axis(
                kind=kind,
                agent=agent,
                cfg=cfg,
                sim_name=sim_name,
                base_domain=base_domain,
                axis="mass_scale",
                values=MASS_SWEEP,
                seed=ep_seed + 3000,
            )
        )
        robustness["sensor_noise_std"].extend(
            _sweep_axis(
                kind=kind,
                agent=agent,
                cfg=cfg,
                sim_name=sim_name,
                base_domain=base_domain,
                axis="sensor_noise_std",
                values=NOISE_SWEEP,
                seed=ep_seed + 4000,
            )
        )

    report: dict[str, Any] = {
        "run_id": run_id,
        "status": "READY",
        "source_sim": cfg.sim,
        "arch": cfg.arch,
        "task": cfg.task,
        "n_episodes": n_ep,
        "checkpoint": str(ckpt.resolve()),
        "source": source,
        "targets": target_rows,
        "robustness": robustness,
    }

    dest = out_dir or transfer_dir(run_id)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "report.json").write_text(json.dumps(report, indent=2))

    if wandb_run is not None:
        try:
            payload: dict[str, Any] = {}
            if source_mean is not None:
                payload["transfer/source_mean"] = source_mean
            for row in target_rows:
                sim = row["sim"]
                if row.get("mean") is not None:
                    payload[f"transfer/{sim}_mean"] = row["mean"]
                if row.get("transfer_ratio") is not None:
                    payload[f"transfer/{sim}_ratio"] = row["transfer_ratio"]
                if row.get("gap") is not None:
                    payload[f"transfer/{sim}_gap"] = row["gap"]
            if payload:
                wandb_run.log(payload)
        except Exception:
            pass

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RoboLab Phase 6 transfer eval")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--targets", default="", help="Comma-separated sims")
    parser.add_argument("--n-episodes", type=int, default=5)
    args = parser.parse_args(argv)

    backend = args.backend_url.rstrip("/")
    run_id = args.run_id
    try:
        raw = yaml.safe_load(args.config.read_text())
        cfg = RunConfig.model_validate(raw)
        targets = [t.strip() for t in args.targets.split(",") if t.strip()] or None
        report = run_transfer_eval(
            cfg=cfg,
            checkpoint=args.checkpoint,
            run_id=run_id,
            targets=targets,
            n_episodes=args.n_episodes,
        )
        _post_json(
            f"{backend}/api/runs/{run_id}/transfer-complete",
            {"status": "READY", "report": report},
        )
        return 0
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        _post_json(
            f"{backend}/api/runs/{run_id}/transfer-fail",
            {"error": err, "traceback": traceback.format_exc()},
        )
        print(err, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
