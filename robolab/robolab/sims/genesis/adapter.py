"""Genesis SimAdapter — real when genesis-world is installed; else actionable error."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from robolab.core.run import DomainParams
from robolab.core.sim import SimAdapter, register_sim
from robolab.core.task import TaskSpec
from robolab.robots.paths import urdf_path
from robolab.tasks.worlds import (
    build_observation,
    enrich_task_info,
    layout_for_task,
)

LIDAR_ANGLES_DEG = (0.0, 45.0, -45.0, 90.0, -90.0)
LIDAR_MAX = 5.0

_GS_INITED = False
_INSTALL_HINT = (
    "Simulator 'genesis' requires the optional package genesis-world "
    "(~80MB wheel + heavy deps). Install with: pip install genesis-world\n"
    "Docs: https://genesis-world.readthedocs.io/ — see docs/PHASE_5B_APIS.md. "
    "CPU backend works for smoke; GPU recommended for training."
)


def genesis_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("genesis") is not None
    except Exception:
        return False


def _prepare_headless_gl() -> None:
    """Genesis pulls pyglet at import time; headless GPU pods have no X display."""
    import os

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "all")
    try:
        import pyglet

        pyglet.options["headless"] = True
    except Exception:
        pass


def _ensure_gs():
    global _GS_INITED
    if not genesis_available():
        raise RuntimeError(_INSTALL_HINT)
    _prepare_headless_gl()
    import genesis as gs

    if not _GS_INITED:
        # Prefer CPU for portability; users with CUDA can set ROBOLAB_GENESIS_GPU=1.
        import os

        use_gpu = os.environ.get("ROBOLAB_GENESIS_GPU", "").strip() in {"1", "true", "yes"}
        backend = gs.gpu if use_gpu else gs.cpu
        try:
            gs.init(backend=backend, precision="32", logging_level="warning")
        except Exception as exc:
            # GPU init can fail on driver/EGL mismatch — fall back to CPU once.
            if use_gpu:
                try:
                    gs.init(backend=gs.cpu, precision="32", logging_level="warning")
                except Exception as exc2:
                    raise RuntimeError(
                        f"genesis.init failed gpu=({exc}) cpu=({exc2}). {_INSTALL_HINT}"
                    ) from exc2
            else:
                raise RuntimeError(
                    f"genesis.init failed ({exc}). {_INSTALL_HINT}"
                ) from exc
        _GS_INITED = True
    return gs


class DiffDriveLidarGenesisEnv(gym.Env):
    """Kinematic diffdrive + Genesis Lidar (5 planar rays) matching MuJoCo angles."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: TaskSpec,
        urdf: Path,
        domain: DomainParams,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.task = task
        self.domain = domain
        self.render_mode = render_mode
        self._urdf = Path(urdf)
        self._layout = layout_for_task(task.name)
        self._path_s = 0.0
        self._v_max = 0.5
        self._w_max = 1.2
        self._steps = 0
        self._success_streak = 0
        self._control_dt = 1.0 / max(1e-6, domain.control_hz)
        self._physics_dt = self._control_dt / max(1, domain.physics_substeps)
        self._xy = np.array([*self._layout.spawn_xy], dtype=np.float64)
        self._yaw = float(self._layout.spawn_yaw)
        self._cam = None

        obs_shape = tuple(task.observation.shape)
        self.observation_space = spaces.Box(
            low=float(task.observation.low),
            high=float(task.observation.high),
            shape=obs_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=float(task.action.low),
            high=float(task.action.high),
            shape=task.action.shape,
            dtype=np.float32,
        )

        gs = _ensure_gs()
        self._gs = gs
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self._physics_dt,
                gravity=(0.0, 0.0, -9.81),
            ),
            show_viewer=False,
        )
        self._scene.add_entity(gs.morphs.Plane())
        for half, pos, _rgba in self._layout.boxes:
            size = (2 * half[0], 2 * half[1], 2 * half[2])
            self._scene.add_entity(
                gs.morphs.Box(size=size, pos=tuple(pos), fixed=True)
            )
        if not self._urdf.is_file():
            raise FileNotFoundError(f"URDF missing: {self._urdf}")
        sx, sy = self._layout.spawn_xy
        self._robot = self._scene.add_entity(
            gs.morphs.URDF(
                file=str(self._urdf.resolve()),
                pos=(sx, sy, 0.05),
                quat=(1.0, 0.0, 0.0, 0.0),
                fixed=False,
            )
        )
        self._lidar = self._scene.add_sensor(
            gs.sensors.Lidar(
                pattern=gs.sensors.SphericalPattern(
                    angles=(list(LIDAR_ANGLES_DEG), [0.0]),
                ),
                entity_idx=self._robot.idx,
                pos_offset=(0.0, 0.0, 0.12),
                max_range=LIDAR_MAX,
                min_range=0.05,
                return_world_frame=False,
                draw_debug=False,
            )
        )
        if render_mode == "rgb_array":
            self._cam = self._scene.add_camera(
                res=(320, 240),
                pos=(sx - 2.0, sy, 2.5),
                lookat=(sx, sy, 0.05),
                fov=60,
                GUI=False,
                debug=True,
            )
        self._scene.build()
        self._apply_pose(self._xy[0], self._xy[1], self._yaw)
        for _ in range(max(1, domain.physics_substeps)):
            self._scene.step()

    def _quat_wxyz(self, yaw: float) -> tuple[float, float, float, float]:
        return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))

    def _apply_pose(self, x: float, y: float, yaw: float) -> None:
        self._robot.set_pos(np.array([x, y, 0.05], dtype=np.float32))
        self._robot.set_quat(np.array(self._quat_wxyz(yaw), dtype=np.float32))

    def _lidar_ranges(self) -> np.ndarray:
        data = self._lidar.read()
        dists = data.distances
        if hasattr(dists, "detach"):
            arr = dists.detach().cpu().numpy()
        else:
            arr = np.asarray(dists)
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        if flat.size < len(LIDAR_ANGLES_DEG):
            # Fallback: pad to 5 rays
            out = np.full(len(LIDAR_ANGLES_DEG), LIDAR_MAX, dtype=np.float32)
            out[: flat.size] = flat
            return out
        out = flat[: len(LIDAR_ANGLES_DEG)].astype(np.float32)
        out = np.clip(out, 0.0, LIDAR_MAX)
        if self.domain.sensor_noise_std > 0:
            out = out + self.np_random.normal(
                0.0, self.domain.sensor_noise_std, size=out.shape
            ).astype(np.float32)
            out = np.clip(out, 0.0, LIDAR_MAX)
        return out

    def _pack(self, ranges: np.ndarray, forward_speed: float) -> tuple[np.ndarray, dict]:
        pos = np.array([self._xy[0], self._xy[1], 0.05], dtype=np.float64)
        extra = enrich_task_info(
            self.task.name,
            self.task.meta,
            position=pos,
            yaw=self._yaw,
            ranges=ranges,
            forward_speed=forward_speed,
            path_s=self._path_s,
        )
        obs = build_observation(self.task.name, ranges, extra["rel_goal"])
        info = {
            "ranges": ranges.copy(),
            "position": pos,
            "forward_speed": forward_speed,
            "steps": self._steps,
            "control_hz": self.domain.control_hz,
            "physics_substeps": self.domain.physics_substeps,
            "control_dt": self._control_dt,
            "physics_dt": self._physics_dt,
            **extra,
        }
        return obs.astype(np.float32), info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        layout = self._layout
        y = float(self.np_random.uniform(-layout.spawn_noise_y, layout.spawn_noise_y))
        yaw = float(
            layout.spawn_yaw
            + self.np_random.uniform(-layout.spawn_noise_yaw, layout.spawn_noise_yaw)
        )
        self._xy = np.array(
            [layout.spawn_xy[0], layout.spawn_xy[1] + y], dtype=np.float64
        )
        self._yaw = yaw
        self._apply_pose(self._xy[0], self._xy[1], self._yaw)
        for _ in range(max(1, self.domain.physics_substeps)):
            self._scene.step()
        self._steps = 0
        self._success_streak = 0
        self._path_s = 0.0
        ranges = self._lidar_ranges()
        return self._pack(ranges, 0.0)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        v = float(action[0]) * self._v_max
        w = float(action[1]) * self._w_max
        dt = self._control_dt
        self._yaw = self._yaw + w * dt
        self._xy[0] += v * np.cos(self._yaw) * dt
        self._xy[1] += v * np.sin(self._yaw) * dt
        self._apply_pose(self._xy[0], self._xy[1], self._yaw)
        for _ in range(max(1, self.domain.physics_substeps)):
            self._scene.step()
        self._apply_pose(self._xy[0], self._xy[1], self._yaw)

        if self.task.name == "figure8_tracking":
            scale = float(self.task.meta.get("path_scale", 1.6))
            self._path_s += (abs(v) * dt) / max(0.2, scale)

        self._steps += 1
        ranges = self._lidar_ranges()
        obs, info = self._pack(ranges, v)
        reward = float(self.task.reward(info)) if self.task.reward else 0.0
        if self.task.success and self.task.success(info):
            self._success_streak += 1
        else:
            self._success_streak = 0
        info["success"] = self._success_streak >= 20
        info["is_success"] = info["success"]
        terminated = bool(self.task.termination(info)) if self.task.termination else False
        truncated = self._steps >= self.task.max_steps
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array" or self._cam is None:
            return None
        self._cam.set_pose(
            pos=(self._xy[0] - 2.5, self._xy[1], 2.5),
            lookat=(self._xy[0], self._xy[1], 0.05),
        )
        rgb, _, _, _ = self._cam.render(rgb=True)
        return np.asarray(rgb, dtype=np.uint8)

    def close(self):
        self._cam = None
        self._scene = None


@register_sim("genesis")
class GenesisAdapter(SimAdapter):
    name = "genesis"

    def load_robot(self, urdf_path_arg: Path, **kw: Any) -> dict[str, Any]:
        robot = kw.get("robot", "diffdrive_lidar")
        u = Path(urdf_path_arg) if urdf_path_arg else urdf_path(robot)
        if not u.exists():
            raise FileNotFoundError(f"URDF missing: {u}")
        return {"urdf": u, "robot": robot}

    def make_env(
        self,
        task: TaskSpec,
        robot: Any,
        domain: DomainParams,
        render: bool = False,
    ) -> gym.Env:
        if not genesis_available():
            raise RuntimeError(_INSTALL_HINT)
        if isinstance(robot, dict):
            model_path = Path(robot["urdf"])
        else:
            model_path = urdf_path(str(robot))
        return DiffDriveLidarGenesisEnv(
            task=task,
            urdf=model_path,
            domain=domain,
            render_mode="rgb_array" if render else None,
        )

    def perturb(self, env: gym.Env, params: DomainParams) -> None:
        if isinstance(env, DiffDriveLidarGenesisEnv):
            env.domain = params

    def render_frame(self, env: gym.Env) -> Any:
        frame = env.render()
        if frame is None:
            raise RuntimeError("Env was not created with render=True")
        return frame

    def capabilities(self) -> set[str]:
        caps = {
            "urdf",
            "lidar",
            "optional_gpu",
            "pip:genesis-world",
            "multi_task",
            "heavy_optional_dep",
        }
        if not genesis_available():
            caps.add("requires_install")
        else:
            caps.add("installed")
        return caps
