"""MuJoCo SimAdapter + Gymnasium wall-follow env."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robolab.core.run import DomainParams
from robolab.core.sim import SimAdapter, register_sim
from robolab.core.task import TaskSpec

ROBOTS_ROOT = Path(__file__).resolve().parents[2] / "robots"
LIDAR_ANGLES_DEG = (0.0, 45.0, -45.0, 90.0, -90.0)
LIDAR_MAX = 5.0


def robot_dir(robot: str) -> Path:
    return ROBOTS_ROOT / robot


def scene_path(robot: str) -> Path:
    return robot_dir(robot) / "scene.xml"


def urdf_path(robot: str) -> Path:
    return robot_dir(robot) / "robot.urdf"


class DiffDriveLidarEnv(gym.Env):
    """Differential-drive robot with ray-cast lidar in a MuJoCo corridor."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: TaskSpec,
        model_path: Path,
        domain: DomainParams,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.task = task
        self.domain = domain
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._apply_domain()

        n_rays = len(LIDAR_ANGLES_DEG)
        self.observation_space = spaces.Box(
            low=0.0,
            high=float(task.observation.high),
            shape=(n_rays,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=float(task.action.low),
            high=float(task.action.high),
            shape=task.action.shape,
            dtype=np.float32,
        )

        self._wheel_radius = 0.05
        self._half_track = 0.14
        self._v_max = 0.5
        self._w_max = 1.2
        self._steps = 0
        self._success_streak = 0
        self._renderer: mujoco.Renderer | None = None
        self._cam: mujoco.MjvCamera | None = None

        self._base_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        # Only cast against world geoms (group 0); robot geoms are group 1.
        self._geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self._geomid = np.zeros(1, dtype=np.int32)
        self._left_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
        self._right_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")

    def _apply_domain(self) -> None:
        # Friction scale on wheel geoms
        for name in ("left_wheel_geom", "right_wheel_geom", "floor"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.model.geom_friction[gid, 0] = self.domain.friction * (
                    1.5 if "wheel" in name else 1.0
                )
        # Control / physics timing
        ctrl_dt = 1.0 / self.domain.control_hz
        self.model.opt.timestep = ctrl_dt / max(1, self.domain.physics_substeps)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Spawn near corridor entrance with slight lateral/yaw noise
        y = float(self.np_random.uniform(-0.25, 0.25))
        yaw = float(self.np_random.uniform(-0.2, 0.2))
        quat = np.array(
            [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
            dtype=np.float64,
        )
        self.data.qpos[:3] = np.array([0.3, y, 0.05])
        self.data.qpos[3:7] = quat
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._success_streak = 0
        obs = self._get_obs()
        return obs, {"ranges": obs.copy()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        v = float(action[0]) * self._v_max
        w = float(action[1]) * self._w_max
        dt = 1.0 / max(1e-6, self.domain.control_hz)

        # Kinematic planar drive (stable for Phase 1 RL). MuJoCo used for
        # kinematics + mj_ray lidar; full wheel dynamics deferred.
        quat = self.data.qpos[3:7]
        yaw = float(
            np.arctan2(
                2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
            )
        )
        yaw = yaw + w * dt
        self.data.qpos[0] += v * np.cos(yaw) * dt
        self.data.qpos[1] += v * np.sin(yaw) * dt
        self.data.qpos[2] = 0.05
        self.data.qpos[3:7] = np.array(
            [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
            dtype=np.float64,
        )
        self.data.qvel[:] = 0.0
        if self._left_joint >= 0 and self._right_joint >= 0:
            left = (v - w * self._half_track) / self._wheel_radius
            right = (v + w * self._half_track) / self._wheel_radius
            qadr_l = self.model.jnt_qposadr[self._left_joint]
            qadr_r = self.model.jnt_qposadr[self._right_joint]
            self.data.qpos[qadr_l] += left * dt
            self.data.qpos[qadr_r] += right * dt
        mujoco.mj_forward(self.model, self.data)

        self._steps += 1
        obs = self._get_obs()
        info = {
            "ranges": obs.copy(),
            "position": self.data.xpos[self._base_body].copy(),
            "forward_speed": v,
            "steps": self._steps,
        }
        reward = float(self.task.reward(info)) if self.task.reward else 0.0
        terminated = bool(self.task.termination(info)) if self.task.termination else False
        truncated = self._steps >= self.task.max_steps
        if self.task.success and self.task.success(info):
            self._success_streak += 1
        else:
            self._success_streak = 0
        info["success"] = self._success_streak >= 20
        info["is_success"] = info["success"]
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        ranges = []
        pos = self.data.xpos[self._base_body].copy()
        # Body x-axis in world
        xmat = self.data.xmat[self._base_body].reshape(3, 3)
        yaw = np.arctan2(xmat[1, 0], xmat[0, 0])
        origin = pos + np.array([0.0, 0.0, 0.12])
        for deg in LIDAR_ANGLES_DEG:
            ang = yaw + np.deg2rad(deg)
            direction = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float64)
            # mj_ray returns distance or -1 if no hit (MuJoCo 3.x signature).
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                origin.astype(np.float64),
                direction.astype(np.float64),
                self._geomgroup,
                True,
                self._base_body,
                self._geomid,
            )
            if dist < 0:
                dist = LIDAR_MAX
            dist = float(np.clip(dist, 0.0, LIDAR_MAX))
            if self.domain.sensor_noise_std > 0:
                dist += float(self.np_random.normal(0.0, self.domain.sensor_noise_std))
                dist = float(np.clip(dist, 0.0, LIDAR_MAX))
            ranges.append(dist)
        return np.asarray(ranges, dtype=np.float32)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)
            self._cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self._cam)
            self._cam.elevation = -35
            self._cam.azimuth = 90
            self._cam.distance = 3.5
        # Track the robot so longer corridors stay in frame for playback.
        self._cam.lookat[:] = self.data.xpos[self._base_body]
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._cam = None


@register_sim("mujoco")
class MujocoAdapter(SimAdapter):
    name = "mujoco"

    def load_robot(self, urdf_path: Path, **kw: Any) -> dict[str, Any]:
        """Return paths; MuJoCo Phase 1 trains from scene.xml mirroring the URDF."""
        robot = kw.get("robot", "diffdrive_lidar")
        u = Path(urdf_path) if urdf_path else urdf_path_fn(robot)
        s = scene_path(robot)
        if not s.exists():
            raise FileNotFoundError(f"MuJoCo scene missing: {s}")
        return {"urdf": u, "scene": s, "robot": robot}

    def make_env(
        self,
        task: TaskSpec,
        robot: Any,
        domain: DomainParams,
        render: bool = False,
    ) -> gym.Env:
        if isinstance(robot, dict):
            model_path = Path(robot["scene"])
        else:
            model_path = scene_path(str(robot))
        return DiffDriveLidarEnv(
            task=task,
            model_path=model_path,
            domain=domain,
            render_mode="rgb_array" if render else None,
        )

    def perturb(self, env: gym.Env, params: DomainParams) -> None:
        if isinstance(env, DiffDriveLidarEnv):
            env.domain = params
            env._apply_domain()

    def render_frame(self, env: gym.Env) -> np.ndarray:
        frame = env.render()
        if frame is None:
            raise RuntimeError("Env was not created with render=True")
        return frame

    def capabilities(self) -> set[str]:
        return {"urdf", "mj_ray", "egl", "osmesa", "headless"}


def urdf_path_fn(robot: str) -> Path:
    return urdf_path(robot)
