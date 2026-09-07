"""MuJoCo SimAdapter + Gymnasium diffdrive env (multi-task worlds)."""

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
from robolab.robots.paths import urdf_path
from robolab.tasks.worlds import (
    build_observation,
    enrich_task_info,
    layout_for_task,
    mujoco_scene_path,
    resolve_wall_collision,
)

LIDAR_ANGLES_DEG = (0.0, 45.0, -45.0, 90.0, -90.0)
LIDAR_MAX = 5.0


class DiffDriveLidarEnv(gym.Env):
    """Differential-drive robot with ray-cast lidar; world selected by task."""

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
        self._layout = layout_for_task(task.name)
        self._path_s = 0.0

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

        self._wheel_radius = 0.05
        self._half_track = 0.14
        self._v_max = 0.5
        self._w_max = 1.2
        self._steps = 0
        self._success_streak = 0
        self._wall_contact = False
        self._renderer: mujoco.Renderer | None = None
        self._cam: mujoco.MjvCamera | None = None

        self._base_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self._geomid = np.zeros(1, dtype=np.int32)
        self._left_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint"
        )
        self._right_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint"
        )

    def _apply_domain(self) -> None:
        for name in ("left_wheel_geom", "right_wheel_geom", "floor"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.model.geom_friction[gid, 0] = self.domain.friction * (
                    1.5 if "wheel" in name else 1.0
                )
        ctrl_dt = 1.0 / max(1e-6, self.domain.control_hz)
        self.model.opt.timestep = ctrl_dt / max(1, self.domain.physics_substeps)
        self._control_dt = ctrl_dt
        self._physics_dt = float(self.model.opt.timestep)

    def _yaw(self) -> float:
        quat = self.data.qpos[3:7]
        return float(
            np.arctan2(
                2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
            )
        )

    def _lidar(self) -> np.ndarray:
        ranges = []
        pos = self.data.xpos[self._base_body].copy()
        xmat = self.data.xmat[self._base_body].reshape(3, 3)
        yaw = np.arctan2(xmat[1, 0], xmat[0, 0])
        origin = pos + np.array([0.0, 0.0, 0.12])
        for deg in LIDAR_ANGLES_DEG:
            ang = yaw + np.deg2rad(deg)
            direction = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float64)
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

    def _pack(self, ranges: np.ndarray, forward_speed: float) -> tuple[np.ndarray, dict]:
        yaw = self._yaw()
        pos = self.data.xpos[self._base_body].copy()
        extra = enrich_task_info(
            self.task.name,
            self.task.meta,
            position=pos,
            yaw=yaw,
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
            "wall_contact": bool(self._wall_contact),
            **extra,
        }
        return obs.astype(np.float32), info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        layout = self._layout
        y = float(self.np_random.uniform(-layout.spawn_noise_y, layout.spawn_noise_y))
        yaw = float(
            layout.spawn_yaw
            + self.np_random.uniform(-layout.spawn_noise_yaw, layout.spawn_noise_yaw)
        )
        quat = np.array(
            [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
            dtype=np.float64,
        )
        self.data.qpos[:3] = np.array(
            [layout.spawn_xy[0], layout.spawn_xy[1] + y, 0.05]
        )
        self.data.qpos[3:7] = quat
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._success_streak = 0
        self._wall_contact = False
        self._path_s = 0.0
        if self.task.name == "figure8_tracking":
            self._path_s = 0.0
        ranges = self._lidar()
        obs, info = self._pack(ranges, 0.0)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        v = float(action[0]) * self._v_max
        w = float(action[1]) * self._w_max
        dt = self._control_dt

        yaw = self._yaw()
        yaw = yaw + w * dt
        prev_xy = [float(self.data.qpos[0]), float(self.data.qpos[1])]
        self.data.qpos[0] += v * np.cos(yaw) * dt
        self.data.qpos[1] += v * np.sin(yaw) * dt
        # Kinematic drive bypasses MuJoCo contacts — resolve layout walls explicitly.
        xy, wall_hit = resolve_wall_collision(
            [self.data.qpos[0], self.data.qpos[1]],
            self._layout.boxes,
            prev_xy=prev_xy,
        )
        self.data.qpos[0] = float(xy[0])
        self.data.qpos[1] = float(xy[1])
        self._wall_contact = wall_hit
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

        if self.task.name == "figure8_tracking":
            # Advance path parameter roughly proportional to speed / path scale.
            scale = float(self.task.meta.get("path_scale", 1.6))
            self._path_s += (abs(v) * dt) / max(0.2, scale)

        self._steps += 1
        ranges = self._lidar()
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
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)
            self._cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self._cam)
            self._cam.elevation = -35
            self._cam.azimuth = 90
            self._cam.distance = 3.5
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

    def load_robot(self, urdf_path_arg: Path, **kw: Any) -> dict[str, Any]:
        robot = kw.get("robot", "diffdrive_lidar")
        task = kw.get("task", "wall_follow")
        u = Path(urdf_path_arg) if urdf_path_arg else urdf_path(robot)
        s = mujoco_scene_path(robot, str(task))
        if not s.exists():
            raise FileNotFoundError(f"MuJoCo scene missing: {s}")
        return {"urdf": u, "scene": s, "robot": robot, "task": task}

    def make_env(
        self,
        task: TaskSpec,
        robot: Any,
        domain: DomainParams,
        render: bool = False,
    ) -> gym.Env:
        if isinstance(robot, dict) and "scene" in robot:
            model_path = Path(robot["scene"])
            # Prefer scene matched to live task (load_robot may have used default).
            model_path = mujoco_scene_path(str(robot.get("robot", "diffdrive_lidar")), task.name)
        else:
            model_path = mujoco_scene_path(str(robot), task.name)
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
        return {"urdf", "mj_ray", "egl", "osmesa", "headless", "multi_task"}
