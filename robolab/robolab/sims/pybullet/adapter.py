"""PyBullet SimAdapter + Gymnasium diffdrive env (multi-task worlds)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
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


class DiffDriveLidarPybulletEnv(gym.Env):
    """Differential-drive robot with ray-cast lidar in a PyBullet task world."""

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

        self._cid: int | None = None
        self._robot_id: int = -1
        self._control_dt = 1.0 / max(1e-6, self.domain.control_hz)
        self._physics_dt = self._control_dt / max(1, self.domain.physics_substeps)
        self._build_world()

    def _build_world(self) -> None:
        if self._cid is not None:
            p.disconnect(physicsClientId=self._cid)
        self._cid = p.connect(p.DIRECT)
        p.resetSimulation(physicsClientId=self._cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self._cid)
        p.setTimeStep(self._physics_dt, physicsClientId=self._cid)

        floor_col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self._cid)
        floor_vis = p.createVisualShape(
            p.GEOM_PLANE,
            rgbaColor=[0.85, 0.88, 0.9, 1.0],
            physicsClientId=self._cid,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_col,
            baseVisualShapeIndex=floor_vis,
            basePosition=[0, 0, 0],
            physicsClientId=self._cid,
        )

        for half, pos, rgba in self._layout.boxes:
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=self._cid
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half,
                rgbaColor=rgba,
                physicsClientId=self._cid,
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
                physicsClientId=self._cid,
            )

        if not self._urdf.is_file():
            raise FileNotFoundError(f"URDF missing: {self._urdf}")
        flags = getattr(p, "URDF_USE_INERTIA_FROM_FILE", 0)
        sx, sy = self._layout.spawn_xy
        self._robot_id = p.loadURDF(
            str(self._urdf.resolve()),
            basePosition=[sx, sy, 0.05],
            baseOrientation=p.getQuaternionFromEuler([0, 0, self._layout.spawn_yaw]),
            useFixedBase=0,
            flags=flags,
            physicsClientId=self._cid,
        )
        if self._robot_id < 0:
            raise RuntimeError(f"loadURDF failed for {self._urdf}")

        n_joints = p.getNumJoints(self._robot_id, physicsClientId=self._cid)
        for j in range(n_joints):
            p.setJointMotorControl2(
                self._robot_id,
                j,
                p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self._cid,
            )
        self._apply_domain()

    def _apply_domain(self) -> None:
        self._control_dt = 1.0 / max(1e-6, self.domain.control_hz)
        self._physics_dt = self._control_dt / max(1, self.domain.physics_substeps)
        if self._cid is not None:
            p.setTimeStep(self._physics_dt, physicsClientId=self._cid)
            p.changeDynamics(
                self._robot_id,
                -1,
                lateralFriction=float(self.domain.friction),
                physicsClientId=self._cid,
            )

    def _lidar(self) -> np.ndarray:
        pos, orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._cid
        )
        yaw = float(p.getEulerFromQuaternion(orn)[2])
        origin = np.array(
            [float(pos[0]), float(pos[1]), float(pos[2]) + 0.12], dtype=np.float64
        )
        ranges: list[float] = []
        for deg in LIDAR_ANGLES_DEG:
            ang = yaw + np.deg2rad(deg)
            direction = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float64)
            end = origin + direction * LIDAR_MAX
            hits = p.rayTest(origin.tolist(), end.tolist(), physicsClientId=self._cid)
            hit = hits[0]
            uid, _link, frac, _hit_pos, _normal = hit
            if uid < 0 or frac >= 1.0 - 1e-9 or uid == self._robot_id:
                dist = LIDAR_MAX
            else:
                dist = float(frac) * LIDAR_MAX
            dist = float(np.clip(dist, 0.0, LIDAR_MAX))
            if self.domain.sensor_noise_std > 0:
                dist += float(self.np_random.normal(0.0, self.domain.sensor_noise_std))
                dist = float(np.clip(dist, 0.0, LIDAR_MAX))
            ranges.append(dist)
        return np.asarray(ranges, dtype=np.float32)

    def _pack(self, ranges: np.ndarray, forward_speed: float) -> tuple[np.ndarray, dict]:
        pos, orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._cid
        )
        yaw = float(p.getEulerFromQuaternion(orn)[2])
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
            "position": np.asarray(pos, dtype=np.float64),
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
        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(
            self._robot_id,
            [layout.spawn_xy[0], layout.spawn_xy[1] + y, 0.05],
            orn,
            physicsClientId=self._cid,
        )
        p.resetBaseVelocity(
            self._robot_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._cid
        )
        n_joints = p.getNumJoints(self._robot_id, physicsClientId=self._cid)
        for j in range(n_joints):
            p.resetJointState(self._robot_id, j, 0.0, 0.0, physicsClientId=self._cid)
        self._steps = 0
        self._success_streak = 0
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

        pos, orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._cid
        )
        yaw = float(p.getEulerFromQuaternion(orn)[2])
        yaw = yaw + w * dt
        x = float(pos[0]) + v * np.cos(yaw) * dt
        y = float(pos[1]) + v * np.sin(yaw) * dt
        new_orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(
            self._robot_id,
            [x, y, 0.05],
            new_orn,
            physicsClientId=self._cid,
        )
        p.resetBaseVelocity(
            self._robot_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._cid
        )
        for _ in range(max(1, self.domain.physics_substeps)):
            p.stepSimulation(physicsClientId=self._cid)
        p.resetBasePositionAndOrientation(
            self._robot_id,
            [x, y, 0.05],
            new_orn,
            physicsClientId=self._cid,
        )

        if self.task.name == "figure8_tracking":
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
        pos, _orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._cid
        )
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[float(pos[0]), float(pos[1]), float(pos[2])],
            distance=3.5,
            yaw=90,
            pitch=-35,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self._cid,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=320 / 240,
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=self._cid,
        )
        _w, _h, rgba, _depth, _seg = p.getCameraImage(
            width=320,
            height=240,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._cid,
        )
        rgb = np.reshape(np.asarray(rgba, dtype=np.uint8), (240, 320, 4))[:, :, :3]
        return np.ascontiguousarray(rgb)

    def close(self):
        if self._cid is not None:
            try:
                p.disconnect(physicsClientId=self._cid)
            except Exception:
                pass
            self._cid = None


@register_sim("pybullet")
class PybulletAdapter(SimAdapter):
    name = "pybullet"

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
        if isinstance(robot, dict):
            model_path = Path(robot["urdf"])
        else:
            model_path = urdf_path(str(robot))
        return DiffDriveLidarPybulletEnv(
            task=task,
            urdf=model_path,
            domain=domain,
            render_mode="rgb_array" if render else None,
        )

    def perturb(self, env: gym.Env, params: DomainParams) -> None:
        if isinstance(env, DiffDriveLidarPybulletEnv):
            env.domain = params
            env._apply_domain()

    def render_frame(self, env: gym.Env) -> np.ndarray:
        frame = env.render()
        if frame is None:
            raise RuntimeError("Env was not created with render=True")
        return frame

    def capabilities(self) -> set[str]:
        return {"urdf", "rayTest", "tiny_renderer", "direct", "headless", "multi_task"}
