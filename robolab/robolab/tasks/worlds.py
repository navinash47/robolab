"""Shared planar world layouts for diffdrive tasks (MuJoCo / PyBullet / Genesis)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robolab.robots.paths import robot_dir

# Boxes: (half_extents_xyz, center_xyz, rgba)
BoxSpec = tuple[list[float], list[float], list[float]]


@dataclass(frozen=True)
class WorldLayout:
    name: str
    boxes: tuple[BoxSpec, ...]
    floor_half_xy: tuple[float, float] = (15.0, 8.0)
    spawn_xy: tuple[float, float] = (0.3, 0.0)
    spawn_yaw: float = 0.0
    spawn_noise_y: float = 0.15
    spawn_noise_yaw: float = 0.15
    mujoco_scene: str = "scene.xml"


# --- Corridor (wall_follow) — matches historical scene.xml ---
CORRIDOR_BOXES: tuple[BoxSpec, ...] = (
    ([6.5, 0.05, 0.25], [6.0, 0.7, 0.25], [0.55, 0.45, 0.35, 1.0]),
    ([6.5, 0.05, 0.25], [6.0, -0.7, 0.25], [0.55, 0.45, 0.35, 1.0]),
    ([0.05, 0.75, 0.25], [12.5, 0.0, 0.25], [0.45, 0.4, 0.35, 1.0]),
    ([0.2, 0.05, 0.25], [-0.4, 0.7, 0.25], [0.55, 0.45, 0.35, 1.0]),
    ([0.2, 0.05, 0.25], [-0.4, -0.7, 0.25], [0.55, 0.45, 0.35, 1.0]),
)

# Open arena perimeter for go_to_goal / figure8
OPEN_PERIMETER: tuple[BoxSpec, ...] = (
    ([0.1, 4.0, 0.25], [-2.0, 0.0, 0.25], [0.5, 0.5, 0.5, 1.0]),
    ([0.1, 4.0, 0.25], [8.0, 0.0, 0.25], [0.5, 0.5, 0.5, 1.0]),
    ([5.1, 0.1, 0.25], [3.0, 4.0, 0.25], [0.5, 0.5, 0.5, 1.0]),
    ([5.1, 0.1, 0.25], [3.0, -4.0, 0.25], [0.5, 0.5, 0.5, 1.0]),
)

# Parallel-park: roadway + bay on +y
DOCK_BOXES: tuple[BoxSpec, ...] = (
    ([5.0, 0.08, 0.25], [3.0, 1.2, 0.25], [0.55, 0.45, 0.35, 1.0]),  # left curb
    ([5.0, 0.08, 0.25], [3.0, -1.2, 0.25], [0.55, 0.45, 0.35, 1.0]),  # right curb
    ([0.08, 0.55, 0.25], [1.5, 0.65, 0.25], [0.4, 0.35, 0.3, 1.0]),  # bay front post
    ([0.08, 0.55, 0.25], [4.0, 0.65, 0.25], [0.4, 0.35, 0.3, 1.0]),  # bay rear post
    ([1.35, 0.08, 0.25], [2.75, 1.15, 0.25], [0.4, 0.35, 0.3, 1.0]),  # bay back wall
)

# Simple maze (S near origin, E at far right)
MAZE_BOXES: tuple[BoxSpec, ...] = (
    ([4.5, 0.08, 0.25], [3.5, 2.2, 0.25], [0.5, 0.4, 0.35, 1.0]),  # outer top
    ([4.5, 0.08, 0.25], [3.5, -2.2, 0.25], [0.5, 0.4, 0.35, 1.0]),  # outer bottom
    ([0.08, 2.3, 0.25], [-1.0, 0.0, 0.25], [0.5, 0.4, 0.35, 1.0]),  # start wall
    ([0.08, 1.4, 0.25], [8.0, -0.8, 0.25], [0.5, 0.4, 0.35, 1.0]),  # end partial (exit gap +y)
    ([0.08, 1.6, 0.25], [2.0, -0.6, 0.25], [0.45, 0.35, 0.3, 1.0]),  # mid blocker
    ([1.8, 0.08, 0.25], [3.5, 0.4, 0.25], [0.45, 0.35, 0.3, 1.0]),  # horizontal bar
    ([0.08, 1.2, 0.25], [5.2, 1.0, 0.25], [0.45, 0.35, 0.3, 1.0]),  # mid vertical
    ([1.5, 0.08, 0.25], [6.5, -0.5, 0.25], [0.45, 0.35, 0.3, 1.0]),  # lower bar
)


WORLDS: dict[str, WorldLayout] = {
    "wall_follow": WorldLayout(
        name="corridor",
        boxes=CORRIDOR_BOXES,
        spawn_xy=(0.3, 0.0),
        mujoco_scene="scene.xml",
    ),
    "go_to_goal": WorldLayout(
        name="open",
        boxes=OPEN_PERIMETER,
        spawn_xy=(0.5, 0.0),
        mujoco_scene="scene_open.xml",
    ),
    "docking": WorldLayout(
        name="dock",
        boxes=DOCK_BOXES,
        spawn_xy=(0.4, 0.0),
        spawn_noise_y=0.1,
        spawn_noise_yaw=0.1,
        mujoco_scene="scene_dock.xml",
    ),
    "maze": WorldLayout(
        name="maze",
        boxes=MAZE_BOXES,
        spawn_xy=(0.2, 0.0),
        spawn_noise_y=0.1,
        spawn_noise_yaw=0.1,
        mujoco_scene="scene_maze.xml",
    ),
    "figure8_tracking": WorldLayout(
        name="open_figure8",
        boxes=OPEN_PERIMETER,
        spawn_xy=(3.0, 0.0),
        spawn_noise_y=0.05,
        spawn_noise_yaw=0.05,
        mujoco_scene="scene_open.xml",
    ),
}


def layout_for_task(task_name: str) -> WorldLayout:
    if task_name not in WORLDS:
        return WORLDS["wall_follow"]
    return WORLDS[task_name]


def mujoco_scene_path(robot: str, task_name: str) -> Path:
    layout = layout_for_task(task_name)
    path = robot_dir(robot) / layout.mujoco_scene
    if not path.exists():
        path = robot_dir(robot) / "scene.xml"
    return path


def figure8_point(t: float, scale: float = 1.6, center: tuple[float, float] = (3.0, 0.0)) -> tuple[float, float, float]:
    """Lemniscate of Bernoulli: returns (x, y, tangent_yaw). t in radians."""
    import numpy as np

    s = np.sin(t)
    c = np.cos(t)
    # x = a sin(t), y = a sin(t) cos(t)
    x = center[0] + scale * s
    y = center[1] + scale * s * c
    # derivative for tangent
    dx = scale * c
    dy = scale * (c * c - s * s)
    yaw = float(np.arctan2(dy, dx))
    return float(x), float(y), yaw


def enrich_task_info(
    task_name: str,
    meta: dict[str, Any],
    *,
    position: Any,
    yaw: float,
    ranges: Any,
    forward_speed: float,
    path_s: float = 0.0,
) -> dict[str, Any]:
    """Extra fields for reward/termination (also used to build extended obs)."""
    import numpy as np

    pos = np.asarray(position, dtype=np.float64).reshape(-1)
    xy = pos[:2]
    info: dict[str, Any] = {
        "yaw": float(yaw),
        "goal_xy": None,
        "dist_to_goal": None,
        "yaw_err": None,
        "crosstrack": None,
        "path_s": float(path_s),
        "rel_goal": np.zeros(3, dtype=np.float32),
    }

    if task_name == "go_to_goal":
        goal = np.asarray(meta.get("goal_xy", [5.0, 0.0]), dtype=np.float64)
        delta = goal - xy
        dist = float(np.linalg.norm(delta))
        desired = float(np.arctan2(delta[1], delta[0]))
        yaw_err = float(_wrap(desired - yaw))
        # robot-frame goal
        c, s = np.cos(yaw), np.sin(yaw)
        rel_x = float(c * delta[0] + s * delta[1])
        rel_y = float(-s * delta[0] + c * delta[1])
        info.update(
            goal_xy=goal,
            dist_to_goal=dist,
            yaw_err=yaw_err,
            rel_goal=np.array([rel_x, rel_y, yaw_err], dtype=np.float32),
        )
    elif task_name == "docking":
        dock = np.asarray(meta.get("dock_xy", [2.75, 0.55]), dtype=np.float64)
        dock_yaw = float(meta.get("dock_yaw", 1.5708))
        delta = dock - xy
        dist = float(np.linalg.norm(delta))
        yaw_err = float(_wrap(dock_yaw - yaw))
        c, s = np.cos(yaw), np.sin(yaw)
        rel_x = float(c * delta[0] + s * delta[1])
        rel_y = float(-s * delta[0] + c * delta[1])
        info.update(
            goal_xy=dock,
            dist_to_goal=dist,
            yaw_err=yaw_err,
            rel_goal=np.array([rel_x, rel_y, yaw_err], dtype=np.float32),
        )
    elif task_name == "maze":
        exit_xy = np.asarray(meta.get("exit_xy", [7.2, 1.4]), dtype=np.float64)
        delta = exit_xy - xy
        dist = float(np.linalg.norm(delta))
        desired = float(np.arctan2(delta[1], delta[0]))
        yaw_err = float(_wrap(desired - yaw))
        c, s = np.cos(yaw), np.sin(yaw)
        rel_x = float(c * delta[0] + s * delta[1])
        rel_y = float(-s * delta[0] + c * delta[1])
        info.update(
            goal_xy=exit_xy,
            dist_to_goal=dist,
            yaw_err=yaw_err,
            rel_goal=np.array([rel_x, rel_y, yaw_err], dtype=np.float32),
        )
    elif task_name == "figure8_tracking":
        scale = float(meta.get("path_scale", 1.6))
        center = tuple(meta.get("path_center", [3.0, 0.0]))  # type: ignore[arg-type]
        # Advance path parameter from progress heuristic
        px, py, tyaw = figure8_point(path_s, scale=scale, center=(float(center[0]), float(center[1])))
        target = np.array([px, py], dtype=np.float64)
        delta = target - xy
        # crosstrack ≈ lateral error in path frame
        c, s = np.cos(tyaw), np.sin(tyaw)
        along = float(c * delta[0] + s * delta[1])
        cross = float(-s * delta[0] + c * delta[1])
        yaw_err = float(_wrap(tyaw - yaw))
        info.update(
            goal_xy=target,
            dist_to_goal=float(np.linalg.norm(delta)),
            yaw_err=yaw_err,
            crosstrack=cross,
            rel_goal=np.array([along, cross, yaw_err], dtype=np.float32),
        )

    ranges_arr = np.asarray(ranges, dtype=np.float64)
    info["min_range"] = float(ranges_arr.min()) if ranges_arr.size else 5.0
    info["forward_speed"] = float(forward_speed)
    return info


def build_observation(task_name: str, ranges: Any, rel_goal: Any) -> Any:
    import numpy as np

    r = np.asarray(ranges, dtype=np.float32).reshape(-1)
    if task_name == "wall_follow":
        return r
    g = np.asarray(rel_goal, dtype=np.float32).reshape(-1)
    return np.concatenate([r, g[:3]], axis=0)


def _wrap(a: float) -> float:
    import numpy as np

    return float((a + np.pi) % (2 * np.pi) - np.pi)
