"""Shared planar world layouts for diffdrive tasks (MuJoCo / PyBullet / Genesis)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robolab.robots.paths import robot_dir

# Boxes: (half_extents_xyz, center_xyz, rgba)
BoxSpec = tuple[list[float], list[float], list[float]]

# RoboMaster-ish chassis ~0.55×0.42 m; circumradius ≈ 0.35. Slightly smaller
# keeps spawn clear of thin corridor walls while still blocking tunneling.
ROBOT_COLLISION_RADIUS = 0.32


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


def resolve_wall_collision(
    xy: Any,
    boxes: tuple[BoxSpec, ...],
    *,
    robot_radius: float = ROBOT_COLLISION_RADIUS,
    iterations: int = 4,
    prev_xy: Any | None = None,
) -> tuple[np.ndarray, bool]:
    """Push a disk robot out of axis-aligned wall boxes.

    All gate sims drive the base kinematically (set pose each step), so engine
    contacts never block motion. This shared resolver makes walls solid for
    MuJoCo / PyBullet / Genesis alike.

    When ``prev_xy`` is set, the motion is sub-sampled so a single large step
    cannot tunnel through a thin wall.
    """
    target = np.asarray(xy, dtype=np.float64).reshape(-1).copy()
    if prev_xy is not None:
        prev = np.asarray(prev_xy, dtype=np.float64).reshape(-1)
        delta = target[:2] - prev[:2]
        dist = float(np.linalg.norm(delta))
        step = max(robot_radius * 0.5, 1e-3)
        n = max(1, int(np.ceil(dist / step)))
        hit_any = False
        cur = prev[:2].astype(np.float64).copy()
        for i in range(1, n + 1):
            probe = prev[:2] + (i / n) * delta
            cur, hit = _resolve_disk_vs_boxes(
                probe, boxes, robot_radius=robot_radius, iterations=iterations
            )
            hit_any = hit_any or hit
            if hit:
                # Stop at first contact; do not continue through the wall.
                break
        out = target.copy()
        out[0], out[1] = float(cur[0]), float(cur[1])
        return out, hit_any

    pos, hit = _resolve_disk_vs_boxes(
        target[:2], boxes, robot_radius=robot_radius, iterations=iterations
    )
    out = target.copy()
    out[0], out[1] = float(pos[0]), float(pos[1])
    return out, hit


def _resolve_disk_vs_boxes(
    xy: Any,
    boxes: tuple[BoxSpec, ...],
    *,
    robot_radius: float,
    iterations: int,
) -> tuple[np.ndarray, bool]:
    pos = np.asarray(xy, dtype=np.float64).reshape(-1).copy()
    x, y = float(pos[0]), float(pos[1])
    hit = False
    r = float(robot_radius)
    r2 = r * r
    for _ in range(max(1, iterations)):
        moved = False
        for half, center, _rgba in boxes:
            wx, wy = float(center[0]), float(center[1])
            hx, hy = float(half[0]), float(half[1])
            minx, maxx = wx - hx, wx + hx
            miny, maxy = wy - hy, wy + hy
            inside = (minx <= x <= maxx) and (miny <= y <= maxy)
            if inside:
                # Deep penetration / tunnel: eject through nearest face.
                left, right = x - minx, maxx - x
                bottom, top = y - miny, maxy - y
                m = min(left, right, bottom, top)
                if m == left:
                    x = minx - r
                elif m == right:
                    x = maxx + r
                elif m == bottom:
                    y = miny - r
                else:
                    y = maxy + r
                hit = True
                moved = True
                continue
            cx = float(np.clip(x, minx, maxx))
            cy = float(np.clip(y, miny, maxy))
            dx, dy = x - cx, y - cy
            d2 = dx * dx + dy * dy
            if d2 >= r2:
                continue
            if d2 <= 1e-16:
                # Center on a corner/edge of the AABB — nudge along +x.
                x = maxx + r
                hit = True
                moved = True
                continue
            dist = float(np.sqrt(d2))
            pen = r - dist
            x += (dx / dist) * pen
            y += (dy / dist) * pen
            hit = True
            moved = True
        if not moved:
            break
    return np.array([x, y], dtype=np.float64), hit


def is_wall_crash(info: dict[str, Any], collision_dist: float = 0.12) -> bool:
    """True if kinematic wall contact or lidar reports an imminent crash."""
    if bool(info.get("wall_contact")):
        return True
    if float(info.get("min_range", 5.0)) < collision_dist:
        return True
    ranges = info.get("ranges")
    if ranges is not None:
        arr = np.asarray(ranges, dtype=np.float64).reshape(-1)
        if arr.size and float(arr.min()) < collision_dist:
            return True
    return False


# --- wall_follow: P2_D3 Gazebo largemaze.world, scaled up ---
# Source topology (m, yaw≈0 or π/2): outer 8×8 box (x,y ∈ [-4,4]) plus
# internal segments for straight / inside-L / outside-L / I-corner / 180° U-turn
# (Fig. 4 in P2_D3). Linear XY scale only; keep thin walls for lidar.
WALL_LAYOUT_SCALE = 4.0  # outer span ≈ 32 m (was 2.5 → 20 m)
_WALL_RGBA = [0.55, 0.45, 0.35, 1.0]
_WALL_H = 0.35
_WALL_HALF_T = 0.06  # half-thickness (m); not scaled — keep lidar/robot sensible


def _segment_boxes(
    segments: tuple[tuple[float, float, float, float], ...],
    *,
    scale: float = 1.0,
) -> tuple[BoxSpec, ...]:
    """(cx, cy, yaw, full_length) → half-extent boxes. yaw≈π/2 → vertical."""
    boxes: list[BoxSpec] = []
    s = float(scale)
    for cx, cy, yaw, length in segments:
        half_len = 0.5 * length * s
        if abs(yaw) > 1.0:
            half = [_WALL_HALF_T, half_len, _WALL_H]
        else:
            half = [half_len, _WALL_HALF_T, _WALL_H]
        center = [cx * s, cy * s, _WALL_H]
        boxes.append((half, center, list(_WALL_RGBA)))
    return tuple(boxes)


def _scaled_wall_boxes(scale: float = WALL_LAYOUT_SCALE) -> tuple[BoxSpec, ...]:
    """Build MuJoCo/PyBullet half-extent boxes from Gazebo full-size segments."""
    # (cx, cy, yaw, full_length_along_local_x) from stingray largemaze.world
    segments = (
        (0.0, -2.0, 0.0, 4.0),  # bottom_inner horizontal
        (-2.5, 0.0, 0.0, 3.0),  # mid_left horizontal
        (-4.0, 0.0, 1.5708, 8.0),  # left vertical (perimeter)
        (2.0, 1.0, 1.5708, 2.0),  # stub vertical (I / U helper)
        (0.0, 4.0, 0.0, 8.0),  # top horizontal (perimeter)
        (4.0, 0.0, 1.5708, 8.0),  # right vertical (perimeter)
        (-1.0, 2.0, 0.0, 6.0),  # mid_top horizontal
        (0.0, -4.0, 0.0, 8.0),  # bottom horizontal (perimeter)
    )
    return _segment_boxes(segments, scale=scale)


CORRIDOR_BOXES: tuple[BoxSpec, ...] = _scaled_wall_boxes(WALL_LAYOUT_SCALE)

# P2_D3 Fig. 4 scenario slices (meters, unscaled local frames; ~RoboMaster corridor).
# Spawns place the robot with the right wall in the PDF medium band (~0.8 m).
_SCENARIO_STRAIGHT = _segment_boxes(
    (
        (0.0, -1.0, 0.0, 12.0),  # right wall (follow this)
        (0.0, 1.2, 0.0, 12.0),  # left wall
    ),
    scale=1.0,
)
_SCENARIO_L_INSIDE = _segment_boxes(
    (
        (0.0, -1.0, 0.0, 8.0),
        (4.0, 1.0, 1.5708, 6.0),
        (1.0, 4.0, 0.0, 6.0),
    ),
    scale=1.0,
)
_SCENARIO_L_OUTSIDE = _segment_boxes(
    (
        (0.0, -1.0, 0.0, 8.0),
        (-1.0, -4.0, 1.5708, 6.0),
        (-4.0, -1.0, 0.0, 6.0),
    ),
    scale=1.0,
)
_SCENARIO_I_CORNER = _segment_boxes(
    (
        (0.0, -1.0, 0.0, 10.0),
        (0.0, 1.2, 0.0, 4.0),
        (3.0, 1.2, 0.0, 4.0),
        (1.5, 0.1, 1.5708, 2.2),  # stub / doorway
    ),
    scale=1.0,
)
_SCENARIO_UTURN = _segment_boxes(
    (
        (0.0, -1.0, 0.0, 8.0),
        (0.0, 1.0, 0.0, 8.0),
        (4.0, 0.0, 1.5708, 2.0),
        (0.0, 3.0, 0.0, 8.0),
        (0.0, 5.0, 0.0, 8.0),
    ),
    scale=1.0,
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
        name="largemaze",
        boxes=CORRIDOR_BOXES,
        # North of mid_left (y=0): right-wall ≈ 0.85 m (PDF medium) when yaw=+x.
        floor_half_xy=(4.0 * WALL_LAYOUT_SCALE + 2.0, 4.0 * WALL_LAYOUT_SCALE + 2.0),
        spawn_xy=(-6.0, 0.85),
        spawn_yaw=0.0,
        spawn_noise_y=0.12,
        spawn_noise_yaw=0.2,
        mujoco_scene="scene.xml",
    ),
    # P2_D3 Fig. 4 wall tests (shared reward with wall_follow).
    "wall_straight": WorldLayout(
        name="straight",
        boxes=_SCENARIO_STRAIGHT,
        floor_half_xy=(8.0, 4.0),
        spawn_xy=(-4.0, -0.2),
        spawn_yaw=0.0,
        spawn_noise_y=0.1,
        spawn_noise_yaw=0.1,
        mujoco_scene="scene_wall_straight.xml",
    ),
    "wall_l_inside": WorldLayout(
        name="l_inside",
        boxes=_SCENARIO_L_INSIDE,
        floor_half_xy=(8.0, 8.0),
        spawn_xy=(-2.5, -0.2),
        spawn_yaw=0.0,
        mujoco_scene="scene_wall_l_inside.xml",
    ),
    "wall_l_outside": WorldLayout(
        name="l_outside",
        boxes=_SCENARIO_L_OUTSIDE,
        floor_half_xy=(8.0, 8.0),
        spawn_xy=(-2.5, -0.2),
        spawn_yaw=0.0,
        mujoco_scene="scene_wall_l_outside.xml",
    ),
    "wall_i_corner": WorldLayout(
        name="i_corner",
        boxes=_SCENARIO_I_CORNER,
        floor_half_xy=(8.0, 4.0),
        spawn_xy=(-3.5, -0.2),
        spawn_yaw=0.0,
        mujoco_scene="scene_wall_i_corner.xml",
    ),
    "wall_uturn": WorldLayout(
        name="uturn",
        boxes=_SCENARIO_UTURN,
        floor_half_xy=(8.0, 8.0),
        spawn_xy=(-2.5, -0.2),
        spawn_yaw=0.0,
        mujoco_scene="scene_wall_uturn.xml",
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


def is_wall_follow_family(task_name: str) -> bool:
    return task_name == "wall_follow" or task_name.startswith("wall_")


def build_observation(task_name: str, ranges: Any, rel_goal: Any) -> Any:
    import numpy as np

    r = np.asarray(ranges, dtype=np.float32).reshape(-1)
    if is_wall_follow_family(task_name):
        return r
    g = np.asarray(rel_goal, dtype=np.float32).reshape(-1)
    return np.concatenate([r, g[:3]], axis=0)


def _wrap(a: float) -> float:
    import numpy as np

    return float((a + np.pi) % (2 * np.pi) - np.pi)
