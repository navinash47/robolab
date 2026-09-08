# RoboLab robots

Canonical robot descriptions live here as **URDF**. Every sim adapter must load the same URDF (or a scene that mirrors it) so sim-to-sim comparisons stay meaningful.

## MuJoCo URDF constraints (document — do not hide)

MuJoCo's URDF compiler supports a **subset** of URDF. Known limits that matter for RoboLab:

1. **No closed kinematic loops** — keep trees only.
2. **Mesh path handling is brittle** — prefer primitive geoms (`box`, `cylinder`, `sphere`) for Phase 1 robots. If meshes are needed later, use absolute or VFS paths and test both MuJoCo and PyBullet.
3. **Root body is welded to world by default** — mobile bases need an explicit floating joint. For MuJoCo we ship `scene.xml` that places the same geometry on a `freejoint` and adds the arena; `robot.urdf` remains the canonical kinematics/geometry for PyBullet (Phase 5).
4. **Sensors** — URDF does not express MuJoCo rangefinders. Lidar is implemented via `mujoco.mj_ray` (MuJoCo), `pybullet.rayTest` (PyBullet), and Genesis `Lidar` + `SphericalPattern(angles=…)` with the same 5 planar angles.
5. **Drive model (Phase 1 / Phase 5):** Adapters use **kinematic planar drive** (pose update) rather than torque/velocity wheel actuators. Wheel joints are visual. This keeps training stable for the lidar tasks; full contact dynamics can be revisited later without changing the URDF or TaskSpec.

## `diffdrive_lidar`

Differential-drive base with 5 virtual range rays (front, ±45°, ±90°). **RoboMaster-ish**
top silhouette (~0.55×0.42 m hull + red/blue armor plates). Primitive geoms only.
Same URDF across sims.

**Tasks:** `wall_follow` + Fig. 4 scenarios (`wall_straight`, `wall_l_inside`,
`wall_l_outside`, `wall_i_corner`, `wall_uturn`), plus `go_to_goal` / `docking` /
`maze` / `figure8_tracking`.

| File | Role |
|---|---|
| `robot.urdf` | Canonical URDF (PyBullet / Genesis) |
| `scene.xml` | MuJoCo P2_D3 largemaze ×4 (`wall_follow`) |
| `scene_wall_*.xml` | Fig. 4 scenario isolates |
| `scene_open.xml` | Open arena (`go_to_goal`, `figure8_tracking`) |
| `scene_dock.xml` | Parallel-park bay (`docking`) |
| `scene_maze.xml` | Obstacle maze (`maze`) |
