# RoboLab robots

Canonical robot descriptions live here as **URDF**. Every sim adapter must load the same URDF (or a scene that mirrors it) so sim-to-sim comparisons stay meaningful.

## MuJoCo URDF constraints (document — do not hide)

MuJoCo's URDF compiler supports a **subset** of URDF. Known limits that matter for RoboLab:

1. **No closed kinematic loops** — keep trees only.
2. **Mesh path handling is brittle** — prefer primitive geoms (`box`, `cylinder`, `sphere`) for Phase 1 robots. If meshes are needed later, use absolute or VFS paths and test both MuJoCo and PyBullet.
3. **Root body is welded to world by default** — mobile bases need an explicit floating joint. For MuJoCo we ship `scene.xml` that places the same geometry on a `freejoint` and adds the arena; `robot.urdf` remains the canonical kinematics/geometry for PyBullet (Phase 5).
4. **Sensors** — URDF does not express MuJoCo rangefinders. Lidar is implemented via `mujoco.mj_ray` in the adapter so both sims can share the same ray geometry later.
5. **Drive model (Phase 1):** MuJoCo `scene.xml` uses **kinematic planar drive** (update freejoint pose + `mj_forward`) rather than torque/velocity wheel actuators. Wheel joints are visual. This keeps training stable for the lidar wall-follow gate; full contact dynamics can be revisited later without changing the URDF or TaskSpec.

## `diffdrive_lidar`

Differential-drive base with 5 virtual range rays (front, ±45°, ±90°). Geometry: box chassis + two cylinder wheels. No meshes.
