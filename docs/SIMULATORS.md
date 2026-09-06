# RoboLab simulators

Human-oriented map of what appears in the New Run **Sim** dropdown, what works today, and what is stubbed.

## In the dropdown

| Name | Status | Install | Notes |
|---|---|---|---|
| **mujoco** | **Real** — default | `mujoco` (core dep) | MJCF scenes under `robots/diffdrive_lidar/`; kinematic drive + `mj_ray` lidar |
| **pybullet** | **Real** | `pybullet` (core dep) | Same URDF; corridor/maze as `createMultiBody` boxes; TinyRenderer |
| **genesis** | **Real** on RunPod (`:genesis` / install-at-start); Mac API host has no `genesis-world` | Optional on Linux GPU: `pip install genesis-world`. **Train + Render video both use RunPod** — do not expect local Mac playback | Same URDF + Box walls; 5-ray Lidar via `SphericalPattern(angles=…)`. Not a default Mac dep |
| **isaaclab** | **Stub** (Mac unsupported; RunPod needs approved `:isaac` image) | NVIDIA Isaac Lab on Isaac Sim | See `docs/ISAAC_INSTALL.md` — do not treat as a passing gate |
| **isaac_sim** | **Stub** | Isaac Sim runtime (separate from Lab) | Naming stub only; same NVIDIA/Linux constraints |

## How to New Run (custom steps + tasks)

1. Open **Experiments → New Run**.
2. Pick **Task** (`wall_follow`, `go_to_goal`, `docking`, `maze`, `figure8_tracking`).
3. Pick **Sim** (`mujoco` / `pybullet` recommended for new tasks; Genesis if installed; Isaac* will error until implemented).
4. **Timesteps:** choose 50k / 5k / 2k, or **Custom** and type any integer **1 … 10 000 000**.
5. Start. Failures remain flaggable; logistics auto-log is unchanged.

## Other engines (researched, not registered)

| Engine | Why not (yet) |
|---|---|
| **Brax** | JAX rigid-body / MJCF-ish for locomotion research; **not** a general robotics URDF + lidar corridor stack for our Gymnasium PPO path |
| **Gazebo / Ignition** | Needs ROS 2 (or similar) process model; wrong packaging for RoboLab’s in-process adapters |
| **Drake** | Strong plant modeling; heavier C++/pydrake story than MuJoCo/PyBullet for this phase |
| **Sapien** | Pip-friendly and URDF-capable; candidate for a future adapter (not stubbed this round) |
| **Bullet via pybullet** | Already covered as `pybullet` |

Blunt TEST note: **mujoco + pybullet** are the gate for all five tasks. Genesis is best-effort when installed. Isaac stubs must not be treated as passing gates.
