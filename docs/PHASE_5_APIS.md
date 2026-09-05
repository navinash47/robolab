# Phase 5 APIs (researched before code)

Exact PyBullet / Gymnasium / DomainParams signatures for the dual-sim wall-follow gate. **Doc wins** if it contradicts the build prompt — differences flagged below.

Sources (checked 2026-09-05):

- [PyBullet Quickstart Guide (HTML)](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html)
- [PyBullet Quickstart PDF](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstartguide.pdf)
- [pip `pybullet`](https://pypi.org/project/pybullet/) (Bullet Physics Python bindings)
- Existing RoboLab MuJoCo adapter (`robolab/sims/mujoco/adapter.py`) + `robots/diffdrive_lidar/robot.urdf`
- [Gymnasium Env / spaces](https://gymnasium.farama.org/api/env/)

## Conflict table (prompt vs docs)

| Prompt assumption | Actual PyBullet docs | What we do |
|---|---|---|
| Load *same* URDF as MuJoCo | `loadURDF` loads the URDF; MuJoCo trains from `scene.xml` that **mirrors** URDF geometry + corridor | Canonical file remains `robots/diffdrive_lidar/robot.urdf`. PyBullet `loadURDF` that path. Corridor walls (not in URDF) built with `createCollisionShape` / `createMultiBody` matching `scene.xml` dims |
| Full dynamics for both | Phase 1 MuJoCo uses **kinematic planar drive** (pose update + `mj_forward`), not wheel motors | PyBullet mirrors the same kinematic drive via `resetBasePositionAndOrientation` so `wall_follow` TaskSpec / obs / actions stay identical. Document; full dynamics deferred |
| GUI for rendering | DIRECT mode has **no OpenGL window**; `getCameraImage` still works via TinyRenderer | Train + playback use `connect(DIRECT)` + `getCameraImage` (software renderer). Never require GUI |
| `physics_substeps` unused | MuJoCo sets `model.opt.timestep` but kinematic step never called `mj_step` | Both sims: `physics_dt = 1 / (control_hz * physics_substeps)`; log both fields; PyBullet `setTimeStep(physics_dt)`; control motion uses `control_dt = 1 / control_hz` |

## Connect / disconnect

**Docs:** [Quickstart — connect, disconnect](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html)

```python
import pybullet as p

# DIRECT: same-process physics, no GUI window.
# Docs: DIRECT still allows getCameraImage via built-in software renderer (TinyRenderer).
cid = p.connect(p.DIRECT)
assert cid >= 0
# ...
p.disconnect(physicsClientId=cid)
```

Optional GUI (debug only, not used in RoboLab train/render):

```python
p.connect(p.GUI)  # OpenGL window; not headless-safe
```

## loadURDF (same robot file)

**Docs:** Quickstart `loadURDF` section.

```python
body_id = p.loadURDF(
    fileName=str(urdf_path),          # absolute path to robots/<robot>/robot.urdf
    basePosition=[0.3, 0.0, 0.05],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),  # [x,y,z,w]
    useFixedBase=0,
    flags=p.URDF_USE_INERTIA_FROM_FILE,  # honor URDF inertial block
    physicsClientId=cid,
)
# Returns body unique id (>=0) or negative on failure.
```

**Important (docs):** joints have motors enabled by default (high friction). For kinematic drive we disable velocity motors:

```python
for j in range(p.getNumJoints(body_id, physicsClientId=cid)):
    p.setJointMotorControl2(
        body_id, j, p.VELOCITY_CONTROL, force=0, physicsClientId=cid
    )
```

## Corridor / world (not in URDF)

Match MuJoCo `scene.xml` corridor (inner ~1.2 m wide, ~12 m usable):

| Geom | Center (x,y,z) | Half-extents (box) |
|---|---|---|
| floor plane | — | `createCollisionShape(GEOM_PLANE)` or large thin box |
| wall_left | (6.0, 0.7, 0.25) | (6.5, 0.05, 0.25) |
| wall_right | (6.0, -0.7, 0.25) | (6.5, 0.05, 0.25) |
| wall_end | (12.5, 0, 0.25) | (0.05, 0.75, 0.25) |
| wall_start_l / r | (-0.4, ±0.7, 0.25) | (0.2, 0.05, 0.25) |

```python
col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=cid)
vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                          rgbaColor=[0.55, 0.45, 0.35, 1], physicsClientId=cid)
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                  baseVisualShapeIndex=vis, basePosition=[x, y, z],
                  physicsClientId=cid)
```

Gravity (matches MuJoCo scene):

```python
p.setGravity(0, 0, -9.81, physicsClientId=cid)
```

## Timestep honesty (`control_hz` / `physics_substeps`)

`DomainParams` already defines:

```python
control_hz: float = 50.0
physics_substeps: int = 5
```

```python
control_dt = 1.0 / domain.control_hz
physics_dt = control_dt / max(1, domain.physics_substeps)
p.setTimeStep(physics_dt, physicsClientId=cid)
# Optional: p.setPhysicsEngineParameter(numSubSteps=1, ...) — we own substeps explicitly
```

Kinematic control step (parity with MuJoCo adapter):

```python
# Each env.step: integrate planar (v, w) over control_dt, then:
p.resetBasePositionAndOrientation(robot_id, [x, y, z], orn_xyzw, physicsClientId=cid)
# Optionally advance physics_substeps of stepSimulation so contacts stay consistent:
for _ in range(domain.physics_substeps):
    p.stepSimulation(physicsClientId=cid)
```

Log both fields into W&B `config` / run `config_json` and surface on the run page (sim-to-sim honesty).

## Lidar via `rayTest` (MuJoCo uses `mj_ray`)

Same 5 planar angles as MuJoCo: `(0°, +45°, -45°, +90°, -90°)` relative to yaw; max range `5.0`.

```python
# rayTest(rayFromPosition, rayToPosition) → list of hits
# Each hit: (objectUniqueId, linkIndex, hitFraction, hitPosition, hitNormal)
# hitFraction in [0,1]; 1.0 = miss (docs / examples).
hit = p.rayTest(origin, origin + direction * LIDAR_MAX, physicsClientId=cid)[0]
uid, link, frac, pos, normal = hit
dist = LIDAR_MAX if uid < 0 or frac >= 1.0 else float(frac) * LIDAR_MAX
```

Filter self-hits: if `uid == robot_id`, treat as miss or cast with a small origin offset and ignore robot body (MuJoCo excludes robot geom group).

Batch alternative (optional): `rayTestBatch(rayFroms, rayTos)`.

## Pose / orientation

```python
pos, orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=cid)
# orn is [x, y, z, w]
yaw = p.getEulerFromQuaternion(orn)[2]
orn = p.getQuaternionFromEuler([0, 0, yaw])
p.resetBasePositionAndOrientation(robot_id, pos, orn, physicsClientId=cid)
```

## Headless RGB render (`getCameraImage`)

**Docs:** DIRECT mode explicitly supports TinyRenderer via `getCameraImage` (no GPU required).

```python
view = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=base_pos,
    distance=3.5,
    yaw=90,
    pitch=-35,
    roll=0,
    upAxisIndex=2,
    physicsClientId=cid,
)
proj = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=320 / 240,
    nearVal=0.1,
    farVal=100.0,
    physicsClientId=cid,
)
w, h, rgba, depth, seg = p.getCameraImage(
    width=320,
    height=240,
    viewMatrix=view,
    projectionMatrix=proj,
    renderer=p.ER_TINY_RENDERER,  # software; works with DIRECT
    physicsClientId=cid,
)
rgb = np.reshape(rgba, (h, w, 4))[:, :, :3].astype(np.uint8)
```

**Flag vs prompt:** Do **not** use `ER_BULLET_HARDWARE_OPENGL` under DIRECT/headless — that path expects a GL context / GUI. TinyRenderer is the documented DIRECT-safe path.

Playback still uses Gymnasium `RecordVideo` (Phase 4) wrapping `render_mode="rgb_array"`.

## Gymnasium spaces (must match MuJoCo)

Identical to MuJoCo `DiffDriveLidarEnv` / `wall_follow` TaskSpec:

```python
observation_space = spaces.Box(low=0.0, high=5.0, shape=(5,), dtype=np.float32)
action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

Report `obs_dim=5`, `act_dim=2` on the run page for **both** sims.

## Sim registry / UI

```python
@register_sim("pybullet")
class PybulletAdapter(SimAdapter): ...
```

- `GET /api/sims` → `{"sims": ["mujoco", "pybullet"]}` (after importing both adapter packages)
- New Run dropdown: `mujoco | pybullet`
- Trainer / render import `robolab.sims.pybullet` so registry is populated

## Dependency (listed in stack)

```toml
# robolab/pyproject.toml
"pybullet>=3.2.6",
```

Install via `uv sync`. No ask required — pybullet is on the Section 3 / phase stack list.

### Conflict: macOS / Xcode 16+ build

| Prompt assumption | Actual | What we do |
|---|---|---|
| `pip/uv install pybullet` just works | Bundled zlib 1.2.8 redefines `fdopen` when `TARGET_OS_MAC` is set; Apple Clang 17 always defines that macro → compile fails ([bullet3#4753](https://github.com/bulletphysics/bullet3/issues/4753)) | On Darwin: `CFLAGS="-fno-define-target-os-macros" uv sync` (or `uv pip install pybullet`). Documented in `PHASE_5_TEST.md` / Makefile note. Linux/RunPod wheels or builds are unaffected. |

## What does *not* change

- `wall_follow` TaskSpec (`robolab/tasks/wall_follow.py`) — reward / termination / obs / action unchanged
- Canonical URDF path and geometry
- Phase 4 RecordVideo / wandb.Video pipeline (sim-agnostic once `render()` works)

## Links

- [PyBullet Quickstart (HTML)](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html)
- [PyBullet Quickstart (PDF)](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstartguide.pdf)
- [PyPI pybullet](https://pypi.org/project/pybullet/)
- [Gymnasium Env](https://gymnasium.farama.org/api/env/)
