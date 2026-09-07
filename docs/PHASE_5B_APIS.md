# Phase 5B APIs (researched before code)

Custom timesteps, additional tasks (`go_to_goal`, `docking`, `maze`, `figure8_tracking`), and extra simulators (Genesis + Isaac stubs). **Doc wins** if it contradicts the build prompt.

Sources (checked 2026-09-05):

- [Genesis Hello / URDF](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html)
- [Genesis Raycaster / Lidar](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/sensors/raycaster.html) (SphericalPattern custom `angles`)
- PyPI [`genesis-world`](https://pypi.org/project/genesis-world/) (~80 MB wheel alone; optional, **not** a hard RoboLab dep)
- NVIDIA Isaac Lab / Isaac Sim (heavy native NVIDIA stack; capability-flagged stubs only)
- Existing RoboLab MuJoCo / PyBullet adapters + `robots/diffdrive_lidar`

## Conflict table

| Prompt assumption | Actual | What we do |
|---|---|---|
| `pip install genesis` | Package name is **`genesis-world`**; import is `import genesis as gs` | Optional install; adapter lazy-imports; clear error if missing |
| Genesis is light | Wheel ~80 MB + PyTorch / Taichi stack | **Flag as heavy** — do not add to default `pyproject.toml` |
| Same URDF everywhere | Genesis `gs.morphs.URDF` supports URDF; primitive-geom `diffdrive_lidar` is OK. Corridor/maze walls are **not** in the URDF — built as `Box` morphs (parity with PyBullet) | Real Genesis adapter for wall-follow-style kinematic drive + 5-ray Lidar when installed |
| Isaac Lab == Isaac Sim | Lab is the RL framework; Sim is the Omniverse runtime | Separate stubs: `isaaclab`, `isaac_sim` |
| Full Isaac in CI | Requires NVIDIA GPU + Omniverse install | Stubs only; launch fails with install message |

## Custom timesteps

`TrainerCfg.timesteps`: integer **1 … 10_000_000** (Pydantic `Field(ge=1, le=10_000_000)`).

UI: New Run presets (50k / 5k / 2k) **plus Custom** free number input → `POST /runs` `trainer.timesteps`.

## Tasks (same robot: `diffdrive_lidar`)

| Task | World (MuJoCo scene / PyBullet boxes) | Obs | Success |
|---|---|---|---|
| `wall_follow` | corridor `scene.xml` | lidar(5) | right-wall standoff + forward |
| `go_to_goal` | open arena + goal | lidar(5)+goal rel(3) | ‖xy − goal‖ < 0.25 m for 20 ticks |
| `docking` | road + parallel bay | lidar(5)+dock rel(3) | in bay, ‖yaw − π/2‖ small |
| `maze` | obstacle maze | lidar(5)+exit bearing(3) | reach exit without crash |
| `figure8_tracking` | open arena (virtual path) | lidar(5)+track errs(3) | complete ≥1 loop with low crosstrack |

No new robot required — document bluntly if a future task needs one.

## Genesis (real adapter, optional dep)

```python
# pip install genesis-world   # heavy; CPU backend works for smoke
import genesis as gs
gs.init(backend=gs.cpu)  # or gs.cuda / gs.gpu
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=physics_dt, gravity=(0,0,-9.81)))
scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.URDF(file=str(urdf), pos=(0.3,0,0.05), fixed=False))
# walls: gs.morphs.Box(size=(sx,sy,sz), pos=..., fixed=True)
lidar = scene.add_sensor(gs.sensors.Lidar(
    pattern=gs.sensors.SphericalPattern(
        angles=([0.0, 45.0, -45.0, 90.0, -90.0], [0.0]),  # match MuJoCo planar rays
    ),
    entity_idx=robot.idx,
    pos_offset=(0.0, 0.0, 0.12),
    max_range=5.0,
))
scene.build()
robot.set_pos(...); robot.set_quat(...)  # kinematic parity with MuJoCo/PyBullet
scene.step()
dists = lidar.read().distances
```

**URDF note:** Our URDF is primitive boxes/cylinders (no mesh files) — Genesis URDF load should work. If a mesh-heavy URDF fails, document and fall back to Box proxy — blunt in TEST.

**Kinematic walls:** MuJoCo / PyBullet / Genesis all set base pose each control step (contacts are overwritten). Wall solidity is enforced by shared `resolve_wall_collision` against `WorldLayout.boxes` (same boxes as PyBullet/Genesis morphs / MuJoCo scene). `info["wall_contact"]` feeds crash penalty/termination — lidar alone is not enough to stop tunneling.

**Capability flags:** `urdf`, `lidar`, `optional_gpu`, `pip:genesis-world`, and `requires_install` when import fails.

## Isaac Lab / Isaac Sim (stubs)

| Registry name | What it is | Status |
|---|---|---|
| `isaaclab` | [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) RL / env framework on Isaac Sim | Stub — `requires_nvidia`, `requires_install`, `stub` |
| `isaac_sim` | Isaac Sim / Omniverse runtime itself | Stub — separate so UI can distinguish runtime vs Lab |

Launch either → `RuntimeError` with install URL + “not implemented in RoboLab yet”.

## Sim registry / UI

`GET /api/sims` → `{ "sims": [...], "meta": { "<name>": { "capabilities": [...] } } }`

`GET /api/tasks` → `{ "tasks": ["docking", "figure8_tracking", "go_to_goal", "maze", "wall_follow"] }`

Dropdown shows all sims; stub/uninstalled ones stay selectable and fail at launch with an actionable message.

## Links

- [Genesis docs](https://genesis-world.readthedocs.io/)
- [PyPI genesis-world](https://pypi.org/project/genesis-world/)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- See also `docs/SIMULATORS.md`, `docs/TASK_CANDIDATES.md`
