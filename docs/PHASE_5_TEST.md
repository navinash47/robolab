# Phase 5 human gate — PyBullet dual-sim

**Blunt status (2026-09-05):** Phase 5 **application code is in place** (PyBullet adapter loading the same `diffdrive_lidar` URDF, kinematic drive + `rayTest` lidar matching MuJoCo spaces, sim dropdown, `control_hz` / `physics_substeps` logged, run page shows `obs_dim` / `act_dim`).

**Local proof run:** PyBullet `wall_follow` mlp **2k** → run `80871d73ddd5` **COMPLETE**, `obs_dim=5`, `act_dim=2`, `control_hz=50`, `physics_substeps=5`, video **READY** (H.264 MP4 served at `/api/runs/80871d73ddd5/video`). MuJoCo COMPLETE run `23421c2da07b` reports the **same** obs/act dims (and domain timing from config) on the run page.

**Human gate NOT passed** — awaiting UI confirm: launch PyBullet wall_follow, render video, compare two videos (same robot, two engines), confirm identical spaces.

## Prerequisite: pybullet install (Darwin)

PyBullet’s bundled zlib breaks on Apple Clang 17 / Xcode 16+ unless `TARGET_OS_MAC` is suppressed ([bullet3#4753](https://github.com/bulletphysics/bullet3/issues/4753)). `make sync` sets this on Darwin:

```bash
make sync   # uses CFLAGS="-fno-define-target-os-macros" on macOS
# or manually:
CFLAGS="-fno-define-target-os-macros" uv sync --all-packages --python 3.11
```

Linux / RunPod: plain `uv sync` is fine.

Confirm:

```bash
uv run python -c "import pybullet as p; print(p.getAPIVersion())"
curl -s http://127.0.0.1:8000/api/sims
# → {"sims":["mujoco","pybullet"]}
```

## Exact clicks (dashboard)

1. `make sync && make dev` → open http://localhost:5173
2. **New Run**
3. Compute: **local**
4. Arch: **mlp**
5. Task: **wall_follow**
6. Sim: **pybullet** (dropdown — not mujoco)
7. Timesteps: **2k (quick)** for a smoke gate, or **50k** for a fuller train
8. **Start**
9. Experiments row: sim column shows `pybullet`; status → `COMPLETE`
10. Click the row (or open run detail): see **obs_dim: 5 · act_dim: 2 · control_hz: 50 · physics_substeps: 5**
11. **Render video** → player appears; robot moves in the corridor (PyBullet TinyRenderer)
12. Open a prior **mujoco** COMPLETE run (e.g. `23421c2da07b`) detail: **same obs_dim / act_dim**
13. Play both videos: same `diffdrive_lidar` robot, two engines (`mujoco` vs `pybullet`)

## API / CLI smoke (optional)

```bash
# Short train (same shape as New Run → pybullet → 2k)
curl -s -X POST http://127.0.0.1:8000/api/runs -H 'Content-Type: application/json' -d @- <<'EOF'
{"name":"wall_follow-mlp-pybullet-smoke","sim":"pybullet","task":"wall_follow","robot":"diffdrive_lidar","arch":"mlp","arch_cfg":{"hidden_sizes":[64,64],"activation":"tanh"},"trainer":{"algo":"ppo","timesteps":2048,"lr":0.0003,"batch_size":64,"n_steps":512,"n_envs":1,"gamma":0.99,"device":"cpu","seed":0},"seeds":[0],"domain":{"control_hz":50,"physics_substeps":5,"friction":1,"mass_scale":1,"sensor_noise_std":0,"action_delay_steps":0},"compute":"local","budget_usd":0}
EOF

# After COMPLETE:
curl -s -X POST http://127.0.0.1:8000/api/runs/<RUN_ID>/render
# Poll until video_status=READY, then open:
open http://localhost:5173
# or: curl -OJ http://127.0.0.1:8000/api/runs/<RUN_ID>/video
```

Config template: `configs/experiments/wall_follow_mlp_pybullet.yaml`

## Checklist

- [ ] `GET /api/sims` lists `mujoco` and `pybullet`
- [ ] New Run sim dropdown has both options
- [ ] PyBullet wall_follow run reaches **COMPLETE**
- [ ] Run page: **obs_dim=5**, **act_dim=2** (identical to MuJoCo wall_follow)
- [ ] Run page logs **control_hz** and **physics_substeps**
- [ ] **Render video** → READY → playable MP4
- [ ] Side-by-side: MuJoCo video + PyBullet video, same robot
- [ ] No `.env` / `videos/` / `checkpoints/` committed

## Known good (local proof)

| Item | Value |
|---|---|
| Run id | `80871d73ddd5` |
| Sim | `pybullet` |
| Spaces | obs 5 / act 2 |
| Timing | control_hz 50 / physics_substeps 5 |
| W&B | https://wandb.ai/avinashnandyala2-umass-amherst/robolab/runs/g4ckfr5a |
| Video | READY, ~1.3 MB H.264 |
| MuJoCo twin (spaces) | `23421c2da07b` → obs 5 / act 2 |

## Stop

**Do not start Phase 6 (transfer)** until the human confirms this gate.
