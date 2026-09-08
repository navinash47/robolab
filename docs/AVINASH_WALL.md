# Avinash Wall Follow (`avinash_wall`) + Q-learning Algo

Tabular wall-following policy from the Robotics course project report
**P2_D3** (Q-learning vs SARSA). Registered as builtin architecture
`avinash_wall` (UI label: **Avinash Wall Follow**).

The same PDF **task mode** (27-state bins for reward, 3 discrete actions)
is also available as **function-approx Q-learning** when New Run sets
**Algo = Q-learning** with Arch = `mlp` | `kan` | `kaf` | `gpkan` | `fan`.
Exploration ε follows training progress (`step / total_timesteps`), not a
fixed episode count — so a custom 2M-step run decays smoothly over the full run.

## World layout (largemaze)

`wall_follow` uses the Gazebo **`largemaze.world`** topology from the course
package (same segments as Fig. 4 scenarios: straight, inside-L, outside-L,
I-corner, 180° U-turn), scaled by **`WALL_LAYOUT_SCALE = 4.0`** in
`robolab/tasks/worlds.py`. Outer span ≈ **32 m × 32 m**. Shared `WorldLayout`
boxes drive MuJoCo `scene.xml`, PyBullet, and Genesis.

Additional tasks mirror Fig. 4 in isolation: `wall_straight`, `wall_l_inside`,
`wall_l_outside`, `wall_i_corner`, `wall_uturn` (same reward / obs / action).

Robot: RoboMaster-ish top silhouette (~0.55×0.42 m) in `robot.urdf` (same URDF
across sims). Render cameras are **top-down**.

PPO continuous reward targets **0.80 m** right standoff (PDF medium band).
Q-learning still uses PDF `pdf_reward`. See `docs/ARCH_HYPERPARAMS.md`.

After a layout/reward change: **retrain** and **re-render** — old checkpoints
were trained on the previous corridor / 0.35 m target.

## Algorithm chosen

**Q-learning** (not SARSA). The report finds both reach similar final J-values
(~6000), but Q-learning converges faster in the final phase, showed smoother
L-corner trajectories, and slightly shorter wall-clock. SARSA remains available
on the tabular path via `arch_cfg.algorithm: sarsa`.

## Spec (from the PDF — not invented)

| Piece | Value |
|-------|--------|
| State | `(front, right, left)` each Near / Medium / Far → **27** states |
| Bins | Near ≤0.7 m, Medium ≤0.9 m, Far >0.9 m |
| Actions | Turn left / Forward / Turn right; **v=0.3 m/s**, **ω=±0.7 or 0** |
| Reward | +20 right=medium; +15 turn-left when front near; −8 front near otherwise; −5 right=far; −1 right=near |
| α / γ | 0.1 / 1.0 (tabular α; neural FA uses Adam `trainer.lr`) |
| ε | **RoboLab default:** linear **1.0 → 0.05** over **100%** of `total_timesteps` (any custom budget, e.g. 2M). PDF reference schedule was episode-based (1.0→0.1 by −0.05/ep, hold to ep 200, then 0) and is not used by trainers. |
| Episode | 1200 steps (or env terminal) |

RoboLab maps 5-ray lidar front / right / left beams to the three PDF sectors.
Action signs use RoboLab’s convention (+angular = left/CCW) so the reward’s
“turning left” matches physical left turns.

## Two trainers

| New Run | Trainer | Checkpoint |
|---------|---------|------------|
| Arch **`avinash_wall`** (Algo forced to Q-learning) | Tabular `q_tabular.py` | `policy.zip` → `q_table.npy` |
| Arch **`kaf`/`kan`/`gpkan`/`fan`/`mlp`** + Algo **Q-learning** | FA `q_fa.py` (arch tower + Q head) | `policy.zip` → `q_fa.pt` |
| Same arches + Algo **PPO** | SB3 PPO (unchanged) | SB3 `policy.zip` |

Function-approx Q uses continuous lidar obs by default (`arch_cfg.obs_mode: lidar`);
set `obs_mode: onehot27` to feed a one-hot of the PDF state index instead.

## Launch on RunPod (wall_follow)

1. Commit + push so the worker can checkout a clean SHA.
2. Dashboard → **New Run**:
   - Compute: **runpod**
   - Arch: **Avinash Wall Follow (`avinash_wall`)** *or* KAF/KAN/… with Algo **Q-learning**
   - Task: **wall_follow**
   - Sim: **mujoco** or **pybullet** (gate sims)
   - Timesteps: **2000–5000** for smoke; full PDF budget is ~200×1200 ≈ **240k**
    steps — or set any custom total (ε scales to that budget). Ask before long trains.
3. Start run; watch Experiments until `COMPLETE`.

If **Avinash Wall Follow** is missing from the Arch dropdown, restart the API
(`make dev` / kill the process on `:8000`) and hard-refresh the dashboard
(Vite may cache the previous `/api/archs` payload).

CLI-shaped config sketch (tabular):

```yaml
sim: mujoco
task: wall_follow
arch: avinash_wall
arch_cfg:
  algorithm: q_learning
  alpha: 0.1
  gamma: 1.0
trainer:
  algo: q_learning
  timesteps: 5000
  lr: 0.1
  gamma: 1.0
  device: cuda
compute: runpod
```

CLI-shaped config sketch (KAF + Q-learning):

```yaml
sim: mujoco
task: wall_follow
arch: kaf
arch_cfg:
  hidden_sizes: [64, 64]
  num_grids: 8
  gamma: 1.0
trainer:
  algo: q_learning
  timesteps: 5000
  lr: 0.0003
  gamma: 1.0
  device: cuda
compute: runpod
```
