# Avinash Wall Follow (`avinash_wall`)

Tabular wall-following policy from the Robotics course project report
**P2_D3** (Q-learning vs SARSA). Registered as builtin architecture
`avinash_wall` (UI label: **Avinash Wall Follow**).

## Algorithm chosen

**Q-learning** (not SARSA). The report finds both reach similar final J-values
(~6000), but Q-learning converges faster in the final phase, showed smoother
L-corner trajectories, and slightly shorter wall-clock. SARSA remains available
via `arch_cfg.algorithm: sarsa`.

## Spec (from the PDF — not invented)

| Piece | Value |
|-------|--------|
| State | `(front, right, left)` each Near / Medium / Far → **27** states |
| Bins | Near ≤0.7 m, Medium ≤0.9 m, Far >0.9 m |
| Actions | Turn left / Forward / Turn right; **v=0.3 m/s**, **ω=±0.7 or 0** |
| Reward | +20 right=medium; +15 turn-left when front near; −8 front near otherwise; −5 right=far; −1 right=near |
| α / γ | 0.1 / 1.0 |
| ε | 1.0 → 0.1 by −0.05/episode, hold to episode 200, then 0 |
| Episode | 1200 steps (or env terminal) |

RoboLab maps 5-ray lidar front / right / left beams to the three PDF sectors.
Action signs use RoboLab’s convention (+angular = left/CCW) so the reward’s
“turning left” matches physical left turns.

Training uses a **dedicated tabular runner** (not SB3 PPO). Checkpoint
`policy.zip` stores `q_table.npy` + `meta.json`.

## Launch on RunPod (wall_follow)

1. Commit + push so the worker can checkout a clean SHA.
2. Dashboard → **New Run**:
   - Compute: **runpod**
   - Arch: **Avinash Wall Follow (`avinash_wall`)**
   - Task: **wall_follow**
   - Sim: **mujoco** or **pybullet** (gate sims)
   - Timesteps: **2000–5000** for smoke; full PDF budget is ~200×1200 ≈ **240k** steps — ask before long trains.
3. Start run; watch Experiments until `COMPLETE`.

CLI-shaped config sketch:

```yaml
sim: mujoco
task: wall_follow
arch: avinash_wall
arch_cfg:
  algorithm: q_learning
  alpha: 0.1
  gamma: 1.0
trainer:
  timesteps: 5000
  lr: 0.1
  gamma: 1.0
  device: cuda
compute: runpod
```
